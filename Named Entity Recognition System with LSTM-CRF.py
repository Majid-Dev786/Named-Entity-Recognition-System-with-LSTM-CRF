import torch
import torch.nn as nn
from TorchCRF import CRF
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import random
import string

# Defining my LSTM-CRF model class for Named Entity Recognition tasks
class LSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim):
        super(LSTM_CRF, self).__init__()
        # Here, I'm creating an embedding layer to convert vocab indices into dense vectors of fixed size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM layer for learning the sequence, bidirectional to capture patterns from both directions
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        # Linear layer to map the output of LSTM into the tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        # The CRF layer for predicting the tag sequence in a way that considers the sequence context
        self.crf = CRF(tagset_size)

    # Forward pass computes the loss based on the inputs and their true tags
    def forward(self, sentences, sentence_masks, tags, tag_masks):
        lstm_feats = self._get_lstm_features(sentences)
        # CRF layer calculates the loss between predicted and true sequences
        loss = self.crf(lstm_feats, tags, mask=tag_masks.bool())
        return loss.mean()

    # For prediction, given a sentence, predict the tag sequence
    def predict(self, sentences, sentence_masks):
        lstm_feats = self._get_lstm_features(sentences)
        return self.crf.decode(lstm_feats, mask=sentence_masks.bool())

    # Helper function to get LSTM features for a sentence
    def _get_lstm_features(self, sentences):
        embeds = self.embedding(sentences)
        lstm_out, _ = self.lstm(embeds)
        return self.hidden2tag(lstm_out)

# Dataset class for handling NER data
class NERDataset(Dataset):
    def __init__(self, sentences, tags, max_length):
        # Preprocessing sentences and tags by padding them to a max length
        self.sentences = [torch.tensor(sentence) for sentence in sentences]
        self.sentences, self.sentence_masks = zip(*[self._pad_sequence(sentence, max_length) for sentence in self.sentences])
        self.tags = [tag.clone().detach() if torch.is_tensor(tag) else torch.tensor(tag) for tag in tags]
        self.tags, self.tag_masks = zip(*[self._pad_sequence(tag, max_length) for tag in self.tags])

    # Returns the total number of samples
    def __len__(self):
        return len(self.sentences)

    # Allows indexing so the dataset can be used in DataLoader
    def __getitem__(self, idx):
        return self.sentences[idx], self.sentence_masks[idx], self.tags[idx], self.tag_masks[idx]

    # Helper function for padding sequences to a fixed length
    def _pad_sequence(self, sequence, max_length):
        sequence = sequence.clone().detach()
        mask = torch.ones(max_length, dtype=torch.bool)
        # Padding logic
        if len(sequence) < max_length:
            sequence = torch.cat([sequence, torch.zeros(max_length - len(sequence), dtype=torch.long)])
            mask[len(sequence):] = 0
        elif len(sequence) > max_length:
            sequence = sequence[:max_length]
            mask = mask[:max_length]
        return sequence, mask

# Main NER class for setting up, training, and evaluating the model
class NER:
    def __init__(self, num_sentences=10, embedding_dim=10, hidden_dim=20, epochs=10, batch_size=32, test_size=0.2):
        # Initializing model parameters and generating dummy data
        self.num_sentences = num_sentences
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.test_size = test_size
        self.sentences, self.tags = self.generate_data(self.num_sentences)

    # Generates dummy sentences and tags for testing purposes
    def generate_data(self, num_sentences):
        sentences = [self.generate_random_sentence() for _ in range(num_sentences)]
        tags = [self.generate_random_tags(len(sentence)) for sentence in sentences]
        return sentences, tags

    # Helper to generate a random sentence
    def generate_random_sentence(self):
        return [''.join(random.choice(string.ascii_letters) for _ in range(random.randint(2, 5))) for _ in range(random.randint(3, 8))]

    # Helper to generate a random tag sequence
    def generate_random_tags(self, sentence_length):
        return [random.randint(0, 2) for _ in range(sentence_length)]

    # Prepares data for training by creating mappings from words/tags to indices
    def prepare_data(self):
        word2idx = {word: i+1 for i, word in enumerate(set(word for sentence in self.sentences for word in sentence))}
        word2idx["X"] = 0  # Adding a default value for unknown words
        idx2word = {i: word for word, i in word2idx.items()}

        tag2idx = {tag: i for i, tag in enumerate(set(tag for tag_sequence in self.tags for tag in tag_sequence))}
        idx2tag = {i: tag for tag, i in tag2idx.items()}

        sentences = [[word2idx[word] for word in sentence] for sentence in self.sentences]
        tags = [torch.tensor(tag_sequence) for tag_sequence in self.tags]

        return word2idx, idx2word, tag2idx, idx2tag, sentences, tags

    # Main function to train the model and evaluate it
    def run(self):
        word2idx, idx2word, tag2idx, idx2tag, sentences, tags = self.prepare_data()
        sentences_train, sentences_val, tags_train, tags_val = train_test_split(sentences, tags, test_size=self.test_size)

        model = LSTM_CRF(len(word2idx), len(tag2idx), self.embedding_dim, self.hidden_dim)
        
        max_length = max(len(sentence) for sentence in sentences)
        train_data = NERDataset(sentences_train, tags_train, max_length)
        val_data = NERDataset(sentences_val, tags_val, max_length)
        train_loader = DataLoader(train_data, batch_size=self.batch_size)
        val_loader = DataLoader(val_data, batch_size=self.batch_size)

        optimizer = Adam(model.parameters())
        for epoch in range(self.epochs):
            for sentences, sentence_masks, tags, tag_masks in train_loader:
                model.zero_grad()
                loss = model(sentences, sentence_masks, tags, tag_masks)
                loss.backward()
                optimizer.step()

        self.validate_model(model, val_loader, idx2word, idx2tag)

    # Validates the model using the validation dataset and prints out predictions
    def validate_model(self, model, val_loader, idx2word, idx2tag):
        with torch.no_grad():
            for sentences, sentence_masks, tags, tag_masks in val_loader:
                predicted_tags = model.predict(sentences, sentence_masks)
                sentences = [[idx2word[idx] for idx in sentence] for sentence in sentences.tolist()]
                predicted_tags = [[idx2tag[idx] for idx in sequence] for sequence in predicted_tags]
                true_tags = [[idx2tag[idx] for idx in sequence.tolist()] for sequence in tags]
                for sentence, true, predicted in zip(sentences, true_tags, predicted_tags):
                    print(f"Sentence: {' '.join(sentence)}")
                    print(f"Predicted tags: {predicted}")

if __name__ == "__main__":
    ner = NER()
    ner.run()
