"""
Text Dataset Utils for IMDB and other text classification tasks
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from collections import Counter
import pickle
import os
from typing import List, Tuple, Dict, Optional


class TextVocabulary:
    """Text vocabulary"""

    def __init__(self, min_freq=2, max_size=10000):
        self.min_freq = min_freq
        self.max_size = max_size
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = {0: '<pad>', 1: '<unk>'}
        self.word_counts = Counter()

    def build_vocab(self, texts: List[str], tokenizer):
        """Build vocabulary"""
        print("Build vocabulary...")

        # Count word frequencies
        for text in texts:
            tokens = tokenizer(text)
            self.word_counts.update(tokens)

        # Sort by frequency, select high-frequency words
        most_common = self.word_counts.most_common(
            self.max_size - 2)  # -2 for pad and unk

        for word, count in most_common:
            if count >= self.min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

        print(f" Vocabulary built: {len(self.word2idx)} words")
        print(f"    Most frequent words: {most_common[:5]}")

    def encode(self, tokens: List[str]) -> List[int]:
        """Convert tokens to indices"""
        return [self.word2idx.get(token, 1) for token in tokens]  # 1 = <unk>

    def decode(self, indices: List[int]) -> List[str]:
        """Convert indices to tokens"""
        return [self.idx2word.get(idx, '<unk>') for idx in indices]

    def __len__(self):
        return len(self.word2idx)

    def save(self, path: str):
        """Save vocabulary"""
        with open(path, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_counts': self.word_counts,
                'min_freq': self.min_freq,
                'max_size': self.max_size
            }, f)

    def load(self, path: str):
        """加载词汇表"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.word2idx = data['word2idx']
            self.idx2word = data['idx2word']
            self.word_counts = data['word_counts']
            self.min_freq = data['min_freq']
            self.max_size = data['max_size']


class TextClassificationDataset(Dataset):
    """Text classification dataset"""

    def __init__(self, texts: List[str], labels: List[int], vocab: TextVocabulary,
                 tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize and encode
        tokens = self.tokenizer(text)
        token_ids = self.vocab.encode(tokens)

        # Truncate or pad
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids.extend(
                [0] * (self.max_length - len(token_ids)))  # 0 = <pad>

        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    texts, labels = zip(*batch)

    # Stack tensors
    # (max_len, batch_size) for Transformer
    texts = torch.stack(texts).transpose(0, 1)
    labels = torch.stack(labels)

    return texts, labels


class IMDBDatasetManager:
    """IMDB dataset manager"""

    def __init__(self, data_dir="./data", vocab_size=10000, max_length=512):
        self.data_dir = data_dir
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = None

        os.makedirs(data_dir, exist_ok=True)

    def prepare_data(self, force_rebuild=False):
        """Prepare IMDB dataset"""
        vocab_path = os.path.join(self.data_dir, 'imdb_vocab.pkl')

        if not force_rebuild and os.path.exists(vocab_path):
            print("Load existing vocabulary...")
            self.vocab = TextVocabulary()
            self.vocab.load(vocab_path)
            print(f" Vocabulary loaded: {len(self.vocab)} words")
        else:
            print("Download and process IMDB dataset...")

            # Download dataset
            train_iter = IMDB(split='train')
            test_iter = IMDB(split='test')

            # Collect all texts for building vocabulary
            all_texts = []
            train_data = []
            test_data = []

            print("Process training set...")
            for label, text in train_iter:
                train_data.append((text, 1 if label == 'pos' else 0))
                all_texts.append(text)

            print("Process test set...")
            for label, text in test_iter:
                test_data.append((text, 1 if label == 'pos' else 0))
                all_texts.append(text)

            # Build vocabulary
            self.vocab = TextVocabulary(min_freq=2, max_size=self.vocab_size)
            self.vocab.build_vocab(all_texts, self.tokenizer)

            # Save vocabulary
            self.vocab.save(vocab_path)

            # Save processed data
            train_path = os.path.join(self.data_dir, 'imdb_train.pkl')
            test_path = os.path.join(self.data_dir, 'imdb_test.pkl')

            with open(train_path, 'wb') as f:
                pickle.dump(train_data, f)
            with open(test_path, 'wb') as f:
                pickle.dump(test_data, f)

        return self._load_datasets()

    def _load_datasets(self):
        """Load datasets"""
        train_path = os.path.join(self.data_dir, 'imdb_train.pkl')
        test_path = os.path.join(self.data_dir, 'imdb_test.pkl')

        with open(train_path, 'rb') as f:
            train_data = pickle.load(f)
        with open(test_path, 'rb') as f:
            test_data = pickle.load(f)

        # Separate texts and labels
        train_texts, train_labels = zip(*train_data)
        test_texts, test_labels = zip(*test_data)

        # Create datasets
        train_dataset = TextClassificationDataset(
            list(train_texts), list(train_labels), self.vocab,
            self.tokenizer, self.max_length
        )

        test_dataset = TextClassificationDataset(
            list(test_texts), list(test_labels), self.vocab,
            self.tokenizer, self.max_length
        )

        print(f" Dataset statistics:")
        print(f"    Training set: {len(train_dataset)} samples")
        print(f"    Test set: {len(test_dataset)} samples")
        print(f"    Vocabulary: {len(self.vocab)} words")
        print(f"    Maximum length: {self.max_length}")

        return train_dataset, test_dataset

    def get_dataloaders(self, batch_size=32, num_workers=4):
        """Get data loader"""
        train_dataset, test_dataset = self.prepare_data()

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

        return train_loader, test_loader


def get_imdb_data(batch_size=32, vocab_size=10000, max_length=512, data_dir="./data"):
    """Convenient function: get IMDB data loader"""
    manager = IMDBDatasetManager(data_dir, vocab_size, max_length)
    train_loader, test_loader = manager.get_dataloaders(batch_size)

    return train_loader, test_loader, manager.vocab


if __name__ == "__main__":
    # Test dataset
    print("Test IMDB dataset...")

    try:
        train_loader, test_loader, vocab = get_imdb_data(
            batch_size=4, max_length=100)

        # Test a batch
        for texts, labels in train_loader:
            print(f" Dataset test successful!")
            print(f"    Text shape: {texts.shape}")  # (seq_len, batch_size)
            print(f"    Label shape: {labels.shape}")  # (batch_size,)
            print(f"    Vocabulary: {len(vocab)}")

            # Display first sample
            sample_text = texts[:, 0]  # First sample
            sample_tokens = vocab.decode(sample_text.tolist())
            print(f"    Sample preview: {' '.join(sample_tokens[:20])}...")
            break

    except Exception as e:
        print(f"Dataset test failed: {e}")
        print("This may be because the IMDB dataset needs to be downloaded")

