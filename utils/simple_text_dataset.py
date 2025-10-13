"""
Simple Text Dataset Utils (No torchtext dependency)
Only supports AG News dataset for news classification.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import re
import pickle
import os
import requests
import pandas as pd
from collections import Counter
from typing import List, Tuple, Dict, Optional


def simple_tokenizer(text: str) -> List[str]:
    """Simple English tokenizer"""
    # Convert to lowercase, remove punctuation, tokenize
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.split()


class SimpleVocabulary:
    """Simple vocabulary"""

    def __init__(self, min_freq=2, max_size=10000):
        self.min_freq = min_freq
        self.max_size = max_size
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = {0: '<pad>', 1: '<unk>'}
        self.word_counts = Counter()

    def build_vocab(self, texts: List[str]):
        """Build vocabulary"""
        print("🔨 Build vocabulary...")

        # Count word frequencies
        for text in texts:
            tokens = simple_tokenizer(text)
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
        """Load vocabulary"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.word2idx = data['word2idx']
            self.idx2word = data['idx2word']
            self.word_counts = data['word_counts']
            self.min_freq = data['min_freq']
            self.max_size = data['max_size']


class SimpleTextDataset(Dataset):
    """Simple text classification dataset"""

    def __init__(self, texts: List[str], labels: List[int], vocab: SimpleVocabulary,
                 max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize and encode
        tokens = simple_tokenizer(text)
        token_ids = self.vocab.encode(tokens)

        # Truncate or pad
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids.extend(
                [0] * (self.max_length - len(token_ids)))  # 0 = <pad>

        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def simple_collate_fn(batch):
    """Custom collate function for DataLoader"""
    texts, labels = zip(*batch)

    # Stack tensors
    # (max_len, batch_size) for Transformer
    texts = torch.stack(texts).transpose(0, 1)
    labels = torch.stack(labels)

    return texts, labels


class SimpleAGNewsManager:
    """AG News dataset manager (4 classes of news classification)"""

    def __init__(self, data_dir="./data", vocab_size=10000, max_length=256):
        self.data_dir = data_dir
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.vocab = SimpleVocabulary(max_size=vocab_size)

        # AG News class mapping
        self.class_names = ['World', 'Sports', 'Business', 'Sci/Tech']

        print(f" Initialize AG News data manager")
        print(f"    Data directory: {data_dir}")
        print(f"    Vocabulary: {vocab_size}")
        print(f"    Maximum length: {max_length}")

    def download_ag_news(self):
        """Download AG News dataset"""
        ag_news_dir = os.path.join(self.data_dir, "ag_news")
        os.makedirs(ag_news_dir, exist_ok=True)

        # AG News dataset download links
        urls = {
            'train': 'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv',
            'test': 'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv'
        }

        for split, url in urls.items():
            file_path = os.path.join(ag_news_dir, f"{split}.csv")

            if not os.path.exists(file_path):
                print(f"📥 Downloading {split} data...")
                try:
                    response = requests.get(url)
                    response.raise_for_status()

                    with open(file_path, 'wb') as f:
                        f.write(response.content)

                    print(f"✅ {split} data downloaded")
                except Exception as e:
                    print(f"❌ Error downloading {split} data: {e}")
                    # Create a small sample dataset for testing
                    self._create_sample_ag_news(file_path, split)
            else:
                print(f"📂 {split} data already exists")

    def _create_sample_ag_news(self, file_path, split):
        """Create a small sample AG News dataset for testing"""
        print(f"🔧 Creating sample {split} data...")

        # Sample data for each class
        sample_data = {
            1: [  # World
                "The United Nations held an emergency meeting today to discuss global climate change policies.",
                "International trade agreements are being renegotiated between major world powers.",
                "A new diplomatic initiative aims to resolve ongoing conflicts in the Middle East.",
            ],
            2: [  # Sports
                "The championship game ended in a thrilling overtime victory for the home team.",
                "Olympic athletes are preparing for the upcoming winter games with intensive training.",
                "A new world record was set in swimming at the international competition yesterday.",
            ],
            3: [  # Business
                "Tech stocks rose sharply following positive earnings reports from major companies.",
                "The Federal Reserve announced new interest rate policies affecting the banking sector.",
                "A startup company received significant funding for its innovative AI technology.",
            ],
            4: [  # Sci/Tech
                "Scientists discovered a new exoplanet that may harbor conditions suitable for life.",
                "Artificial intelligence breakthrough enables more accurate medical diagnoses.",
                "A revolutionary battery technology promises to extend electric vehicle range significantly.",
            ]
        }

        # Create CSV data
        rows = []
        for class_id, texts in sample_data.items():
            for text in texts:
                # Repeat samples to create larger dataset
                for _ in range(100 if split == 'train' else 20):
                    rows.append(
                        [class_id, f"Sample title for class {class_id}", text])

        # Save as CSV
        import csv
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)

        print(f"✅ Sample {split} data created with {len(rows)} samples")

    def load_ag_news_data(self):
        """Load AG News data"""
        self.download_ag_news()

        ag_news_dir = os.path.join(self.data_dir, "ag_news")
        train_path = os.path.join(ag_news_dir, "train.csv")
        test_path = os.path.join(ag_news_dir, "test.csv")

        def load_csv(file_path):
            texts, labels = [], []
            try:
                df = pd.read_csv(file_path, header=None, names=[
                                 'label', 'title', 'description'])

                # Combine title and description
                for _, row in df.iterrows():
                    text = f"{row['title']} {row['description']}"
                    texts.append(text)
                    # Convert to 0-based indexing
                    labels.append(int(row['label']) - 1)

            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
                # Fallback to manual CSV parsing
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split(',', 2)
                        if len(parts) >= 3:
                            label = int(parts[0]) - 1  # Convert to 0-based
                            title = parts[1].strip('"')
                            description = parts[2].strip('"')
                            text = f"{title} {description}"
                            texts.append(text)
                            labels.append(label)

            return texts, labels

        train_texts, train_labels = load_csv(train_path)
        test_texts, test_labels = load_csv(test_path)

        print(f"📊 AG News data loaded:")
        print(f"    Training: {len(train_texts)} samples")
        print(f"    Test: {len(test_texts)} samples")
        print(f"    Classes: {self.class_names}")

        return train_texts, train_labels, test_texts, test_labels

    def prepare_data(self, force_rebuild=False):
        """Prepare AG News dataset"""
        vocab_path = os.path.join(self.data_dir, 'ag_news_vocab.pkl')

        # Load data
        train_texts, train_labels, test_texts, test_labels = self.load_ag_news_data()

        # Build or load vocabulary
        if not force_rebuild and os.path.exists(vocab_path):
            print("📂 Loading existing vocabulary...")
            self.vocab.load(vocab_path)
            print(f"✅ Vocabulary loaded: {len(self.vocab)} words")
        else:
            print("🔨 Building new vocabulary...")
            self.vocab.build_vocab(train_texts + test_texts)
            self.vocab.save(vocab_path)
            print(f"💾 Vocabulary saved to {vocab_path}")

        # Create datasets
        train_dataset = SimpleTextDataset(
            train_texts, train_labels, self.vocab, self.max_length
        )
        test_dataset = SimpleTextDataset(
            test_texts, test_labels, self.vocab, self.max_length
        )

        print(f"✅ Datasets created:")
        print(f"    Training: {len(train_dataset)} samples")
        print(f"    Test: {len(test_dataset)} samples")
        print(f"    Vocabulary: {len(self.vocab)} words")
        print(f"    Max length: {self.max_length}")

        return train_dataset, test_dataset

    def get_dataloaders(self, batch_size=32, num_workers=2):
        """Get data loaders"""
        train_dataset, test_dataset = self.prepare_data()

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=simple_collate_fn,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=simple_collate_fn,
            pin_memory=True
        )

        return train_loader, test_loader


def get_simple_ag_news_data(batch_size=32, vocab_size=10000, max_length=256, data_dir="./data"):
    """Convenient function: get AG News data loader"""
    manager = SimpleAGNewsManager(data_dir, vocab_size, max_length)
    train_loader, test_loader = manager.get_dataloaders(batch_size)

    # Create validation loader by splitting train data
    from torch.utils.data import random_split
    train_dataset = train_loader.dataset
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_split, val_split = random_split(
        train_dataset, [train_size, val_size])

    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_split,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=simple_collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_split,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=simple_collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, manager.vocab


# For backward compatibility - remove IMDB references
def get_simple_imdb_data(*args, **kwargs):
    """IMDB support has been removed. Use AG News instead."""
    raise NotImplementedError(
        "IMDB dataset support has been removed. Please use get_simple_ag_news_data() instead.")
