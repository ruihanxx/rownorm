"""
Simple CIFAR-10 data loader that avoids torchvision compatibility issues
"""
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os


class SimpleCIFAR10Dataset(Dataset):
    """Simple CIFAR-10 dataset without torchvision dependency"""

    def __init__(self, data_dir, train=True, transform=None):
        self.transform = transform
        self.train = train

        # Load CIFAR-10 data
        if train:
            data_files = [f'data_batch_{i}' for i in range(1, 6)]
        else:
            data_files = ['test_batch']

        self.data = []
        self.targets = []

        for file in data_files:
            file_path = os.path.join(data_dir, 'cifar-10-batches-py', file)
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    batch = pickle.load(f, encoding='bytes')
                    self.data.append(batch[b'data'])
                    self.targets.extend(batch[b'labels'])

        if self.data:
            self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
            self.data = self.data.transpose((0, 2, 3, 1))  # Convert to HWC
        else:
            # Create dummy data if CIFAR-10 is not available
            print("Warning: CIFAR-10 data not found, creating dummy data")
            num_samples = 50000 if train else 10000
            self.data = np.random.randint(
                0, 256, (num_samples, 32, 32, 3), dtype=np.uint8)
            self.targets = np.random.randint(0, 10, num_samples).tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]

        # Convert to PIL Image format for transforms
        img = torch.from_numpy(img).float() / 255.0
        img = img.permute(2, 0, 1)  # Convert to CHW

        if self.transform:
            img = self.transform(img)

        return img, target


def get_simple_cifar10_dataloaders(data_dir: str, batch_size: int = 128, num_workers: int = 4):
    """Get CIFAR-10 dataloaders without torchvision dependency"""

    # Simple transforms
    def train_transform(x):
        # Normalize
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(-1, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(-1, 1, 1)
        x = (x - mean) / std
        return x

    def test_transform(x):
        # Normalize
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(-1, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(-1, 1, 1)
        x = (x - mean) / std
        return x

    # Create datasets
    train_dataset = SimpleCIFAR10Dataset(
        data_dir, train=True, transform=train_transform)
    test_dataset = SimpleCIFAR10Dataset(
        data_dir, train=False, transform=test_transform)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader
