"""
Simple Transformer for Text Classification
A lightweight Transformer implementation for text classification tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional Encoding"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerBlock(nn.Module):
    """Single Transformer Block"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # Self-attention
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feed forward
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class SimpleTransformerClassifier(nn.Module):
    """Simple Transformer Text Classifier"""

    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=2,
                 num_classes=2, max_len=512, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        # Create embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, d_model * 4, dropout)
            for _ in range(num_layers)
        ])

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0, 0.1)

    def forward(self, src, src_mask=None):
        # src: (seq_len, batch_size)
        seq_len, batch_size = src.size()

        # Embedding + positional encoding
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src = self.dropout(src)

        # Transformer layers
        for layer in self.transformer_layers:
            src = layer(src, src_mask)

        # Global average pooling
        # src: (seq_len, batch_size, d_model) -> (batch_size, d_model)
        src = src.mean(dim=0)

        # Classification
        output = self.classifier(src)

        return output


def create_padding_mask(seq, pad_token=0):
    """创建padding mask"""
    # seq: (batch_size, seq_len)
    mask = (seq == pad_token).transpose(0, 1)  # (seq_len, batch_size)
    return mask


# 模型配置字典
TRANSFORMER_CONFIGS = {
    'tiny': {
        'd_model': 64,
        'nhead': 4,
        'num_layers': 2,
        'dropout': 0.1
    },
    'small': {
        'd_model': 128,
        'nhead': 8,
        'num_layers': 3,
        'dropout': 0.1
    },
    'base': {
        'd_model': 256,
        'nhead': 8,
        'num_layers': 4,
        'dropout': 0.1
    }
}


def get_transformer_model(config_name='small', vocab_size=10000, num_classes=2):
    """Get a Transformer model with predefined configurations"""
    if config_name not in TRANSFORMER_CONFIGS:
        raise ValueError(
            f"Unknown config: {config_name}. Available: {list(TRANSFORMER_CONFIGS.keys())}")

    config = TRANSFORMER_CONFIGS[config_name]

    model = SimpleTransformerClassifier(
        vocab_size=vocab_size,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        num_classes=num_classes,
        dropout=config['dropout']
    )

    return model


if __name__ == "__main__":
    # Testing model
    print("🔍 Testing Transformer model...")

    # Creating a small model
    model = get_transformer_model('small', vocab_size=5000, num_classes=2)

    # Creating test data
    batch_size, seq_len = 4, 100
    test_input = torch.randint(1, 5000, (seq_len, batch_size))

    # Forward propagation
    with torch.no_grad():
        output = model(test_input)

    print(f"✅ Model testing successful!")
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(
        f"   Parameter count: {sum(p.numel() for p in model.parameters()):,}")

    # Displaying parameter counts for different configurations
    print("\n📊 Parameter count comparison for different configurations:")
    for config_name in TRANSFORMER_CONFIGS:
        model = get_transformer_model(config_name, vocab_size=5000)
        params = sum(p.numel() for p in model.parameters())
        print(f"   {config_name:5s}: {params:,} parameters")
