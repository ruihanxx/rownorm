# Muon Optimizer Comparison Experiments

A comprehensive experimental framework for comparing different optimizers (AdamW, Muon, RowNorm) across multiple datasets and model architectures.

##  Overview

This project provides systematic benchmarking of optimizer performance on:
- **CIFAR-10** with ResNet18 (computer vision)
- **AG News text classification** with Transformer (NLP)

##  Experiments

### 1. CIFAR-10 Vision Experiment
- **Dataset**: CIFAR-10 (10-class image classification)
- **Model**: ResNet18 (modified for CIFAR-10)
- **Optimizers**: AdamW, Muon, RowNorm, SGD, SGDVariant
- **Training**: 200 epochs with cosine learning rate scheduling
- **Key Features**: Mixed precision training, gradient clipping, data augmentation

**Configuration**:
```bash
python train.py --preset auto
```

### 2. AG News Transformer Experiment
- **Dataset**: AG News (4-class news classification)
- **Model**: Base Transformer (4 layers, 8 heads, 256d)
- **Optimizers**: AdamW, Muon, RowNorm
- **Training**: 50 epochs with cosine scheduling + warmup
- **Batch Size**: 16 (for large model compatibility)

**Configuration**:
```bash
python run_ag_news_experiments.py
```

## Architecture

### Core Components
```
├── train.py                 # Main CIFAR-10 training script
├── train_transformer.py     # Transformer training script
├── eval.py                 # Model evaluation utilities
├── optim/                  # Custom optimizer implementations
│   ├── muon_wrap.py        # Muon optimizer wrapper
│   ├── rownorm.py          # RowNorm SGD implementation
│   ├── normmomentum.py     # Normalized momentum optimizer
│   └── ...                 # Other optimizer variants
├── models/                 # Model architectures
│   ├── resnet_cifar.py     # ResNet18 for CIFAR-10
│   └── simple_transformer.py # Transformer implementation
├── utils/                  # Utility functions
│   ├── common.py           # Data loading utilities
│   ├── metrics.py          # Performance metrics
│   └── logging.py          # Training logging utilities
└── configs/                # Configuration files
    └── cifar10_resnet18.yaml
```

### Experiment Runners

```
├── run_ag_news_experiments.py          # AG News experiment runner
├── create_ag_news_comparison_plots.py  # AG News visualization
└── analyze_ag_news_results.py          # Results analysis
```

## ⚙️ Optimizer Configurations

### Learning Rate Settings

| Experiment | AdamW | Muon | RowNorm | Notes |
|------------|-------|------|---------|-------|
| **CIFAR-10** | 3e-4 | 0.05 | 0.05 | Higher LR for simpler task |
| **AG News** | 3e-4 | 3e-4 | 3e-4 | Lower LR for larger model |

### Key Hyperparameters

| Optimizer | Weight Decay | Momentum | Special Settings |
|-----------|--------------|----------|------------------|
| **AdamW** | 0.01-0.05 | N/A | β₁=0.9, β₂=0.999 |
| **Muon** | 0.0-0.01 | 0.9-0.95 | Orthogonal updates |
| **RowNorm** | 5e-4-0.01 | 0.9 | Nesterov momentum |

## Results & Analysis

### Performance Metrics
- **Accuracy**: Classification accuracy on validation/test sets
- **Loss**: Cross-entropy loss curves
- **Training Speed**: Time per epoch and convergence rate
- **Memory Usage**: GPU memory efficiency

### Visualization
All experiments generate comprehensive plots:
- Training/validation loss curves
- Accuracy progression
- Learning rate schedules
- Performance comparison charts

Results are saved to:
- `./plots/` (CIFAR-10 results)
- `./plots_AG/` (AG News results)

## 🛠️ Installation & Setup

### Requirements
```bash
pip install torch torchvision matplotlib pandas seaborn numpy pyyaml
```

### Dataset Preparation
- **CIFAR-10**: Automatically downloaded via torchvision
- **AG News**: CSV files in `./data/ag_news/`

### Running Experiments

1. **CIFAR-10 Experiment**:
```bash
python train.py --preset auto
```

2. **AG News Experiment**:
```bash
python run_ag_news_experiments.py
```

4. **Generate Analysis Plots**:
```bash
python create_ag_news_comparison_plots.py
python generate_transformer_comparison.py
```

All experiments generate detailed logs:
- `*.log` files: Complete training logs
- `*_config.json`: Hyperparameter configurations  
- `*_curves.png`: Training visualization plots
- `*_stats.csv`: Performance statistics

## Quick Start

1. **Clone and setup**:
```bash
cd Muon_Experiment
pip install -r requirements.txt  # Create this if needed
```

2. **Run quick test**:
```bash
python train.py --epochs 5 --opt adamw  # Quick CIFAR test
```

3. **Full benchmark**:
```bash
bash run_all_experiments.sh  # Run all experiments (create this script)
```
