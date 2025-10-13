"""
Transformer Training Script for Text Classification
"""

import argparse
import os
import csv
import json
import random
import yaml
from pathlib import Path
from datetime import datetime
import math

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_

# Optional plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # headless
    import matplotlib.pyplot as plt
    _MATPLOTLIB_OK = True
except Exception:
    _MATPLOTLIB_OK = False

# Import our modules
from models.simple_transformer import get_transformer_model, TRANSFORMER_CONFIGS
from utils.simple_text_dataset import get_simple_ag_news_data
from utils.metrics import accuracy
from utils.logging import SmoothedValue, Timer
from optim.rownorm import RowNormSGD
from optim.signsgd import SignSGD
from optim.muon_wrap import build_single_device_muon_with_aux_adam
from optim.normmomentum import NormMomentum
from optim.muon_v2 import MuonV2
from optim.sgd_variant import SGDVariant


def build_weight_decay_param_groups(model: nn.Module, weight_decay: float):
    """Apply weight decay to weight matrices (ndim>=2). Disable decay for biases/BN."""
    decay_params = []
    no_decay_params = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            decay_params.append(p)
        else:
            no_decay_params.append(p)

    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    return param_groups


def get_optimizer(model, name: str, lr: float, weight_decay: float, momentum: float, nesterov_mom: float, alpha: float = 0.1, gamma: float = 0.0, norm_pq=(2,torch.inf)):
    """Create optimizer based on name"""
    name = name.lower()

    if name == "muon":
        return build_single_device_muon_with_aux_adam(model, muon_lr=lr, muon_weight_decay=weight_decay, muon_momentum=momentum)

    param_groups = build_weight_decay_param_groups(model, weight_decay)

    if name == "sgd":
        return optim.SGD(param_groups, lr=lr, momentum=momentum, nesterov=nesterov_mom)
    if name == "rownorm":
        return RowNormSGD(param_groups, lr=lr, momentum=momentum, weight_decay=0.0, nesterov_mom=nesterov_mom, max_grad_norm=1.0, p_exp=norm_pq[0], q_exp = norm_pq[1])
    if name == "signsgd":
        return SignSGD(param_groups, lr=lr, momentum=momentum, weight_decay=0.0)
    if name == "muonv2":
        return MuonV2(param_groups, lr=lr, beta=1-alpha, gamma=gamma, weight_decay=0.0)
    if name == "sgdvariant":
        return SGDVariant(param_groups, lr=lr, momentum=momentum, weight_decay=0.0)
    if name == "adamw":
        return optim.AdamW(param_groups, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def train_one_epoch(model, loader, optimizer, device, scaler: GradScaler, max_grad_norm: float = 1.0, amp_enabled: bool = True):
    """Train for one epoch"""
    model.train()
    ce = nn.CrossEntropyLoss()
    loss_meter = SmoothedValue()
    top1_meter = SmoothedValue()
    timer = Timer()

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        if amp_enabled:
            with autocast():
                outputs = model(inputs)
                loss = ce(outputs, targets)
            scaler.scale(loss).backward()
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = ce(outputs, targets)
            loss.backward()
            if max_grad_norm > 0:
                clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        acc = accuracy(outputs, targets)[0]
        loss_meter.update(loss.item(), inputs.size(0))
        top1_meter.update(acc.item(), inputs.size(0))

    return loss_meter.avg, top1_meter.avg, timer.elapsed()


def evaluate(model, loader, device):
    """Evaluate model"""
    model.eval()
    ce = nn.CrossEntropyLoss()
    loss_meter = SmoothedValue()
    top1_meter = SmoothedValue()

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = ce(outputs, targets)
            acc = accuracy(outputs, targets)[0]

            loss_meter.update(loss.item(), inputs.size(0))
            top1_meter.update(acc.item(), inputs.size(0))

    return loss_meter.avg, top1_meter.avg


def cosine_scheduler(optimizer, epoch, total_epochs, base_lr, min_lr, warmup_epochs):
    """Cosine learning rate scheduler with warmup"""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr = min_lr + (base_lr - min_lr) * 0.5 * \
            (1 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description='Transformer Text Classification Training')

    # Basic arguments
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--data_dir', type=str,
                        default=str(Path(__file__).resolve().parent/'data'/'agnews'))
    parser.add_argument('--model', type=str, default='transformer_small',
                        choices=['transformer_tiny', 'transformer_small', 'transformer_base'])
    parser.add_argument('--dataset', type=str,
                        default='agnews', choices=['agnews'])

    # Training parameters
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov_mom', type=float, default=0.0)

    # Optimizer
    parser.add_argument('--opt', type=str, default='adamw',
                        choices=['sgd', 'adamw', 'muon', 'muonv2', 'rownorm', 'sgdvariant'])

    # Scheduler
    parser.add_argument('--sched', type=str, default='cosine',
                        choices=['cosine', 'none'])
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--min_lr', type=float, default=0.0)

    # Regularization
    parser.add_argument('--clip', type=float, default=1.0,
                        help='Gradient clipping')
    parser.add_argument('--dropout', type=float, default=0.1)

    # Data parameters
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--max_length', type=int, default=512)

    # Hardware
    parser.add_argument('--amp', action='store_true',
                        help='Use automatic mixed precision')
    parser.add_argument('--num_workers', type=int, default=4)

    # Logging
    parser.add_argument('--out_dir', type=str, default='./logs')
    parser.add_argument('--plot_dir', type=str, default='./plots')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--p_exp', type=int,
                        default=2)
    parser.add_argument('--q_exp', type=int,
                        default=torch.inf)
    parser.add_argument('--trials', type=int,
                        default=1)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.0)
    args = parser.parse_args()
    args.normpq = [args.p_exp,args.q_exp]
    
    # Load config if provided
    if args.config:
        config = load_config(args.config)
        for key, value in config.items():
            if hasattr(args, key) and getattr(args, key) is None:
                setattr(args, key, value)

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f" Using device: {device}")

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Load dataset
    print(f"📥 Loading {args.dataset} dataset...")
    if args.dataset == 'agnews':
        train_loader,val_loader, test_loader, vocab = get_simple_ag_news_data(
            batch_size=args.batch_size,
            vocab_size=args.vocab_size,
            max_length=args.max_length,
            data_dir=args.data_dir
        )
        num_classes = 4  # AG News has 4 classes
        actual_vocab_size = len(vocab)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(
        f" Dataset loaded: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test")

    # Create model
    print(f" Creating {args.model} model...")
    model_config = args.model.replace('transformer_', '')
    model = get_transformer_model(
        config_name=model_config,
        vocab_size=actual_vocab_size,
        num_classes=num_classes
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f" Model created: {total_params:,} parameters")

    # Create optimizer
    optimizer = get_optimizer(
        model, args.opt, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov_mom=args.nesterov_mom, alpha=args.alpha, gamma=args.gamma, norm_pq=args.normpq)

    scaler = GradScaler(enabled=(device == 'cuda' and args.amp))

    # Prepare logging
    default_run_name = (
        f"{args.dataset}_{args.model}_{args.opt}_bs{args.batch_size}_lr{args.lr}_"
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    run_name = args.run_name or default_run_name

    out_dir = Path(args.out_dir)
    plot_dir = Path(args.plot_dir)
    out_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    log_file = out_dir / f"{run_name}.csv"
    if not args.trials:
        # Save config
        config_file = out_dir / f"{run_name}_config.json"
        with open(config_file, 'w') as f:
            json.dump(vars(args), f, indent=2)

    # Training loop
    print(f" Starting training for {args.epochs} epochs...")

    history = []
    best_val_acc = 0.0

    with open(log_file, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'lr', 'train_loss',
                      'train_acc', 'val_loss', 'val_acc', 'time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(args.epochs):
            # Adjust learning rate
            if args.sched == 'cosine':
                lr_epoch = cosine_scheduler(
                    optimizer, epoch, args.epochs, args.lr, args.min_lr, args.warmup_epochs
                )
            else:
                lr_epoch = args.lr

            # Train
            train_loss, train_acc, train_time = train_one_epoch(
                model, train_loader, optimizer, device, scaler,
                max_grad_norm=args.clip, amp_enabled=args.amp
            )

            # Evaluate
            val_loss, val_acc = evaluate(model, test_loader, device)

            # Log
            print(f"Epoch {epoch+1:3d}/{args.epochs}: lr={lr_epoch:.5f} "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% time={train_time:.1f}s")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), out_dir /
                           f"{run_name}_best.pth")

            # Save checkpoint
            if (epoch + 1) % args.save_freq == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                }, out_dir / f"{run_name}_checkpoint_epoch_{epoch+1}.pth")

            # Log to CSV
            if not args.trials :
                writer.writerow({
                    'epoch': epoch + 1,
                    'lr': lr_epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'time': train_time
                })
                csvfile.flush()

            # Store for plotting
            history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            })

    print(
        f" Training completed! Best validation accuracy: {best_val_acc:.2f}%")

    # Generate plots
    if _MATPLOTLIB_OK and history and not args.trials:
        print(" Generating plots...")

        epochs = [h['epoch'] for h in history]
        train_losses = [h['train_loss'] for h in history]
        val_losses = [h['val_loss'] for h in history]
        train_accs = [h['train_acc'] for h in history]
        val_accs = [h['val_acc'] for h in history]

        # Loss plot
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, 'b-', label='Train')
        plt.plot(epochs, val_losses, 'r-', label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss')
        plt.legend()
        plt.grid(True)

        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accs, 'b-', label='Train')
        plt.plot(epochs, val_accs, 'r-', label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(plot_dir / f"{run_name}_curves.png",
                    dpi=150, bbox_inches='tight')
        plt.close()

        print(f" Plots saved to {plot_dir}")

    print(f"Logs saved to {log_file}")
    print(f"Best model saved to {out_dir / f'{run_name}_best.pth'}")


if __name__ == "__main__":
    main()
