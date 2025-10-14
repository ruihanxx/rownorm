import argparse
import os
import csv
import json
import random
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

from models.resnet_cifar import resnet18_cifar
from utils.common import get_cifar10_dataloaders
from utils.metrics import accuracy
from utils.logging import SmoothedValue, Timer
from optim.rownorm import RowNormSGD
from optim.signsgd import SignSGD
from optim.muon_wrap import build_single_device_muon_with_aux_adam
from optim.normmomentum import NormMomentum
from optim.muon_v2 import MuonV2
from optim.sgd_variant import SGDVariant
from optim.adamw_rescale import AdamWScale


def build_weight_decay_param_groups(model: nn.Module, weight_decay: float):
    """
    Apply weight decay to weight matrices (ndim>=2). Disable decay for biases/BN.
    """
    decay_params = []
    no_decay_params = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            decay_params.append(p)
        else:
            no_decay_params.append(p)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

def linear_decay_scheduler(optimizer, total_epochs, warmup_epochs=0, min_lr=0.0):
    total = max(1, int(total_epochs))
    warm = max(0, int(warmup_epochs))

    if total <= warm and warm > 0:
        # 只有 warmup
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=(1.0 / float(max(1, warm))),
            end_factor=1.0,
            total_iters=warm
        )

    schedulers = []
    milestones = []

    if warm > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=(1.0 / float(max(1, warm))),
            end_factor=1.0,
            total_iters=warm
        )
        schedulers.append(warmup)
        milestones.append(warm)

    decay_epochs = max(1, total - warm)

    # 为每个 param group 精确算出从 base_lr 线性到 min_lr 的倍率函数
    base_lrs = [pg['lr'] for pg in optimizer.param_groups]
    min_factors = [float(min_lr) / float(b) for b in base_lrs]

    def make_linear_fn(factor_min):
        # epoch 从 0 到 decay_epochs
        def fn(epoch):
            t = float(epoch) / float(max(1, decay_epochs))
            # 线性插值：1 -> factor_min
            return (1.0 - t) * 1.0 + t * factor_min
        return fn

    lambdas = [make_linear_fn(fm) for fm in min_factors]
    linear = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)
    schedulers.append(linear)

    if warm > 0:
        from torch.optim.lr_scheduler import SequentialLR
        return SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)
    else:
        return linear

def two_stage_linear_scheduler(optimizer, total_epochs, warmup_epochs=0, min_lr=0.0, k=1.0):

    assert k > 0 
    total = max(1, int(total_epochs))
    warm = max(0, int(warmup_epochs))

    # 只有 warmup
    if total <= warm and warm > 0:
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=(1.0 / float(max(1, warm))),
            end_factor=1.0,
            total_iters=warm
        )

    schedulers, milestones = [], []

    # warmup（可选）
    if warm > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=(1.0 / float(max(1, warm))),
            end_factor=1.0,
            total_iters=warm
        )
        schedulers.append(warmup)
        milestones.append(warm)

    # 两段式线性衰减
    T = max(1, total - warm)             # 衰减总长度（epoch 数）
    T1 = T // 2                          # 前半段时长
    T2 = T - T1                          # 后半段时长（保证 T1+T2=T，奇数时后半多 1）

    base_lrs = [pg['lr'] for pg in optimizer.param_groups]
    # 目标末端倍率（<=1），允许 min_lr=0
    fmins = [float(min_lr) / float(b) if b > 0 else 0.0 for b in base_lrs]
    # 每个 group 的总下降量 D = 1 - fmin
    Ds = [max(0.0, 1.0 - fmin) for fmin in fmins]

    # 斜率约束： (d1/T1) = k * (d2/T2)，且 d1 + d2 = D
    # => d2 = D / (1 + k*T1/T2)；d1 = D - d2
    # 需要考虑 T1 或 T2 可能为 0 的边界（当 T=1 时）
    def split_d(D, T1, T2, k):
        if T1 == 0:      # 全在后半段
            return 0.0, D
        if T2 == 0:      # 全在前半段
            return D, 0.0
        d2 = D / (1.0 + k * (float(T1) / float(T2)))
        d1 = D - d2
        return d1, d2

    d_pairs = [split_d(D, T1, T2, k) for D in Ds]

    def make_piecewise_fn(d1, d2, T1, T2):
        # 分段线性：factor(0)=1，前半降 d1，后半再降 d2，末端 1-(d1+d2)=fmin
        s1 = (d1 / max(1, T1)) if T1 > 0 else 0.0  # 前半每个 epoch 下跌量
        s2 = (d2 / max(1, T2)) if T2 > 0 else 0.0  # 后半每个 epoch 下跌量
        def fn(epoch):  # epoch: 0..T-1
            if T1 > 0 and epoch < T1:
                return 1.0 - s1 * float(epoch)
            # 第二段：从 1-d1 继续降
            e2 = float(epoch - T1)
            return 1.0 - d1 - s2 * max(0.0, e2)
        return fn

    lambdas = [make_piecewise_fn(d1, d2, T1, T2) for (d1, d2) in d_pairs]
    decay = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)
    schedulers.append(decay)

    if warm > 0:
        from torch.optim.lr_scheduler import SequentialLR
        return SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)
    else:
        return decay

def get_optimizer(model, name: str, lr: float, weight_decay: float, momentum: float, nesterov_mom: float, alpha: float = 0.1, gamma: float = 0.0, scale_type: str = "none", norm_pq=(2,torch.inf)):
    name = name.lower()
    if name == "muon":
        return build_single_device_muon_with_aux_adam(model, muon_lr=lr, muon_weight_decay=weight_decay, muon_momentum=momentum)

    param_groups = build_weight_decay_param_groups(model, weight_decay)
    if name == "sgd":
        return optim.SGD(param_groups, lr=lr, momentum=momentum, nesterov=nesterov_mom)
    if name == "rownorm":
        return RowNormSGD(param_groups, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov_mom=nesterov_mom, max_grad_norm=1.0, p_exp=norm_pq[0], q_exp = norm_pq[1])
    if name == "signsgd":
        return SignSGD(param_groups, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adamw":
        return optim.AdamW(param_groups, lr=lr)
    if name == "normmom":
        return NormMomentum(param_groups, lr=lr, alpha=alpha, gamma=gamma)
    if name == "muonv2":
        return MuonV2(param_groups, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov_mom)
    if name == "sgdvariant":
        return SGDVariant(param_groups, lr=lr, momentum=momentum, weight_decay=0.0)
    if name == "adamw_rescale":
        return AdamWScale(param_groups, lr=lr, betas=(momentum, 1-(1-momentum)**2), weight_decay=weight_decay, p_exp=norm_pq[0], q_exp = norm_pq[1])
    raise ValueError(f"Unknown optimizer: {name}")


def train_one_epoch(model, loader, optimizer, device, scaler: GradScaler, max_grad_norm: float = 1.0, amp_enabled: bool = True):
    model.train()
    ce = nn.CrossEntropyLoss()
    loss_meter = SmoothedValue()
    top1_meter = SmoothedValue()
    timer = Timer()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp_enabled):
            outputs = model(images)
            loss = ce(outputs, targets)
        scaler.scale(loss).backward()
        if max_grad_norm is not None and max_grad_norm > 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        acc1, = accuracy(outputs, targets, topk=(1,))
        loss_meter.update(loss.item(), images.size(0))
        top1_meter.update(acc1.item(), images.size(0))

    return loss_meter.avg, top1_meter.avg, timer.elapsed()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    loss_meter = SmoothedValue()
    top1_meter = SmoothedValue()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = ce(outputs, targets)
        acc1, = accuracy(outputs, targets, topk=(1,))
        loss_meter.update(loss.item(), images.size(0))
        top1_meter.update(acc1.item(), images.size(0))
    return loss_meter.avg, top1_meter.avg
# >>> 新增：枚举指定前缀（如 'layer2'）的参数
def _iter_layer_params(model: nn.Module, prefix: str):
    for name, p in model.named_parameters():
        if name.startswith(prefix) and p.requires_grad:
            yield name, p

# >>> 新增：保存快照（权重 + 梯度）到 .pt，并可选写一行 CSV 统计
def _save_layer_snapshot(model: nn.Module, epoch: int, out_dir: Path, run_name: str,
                         layer_prefix: str, write_stats_csv: bool = False):
    # 1) 打包张量
    payload = {}
    for name, p in _iter_layer_params(model, layer_prefix):
        payload[name] = {
            "weight": p.detach().cpu(),
            "grad": (p.grad.detach().cpu() if (p.grad is not None) else None),
            "shape": tuple(p.shape),
            "dtype": str(p.dtype)
        }

    # 2) 保存为 .pt（单个大文件，便于完整回放）
    snap_dir = out_dir / f"{run_name}_layer_snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap_path = snap_dir / f"epoch{epoch:04d}_{layer_prefix.replace('.', '-')}.pt"
    torch.save(payload, snap_path)

    # 3) 可选：写一份轻量 CSV 统计
    if write_stats_csv:
        import csv, math
        csv_path = snap_dir / f"{layer_prefix.replace('.', '-')}_stats.csv"
        new_file = not csv_path.exists()
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "epoch", "param_name", "shape",
                "w_mean", "w_std", "w_l2", "w_absmax",
                "g_mean", "g_std", "g_l2", "g_absmax", "has_grad"
            ])
            if new_file:
                writer.writeheader()

            for name, p in _iter_layer_params(model, layer_prefix):
                w = p.detach()
                w_mean = float(w.mean().item())
                w_std  = float(w.std(unbiased=False).item())
                w_l2   = float(w.norm(p=2).item())
                w_absmax = float(w.abs().max().item())
                if p.grad is not None:
                    g = p.grad.detach()
                    g_mean = float(g.mean().item())
                    g_std  = float(g.std(unbiased=False).item())
                    g_l2   = float(g.norm(p=2).item())
                    g_absmax = float(g.abs().max().item())
                    has_grad = True
                else:
                    g_mean = g_std = g_l2 = g_absmax = float('nan')
                    has_grad = False

                writer.writerow({
                    "epoch": epoch,
                    "param_name": name,
                    "shape": str(tuple(p.shape)),
                    "w_mean": w_mean, "w_std": w_std, "w_l2": w_l2, "w_absmax": w_absmax,
                    "g_mean": g_mean, "g_std": g_std, "g_l2": g_l2, "g_absmax": g_absmax,
                    "has_grad": has_grad
                })

    return snap_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default=str(Path(__file__).resolve().parent/'data'/'cifar'))
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--opt', type=str, default='rownorm',
                        choices=['sgd', 'rownorm', 'signsgd', 'muon', 'adamw', 'normmom', 'muonv2', 'sgdvariant', 'adamw_rescale'])
    parser.add_argument('--clip', type=float, default=0.0)
    parser.add_argument('--out_dir', type=str,
                        default=str(Path(__file__).resolve().parent/'logs'))
    parser.add_argument('--plot_dir', type=str,
                        default=str(Path(__file__).resolve().parent/'plots'))
    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--amp', dest='amp', action='store_true')
    parser.add_argument('--no_amp', dest='amp', action='store_false')
    parser.set_defaults(amp=True)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov_mom', type=float, default=0.0)
    parser.add_argument('--sched', type=str, default='cosine',
                        choices=['none', 'cosine', 'inverse', 'linear'])
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--min_lr', type=float, default=0.0)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--preset', type=str,
                        default='none', choices=['none', 'auto'])
    parser.add_argument('--model', type=str,
                        default='resnet18_cifar', choices=['resnet18_cifar'])
    parser.add_argument('--base_width', type=int, default=64)
    parser.add_argument('--p_exp', type=int,
                        default=2)
    parser.add_argument('--q_exp', type=int,
                        default=torch.inf)
    parser.add_argument('--trials', type=int,
                        default=1)
    parser.add_argument('--lmo_init', type=int,
                        default=0)
    parser.add_argument('--layer_log', default=False)
    parser.add_argument('--layer_log_prefix', type=str, default='layer2',
                        help="Module name prefix to log (e.g., 'layer2', 'layer3.0.conv1').")
    parser.add_argument('--layer_log_stats', action='store_true',
                        help="Also append a CSV with summary stats per parameter (mean/std/norm, etc.).")
    args = parser.parse_args()
    args.normpq = [args.p_exp,args.q_exp]
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    # Seeding for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Optional auto presets for more stable baselines
    if args.preset == 'auto':
        if args.opt == 'signsgd':
            # Conservative, stable settings for vision
            args.lr = 0.01 if args.lr == 0.1 else args.lr
            args.momentum = 0.0
            args.amp = False
            args.clip = 0.0
            args.weight_decay = 1e-4
            if args.sched == 'cosine':
                args.warmup_epochs = max(args.warmup_epochs, 3)
        elif args.opt == 'rownorm':
            args.lr = 0.05 if args.lr == 0.1 else args.lr
            args.momentum = 0.9
            args.nesterov_mom = 0.9
            args.clip = 0.0
            args.weight_decay = 5e-4
            if args.sched == 'cosine':
                args.warmup_epochs = max(args.warmup_epochs, 3)
        elif args.opt == 'muon':
            # Typical stable Muon settings
            args.lr = 0.05 if args.lr == 0.1 else args.lr
            args.momentum = 0.95
            args.weight_decay = 0.0
            args.amp = True
            args.clip = 0.0
            if args.sched == 'cosine':
                args.warmup_epochs = max(args.warmup_epochs, 3)
        elif args.opt == 'adamw':
            args.lr = 3e-4 if args.lr == 0.1 else args.lr
            args.weight_decay = 0.05 if args.weight_decay == 5e-4 else args.weight_decay
        elif args.opt == 'muonv2':
            # Similar to original Muon but with modified update rule
            args.lr = 0.05 if args.lr == 0.1 else args.lr
            args.alpha = 0.1  # momentum blend factor
            args.gamma = 0.0  # parameter shrink factor
            args.weight_decay = 0.0
            args.amp = True
            args.clip = 0.0
            if args.sched == 'cosine':
                args.warmup_epochs = max(args.warmup_epochs, 3)
        elif args.opt == 'sgdvariant':
            # Teacher's suggested SGD variant with normalized updates
            args.lr = 0.05 if args.lr == 0.1 else args.lr
            args.momentum = 0.9
            args.weight_decay = 0.0
            args.nesterov_mom = 0.0
            args.amp = True
            args.clip = 0.0
            if args.sched == 'cosine':
                args.warmup_epochs = max(args.warmup_epochs, 3)

    train_loader, test_loader = get_cifar10_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    # Prepare run naming and output locations
    default_run_name = (
        f"cifar10_resnet18_{args.opt}_bs{args.batch_size}_lr{args.lr}_clip{args.clip}_"
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    run_name = args.run_name or default_run_name
    out_dir = Path(args.out_dir)
    plot_dir = Path(args.plot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    if not args.trials:
        metrics_csv_path = out_dir / f"{run_name}.csv"
        config_json_path = out_dir / f"{run_name}.json"

        # Save run configuration
        config_to_save = {
            "dataset": "CIFAR-10",
            "model": "ResNet18-CIFAR",
            "optimizer": args.opt,
            "lr": args.lr,
            "normpq": args.normpq,
            "lmo_init": args.lmo_init,
            "weight_decay": args.weight_decay,
            "momentum": args.momentum if hasattr(args, 'momentum') else 0.9,
            "nesterov_mom": bool(args.nesterov_mom) if hasattr(args, 'nesterov_mom') else False,
            "batch_size": args.batch_size,
            "clip": args.clip,
            "device": device,
            "amp": bool(args.amp),
            "seed": int(args.seed),
            "sched": args.sched if hasattr(args, 'sched') else 'none',
            "warmup_epochs": int(args.warmup_epochs) if hasattr(args, 'warmup_epochs') else 0,
            "min_lr": float(args.min_lr) if hasattr(args, 'min_lr') else 0.0,
            "epochs": args.epochs,
            "num_workers": int(args.num_workers) if hasattr(args, 'num_workers') else 4,
            "preset": args.preset if hasattr(args, 'preset') else 'none',
            "data_dir": args.data_dir,
            "run_name": run_name,
            "timestamp": datetime.now().isoformat(timespec='seconds')
        }
        with open(config_json_path, 'w') as f:
            json.dump(config_to_save, f, indent=2)

        # Prepare CSV with header
        csv_fieldnames = [
            "epoch", "train_loss", "train_acc", "train_time_s", "val_loss", "val_acc",
            "throughput_img_per_s", "cumulative_time_s", "optimizer", "lr", "lr_epoch", "weight_decay",
            "batch_size", "clip", "device"
        ]
        if not metrics_csv_path.exists():
            with open(metrics_csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
                writer.writeheader()

    def append_csv_row(row: dict):
        with open(metrics_csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            writer.writerow(row)
    if args.model == 'resnet18_cifar':
        model = resnet18_cifar(
            num_classes=10, base_width=args.base_width, lmo_p=args.p_exp, lmo_q=args.q_exp, lmo_enable = args.lmo_init).to(device)
        # use lots of gpu
        if device == 'cuda' and torch.cuda.device_count() > 1: 
            model = nn.DataParallel(model)   
        model = model.to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    def set_fans_resnet_cifar(model: nn.Module):
        
        def infer_hw_from_name(module_name: str):
            if module_name == "conv1" or module_name.startswith("layer1"):
                return 32, 32
            if module_name.startswith("layer2"):
                return 16, 16
            if module_name.startswith("layer3"):
                return 8, 8
            if module_name.startswith("layer4"):
                return 4, 4
            return 1, 1

        for name, m in model.named_modules():
            # ---- Conv2d ----
            if isinstance(m, nn.Conv2d):
                H, W = infer_hw_from_name(name)
                fin  = int(m.in_channels * H * W)
                fout = int(m.out_channels)
                
                m.weight.fan_in_override  = fin
                m.weight.fan_out_override = fout
                if m.bias is not None:
                    m.bias.fan_in_override  = 1
                    m.bias.fan_out_override = fout

            # ---- Linear ----
            elif isinstance(m, nn.Linear):
                fin  = int(m.in_features)
                fout = int(m.out_features)
                m.weight.fan_in_override  = fin
                m.weight.fan_out_override = fout
                if m.bias is not None:
                    m.bias.fan_in_override  = 1
                    m.bias.fan_out_override = fout

    set_fans_resnet_cifar(model)


    optimizer = get_optimizer(
        model, args.opt, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov_mom=args.nesterov_mom, alpha=args.alpha, gamma=args.gamma, norm_pq=args.normpq)
    scaler = GradScaler(enabled=(device == 'cuda' and args.amp))

    num_train_images = len(train_loader.dataset)
    cumulative_time = 0.0
    history = []  # for plotting

    # Official scheduler: Linear warmup + CosineAnnealingLR
    scheduler = None
    if args.sched != 'none':
        warm = max(0, int(args.warmup_epochs))
        total = max(1, int(args.epochs))
        if args.sched == 'cosine':
            if total <= warm and warm > 0:
                scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=(1.0 / float(max(1, warm))), total_iters=warm)
            else:
                if warm > 0:
                    warmup = torch.optim.lr_scheduler.LinearLR(
                        optimizer, start_factor=(1.0 / float(max(1, warm))), total_iters=warm)
                    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=max(1, total - warm), eta_min=args.min_lr)
                    from torch.optim.lr_scheduler import SequentialLR
                    scheduler = SequentialLR(optimizer, schedulers=[
                                            warmup, cosine], milestones=[warm])
                else:
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=total, eta_min=args.min_lr)
        elif args.sched == 'linear':
            scheduler = two_stage_linear_scheduler(
                optimizer,
                total_epochs=total,
                warmup_epochs=warm,
                min_lr=args.min_lr, k=3.0
            )
        else:
            raise ValueError(f"Unknown scheduler: {args.sched}")

    # Ensure first epoch uses warmed LR when applicable
    if scheduler is not None:
        scheduler.step()

    for epoch in range(args.epochs):
        lr_epoch = float(optimizer.param_groups[0]["lr"]) if len(
            optimizer.param_groups) > 0 else float(args.lr)
        train_loss, train_acc, train_time = train_one_epoch(
            model, train_loader, optimizer, device, scaler, max_grad_norm=args.clip, amp_enabled=args.amp)
        val_loss, val_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}: lr={lr_epoch:.5f} train_loss={train_loss:.4f} acc={train_acc:.2f}% time={train_time:.1f}s | val_loss={val_loss:.4f} acc={val_acc:.2f}%")
        # >>> 新增：epoch 末尾导出指定层的权重与梯度（使用“最后一个 batch”的梯度）
        if args.layer_log:
            snap_path = _save_layer_snapshot(
                model=model,
                epoch=epoch + 1,
                out_dir=out_dir,
                run_name=run_name,
                layer_prefix=args.layer_log_prefix,
                write_stats_csv=args.layer_log_stats
            )

        cumulative_time += train_time
        throughput = float(num_train_images) / max(1e-9, float(train_time))
        row = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "train_time_s": float(train_time),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "throughput_img_per_s": float(throughput),
            "cumulative_time_s": float(cumulative_time),
            "optimizer": args.opt,
            "lr": float(args.lr),
            "lr_epoch": float(lr_epoch),
            "weight_decay": float(args.weight_decay),
            "batch_size": int(args.batch_size),
            "clip": float(args.clip),
            "device": device,
        }
        if not args.trials:
            append_csv_row(row)
        if scheduler is not None:
            scheduler.step()
        history.append(row)

    # Save plots
    if _MATPLOTLIB_OK and len(history) > 0 and not args.trials:
        # Accuracy vs epoch
        epochs = [r["epoch"] for r in history]
        train_accs = [r["train_acc"] for r in history]
        val_accs = [r["val_acc"] for r in history]
        train_losses = [r["train_loss"] for r in history]
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, train_accs, label='Train Top-1')
        plt.plot(epochs, val_accs, label='Val Top-1')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title(run_name)
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend()
        plt.tight_layout()
        acc_plot_path = plot_dir / f"{run_name}_acc.png"
        plt.savefig(acc_plot_path)
        plt.close()

        # Accuracy vs cumulative train time
        times = [r["cumulative_time_s"] for r in history]
        plt.figure(figsize=(6, 4))
        plt.plot(times, val_accs, marker='o')
        plt.xlabel('Cumulative Train Time (s)')
        plt.ylabel('Val Accuracy (%)')
        plt.title(f"{run_name} (speed vs accuracy)")
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        acc_time_plot_path = plot_dir / f"{run_name}_acc_vs_time.png"
        plt.savefig(acc_time_plot_path)
        plt.close()

        # Training loss vs epoch
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, train_losses, label='Train Loss', color='#1f77b4')
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title(f"{run_name} (training loss)")
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend()
        plt.tight_layout()
        loss_plot_path = plot_dir / f"{run_name}_train_loss.png"
        plt.savefig(loss_plot_path)
        plt.close()


if __name__ == '__main__':
    main()
