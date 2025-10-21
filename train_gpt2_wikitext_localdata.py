
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
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_

from utils.logging import SmoothedValue, Timer
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

# Import model module
from models.gpt2_model import load_tokenizer, build_model_from_name, build_gpt2, init_model_params_rowlmo
from set_fans import set_fans_gpt2
# ---------- Local file helpers ----------
def _read_local_texts(data_dir: Path):
    mapping = {'train':'train.txt', 'validation':'valid.txt', 'test':'test.txt'}
    out = {}
    for split, fname in mapping.items():
        fpath = data_dir / fname
        if fpath.exists():
            out[split] = fpath.read_text(encoding='utf-8')
    return out

def _tokenize_concat_to_blocks(text: str, tokenizer, block_size: int):
    tokenized = tokenizer(text, return_attention_mask=False)
    ids = tokenized['input_ids']
    n = (len(ids) // block_size) * block_size
    if n == 0:
        return [], []
    ids = ids[:n]
    input_ids = [ids[i:i+block_size] for i in range(0, n, block_size)]
    attention_mask = [[1]*block_size for _ in range(0, n, block_size)]
    return input_ids, attention_mask

class LMIterable(torch.utils.data.Dataset):
    def __init__(self, tensor_ids, tensor_masks):
        self.ids = tensor_ids
        self.masks = tensor_masks
    def __len__(self):
        return self.ids.size(0)
    def __getitem__(self, idx):
        return {"input_ids": self.ids[idx],
                "attention_mask": self.masks[idx],
                "labels": self.ids[idx]}

class GradStatsCSVLogger:
    def __init__(self, path: str):
        self.path = path
        self.fieldnames = ["global_step", "epoch", "iter_in_epoch", "param_name", "grad_norm", "grad_abs_max", "lr"]
        need_header = (not os.path.exists(self.path)) or os.path.getsize(self.path) == 0
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if need_header:
            with open(self.path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.fieldnames)
                w.writeheader()

    def write_many(self, rows: list[dict]):
        if not rows:
            return
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            for r in rows:
                # 只保留字段集合内的键
                r = {k: r.get(k, None) for k in self.fieldnames}
                w.writerow(r)


def grad_stats(model):
    stats=[]
    for name,p in model.named_parameters():
        if p.grad is None: continue
        g = p.grad.detach()
        stats.append((name, float(g.norm()), float(g.abs().max())))
    return stats

def check_nan(model):
    for name,p in model.named_parameters():
        if torch.isnan(p).any() or torch.isinf(p).any():
            print("param bad:", name)
        if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
            print("grad bad:", name)

# torch.autograd.set_detect_anomaly(True)


def get_wikitext_dataloaders(dataset_name: str, model_name: str, block_size: int, batch_size: int, num_workers: int, data_dir: str = ""):
    
    
    tokenizer = load_tokenizer(model_name)
    
    
    # Prefer local files if provided
    if data_dir:
        d = Path(data_dir)
        local = _read_local_texts(d)
        if local.get('train') and local.get('validation'):
            train_ids, train_masks = _tokenize_concat_to_blocks(local['train'], tokenizer, block_size)
            valid_ids, valid_masks = _tokenize_concat_to_blocks(local['validation'], tokenizer, block_size)
            train_data = LMIterable(torch.tensor(train_ids, dtype=torch.long),
                                    torch.tensor(train_masks, dtype=torch.long))
            valid_data = LMIterable(torch.tensor(valid_ids, dtype=torch.long),
                                    torch.tensor(valid_masks, dtype=torch.long))
        else:
            raise FileNotFoundError(f"Missing train.txt/valid.txt under {data_dir}. Use prepare_wikitext_local.py to export.")
    else:
        # Fallback: download (requires internet)
        from datasets import load_dataset
        subset = "wikitext-2-raw-v1" if dataset_name == "wikitext-2" else "wikitext-103-raw-v1"
        ds = load_dataset("wikitext", subset)
        def _tokenize_function(examples, tokenizer, block_size):
            text = "\\n\\n".join(examples["text"])
            tokenized = tokenizer(text, return_attention_mask=False)
            ids = tokenized["input_ids"]
            n = (len(ids) // block_size) * block_size
            if n == 0:
                return {"input_ids": [], "attention_mask": []}
            ids = ids[:n]
            input_ids = [ids[i:i+block_size] for i in range(0, n, block_size)]
            attention_mask = [[1]*block_size for _ in range(0, n, block_size)]
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        mapped_train = ds["train"].map(
            lambda batch: _tokenize_function(batch, tokenizer, block_size),
            batched=True, remove_columns=ds["train"].column_names
        ).filter(lambda example: len(example["input_ids"]) > 0)
        mapped_valid = ds["validation"].map(
            lambda batch: _tokenize_function(batch, tokenizer, block_size),
            batched=True, remove_columns=ds["validation"].column_names
        ).filter(lambda example: len(example["input_ids"]) > 0)

        train_data = LMIterable(torch.tensor(mapped_train["input_ids"], dtype=torch.long),
                                torch.tensor(mapped_train["attention_mask"], dtype=torch.long))
        valid_data = LMIterable(torch.tensor(mapped_valid["input_ids"], dtype=torch.long),
                                torch.tensor(mapped_valid["attention_mask"], dtype=torch.long))

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, valid_loader, tokenizer

# ---------- Schedulers & optim (minimal imports to keep parity) ----------
def build_weight_decay_param_groups(model: nn.Module, weight_decay: float):
    decay_params, no_decay_params = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay_params if p.ndim >= 2 else no_decay_params).append(p)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

def get_optimizer(model, name: str, lr: float, weight_decay: float, momentum: float, nesterov_mom: float, norm_pq=(1,torch.inf)):
    import torch.optim as optim
    param_groups = build_weight_decay_param_groups(model, weight_decay)
    name = name.lower()
    if name.lower() == 'adamw_rescale':
        return optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))  # paper β2=0.95
    if name == "rownorm":
        return RowNormSGD(param_groups, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov_mom=nesterov_mom, max_grad_norm=1.0, p_exp=norm_pq[0], q_exp = norm_pq[1])
    if name == "signsgd":
        return SignSGD(param_groups, lr=lr, momentum=momentum, weight_decay=weight_decay, p_exp=norm_pq[0], q_exp = norm_pq[1], use_fan_scaling=True)
    if name == "adamw":
        return optim.AdamW(param_groups, lr=lr)
    if name == "muonv2":
        return MuonV2(param_groups, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov_mom)
    if name == "sgdvariant":
        return SGDVariant(param_groups, lr=lr, momentum=momentum, weight_decay=0.0)
    if name == "adamw_rescale":
        return AdamWScale(param_groups, lr=lr, betas=(momentum, 1-(1-momentum)*2), weight_decay=weight_decay, p_exp=norm_pq[0], q_exp = norm_pq[1])
    raise ValueError(f"Unknown optimizer: {name}")
    raise ValueError(f"Unknown optimizer {name}")

# ---------- Train/Eval ----------
class StepCSVLogger:
    def __init__(self, path: str, fieldnames: list[str]):
        self.path = path
        self.fieldnames = fieldnames
        need_header = (not os.path.exists(self.path)) or os.path.getsize(self.path) == 0
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if need_header:
            with open(self.path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.fieldnames)
                w.writeheader()

    def write(self, row: dict):
        # 只写出 fieldnames 里有的键，避免额外字段导致错位
        row = {k: row.get(k, None) for k in self.fieldnames}
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            w.writerow(row)

def train_one_epoch(model, loader, optimizer, device, scaler: GradScaler,
                    max_grad_norm: float = 0.0, amp_enabled: bool = True,
                    epoch_idx: int = 1, step_logger=None, log_interval: int = 500, val_loader=None, scheduler=None, trials=0, grad_logger = None, grad_accum_steps: int = 1):
    model.train()
    loss_meter = SmoothedValue()
    timer = Timer()
    tokens_seen = 0
    for it, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        if (it - 1) % grad_accum_steps == 0:
            optimizer.zero_grad(set_to_none=True)
        
        with autocast(enabled=amp_enabled and (device == 'cuda')):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        scaled_loss = loss / max(1, grad_accum_steps)
        scaler.scale(scaled_loss).backward()
        step_loss = float(loss.item())
        
        
        if it % 50 == 0 and False:
            
            watch_keywords = ("attn", "mlp", "wte", "wpe")
            all_stats = grad_stats(model)
            s = [(n, gn, gm) for (n, gn, gm) in all_stats
                if any(k in n for k in watch_keywords)]
            if False:
                for name, gnorm, gmax in s[:10]:
                    print(f"[{epoch_idx * len(loader) + it}] {name:40s} || ||g||={gnorm:.2e}  g_max={gmax:.2e}")
            if grad_logger is not None and True:
                cur_lr = float(optimizer.param_groups[0].get("lr", 0.0))
                global_step = int(epoch_idx * len(loader) + it)
                rows = [{
                    "global_step": global_step,
                    "epoch": int(epoch_idx),
                    "iter_in_epoch": int(it),
                    "param_name": n,
                    "grad_norm": f"{gn:.6e}",
                    "grad_abs_max": f"{gm:.6e}",
                    "lr": cur_lr,
                } for (n, gn, gm) in all_stats]
                grad_logger.write_many(rows)
        


        if (it % grad_accum_steps == 0) or (it == len(loader)):
            if max_grad_norm and max_grad_norm > 0.0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

        if it % int(log_interval) == 0 and not trials:
            cur_lr = float(optimizer.param_groups[0].get("lr", 0.0))
            global_step = int(epoch_idx * len(loader) + it)
            tokens_this_batch = int(batch["labels"].numel()) if "labels" in batch else None
            was_training = model.training
            # val_loss_to_log,_ = evaluate(model, val_loader, device, max_batches=4)
            val_loss_to_log,_ = evaluate(model, val_loader, device)
            if was_training:
                model.train()
            
            step_logger.write({
                "global_step": global_step,
                "epoch": int(epoch_idx),
                "loss": step_loss,
                "val_loss": val_loss_to_log,
                "lr": cur_lr,
                "tokens": tokens_this_batch,
                "iter_in_epoch": int(it),
            })
        
        bs, seqlen = input_ids.size(0), input_ids.size(1)
        tokens_seen += int(bs * seqlen)
        loss_meter.update(loss.item(), bs)
    return loss_meter.avg, tokens_seen, timer.elapsed()

@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    model.eval()
    loss_meter = SmoothedValue()
    tokens = 0
    n_batch = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        bs, seqlen = input_ids.size(0), input_ids.size(1)
        tokens += int(bs * seqlen)
        loss_meter.update(loss.item(), bs)
        n_batch += 1
        if (max_batches is not None) and (n_batch>max_batches):
            break
    return loss_meter.avg, tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=str, default='')  # <-- set to ./data/wikitext-2 or ./data/wikitext-103
    ap.add_argument('--dataset', type=str, default='wikitext-2', choices=['wikitext-2', 'wikitext-103'])
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--block_size', type=int, default=512)
    ap.add_argument('--lr', type=float, default=6e-4)
    ap.add_argument('--warmup_steps', type=int, default=0)
    ap.add_argument('--min_lr', type=float, default=0)
    ap.add_argument('--weight_decay', type=float, default=0.1)
    ap.add_argument('--opt', type=str, default='adamw_rescale', choices=['adamw_rescale','rownorm','muonv2','adamw','signsgd'])
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--out_dir', type=str, default=str(Path(__file__).resolve().parent/'logs_llm'))
    ap.add_argument('--run_name', type=str, default='')
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--momentum', type=float, default=0.9)
    ap.add_argument('--nesterov_mom', type=float, default=0)
    ap.add_argument('--model_name', type=str, default='small') 
    ap.add_argument('--p_exp', type=int,
                        default=1)
    ap.add_argument('--q_exp', type=int,
                        default=torch.inf)
    ap.add_argument('--trials', type=int,
                        default=1)
    ap.add_argument('--lmo_init', type=int,
                        default=0)
    ap.add_argument("--log_step_metrics", type=int, default=1,
                        help="Whether to log per-step metrics to CSV (1=yes, 0=no).")
    ap.add_argument("--step_csv", type=str, default=None,
                        help="CSV path for per-step logs; if None, auto-generate next to metrics_csv.")
    ap.add_argument("--log_step_interval", type=int, default=50,
                    help="Log once every N steps (batches). Default=500.")
    ap.add_argument('--grad_accum_steps', type=int, default=1)
    args = ap.parse_args()
    args.normpq = [args.p_exp,args.q_exp]   
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    # Data
    train_loader, valid_loader, tokenizer = get_wikitext_dataloaders(
        dataset_name=args.dataset, model_name=args.model_name, block_size=args.block_size,
        batch_size=args.batch_size, num_workers=2, data_dir=args.data_dir
    )

    # Model
    from models.gpt2_model import build_gpt2
    model, tokenizer = build_gpt2(device=device, gpt2name = args.model_name, data_parallel=True)
    set_fans_gpt2(model)
    init_model_params_rowlmo(model,args.p_exp)
    # Optimizer
    optimizer = get_optimizer(model, args.opt, lr=args.lr, weight_decay=args.weight_decay,
                              momentum=args.momentum, nesterov_mom=args.nesterov_mom, norm_pq=args.normpq)
    scaler = GradScaler(enabled=(device=='cuda'))

    #lr decay
    scheduler = None
    steps_per_epoch = int(len(train_loader)/args.grad_accum_steps)+1
    warm_steps = max(0, int(args.warmup_steps/args.grad_accum_steps)+1)
    
    total_steps = max(1, args.epochs * steps_per_epoch)
    
    
    if total_steps <= warm_steps and warm_steps > 0:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=(1.0 / float(max(1, warm_steps))), total_iters=warm_steps)
    else:
        if warm_steps > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=(1.0 / float(max(1, warm_steps))), total_iters=warm_steps)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, total_steps - warm_steps), eta_min=args.min_lr)
            from torch.optim.lr_scheduler import SequentialLR
            scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warm_steps])
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps, eta_min=args.min_lr)

    # Ensure first epoch uses warmed LR when applicable
    # if scheduler is not None:
    #     scheduler.step()

    # Outputs
    run_name = args.run_name or f"{args.dataset}_{args.model_name}_{args.opt}_bs{args.batch_size}_seq{args.block_size}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv_path = out_dir / f"{run_name}.csv"
    config_json_path = out_dir / f"{run_name}.json"
    # ---- Grad stats outputs ----
    grad_stats_dir = out_dir / "grad_stats"
    grad_stats_dir.mkdir(parents=True, exist_ok=True)
    # 文件名包含 run_name 与 optimizer 名称，便于多实验区分
    grad_csv_path = grad_stats_dir / f"{run_name}_gradstats_{args.opt}.csv"
    grad_logger = GradStatsCSVLogger(str(grad_csv_path))


    if args.step_csv is None:
        
        base, ext = os.path.splitext(metrics_csv_path)
        step_csv_path = base + "_steps" + ext
    else:
        step_csv_path = args.step_csv

    step_logger = None
    if int(args.log_step_metrics) == 1:
        step_logger = StepCSVLogger(
            step_csv_path,
            fieldnames=[
                "global_step", "epoch", 
                "loss","val_loss", "lr", "tokens", "iter_in_epoch"
            ],
        )

    # Save config
    if not args.trials:
        with open(config_json_path, 'w') as f:
            json.dump(vars(args) | {"device": device, "run_name": run_name}, f, indent=2)

        # CSV header
        with open(metrics_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "epoch","train_loss","train_ppl","train_time_s","val_loss","val_ppl","throughput_tok_per_s","cumulative_time_s",
                "optimizer","lr","weight_decay","batch_size","block_size","clip","device"
            ])
            writer.writeheader()

    cumulative_time = 0.0
    history = []
    for epoch in range(args.epochs):
        lr_epoch = float(optimizer.param_groups[0]["lr"]) if len(
            optimizer.param_groups) > 0 else float(args.lr)
        train_loss, train_tokens, train_time = train_one_epoch(model, train_loader, optimizer, device, scaler, max_grad_norm=args.clip, amp_enabled=True,
        epoch_idx=epoch, step_logger=step_logger, log_interval=args.log_step_interval, val_loader=valid_loader, scheduler=scheduler, trials=args.trials,  grad_logger=grad_logger, grad_accum_steps=args.grad_accum_steps)
        val_loss, _ = evaluate(model, valid_loader, device)

        train_ppl = float(math.exp(min(20.0, train_loss)))
        val_ppl = float(math.exp(min(20.0, val_loss)))

        print(f"Epoch {epoch+1}: lr={lr_epoch:.5f} train_loss={train_loss:.4f} ppl={train_ppl:.2f} time={train_time:.1f}s | val_loss={val_loss:.4f} ppl={val_ppl:.2f}")
        # if scheduler is not None:
        #     scheduler.step()
        cumulative_time += train_time
        throughput = float(train_tokens) / max(1e-9, float(train_time))
        row = {
            "epoch": epoch+1,
            "train_loss": float(train_loss),
            "train_ppl": float(train_ppl),
            "train_time_s": float(train_time),
            "val_loss": float(val_loss),
            "val_ppl": float(val_ppl),
            "throughput_tok_per_s": float(throughput),
            "cumulative_time_s": float(cumulative_time),
            "optimizer": args.opt,
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "batch_size": int(args.batch_size),
            "block_size": int(args.block_size),
            "clip": float(args.clip),
            "device": device,
        }
        if not args.trials:
            with open(metrics_csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writerow(row)
        history.append(row)

if __name__ == '__main__':
    main()
