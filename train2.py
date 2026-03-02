#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import math
import csv
import argparse
import random
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
from ddp_utils import init_distributed, is_main_process, get_rank, get_world_size, reduce_sum, reduce_mean, barrier
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.logging import SmoothedValue, Timer
from utils.common import get_cifar10_dataloaders
from utils.metrics import accuracy
from utils.logging import SmoothedValue, Timer
from optim.rownorm import RowNormSGD
from optim.signsgd import SignSGD


from optim.muon_v2 import MuonV2

from optim.adamw_rescale import AdamWScale
from optim.rowmix import RowmixSGD
import os, math, csv, datetime
from typing import Optional, Dict, Any, Iterable
try:
    from transformers.models.gpt2.modeling_gpt2 import Conv1D as HFConv1D
except Exception:
    HFConv1D = None
try:
    from transformers.pytorch_utils import Conv1D as HFConv1D_Alt
    if HFConv1D is None:
        HFConv1D = HFConv1D_Alt
except Exception:
    pass


# Import model module
from models.gpt2_model_nanogpt import build_gpt2, init_model_params_rowlmo, init_model_params_rowlmo_rowmix
from set_fans import set_fans_gpt2

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

def get_optimizer(model, name: str, lr: float, weight_decay: float, momentum: float, nesterov_mom: float, norm_pq=(1,torch.inf), use_fan_scaling=True):
    import torch.optim as optim
    param_groups = build_weight_decay_param_groups(model, weight_decay)
    name = name.lower()
    if name == 'rowmix':
        return RowmixSGD(param_groups, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov_mom=nesterov_mom, max_grad_norm=1)
    if name == "rownorm":
        return RowNormSGD(param_groups, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov_mom=nesterov_mom, max_grad_norm=1.0, p_exp=norm_pq[0], q_exp = norm_pq[1], use_fan_scaling = use_fan_scaling)
    if name == "signsgd":
        return SignSGD(param_groups, lr=lr, momentum=momentum, weight_decay=weight_decay, p_exp = 1, use_fan_scaling = use_fan_scaling)
    if name == "adamw":
        if nesterov_mom > 0:
            print('using NADAMW')
            return optim.NAdam(param_groups, lr=lr, weight_decay = weight_decay, decoupled_weight_decay=True, betas=(0.9, 0.95))
        else:
            print('using ADAMW')
            return optim.AdamW(param_groups, lr=lr, weight_decay = weight_decay, betas=(0.9, 0.95))
    if name == "muonv2":
        return MuonV2(param_groups, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov_mom)
    if name == "sgdvariant":
        return SGDVariant(param_groups, lr=lr, momentum=momentum, weight_decay=0.0)
    if name == "adamw_rescale":
        return AdamWScale(param_groups, lr=lr, betas=(momentum, 1-(1-momentum)*2), weight_decay=weight_decay, p_exp=1, q_exp = norm_pq[1])
    raise ValueError(f"Unknown optimizer: {name}")
    raise ValueError(f"Unknown optimizer {name}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="nanoGPT-style GPT-2 training loop with DDP + cosine LR decay, "
                    "using custom build_gpt2 / get_optimizer, and logging to CSV."
    )

    # I/O
    parser.add_argument("--out_dir", type=str, default="out",
                        help="Directory to save checkpoints.")
    parser.add_argument("--dataset_dir", type=str, default="data/openwebtext-bin",
                        help="Path to folder containing train.bin and val.bin (relative to this script or absolute).")
    parser.add_argument("--log_every", type=int, default=1,
                        help="How often (in iterations) to print training loss.")
    parser.add_argument("--eval_interval", type=int, default=2000,
                        help="How often (in iterations) to run evaluation + checkpoint + CSV log.")
    parser.add_argument("--eval_iters", type=int, default=200,
                        help="Number of batches per split for evaluation.")
    parser.add_argument("--max_iters", type=int, default=600000,
                        help="Total number of optimizer steps (iterations).")
    parser.add_argument("--save_step", type=int, default=None,
                        help="Global iteration step at which to save a full training state checkpoint (0-based).")
    parser.add_argument("--start_from_past", action="store_true",
                        help="If set, resume training from a previously saved training state in out_dir.")

    # Data / model size
    parser.add_argument("--batch_size", type=int, default=12,
                        help="Micro-batch size per process (per GPU).")
    parser.add_argument("--block_size", type=int, default=1024,
                        help="Sequence length / context length.")
    parser.add_argument("--model_name", type=str, default="small",
                        help="Name passed as gpt2name to build_gpt2 (e.g. gpt2, gpt2-medium).")

    # Gradient accumulation (global, across all GPUs)
    parser.add_argument("--grad_accum_steps", type=int, default=40,
                        help="Global gradient accumulation steps across all GPUs (like nanoGPT's gradient_accumulation_steps).")

    # Optimizer / LR
    parser.add_argument("--opt", type=str, default="adamw",
                        help="Optimizer name passed to get_optimizer.")
    parser.add_argument("--lr", type=float, default=6e-4,
                        help="Peak learning rate.")
    parser.add_argument("--min_lr", type=float, default=6e-5,
                        help="Minimum learning rate after cosine decay.")
    parser.add_argument("--warmup_iters", type=int, default=2000,
                        help="Iterations for linear LR warmup.")
    parser.add_argument("--lr_decay_iters", type=int, default=600000,
                        help="Iterations over which to decay LR with cosine (usually ~= max_iters).")
    parser.add_argument("--weight_decay", type=float, default=1e-1,
                        help="Weight decay passed to get_optimizer.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum passed to get_optimizer (if optimizer uses it).")
    parser.add_argument("--nesterov_mom",type=float, default=0)
    
    parser.add_argument('--use_fan_scaling', type=float, default=0)
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Max gradient norm for clipping (<=0 disables clipping).")

    # System / DDP / precision
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device when not using torchrun (cpu, cuda, cuda:0, mps, or auto).")
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "float32", "bfloat16", "float16"],
                        help="Training dtype; auto picks bfloat16 if available else float32.")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile to optimize the model (PyTorch 2.0+).")
    parser.add_argument("--backend", type=str, default="nccl",
                        help="DDP backend (nccl, gloo, etc.).")
    parser.add_argument("--seed", type=int, default=1337,
                        help="Base random seed (will be offset by rank in DDP).")
    parser.add_argument('--p_exp', type=float,
                        default=1)
    parser.add_argument('--q_exp', type=int,
                        default=torch.inf)
    args = parser.parse_args()                    
    args.normpq = [args.p_exp,args.q_exp] 
    return args


def setup_ddp(args):
    """
    初始化分布式环境（如果是用 torchrun 启动的话）。
    返回：device, ddp, rank, local_rank, world_size, master_process, seed_offset
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Launched by torchrun: multi-process multi-GPU
        ddp = True
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ["WORLD_SIZE"])

        if not torch.cuda.is_available():
            raise RuntimeError("DDP with torchrun currently expects CUDA GPUs to be available.")

        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        dist.init_process_group(backend=args.backend)

        master_process = (rank == 0)
        seed_offset = rank
    else:
        # Single-process (no torchrun)
        ddp = False
        rank = 0
        local_rank = 0
        world_size = 1

        if args.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(args.device)

        master_process = True
        seed_offset = 0

    return device, ddp, rank, local_rank, world_size, master_process, seed_offset


def main():
    args = parse_args()

    # ----------------- DDP setup -----------------
    device, ddp, rank, local_rank, world_size, master_process, seed_offset = setup_ddp(args)
    device_type = "cuda" if device.type == "cuda" else "cpu"

    # ----------------- Seeding & TF32 -----------------
    torch.manual_seed(args.seed + seed_offset)
    np.random.seed(args.seed + seed_offset)
    if device_type == "cuda":
        torch.cuda.manual_seed_all(args.seed + seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ----------------- Dtype & autocast ctx -----------------
    if args.dtype == "auto":
        if device_type == "cuda" and torch.cuda.is_bf16_supported():
            dtype = "bfloat16"
        else:
            dtype = "float32"
    else:
        dtype = args.dtype

    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]

    if device_type == "cpu" or dtype == "float32":
        ctx = nullcontext()
    else:
        ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # ----------------- Paths -----------------
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # out_dir (for checkpoints)
    out_dir = args.out_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(script_dir, out_dir)

    # dataset_dir
    if os.path.isabs(args.dataset_dir):
        data_dir = args.dataset_dir
    else:
        data_dir = os.path.join(script_dir, args.dataset_dir)

    train_bin = os.path.join(data_dir, "train.bin")
    val_bin = os.path.join(data_dir, "val.bin")
    if not os.path.isfile(train_bin):
        raise FileNotFoundError(f"train.bin not found at: {train_bin}")
    if not os.path.isfile(val_bin):
        raise FileNotFoundError(f"val.bin not found at: {val_bin}")

    if master_process:
        os.makedirs(out_dir, exist_ok=True)

    # logs_llm dir for CSV
    logs_dir = os.path.join(script_dir, "logs_llm")
    if master_process:
        os.makedirs(logs_dir, exist_ok=True)

        # ===== 关键修改：使用 "opt_model_name_lr" 作为文件名，并且每次运行都 append =====
        lr_str = f"{args.lr:g}"  # 例如 0.0006 -> '0.0006', 6e-4 -> '0.0006'
        csv_filename = f"{args.model_name}_{args.opt}_{lr_str}_p{args.p_exp}_{args.use_fan_scaling}_newmuon_test.csv"
        csv_path = os.path.join(logs_dir, csv_filename)

        # 如果文件已存在则追加，否则新建并写表头
        file_exists = os.path.exists(csv_path)
        csv_file = open(csv_path, mode="a", newline="")
        csv_writer = csv.writer(csv_file)

        if (not file_exists) or os.stat(csv_path).st_size == 0:
            csv_writer.writerow(
                ["step", "train_loss", "val_loss", "lr", "tokens_seen_per_rank", "step_time_sec"]
            )
            csv_file.flush()
    else:
        csv_file = None
        csv_writer = None

    # ----------------- Data: memmap + get_batch -----------------
    batch_size = args.batch_size
    block_size = args.block_size

    train_data = np.memmap(train_bin, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_bin, dtype=np.uint16, mode="r")

    def get_batch(split: str):
        data = train_data if split == "train" else val_data
        # sample random start positions
        if len(data) <= block_size + 1:
            raise ValueError(
                f"{split} data too small for block_size={block_size}. Found {len(data)} tokens."
            )
        ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
        x_list = []
        y_list = []
        for i in ix.tolist():
            x_np = np.array(data[i : i + block_size], dtype=np.int64)
            y_np = np.array(data[i + 1 : i + 1 + block_size], dtype=np.int64)
            x_list.append(torch.from_numpy(x_np))
            y_list.append(torch.from_numpy(y_np))
        x = torch.stack(x_list)  # (B, T)
        y = torch.stack(y_list)  # (B, T)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        return x, y

    # ----------------- Gradient accumulation steps (global -> local) -----------------
    global_grad_accum = args.grad_accum_steps
    if ddp:
        if global_grad_accum % world_size != 0:
            raise ValueError(
                f"grad_accum_steps={global_grad_accum} must be divisible by world_size={world_size} "
                "to keep effective batch size consistent."
            )
        grad_accum_steps = global_grad_accum // world_size
    else:
        grad_accum_steps = global_grad_accum

    tokens_per_iter = grad_accum_steps * world_size * batch_size * block_size
    
    if master_process:
        print(f"tokens per iteration will be: {tokens_per_iter:,}")

    # ----------------- Model & optimizer -----------------
    # model, tokenizer = build_gpt2(...)
    if args.opt == 'rownorm':
        
        model, tokenizer = build_gpt2(
            vocab_size=50304,
            device=device,
            gpt2name=args.model_name,
            block_size=args.block_size,
        )

        set_fans_gpt2(model)
        # init_model_params_rowlmo(model,args.p_exp, args.weight_decay)
    else:
        model, tokenizer = build_gpt2(
            vocab_size=50304,
            device=device,
            gpt2name=args.model_name,
            block_size=args.block_size,
        )

        set_fans_gpt2(model)
    print('tied:::::',model.get_output_embeddings().weight is model.get_input_embeddings().weight)
    # model.to(device)

    # Optionally compile before wrapping with DDP
    if args.compile:
        if master_process:
            print("Compiling the model with torch.compile() (PyTorch 2.0+ required)...")
        model = torch.compile(model)  # type: ignore

    # Wrap with DDP (if needed)
    if ddp:
        model = DDP(model, device_ids=[local_rank])

    # Optimizer via your custom get_optimizer
    if args.opt == 'muon':
        muon_params = [
            p for name, p in model.named_parameters()
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name and "wpe" not in name and "wte" not in name
        ]
        adamw_params = [
            p for name, p in model.named_parameters()
            if p.ndim < 2 or "embed_tokens" in name or "lm_head" in name or "wpe" in name or "wte" in name
        ]
        betas_1d=(0.9, 0.95)
        # print(adamw_params)
        from optim.moonlight import Moonlight
        optimizer = Moonlight(lr=args.lr, wd=args.weight_decay, muon_params=muon_params, adamw_params=adamw_params,
                        momentum=args.momentum, adamw_betas=betas_1d)
    if args.opt == 'muons':
        muon_params = [
            p for name, p in model.named_parameters()
            if p.ndim >= 2 
        ]
        adamw_params = [
            p for name, p in model.named_parameters()
            if p.ndim < 2 
        ]
        betas_1d=(0.9, 0.95)
        # print(adamw_params)
        from optim.muons import MuonS
        optimizer = MuonS(lr=args.lr, wd=args.weight_decay, muon_params=muon_params, adamw_params=adamw_params,
                        momentum=args.momentum, adamw_betas=betas_1d)
    else:
        optimizer = get_optimizer(
            model,
            args.opt,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            nesterov_mom=args.nesterov_mom,
            norm_pq=args.normpq,
            use_fan_scaling=args.use_fan_scaling,
        )

    # ----------------- LR scheduler (cosine with warmup, like nanoGPT) -----------------
    def get_lr(it: int) -> float:
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            return args.lr * it / max(1, args.warmup_iters)
        # 2) if it > lr_decay_iters, return min learning rate
        if it > args.lr_decay_iters:
            return args.min_lr
        # 3) in between, use cosine decay from lr to min_lr
        decay_ratio = (it - args.warmup_iters) / max(1, args.lr_decay_iters - args.warmup_iters)
        decay_ratio = max(0.0, min(1.0, decay_ratio))
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # ranges 1..0
        return args.min_lr + coeff * (args.lr - args.min_lr)

    # ----------------- Evaluation function (nanoGPT-style estimate_loss) -----------------
    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = []
            for _ in range(args.eval_iters):
                x, y = get_batch(split)
                with ctx:
                    outputs = model(input_ids=x)
                    logits = getattr(outputs, "logits", None)
                    if logits is None:
                        # fall back for plain torch Module returning logits directly
                        if isinstance(outputs, (tuple, list)):
                            logits = outputs[0]
                        else:
                            logits = outputs
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1),
                    )
                losses.append(loss.item())
            out[split] = float(np.mean(losses))
        model.train()
        return out

    # ----------------- Training loop -----------------
    # ----------------- Training loop -----------------
    iter_num = 0
    best_val_loss = float("inf")
    tokens_seen = 0

    # Optionally resume from a previously saved full training state
    if args.start_from_past:
        ckpt_path = os.path.join(out_dir, f"ckpt_rank{rank}.pt")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found for rank {rank}: {ckpt_path}")
        # map_location 确保在当前 device 上加载
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Restore model parameters
        if ddp:
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

        # Restore optimizer state
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore scalar training state
        iter_num = checkpoint.get("iter_num", iter_num)
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
        tokens_seen = checkpoint.get("tokens_seen", tokens_seen)

        # Restore RNG states (data sampling, dropout 等都会受影响)
        cpu_rng_state = checkpoint.get("cpu_rng_state", None)
        if cpu_rng_state is not None:
            # 注意：因为我们在 torch.load 里用了 map_location=device，
            # 这里的 cpu_rng_state 可能已经被搬到 GPU 上了，需要强制挪回 CPU
            if isinstance(cpu_rng_state, torch.Tensor):
                cpu_rng_state = cpu_rng_state.cpu()
            torch.set_rng_state(cpu_rng_state)

        numpy_rng_state = checkpoint.get("numpy_rng_state", None)
        if numpy_rng_state is not None:
            np.random.set_state(numpy_rng_state)

        python_rng_state = checkpoint.get("python_rng_state", None)
        if python_rng_state is not None:
            random.setstate(python_rng_state)

        if device_type == "cuda":
            cuda_rng_state = checkpoint.get("cuda_rng_state", None)
            if cuda_rng_state is not None:
                # 因为 torch.load 时用了 map_location=device，这里可能是 torch.cuda.ByteTensor，
                # 但 torch.cuda.set_rng_state 要的是 CPU 的 torch.ByteTensor，
                # 所以这里也必须先挪回 CPU。
                if isinstance(cuda_rng_state, torch.Tensor):
                    cuda_rng_state = cuda_rng_state.cpu()
                torch.cuda.set_rng_state(cuda_rng_state)

        if master_process:
            print(
                f"[rank {rank}] resumed training from {ckpt_path} "
                f"at iter {iter_num} (tokens_seen={tokens_seen}, best_val_loss={best_val_loss})"
            )

    # 重新开始计时（不从 checkpoint 中恢复 wall-clock 时间）
    t0 = time.time()

    try:
        while iter_num < args.max_iters:
            # Determine learning rate for this iteration (cosine decay with warmup)
            lr = get_lr(iter_num)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Periodic evaluation + checkpoint + CSV logging (master only)
            if master_process and (iter_num % args.eval_interval == 0 or iter_num == args.max_iters - 1):
                losses = estimate_loss()
                train_loss = losses["train"]
                val_loss = losses["val"]
                print(
                    f"[rank {rank}] step {iter_num}: "
                    f"train loss {train_loss:.4f}, val loss {val_loss:.4f}, lr {lr:.3e}"
                )
                if csv_writer is not None:
                    step_time = time.time() - t0
                    csv_writer.writerow(
                        [
                            iter_num,
                            f"{train_loss:.6f}",
                            f"{val_loss:.6f}",
                            f"{lr:.6e}",
                            tokens_seen,
                            f"{step_time:.6f}",
                        ]
                    )
                    csv_file.flush()

                # Save checkpoint if val improves (or always if you prefer)
                # if val_loss < best_val_loss:
                #     best_val_loss = val_loss
                #     raw_model = model.module if ddp else model
                #     ckpt = {
                #         "model_state_dict": raw_model.state_dict(),
                #         "optimizer_state_dict": optimizer.state_dict(),
                #         "iter_num": iter_num,
                #         "best_val_loss": best_val_loss,
                #         "config": vars(args),
                #     }
                #     ckpt_path = os.path.join(out_dir, "ckpt.pt")
                #     print(f"[rank {rank}] saving checkpoint to {ckpt_path}")
                #     torch.save(ckpt, ckpt_path)

            # Sync all ranks here before continuing (optional but nice)
            # if ddp:
            #     dist.barrier()

            # Forward/backward with gradient accumulation (nanoGPT-style)
            model.train()
            optimizer.zero_grad(set_to_none=True)

            for micro_step in range(grad_accum_steps):
                x, y = get_batch("train")
                tokens_seen += x.numel()

                if ddp and micro_step < grad_accum_steps - 1:
                    grad_ctx = model.no_sync()
                else:
                    grad_ctx = nullcontext()

                with ctx, grad_ctx:
                    outputs = model(input_ids=x)
                    logits = getattr(outputs, "logits", None)
                    if logits is None:
                        if isinstance(outputs, (tuple, list)):
                            logits = outputs[0]
                        else:
                            logits = outputs
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1),
                    )
                    # average loss over micro-steps
                    loss = loss / grad_accum_steps

                loss.backward()

            # Gradient clipping
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Timing & logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            if master_process and (iter_num % args.log_every == 0):
                loss_val = float(loss.item())
                print(
                    f"[rank {rank}] iter {iter_num}: loss {loss_val:.4f}, "
                    f"lr {lr:.3e}, time {dt * 1000:.2f}ms"
                )

            iter_num += 1

            # Save full training state at the specified global step (if requested)
            # 注意：save_step 代表已经完成的 global iter 数（1 次 optimizer.step 算 1 个 iter）
            if args.save_step is not None and iter_num == args.save_step:
                ckpt = {
                    "model_state_dict": (model.module if ddp else model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "tokens_seen": tokens_seen,
                    # RNG states
                    "cpu_rng_state": torch.get_rng_state(),
                    "numpy_rng_state": np.random.get_state(),
                    "python_rng_state": random.getstate(),
                }
                if device_type == "cuda":
                    ckpt["cuda_rng_state"] = torch.cuda.get_rng_state()

                ckpt_path = os.path.join(out_dir, f"ckpt_rank{rank}.pt")
                # 每个 rank 写自己的 checkpoint（多卡时 RNG 状态不同）
                if master_process:
                    print(f"[rank {rank}] saving training state to {ckpt_path} at iter {iter_num}")
                torch.save(ckpt, ckpt_path)

    finally:
        if master_process and csv_file is not None:
            csv_file.close()
        if ddp and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
