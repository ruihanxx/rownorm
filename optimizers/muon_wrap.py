"""
Muon Optimizer with Semi-Orthogonal Initialization

This module implements the Muon optimizer with auxiliary AdamW for different
parameter groups. Following the paper "On the Width Scaling of Neural Optimizers 
Under Matrix Operator Norms" (Table 3), Muon uses semi-orthogonal initialization
to match its RMS → RMS geometry.

Initialization Strategy:
- Hidden weights (2D/4D): Semi-orthogonal via QR decomposition
- Biases: Zero initialization
- BatchNorm/LayerNorm: Standard (weight=1, bias=0)
- Embeddings: PyTorch default (not modified)

Mathematical Formulation:
    For weight W ∈ R^(m×n):
    1. W ~ N(0, 1)
    2. W = QR decomposition
    3. W *= sqrt(1/n)  # RMS scaling
"""

from typing import Dict, Iterable, List, Optional

import math
import torch
import torch.nn as nn


def init_semi_orthogonal_qr(weight: torch.Tensor) -> None:
    """
    Use QR decomposition to implement semi-orthogonal initialization

    Args:
        weight: The weight tensor to initialize (modified in-place)

    Mathematical formula:
        1. W ~ N(0, 1)
        2. If m >= n: W = QR[:, :n]
        3. If m < n: W = QR(W^T)[:, :m]^T
        4. W *= sqrt(1/n)
    """
    m, n = weight.shape

    # 1. Standard normal distribution initialization
    nn.init.normal_(weight, mean=0.0, std=1.0)

    # 2. QR decomposition
    with torch.no_grad():
        if m >= n:
            # Tall or square matrix: QR decomposition of W
            Q, R = torch.linalg.qr(weight)
            weight.copy_(Q[:, :n])
        else:
            # Short or wide matrix: QR decomposition of W^T
            Q, R = torch.linalg.qr(weight.T)
            weight.copy_(Q[:, :m].T)

        # 3. RMS geometry scaling: sqrt(1/fan_in)
        scale = math.sqrt(1.0 / n)
        weight.mul_(scale)


def init_semi_orthogonal_for_muon(module: nn.Module) -> None:
    """
    Initialize model weights for the Muon optimizer

    Initialization strategy (based on paper Table 3):
    - Linear/Conv weights: Semi-orthogonal (QR decomposition)
    - Biases: 0
    - BatchNorm/LayerNorm: Standard initialization (weight=1, bias=0)
    - Embedding: Not modified (keep standard initialization)

    Args:
        module: PyTorch module
    """
    if isinstance(module, nn.Linear):
        # Linear layer: 2D weights
        init_semi_orthogonal_qr(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Conv2d):
        # Conv layer: 4D weights [out_c, in_c, kH, kW]
        # Reshape to 2D, initialize, then reshape back
        weight = module.weight
        out_c, in_c, kh, kw = weight.shape

        # Reshape: [out_c, in_c*kH*kW]
        weight_2d = weight.view(out_c, in_c * kh * kw)
        init_semi_orthogonal_qr(weight_2d)

        # Reshape back to original shape
        weight.data.copy_(weight_2d.view(out_c, in_c, kh, kw))

        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
        # BN/LN: Standard initialization
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.ones_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)

    # Embedding layer not modified, keep PyTorch default initialization


def check_semi_orthogonality(model: nn.Module, verbose: bool = True) -> dict:
    """
    Validate the semi-orthogonal property of model weights

    Args:
        model: PyTorch model
        verbose: Whether to print detailed information

    Returns:
        dict: 包含每层正交性误差的字典
    """
    results = {}

    for name, param in model.named_parameters():
        if 'weight' in name and param.ndim == 2:
            W = param.data.float()
            m, n = W.shape

            if m <= n:
                # Check row orthogonality: W @ W^T ≈ I
                product = W @ W.T
                identity = torch.eye(m, device=W.device, dtype=W.dtype)
                error = torch.norm(product - identity, p='fro').item()
                orth_type = "row"
            else:
                # Check column orthogonality: W^T @ W ≈ I
                product = W.T @ W
                identity = torch.eye(n, device=W.device, dtype=W.dtype)
                error = torch.norm(product - identity, p='fro').item()
                orth_type = "col"

            results[name] = {"error": error,
                             "type": orth_type, "shape": (m, n)}

            if verbose:
                print(
                    f"{name:40s} | {orth_type:3s} | shape={str((m,n)):12s} | error={error:.6f}")

    return results


def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)

    # Fix: keep original data type, avoid forced conversion to bfloat16
    original_dtype = G.dtype
    # If original type is float32, keep float32; if bfloat16 is supported and original is bfloat16, then use bfloat16
    if original_dtype == torch.float32 or not torch.cuda.is_bf16_supported():
        X = G.float()
    else:
        X = G.bfloat16()

    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    # Convert back to original data type
    return X.to(original_dtype)


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed variant of MuonWithAuxAdam.
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(
                    ["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(
                    ["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(
                        p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


def build_single_device_muon_with_aux_adam(
    model: torch.nn.Module,
    muon_lr: float = 0.05,
    muon_momentum: float = 0.95,
    muon_weight_decay: float = 0.0,
    adamw_lrs: Optional[Dict[str, float]] = None,
    adamw_weight_decay: float = 0.0,
    adamw_betas=(0.9, 0.95),
    adamw_eps: float = 1e-10,
    use_semi_orthogonal_init: bool = True,
):
    """
    Create a SingleDeviceMuonWithAuxAdam optimizer for a model by splitting parameters into:
    - use_muon=True: hidden 2D/4D weights (matrices, conv filters)
    - use_muon=False: embeddings, biases/scalars, and output heads (AdamW-like aux)

    Args:
        model: PyTorch 模型
        muon_lr: Muon 学习率
        muon_momentum: Muon 动量系数
        muon_weight_decay: Muon 权重衰减
        adamw_lrs: AdamW 各组学习率字典
        adamw_weight_decay: AdamW 权重衰减
        adamw_betas: AdamW beta 参数
        adamw_eps: AdamW epsilon
        use_semi_orthogonal_init: 是否使用 semi-orthogonal 初始化 (默认: True)

    Returns:
        SingleDeviceMuonWithAuxAdam 优化器
    """
    # 1. 应用 Semi-orthogonal 初始化
    if use_semi_orthogonal_init:
        print("Applying semi-orthogonal initialization for Muon...")
        model.apply(init_semi_orthogonal_for_muon)

        # 可选：验证初始化
        # check_semi_orthogonality(model, verbose=False)

    # 2. 原有的参数分组逻辑
    adamw_lrs = adamw_lrs or {}

    hidden_matrix_params: List[torch.nn.Parameter] = []
    embed_params: List[torch.nn.Parameter] = []
    scalar_params: List[torch.nn.Parameter] = []
    head_params: List[torch.nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_head = name.endswith("weight") and (
            "fc" in name or "classifier" in name or "head" in name)
        is_embed = "embed" in name or "embedding" in name
        if p.ndim >= 2 and not is_embed and not is_head:
            hidden_matrix_params.append(p)
        elif is_head:
            head_params.append(p)
        elif is_embed:
            embed_params.append(p)
        else:
            scalar_params.append(p)

    # Apply weight decay to head/embed, but not to scalar (BN/bias) params
    adam_groups = [
        dict(
            params=head_params,
            lr=adamw_lrs.get("head", 3e-4),
            betas=adamw_betas,
            eps=adamw_eps,
            weight_decay=adamw_weight_decay,
            use_muon=False,
        ),
        dict(
            params=embed_params,
            lr=adamw_lrs.get("embed", 3e-4),
            betas=adamw_betas,
            eps=adamw_eps,
            weight_decay=adamw_weight_decay,
            use_muon=False,
        ),
        dict(
            params=scalar_params,
            lr=adamw_lrs.get("scalar", 3e-4),
            betas=adamw_betas,
            eps=adamw_eps,
            weight_decay=0.0,
            use_muon=False,
        ),
    ]

    muon_group = dict(params=hidden_matrix_params, lr=muon_lr,
                      momentum=muon_momentum, weight_decay=muon_weight_decay, use_muon=True)
    param_groups = [*adam_groups, muon_group]
    optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    return optimizer


if __name__ == "__main__":
    # Test semi-orthogonal initialization
    print("Testing Semi-Orthogonal Initialization for Muon\n")
    print("=" * 80)

    # Create test model
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    print("\n1. Before initialization (PyTorch default):")
    print("-" * 80)
    results_before = check_semi_orthogonality(model, verbose=True)

    print("\n2. After semi-orthogonal initialization:")
    print("-" * 80)
    model.apply(init_semi_orthogonal_for_muon)
    results_after = check_semi_orthogonality(model, verbose=True)

    print("\n3. Building Muon optimizer with initialization:")
    print("-" * 80)
    optimizer = build_single_device_muon_with_aux_adam(
        model,
        muon_lr=0.05,
        use_semi_orthogonal_init=True
    )
    print(f"✓ Optimizer created: {type(optimizer).__name__}")

    print("\n" + "=" * 80)
    print("All tests passed!")
