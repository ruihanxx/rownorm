
import math
import logging
from typing import Iterable, List, Optional

import torch
import torch.nn as nn


def init_rownorm_weights(module: nn.Module, p: float = 2.0) -> None:
    """
    Initialize model weights for the RowNormOpNormConstraintPureSignGD optimizer.

    Initialization strategy:
    - Linear/Conv weights: Row-normalized with scaling
    - Bias: 0
    - BatchNorm/LayerNorm: Standard initialization
    - Embedding: No modification

    Args:
        module: PyTorch module
        p: Norm parameter (default: 2.0 for RowNormOpNormConstraintPureSignGD)
    """
    if isinstance(module, nn.Linear):
        weight = module.weight
        m, n = weight.shape

        # 1. Standard normal initialization
        nn.init.normal_(weight, mean=0.0, std=1.0)

        # 2. Row-wise p-norm normalization
        with torch.no_grad():
            for i in range(m):
                row = weight[i, :]
                row_norm = torch.norm(row, p=p)
                if row_norm > 1e-8:
                    weight[i, :] = row / row_norm

            # 3. fan_in scaling
            fan_in = n
            scale = fan_in ** (-1.0 / p)
            weight.mul_(scale)

        # Bias initialized to 0
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Conv2d):
        weight = module.weight
        out_c, in_c, kh, kw = weight.shape

        # Reshape to 2D
        fan_in = in_c * kh * kw
        weight_2d = weight.view(out_c, fan_in)

        # 1. Standard normal initialization
        nn.init.normal_(weight_2d, mean=0.0, std=1.0)

        # 2. Row-wise normalization
        with torch.no_grad():
            for i in range(out_c):
                row = weight_2d[i, :]
                row_norm = torch.norm(row, p=p)
                if row_norm > 1e-8:
                    weight_2d[i, :] = row / row_norm

            # 3. fan_in scaling
            scale = fan_in ** (-1.0 / p)
            weight_2d.mul_(scale)

        # Reshape back to original shape
        weight.data.copy_(weight_2d.view(out_c, in_c, kh, kw))

        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.ones_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)


def check_row_normalization(model: nn.Module, p: float = 2.0, verbose: bool = True) -> dict:
    """
    Validate row normalization of model weights

    Args:
        model: PyTorch model
        p: Norm parameter
        verbose: Whether to print detailed information

    Returns:
        dict: Dictionary containing row norm statistics for each layer
    """
    results = {}

    for name, param in model.named_parameters():
        if 'weight' in name and param.ndim == 2:
            W = param.data.float()
            m, n = W.shape

            # Calculate p-norm of each row
            row_norms = torch.norm(W, p=p, dim=1)

            # Statistics
            mean_norm = row_norms.mean().item()
            std_norm = row_norms.std().item()
            min_norm = row_norms.min().item()
            max_norm = row_norms.max().item()

            results[name] = {
                "mean": mean_norm,
                "std": std_norm,
                "min": min_norm,
                "max": max_norm,
                "shape": (m, n)
            }

            if verbose:
                print(f"{name:40s} | shape={str((m,n)):12s} | "
                      f"row_norm: mean={mean_norm:.6f} std={std_norm:.6f} "
                      f"[{min_norm:.6f}, {max_norm:.6f}]")

    return results


def _fan_in_of_weight(param):
    """Calculate fan_in value for weight parameters."""
    if param.ndim == 2:
        return param.shape[1]
    elif param.ndim == 4:
        return param.shape[1] * param.shape[2] * param.shape[3]
    return None


def _is_weight_for_rownorm(name, param):
    """Determine if this parameter should use RowNorm."""
    name_lower = name.lower()

    # 2D weights
    if param.ndim == 2:
        # Transformer: attention and FFN weights
        if any(key in name_lower for key in ['attn', 'attention']):
            if 'weight' in name_lower:
                return True

        # Transformer/ResNet: linear/FC/MLP weights
        if 'linear' in name_lower and name.endswith('.weight'):
            return True
        if 'mlp' in name_lower and name.endswith('.weight'):
            return True
        if any(key in name_lower for key in ['fc', 'classifier']) and name.endswith('.weight'):
            return True

    # 4D weights (Conv layers for ResNet)
    elif param.ndim == 4:
        if 'conv' in name_lower and 'weight' in name_lower:
            return True
        if 'downsample' in name_lower and 'weight' in name_lower:
            return True

    return False


def _is_1d_for_signgd(name, param):
    """Determine if this 1D parameter should use SignGD."""
    if param.ndim != 1:
        return False

    name_lower = name.lower()

    # Biases
    if 'bias' in name_lower:
        return True

    # BatchNorm (ResNet)
    if 'bn' in name_lower or 'batchnorm' in name_lower:
        return True

    # LayerNorm (Transformer)
    if 'norm' in name_lower or 'layernorm' in name_lower or 'layer_norm' in name_lower:
        return True

    return False


class RowNormOpNormConstraintPureSignGD(torch.optim.Optimizer):
    """
    RowNorm with OpNorm Constraint + Pure SignGD for 1D parameters.

    This optimizer uses (2,mean) → ℓ∞ geometry with 1/sqrt(fan_in) scaling.
    Weights should be initialized with row-normalized Gaussian to match this geometry.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        eps: float = 1e-8,
        max_grad_norm: float = 1.0,
        named_parameters: Optional[List] = None,
        log_interval: int = 50,
        use_row_normalized_init: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum > 0")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if max_grad_norm <= 0.0:
            raise ValueError(f"Invalid max_grad_norm: {max_grad_norm}")

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        nesterov=nesterov, eps=eps, max_grad_norm=max_grad_norm,
                        log_interval=log_interval,
                        use_row_normalized_init=use_row_normalized_init)
        super().__init__(params, defaults)

        self.named_parameters = named_parameters or []
        self.logger = logging.getLogger(self.__class__.__name__)
        self._step = 0

        self._classify_parameters()

        # Statistics and print QKV parameter information
        if self.named_parameters:
            qkv_params = [name for name, p in self.named_parameters if getattr(
                p, '_is_qkv', False)]
            if qkv_params:
                self.logger.info(
                    f"Detected {len(qkv_params)} QKV parameter(s):")
                for name in qkv_params:
                    self.logger.info(f"  - {name}")

        # If needed, apply row-normalized initialization
        if use_row_normalized_init and self.named_parameters:
            self._apply_row_normalized_init()

    def _apply_row_normalized_init(self):
        """Apply row-normalized initialization to all parameters"""
        self.logger.info("Applying row-normalized initialization (p=2.0)...")

        for name, param in self.named_parameters:
            if param.requires_grad:
                # Create a temporary module to apply initialization
                if param.ndim == 2:
                    # Simulate Linear layer
                    temp_module = nn.Linear(
                        param.size(1), param.size(0), bias=False)
                    temp_module.weight = param
                    init_rownorm_weights(temp_module, p=2.0)
                elif param.ndim == 4:
                    # Simulate Conv layer
                    temp_module = nn.Conv2d(
                        param.size(1), param.size(0),
                        param.size(2), bias=False
                    )
                    temp_module.weight = param
                    init_rownorm_weights(temp_module, p=2.0)

    def _classify_parameters(self):
        """Classify parameters including QKV special handling"""
        if not self.named_parameters:
            return

        # Initialize all parameters
        for group in self.param_groups:
            for param in group['params']:
                param._optimizer_param_type = 'other'
                param._is_qkv = False

        # Classify parameters
        for name, param in self.named_parameters:
            # Detect QKV-related layers
            is_qkv = any(pattern in name.lower() for pattern in [
                'q_proj', 'k_proj', 'v_proj',  # Standard naming
                'c_attn',  # GPT-2 style (contains QKV)
                '.qkv.',   # Some implementations use this
                'attn.in_proj',  
            ])

            if is_qkv and param.ndim == 2:  # Only process 2D weight matrices
                param._is_qkv = True
                param._optimizer_param_type = 'weight_qkv_special'
            elif _is_weight_for_rownorm(name, param):
                param._is_qkv = False
                param._optimizer_param_type = 'weight_rownorm'
            elif _is_1d_for_signgd(name, param):
                param._optimizer_param_type = 'param_1d_signgd'
            elif 'embed' in name.lower():
                param._optimizer_param_type = 'embedding'
            else:
                param._optimizer_param_type = 'other'

    def _signgd_1d_pure(self, grad: torch.Tensor, eps: float, name: str = "") -> torch.Tensor:
        """Apply Pure SignGD: g = sign(g)"""
        if grad is None or grad.ndim != 1:
            return grad

        grad_float = grad.float()
        grad_norm = torch.norm(grad_float)

        if grad_norm < eps:
            return grad

        grad_sign = torch.sign(grad_float)
        grad.copy_(grad_sign.to(grad.dtype))

        if self._step % self.defaults.get('log_interval', 50) == 0:
            d = grad.numel()
            self.logger.info(
                f"[PureSignGD] name={name} dim={d} grad_norm={grad_norm:.6f} pure_sign=True")

        return grad

    def _colnorm_for_qkv(self, grad: torch.Tensor, eps: float, param: torch.nn.Parameter = None, name: str = "") -> torch.Tensor:
        """
        Column normalization for QKV matrices (because they are transposed)
        Column normalization for QKV matrices (because they are transposed)
        """
        if grad is None:
            return grad

        original_shape = grad.shape

        # Handle 2D matrices
        if grad.ndim == 2:
            g2d_float = grad.float()
        else:
            return grad  # QKV is usually 2D

        # Column normalization (normalize along dimension 0)
        # Column normalization (normalize along dimension 0)
        norms = g2d_float.norm(dim=0, keepdim=True)  # Shape: [1, fan_in]

        adaptive_eps = eps * torch.ones_like(norms)
        adaptive_eps = torch.max(adaptive_eps, norms * 1e-6)
        norms = norms.clamp_min(adaptive_eps)

        normalized = g2d_float / norms

        # CRITICAL: Swap fan_in and fan_out for QKV
        # Critical: Swap fan_in and fan_out for QKV
        fan_in_actual = param.shape[0]   # QKV: the first dimension is the actual fan_in
        fan_out_actual = param.shape[1]  # QKV: the second dimension is the actual fan_out

        # For RowNorm using (2,mean) → ℓ∞ geometry, when applied to QKV:
        # We use column normalization, so the scaling becomes:
        # scale = 1 / fan_in_actual^(1/2) (p=2 for this optimizer)
        # But since we're normalizing columns (which are the actual outputs),
        # we use: scale = fan_out_actual^(1/2) / fan_in_actual

        p = 2.0  # RowNormOpNormConstraintPureSignGD uses p=2.0

        # Use column normalization scaling formula
        scale = math.pow(fan_out_actual, 1.0/p) / fan_in_actual
        normalized = normalized * scale

        if self._step % self.defaults.get('log_interval', 50) == 0:
            self.logger.info(
                f"[ColNorm-QKV] name={name} shape={original_shape} "
                f"fan_in(actual)={fan_in_actual} fan_out(actual)={fan_out_actual} "
                f"scale={scale:.6f} p={p:.1f}"
            )

        grad.copy_(normalized.to(grad.dtype))
        return grad

    def _rownorm_inplace(self, grad: torch.Tensor, eps: float, param: torch.nn.Parameter = None, name: str = "") -> torch.Tensor:
        """Row-normalize with opnorm_constraint: g[i,:] = (g[i,:] / ||g[i,:]||_2) × (1/√fan_in)"""
        if grad is None:
            return grad

        original_shape = grad.shape

        # Handle 2D (Linear/Attention) and 4D (Conv)
        if grad.ndim == 2:
            g2d_float = grad.float()
        elif grad.ndim == 4:
            # Reshape [out_c, in_c, kH, kW] -> [out_c, in_c*kH*kW]
            out_channels = grad.size(0)
            g2d_float = grad.reshape(out_channels, -1).float()
        else:
            return grad

        # Row normalization
        norms = g2d_float.norm(dim=1, keepdim=True)
        adaptive_eps = eps * torch.ones_like(norms)
        adaptive_eps = torch.max(adaptive_eps, norms * 1e-6)
        norms = norms.clamp_min(adaptive_eps)

        normalized = g2d_float / norms

        # Apply opnorm_constraint scaling: 1/√fan_in
        fan_in = _fan_in_of_weight(param) if param is not None else None
        if fan_in is not None:
            opnorm_scale = 1.0 / math.sqrt(fan_in)
            normalized = normalized * opnorm_scale

            if self._step % self.defaults.get('log_interval', 50) == 0:
                self.logger.info(
                    f"[RowNorm] name={name} shape={original_shape} fan_in={fan_in} scale=1/sqrt({fan_in})={opnorm_scale:.6f}")
        else:
            self.logger.warning(f"Cannot compute fan_in for {name}")

        # Reshape back to original
        if grad.ndim == 4:
            normalized = normalized.reshape(original_shape)

        grad.copy_(normalized.to(grad.dtype))
        return grad

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            lr = group["lr"]
            eps = group["eps"]
            max_grad_norm = group["max_grad_norm"]

            # Global gradient clipping
            gradients = [p.grad for p in group["params"] if p.grad is not None]
            if gradients and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(gradients, max_grad_norm)

            # Parameter name mapping
            param_names = {}
            if self.named_parameters:
                for name, param in self.named_parameters:
                    param_names[id(param)] = name

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                param_type = getattr(param, '_optimizer_param_type', 'other')
                is_qkv = getattr(param, '_is_qkv', False)
                param_name = param_names.get(id(param), f"param_{id(param)}")

                # Apply optimization strategy based on parameter type
                if param_type == 'weight_qkv_special':
                    # QKV 特殊处理：RowNorm 优化器对 QKV 用列归一化
                    self._colnorm_for_qkv(grad, eps, param, param_name)
                elif param_type == 'weight_rownorm':
                    # 标准权重：RowNorm
                    self._rownorm_inplace(grad, eps, param, param_name)
                elif param_type == 'param_1d_signgd':
                    self._signgd_1d_pure(grad, eps, param_name)

                # Weight decay (AdamW-style)
                if weight_decay != 0:
                    param.mul_(1 - lr * weight_decay)

                # Momentum
                param_state = self.state[param]
                if len(param_state) == 0:
                    param_state["momentum_buffer"] = torch.zeros_like(param)

                buf = param_state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)

                if nesterov:
                    param.add_(grad + momentum * buf, alpha=-lr)
                else:
                    param.add_(buf, alpha=-lr)

        self._step += 1
        return loss


def create_rownorm_opnorm_optimizer(
    model: nn.Module,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    use_row_normalized_init: bool = True,
    **kwargs
) -> RowNormOpNormConstraintPureSignGD:
    if use_row_normalized_init:
        model.apply(lambda m: init_rownorm_weights(m, p=2.0))

    named_params = list(model.named_parameters())
    param_list = [param for name, param in named_params if param.requires_grad]

    return RowNormOpNormConstraintPureSignGD(
        param_list,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        named_parameters=named_params,
        use_row_normalized_init=False,  # Already initialized
        **kwargs
    )


if __name__ == "__main__":
    print("=" * 80)
    print("Testing Row-Normalized Initialization for RowNorm OpNorm")
    print("=" * 80)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(name)s - %(levelname)s - %(message)s'
    )

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
    check_row_normalization(model, p=2.0, verbose=True)

    print("\n2. After row-normalized initialization:")
    print("-" * 80)
    model.apply(lambda m: init_rownorm_weights(m, p=2.0))
    check_row_normalization(model, p=2.0, verbose=True)

    print("\n3. Creating RowNorm optimizer with initialization:")
    print("-" * 80)
    optimizer = create_rownorm_opnorm_optimizer(
        model,
        lr=0.01,
        use_row_normalized_init=True
    )
    print(f"✓ Optimizer created: {type(optimizer).__name__}")

    print("\n" + "=" * 80)
    print("All tests completed!")
