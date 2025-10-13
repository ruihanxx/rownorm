"""
ColNorm + Pure SignGD Optimizer
Universal support for ResNet (Conv) and Transformer architectures

Hyperparameters:
---------------
- lr: Learning rate (default: 0.01)
- beta_1: Momentum coefficient for normalization buffer (default: 0.9)
- beta_2: Momentum coefficient for main buffer (default: 0.9)
- weight_decay: Weight decay coefficient (default: 0.0)
- nesterov: Use Nesterov momentum (default: True)
- eps: Epsilon for numerical stability (default: 1e-8)
- max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
- log_interval: Steps between logging (default: 50)

Dual Momentum System:
---------------------
M_t+1 = beta_2 * M_t + (1-beta_2) * grad        # Main momentum buffer
M̃_t+1 = beta_1 * M̃_t + (1-beta_1) * grad        # Normalization buffer
normalized_grad = normalize(M̃_t+1)               # Only normalize the second buffer
"""

import math
import logging
from typing import Iterable, List, Optional

import torch
import torch.nn as nn


def init_colnorm_weights(module: nn.Module) -> None:
    """
    Initialize model weights for the ColNorm optimizer.

    Initialization strategy:
    - Linear/Conv weights: Column-normalized with sqrt(fan_out)/fan_in scaling
    - Bias: 0
    - BatchNorm/LayerNorm: Standard initialization
    - Embedding: No modification

    Args:
        module: PyTorch module

    Mathematical formula:
        For 2D weight W ∈ R^(fan_out × fan_in):
        1. W ~ N(0, 1)
        2. W[:,j] = W[:,j] / ||W[:,j]||_2  (column normalization)
        3. W = W * sqrt(fan_out) / fan_in   (scaling)
    """
    if isinstance(module, nn.Linear):
        weight = module.weight
        fan_out, fan_in = weight.shape

        # 1. Standard normal initialization
        nn.init.normal_(weight, mean=0.0, std=1.0)

        # 2. Column-wise L2 normalization
        with torch.no_grad():
            for j in range(fan_in):
                col = weight[:, j]
                col_norm = torch.norm(col, p=2)
                if col_norm > 1e-8:
                    weight[:, j] = col / col_norm

            # 3. Double scaling: sqrt(fan_out) / fan_in
            scale = math.sqrt(fan_out) / fan_in
            weight.mul_(scale)

        # Bias initialized to 0
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Conv2d):
        weight = module.weight
        out_c, in_c, kh, kw = weight.shape

        # Reshape to 2D: [out_c, in_c*kH*kW]
        fan_out = out_c
        fan_in = in_c * kh * kw
        weight_2d = weight.view(fan_out, fan_in)

        # 1. Standard normal initialization
        nn.init.normal_(weight_2d, mean=0.0, std=1.0)

        # 2. Column-wise normalization
        with torch.no_grad():
            for j in range(fan_in):
                col = weight_2d[:, j]
                col_norm = torch.norm(col, p=2)
                if col_norm > 1e-8:
                    weight_2d[:, j] = col / col_norm

            # 3. Double scaling
            scale = math.sqrt(fan_out) / fan_in
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


def check_column_normalization(model: nn.Module, verbose: bool = True) -> dict:
    """
    Validate column normalization of model weights

    """
    results = {}

    for name, param in model.named_parameters():
        if 'weight' in name and param.ndim == 2:
            W = param.data.float()
            fan_out, fan_in = W.shape

            # 计算每列的 L2 范数
            col_norms = torch.norm(W, p=2, dim=0)

            # 期望的范数值（考虑缩放）
            expected_norm = math.sqrt(fan_out) / fan_in

            # 统计
            mean_norm = col_norms.mean().item()
            std_norm = col_norms.std().item()
            min_norm = col_norms.min().item()
            max_norm = col_norms.max().item()

            results[name] = {
                "mean": mean_norm,
                "std": std_norm,
                "min": min_norm,
                "max": max_norm,
                "expected": expected_norm,
                "shape": (fan_out, fan_in)
            }

            if verbose:
                print(f"{name:40s} | shape={str((fan_out, fan_in)):12s} | "
                      f"col_norm: mean={mean_norm:.6f} (expected={expected_norm:.6f}) "
                      f"std={std_norm:.6f} [{min_norm:.6f}, {max_norm:.6f}]")

    return results


def _fan_in_of_weight(param):
    """Calculate fan_in value for weight parameters.

    Linear: [out, in] -> fan_in = in
    Conv2d: [out_c, in_c, kH, kW] -> fan_in = in_c * kH * kW
    Other: return None
    """
    if param.ndim == 2:
        return param.shape[1]
    elif param.ndim == 4:
        return param.shape[1] * param.shape[2] * param.shape[3]
    return None


def _fan_out_of_weight(param):
    """Calculate fan_out value for weight parameters.

    Linear: [out, in] -> fan_out = out
    Conv2d: [out_c, in_c, kH, kW] -> fan_out = out_c
    Other: return None
    """
    if param.ndim >= 2:
        return param.shape[0]
    return None


def _is_weight_for_colnorm(name, param):
    """Determine if this parameter should use ColNorm.

    Applies to:
    - 2D: Transformer attention/FFN weights, ResNet FC layers
    - 4D: ResNet conv layers
    """
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


class ColNormPureSignGD(torch.optim.Optimizer):
    """
    Column Normalization with sqrt(fan_out)/fan_in scaling + Pure SignGD for 1D.

    This optimizer uses (1,mean) → (2,mean) geometry.
    Weights should be initialized with column-normalized Gaussian to match this geometry.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.01,
        beta_1: float = 0.9,  # Momentum for normalization buffer
        beta_2: float = 0.9,  # Momentum for main buffer
        weight_decay: float = 0.0,
        nesterov: bool = True,
        eps: float = 1e-8,
        max_grad_norm: float = 1.0,
        named_parameters: Optional[List] = None,
        log_interval: int = 50,
        use_column_normalized_init: bool = False,
        # Backward compatibility
        momentum: Optional[float] = None,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        # Backward compatibility: if momentum is provided, use it for both beta_1 and beta_2
        if momentum is not None:
            beta_1 = momentum
            beta_2 = momentum

        if beta_1 < 0.0 or beta_1 > 1.0:
            raise ValueError(f"Invalid beta_1 value: {beta_1}")
        if beta_2 < 0.0 or beta_2 > 1.0:
            raise ValueError(f"Invalid beta_2 value: {beta_2}")
        if nesterov and (beta_1 <= 0 or beta_2 <= 0):
            raise ValueError("Nesterov momentum requires beta_1 > 0 and beta_2 > 0")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if max_grad_norm <= 0.0:
            raise ValueError(f"Invalid max_grad_norm: {max_grad_norm}")

        defaults = dict(lr=lr, beta_1=beta_1, beta_2=beta_2, weight_decay=weight_decay,
                        nesterov=nesterov, eps=eps, max_grad_norm=max_grad_norm,
                        log_interval=log_interval,
                        use_column_normalized_init=use_column_normalized_init)
        super().__init__(params, defaults)

        self.named_parameters = named_parameters or []
        self.logger = logging.getLogger(self.__class__.__name__)
        self._step = 0

        self._classify_parameters()

        # Apply column-normalized initialization
        if use_column_normalized_init and self.named_parameters:
            self._apply_column_normalized_init()

    def _apply_column_normalized_init(self):
        """Apply column-normalized initialization to all parameters"""
        self.logger.info("Applying column-normalized initialization...")

        for name, param in self.named_parameters:
            if param.requires_grad:
                if param.ndim == 2:
                    temp_module = nn.Linear(
                        param.size(1), param.size(0), bias=False)
                    temp_module.weight = param
                    init_colnorm_weights(temp_module)
                elif param.ndim == 4:
                    temp_module = nn.Conv2d(
                        param.size(1), param.size(0),
                        param.size(2), bias=False
                    )
                    temp_module.weight = param
                    init_colnorm_weights(temp_module)

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
                'attn.in_proj',  # Another common naming
            ])

            if is_qkv and param.ndim == 2:  # Only process 2D weight matrices
                param._is_qkv = True
                param._optimizer_param_type = 'weight_qkv_special'
            elif _is_weight_for_colnorm(name, param):
                param._is_qkv = False
                param._optimizer_param_type = 'weight_colnorm'
            elif _is_1d_for_signgd(name, param):
                param._optimizer_param_type = 'param_1d_signgd'
            elif 'embed' in name.lower():
                param._optimizer_param_type = 'embedding'
            else:
                param._optimizer_param_type = 'other'

        # Print QKV parameter information
        if self.named_parameters:
            qkv_params = [name for name, p in self.named_parameters if getattr(
                p, '_is_qkv', False)]
            if qkv_params:
                self.logger.info(
                    f"Detected {len(qkv_params)} QKV parameter(s):")
                for name in qkv_params:
                    self.logger.info(f"  - {name}")

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

    def _rownorm_for_qkv(self, grad: torch.Tensor, eps: float, param: torch.nn.Parameter = None, name: str = "") -> torch.Tensor:
        """
        Row normalization for QKV matrices (because they are transposed)
        When using ColNorm on standard layers, use RowNorm on QKV
        """
        if grad is None:
            return grad

        original_shape = grad.shape

        # Handle 2D matrices
        if grad.ndim == 2:
            g2d_float = grad.float()
        else:
            return grad

        # Row normalization (normalize along dimension 1)
        norms = g2d_float.norm(dim=1, keepdim=True)  # Shape: [fan_out, 1]

        adaptive_eps = eps * torch.ones_like(norms)
        adaptive_eps = torch.max(adaptive_eps, norms * 1e-6)
        norms = norms.clamp_min(adaptive_eps)

        normalized = g2d_float / norms

        # CRITICAL: Swap fan_in and fan_out for QKV
        fan_in_actual = param.shape[0]   # QKV: the first dimension is the actual fan_in
        fan_out_actual = param.shape[1]  # QKV: the second dimension is the actual fan_out

        # For ColNorm using (1,mean) → (2,mean) geometry, when applied to QKV:
        # We use row normalization with swapped fan values
        p = 2.0  # ColNorm typically uses p=2
        scale = 1.0 / math.pow(fan_in_actual, 1.0/p)
        normalized = normalized * scale

        if self._step % self.defaults.get('log_interval', 50) == 0:
            self.logger.info(
                f"[RowNorm-QKV] name={name} shape={original_shape} "
                f"fan_in(actual)={fan_in_actual} fan_out(actual)={fan_out_actual} "
                f"scale={scale:.6f}"
            )

        grad.copy_(normalized.to(grad.dtype))
        return grad

    def _colnorm_inplace(self, grad: torch.Tensor, eps: float, param: torch.nn.Parameter = None, name: str = "") -> torch.Tensor:
        """Column-normalize with scaling: g[:,j] = (g[:,j] / ||g[:,j]||_2) × √(fan_out)/fan_in"""
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

        # Column normalization (normalize along dimension 0)
        norms = g2d_float.norm(dim=0, keepdim=True)  # Shape: [1, in_features]

        adaptive_eps = eps * torch.ones_like(norms)
        adaptive_eps = torch.max(adaptive_eps, norms * 1e-6)
        norms = norms.clamp_min(adaptive_eps)

        normalized = g2d_float / norms

        # Apply scaling: √(fan_out)/fan_in
        fan_in = _fan_in_of_weight(param) if param is not None else None
        fan_out = _fan_out_of_weight(param) if param is not None else None

        if fan_in is not None and fan_out is not None:
            scale = math.sqrt(fan_out) / fan_in
            normalized = normalized * scale

            if self._step % self.defaults.get('log_interval', 50) == 0:
                self.logger.info(
                    f"[ColNorm] name={name} shape={original_shape} "
                    f"fan_in={fan_in} fan_out={fan_out} scale=sqrt({fan_out})/{fan_in}={scale:.6f}")
        else:
            self.logger.warning(f"Cannot compute fan_in/fan_out for {name}")

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
            beta_1 = group["beta_1"]  # For normalization buffer
            beta_2 = group["beta_2"]  # For main buffer
            nesterov = group["nesterov"]
            lr = group["lr"]
            eps = group["eps"]
            max_grad_norm = group["max_grad_norm"]

            # Parameter name mapping
            param_names = {}
            if self.named_parameters:
                for name, param in self.named_parameters:
                    param_names[id(param)] = name

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad  # Raw gradient - do not modify!
                param_type = getattr(param, '_optimizer_param_type', 'other')
                is_qkv = getattr(param, '_is_qkv', False)
                param_name = param_names.get(id(param), f"param_{id(param)}")

                # 1. Dual Momentum System
                param_state = self.state[param]

                if beta_1 > 0 or beta_2 > 0:
                    # Initialize two momentum buffers
                    if len(param_state) == 0:
                        param_state["momentum_buffer_1"] = torch.zeros_like(param)  # M̃_t (for normalization)
                        param_state["momentum_buffer_2"] = torch.zeros_like(param)  # M_t (main buffer)

                    buf_1 = param_state["momentum_buffer_1"]  # Normalization buffer
                    buf_2 = param_state["momentum_buffer_2"]  # Main buffer

                    # Update both momentum buffers with raw gradient
                    buf_1.mul_(beta_1).add_(grad, alpha=1-beta_1)  # M̃_t+1 = beta_1 * M̃_t + (1-beta_1) * grad
                    buf_2.mul_(beta_2).add_(grad, alpha=1-beta_2)  # M_t+1 = beta_2 * M_t + (1-beta_2) * grad

                    # Use buf_1 for normalization (critical: only normalize the second buffer!)
                    effective_grad = buf_1.clone()
                else:
                    # No momentum
                    effective_grad = grad.clone()

                # 2. Now apply normalization to the effective gradient (in-place)
                # IMPORTANT: This only normalizes buf_1 (M̃_t+1), not buf_2 (M_t+1)
                if param_type == 'weight_qkv_special':
                    # QKV special handling: ColNorm optimizer uses row normalization for QKV
                    self._rownorm_for_qkv(effective_grad, eps, param, param_name)
                elif param_type == 'weight_colnorm':
                    # Standard weights: ColNorm
                    self._colnorm_inplace(effective_grad, eps, param, param_name)
                elif param_type == 'param_1d_signgd':
                    self._signgd_1d_pure(effective_grad, eps, param_name)
                # For embeddings and others: no normalization

                # 3. Clip the normalized effective gradient (not raw gradient!)
                if max_grad_norm > 0:
                    effective_grad_norm = torch.norm(effective_grad)
                    if effective_grad_norm > max_grad_norm:
                        effective_grad.mul_(max_grad_norm / (effective_grad_norm + eps))

                # 4. Weight decay (AdamW-style)
                if weight_decay != 0:
                    param.mul_(1 - lr * weight_decay)

                # 5. Parameter update
                param.add_(effective_grad, alpha=-lr)

        self._step += 1
        return loss


def create_colnorm_optimizer(
    model: nn.Module,
    lr: float = 0.01,
    beta_1: float = 0.9,
    beta_2: float = 0.9,
    weight_decay: float = 0.0,
    use_column_normalized_init: bool = True,
    momentum: Optional[float] = None,  # Backward compatibility
    **kwargs
) -> ColNormPureSignGD:
    if use_column_normalized_init:
        model.apply(init_colnorm_weights)

    named_params = list(model.named_parameters())
    param_list = [param for name, param in named_params if param.requires_grad]

    return ColNormPureSignGD(
        param_list,
        lr=lr,
        beta_1=beta_1,
        beta_2=beta_2,
        weight_decay=weight_decay,
        named_parameters=named_params,
        use_column_normalized_init=False,  # 已经初始化过了
        momentum=momentum,  # For backward compatibility
        **kwargs
    )


# Backwards-compatible alias
ColNormSGD = ColNormPureSignGD


if __name__ == "__main__":
    print("=" * 80)
    print("Testing Column-Normalized Initialization for ColNorm")
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
    check_column_normalization(model, verbose=True)

    print("\n2. After column-normalized initialization:")
    print("-" * 80)
    model.apply(init_colnorm_weights)
    check_column_normalization(model, verbose=True)

    print("\n3. Creating ColNorm optimizer with initialization:")
    print("-" * 80)
    optimizer = create_colnorm_optimizer(
        model,
        lr=0.01,
        use_column_normalized_init=True
    )
    print(f"✓ Optimizer created: {type(optimizer).__name__}")

    print("\n" + "=" * 80)
    print("All tests completed!")
