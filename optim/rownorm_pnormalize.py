"""
Hyperparameters:
---------------
- lr: Learning rate (default: 0.1)
- beta_1: Momentum coefficient for normalization buffer (default: 0.9)
- beta_2: Momentum coefficient for main buffer (default: 0.9)
- p: Power parameter, must be >= 2 (default: 2.0)
- use_width_scaling: Whether to use fan_in^{-1/p} scaling (default: True)
- weight_decay: Weight decay coefficient, decoupled (default: 0.0)
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


def init_rownorm_weights(module: nn.Module, p: float = 2.0) -> None:
    """
    Initialize model weights for RowNorm optimizers.

    Initialization strategy:
    - Linear/Conv weights: Row-normalized with scaling
    - Bias: 0
    - BatchNorm/LayerNorm: Standard initialization
    - Embedding: No modification

    Args:
        module: PyTorch module
        p: Norm parameter (default: 2.0)
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


def _is_weight_for_rownorm(name, param):
    """Determine if this parameter should use RowNorm.

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


class RowNormPNormalize(torch.optim.Optimizer):
    """
    For weight matrices, the update for each row i is:
    - With width scaling:
      Θ_i ← Θ_i - η * fan_in^{-1/p} * sign(M_i)|M_i|^{p-1} / ||M_i||_p^{p-1}
    - Without width scaling:
      Θ_i ← Θ_i - η * sign(M_i)|M_i|^{p-1} / ||M_i||_p^{p-1}

    Args:
        params: Iterable of parameters or dicts defining parameter groups
        lr: Learning rate (default: 0.1)
        momentum: Momentum coefficient (default: 0.9)
        p: Power parameter, must be >= 2 (default: 2.0)
        use_width_scaling: Whether to use fan_in^{-1/p} scaling (default: True)
        weight_decay: Weight decay coefficient, decoupled (default: 0.0)
        nesterov: Use Nesterov momentum (default: True)
        eps: Epsilon for numerical stability (default: 1e-8)
        max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
        named_parameters: List of named parameters for logging (optional)
        log_interval: Logging interval in steps (default: 50)

    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.1,
        beta_1: float = 0.9,  # Momentum for normalization buffer
        beta_2: float = 0.9,  # Momentum for main buffer
        p: float = 2.0,
        use_width_scaling: bool = True,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        eps: float = 1e-8,
        max_grad_norm: float = 1.0,
        named_parameters: Optional[List] = None,
        log_interval: int = 50,
        # Backward compatibility
        momentum: Optional[float] = None,
    ) -> None:
        if p < 2.0 and not math.isinf(p):
            raise ValueError(f"p must be >= 2 or inf, got {p}")
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

        defaults = dict(
            lr=lr,
            beta_1=beta_1,
            beta_2=beta_2,
            p=p,
            use_width_scaling=use_width_scaling,
            weight_decay=weight_decay,
            nesterov=nesterov,
            eps=eps,
            max_grad_norm=max_grad_norm,
            log_interval=log_interval
        )
        super().__init__(params, defaults)

        self.named_parameters = named_parameters or []
        self.logger = logging.getLogger(self.__class__.__name__)
        self._step = 0

        self._classify_parameters()

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

            if is_qkv and param.ndim == 2:  # 只处理 2D 权重矩阵
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

        # 统计并打印 QKV 参数信息
        if self.named_parameters:
            qkv_params = [name for name, p in self.named_parameters if getattr(
                p, '_is_qkv', False)]
            if qkv_params:
                self.logger.info(
                    f"Detected {len(qkv_params)} QKV parameter(s):")
                for name in qkv_params:
                    self.logger.info(f"  - {name}")

    def _colnorm_for_qkv(self, grad: torch.Tensor, eps: float, p: float, use_width_scaling: bool, param: torch.nn.Parameter = None, name: str = "") -> torch.Tensor:
        """
        Column normalization for QKV matrices (because they are transposed)
        """
        if grad is None:
            return grad

        original_shape = grad.shape

        # Handle 2D matrices
        if grad.ndim == 2:
            g2d_float = grad.float()
        else:
            return grad  # QKV 通常是 2D

        num_rows, fan_in = g2d_float.shape

        # Column normalization with p-norm power transform
        for j in range(fan_in):
            col = g2d_float[:, j]

            # Special handling for p=inf (L∞ norm)
            if math.isinf(p):
                # 1. Compute L∞ norm (max absolute value)
                col_norm_inf = torch.max(torch.abs(col))

                # Numerical stability: if norm is too small, set to zero
                if col_norm_inf < eps:
                    g2d_float[:, j] = torch.zeros_like(col)
                    continue

                # 2. Normalize by L∞ norm: col / ||col||_∞
                normalized = col / col_norm_inf
                g2d_float[:, j] = normalized
            else:
                # Standard p-norm case (p >= 2)
                # 1. First compute p-norm and check for numerical stability
                col_norm_p = torch.norm(col, p=p)

                # Numerical stability: if norm is too small, set to zero
                if col_norm_p < eps:
                    g2d_float[:, j] = torch.zeros_like(col)
                    continue

                # 2. Power transform: sign(col) * |col|^(p-1)
                col_sign = torch.sign(col)
                col_abs = torch.abs(col)
                powered = col_sign * \
                    torch.pow(torch.clamp(col_abs, min=1e-20), p - 1)

                # 3. Normalize: powered / ||col||_p^(p-1)
                normalized = powered / torch.pow(col_norm_p, p - 1)

                g2d_float[:, j] = normalized

        # CRITICAL: Swap fan_in and fan_out for QKV
        fan_in_actual = param.shape[0]   # QKV: 第一维是实际的 fan_in
        fan_out_actual = param.shape[1]  # QKV: 第二维是实际的 fan_out

        # 4. Optional width scaling for QKV (using swapped dimensions)
        if use_width_scaling and not math.isinf(p):
            # For QKV column normalization, we scale by fan_out_actual^(-1/p)
            # Note: For p=inf, fan_out^(-1/inf) = fan_out^0 = 1 (no scaling)
            scale = torch.pow(torch.tensor(
                fan_out_actual, dtype=torch.float32), -1.0/p)
            g2d_float = g2d_float * scale

        if self._step % self.defaults.get('log_interval', 50) == 0:
            if use_width_scaling and not math.isinf(p):
                scale_value = fan_out_actual ** (-1.0/p)
                scale_info = f"scale={scale_value:.6f}"
            else:
                scale_info = "no_scale" if not use_width_scaling else "p=inf (scale=1)"

            p_str = "inf" if math.isinf(p) else f"{p:.1f}"
            self.logger.info(
                f"[ColNorm-QKV] name={name} shape={original_shape} "
                f"fan_in(actual)={fan_in_actual} fan_out(actual)={fan_out_actual} "
                f"p={p_str} {scale_info}"
            )

        grad.copy_(g2d_float.to(grad.dtype))
        return grad

    def _apply_pnormalize_row(
        self,
        grad: torch.Tensor,
        eps: float,
        p: float,
        use_width_scaling: bool,
        param: torch.nn.Parameter = None,
        param_name: str = ""
    ) -> torch.Tensor:
        """
        Apply row-wise p-normalization with power transform.

        Args:
            grad: Gradient tensor [num_rows, fan_in]
            eps: Numerical stability constant
            p: Power parameter
            use_width_scaling: Whether to use width scaling
            param: Parameter (for computing fan_in)
            param_name: Parameter name (for logging)

        Returns:
            Processed gradient
        """
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

        num_rows, fan_in = g2d_float.shape

        # Process each row
        for i in range(num_rows):
            row = g2d_float[i, :]

            # Special handling for p=inf (L∞ norm)
            if math.isinf(p):
                # 1. Compute L∞ norm (max absolute value)
                row_norm_inf = torch.max(torch.abs(row))

                # Numerical stability: if norm is too small, set to zero
                if row_norm_inf < eps:
                    g2d_float[i, :] = torch.zeros_like(row)
                    continue

                # 2. Normalize by L∞ norm: row / ||row||_∞
                normalized = row / row_norm_inf

                # Note: For p=inf, fan_in^(-1/inf) = fan_in^0 = 1 (no scaling effect)
                # So we don't apply width scaling for p=inf

                g2d_float[i, :] = normalized
            else:
                # Standard p-norm case (p >= 2)
                # 1. First compute p-norm and check for numerical stability
                row_norm_p = torch.norm(row, p=p)

                # Numerical stability: if norm is too small, set to zero
                if row_norm_p < eps:
                    g2d_float[i, :] = torch.zeros_like(row)
                    continue

                # 2. Power transform: sign(row) * |row|^(p-1)
                # Use clamp for numerical stability without changing mathematical form
                row_sign = torch.sign(row)
                row_abs = torch.abs(row)
                powered = row_sign * \
                    torch.pow(torch.clamp(row_abs, min=1e-20), p - 1)

                # 3. Normalize: powered / ||row||_p^(p-1)
                # No need to add eps since we already checked row_norm_p >= eps
                normalized = powered / torch.pow(row_norm_p, p - 1)

                # 4. Optional width scaling
                if use_width_scaling:
                    scale = torch.pow(torch.tensor(
                        fan_in, dtype=torch.float32), -1.0/p)
                    normalized = normalized * scale

                g2d_float[i, :] = normalized

        # Logging
        if self._step % self.defaults.get('log_interval', 50) == 0:
            if use_width_scaling and not math.isinf(p):
                scale_value = fan_in ** (-1.0/p)
                scale_info = f"scale={scale_value:.6f}"
            else:
                scale_info = "no_scale" if not use_width_scaling else "p=inf (scale=1)"

            p_str = "inf" if math.isinf(p) else f"{p:.1f}"
            self.logger.info(
                f"[PNormalize] name={param_name} shape={original_shape} "
                f"p={p_str} {scale_info} fan_in={fan_in}"
            )

        # Reshape back to original if needed
        if grad.ndim == 4:
            g2d_float = g2d_float.reshape(original_shape)

        grad.copy_(g2d_float.to(grad.dtype))
        return grad

    def _signgd_1d(self, grad: torch.Tensor, eps: float, param_name: str = "") -> torch.Tensor:
        """Apply SignGD to 1D parameters: g = sign(g)"""
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
                f"[SignGD] name={param_name} dim={d} grad_norm={grad_norm:.6f}"
            )

        return grad

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        """Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta_1 = group['beta_1']  # For normalization buffer
            beta_2 = group['beta_2']  # For main buffer
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']
            eps = group['eps']
            max_grad_norm = group['max_grad_norm']
            p = group['p']
            use_width_scaling = group['use_width_scaling']

            # 1. Create parameter name mapping
            param_names = {}
            if self.named_parameters:
                for name, param in self.named_parameters:
                    param_names[id(param)] = name

            # 2. Process each parameter
            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad  # Raw gradient - do not modify!
                param_type = getattr(param, '_optimizer_param_type', 'other')
                is_qkv = getattr(param, '_is_qkv', False)
                param_name = param_names.get(id(param), f"param_{id(param)}")

                # 3. Dual Momentum System
                state = self.state[param]

                if beta_1 > 0 or beta_2 > 0:
                    # Initialize two momentum buffers
                    if len(state) == 0:
                        state["momentum_buffer_1"] = torch.zeros_like(param)  # M̃_t (for normalization)
                        state["momentum_buffer_2"] = torch.zeros_like(param)  # M_t (main buffer)

                    buf_1 = state["momentum_buffer_1"]  # Normalization buffer
                    buf_2 = state["momentum_buffer_2"]  # Main buffer

                    # Update both momentum buffers with raw gradient
                    buf_1.mul_(beta_1).add_(grad, alpha=1-beta_1)  # M̃_t+1 = beta_1 * M̃_t + (1-beta_1) * grad
                    buf_2.mul_(beta_2).add_(grad, alpha=1-beta_2)  # M_t+1 = beta_2 * M_t + (1-beta_2) * grad

                    # Use buf_1 for normalization (critical: only normalize the second buffer!)
                    effective_grad = buf_1.clone()
                else:
                    # No momentum
                    effective_grad = grad.clone()

                # 4. Now apply normalization to the effective gradient (in-place)
                # IMPORTANT: This only normalizes buf_1 (M̃_t+1), not buf_2 (M_t+1)
                if param_type == 'weight_qkv_special':
                    # QKV special handling: RowNorm optimizer uses column normalization for QKV
                    self._colnorm_for_qkv(
                        effective_grad, eps, p, use_width_scaling, param, param_name)
                elif param_type == 'weight_rownorm':
                    # Standard weights: RowNorm
                    self._apply_pnormalize_row(
                        effective_grad, eps, p, use_width_scaling, param, param_name
                    )
                elif param_type == 'param_1d_signgd':
                    self._signgd_1d(effective_grad, eps, param_name)
                # For embeddings and others: no normalization

                # 5. Clip the normalized effective gradient (not raw gradient!)
                if max_grad_norm > 0:
                    effective_grad_norm = torch.norm(effective_grad)
                    if effective_grad_norm > max_grad_norm:
                        effective_grad.mul_(max_grad_norm / (effective_grad_norm + eps))

                # 6. Decoupled weight decay (AdamW-style)
                if weight_decay != 0:
                    param.mul_(1.0 - lr * weight_decay)

                # 7. Parameter update
                param.add_(effective_grad, alpha=-lr)

        self._step += 1
        return loss


def create_rownorm_pnormalize_optimizer(
    model: torch.nn.Module,
    lr: float = 0.1,
    beta_1: float = 0.9,
    beta_2: float = 0.9,
    p: float = 2.0,
    use_width_scaling: bool = True,
    weight_decay: float = 0.0,
    momentum: Optional[float] = None,  # Backward compatibility
    **kwargs
) -> RowNormPNormalize:
    """
    Convenience function to create RowNormPNormalize optimizer for a model.
    Args:
        model: PyTorch model
        lr: Learning rate
        beta_1: Momentum coefficient for normalization buffer
        beta_2: Momentum coefficient for main buffer
        p: Power parameter (>= 2)
        use_width_scaling: Whether to use fan_in^{-1/p} scaling
        weight_decay: Weight decay coefficient
        momentum: Legacy parameter (for backward compatibility)
        **kwargs: Additional arguments to pass to the optimizer

    Returns:
        RowNormPNormalize optimizer instance
    """
    named_params = list(model.named_parameters())
    param_list = [param for name, param in named_params if param.requires_grad]

    return RowNormPNormalize(
        param_list,
        lr=lr,
        beta_1=beta_1,
        beta_2=beta_2,
        p=p,
        use_width_scaling=use_width_scaling,
        weight_decay=weight_decay,
        named_parameters=named_params,
        momentum=momentum,  # For backward compatibility
        **kwargs
    )


if __name__ == "__main__":
    print("=" * 80)
    print("Testing RowNormPNormalize optimizer")
    print("=" * 80)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(name)s - %(levelname)s - %(message)s'
    )

    # Create test model
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 10)
    )

    # Test cases
    test_cases = [
        (2.0, True, "p=2.0 with width scaling (default, paper version)"),
        (2.0, False, "p=2.0 without width scaling (ablation)"),
        (3.0, True, "p=3.0 with width scaling"),
        (4.0, True, "p=4.0 with width scaling"),
    ]

    x = torch.randn(32, 128)
    y = torch.randint(0, 10, (32,))

    for p, use_scaling, description in test_cases:
        print(f"\n{'='*80}")
        print(f"Test: {description}")
        print(f"{'='*80}")

        # Reset model
        model = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10)
        )

        optimizer = create_rownorm_pnormalize_optimizer(
            model,
            lr=0.1,
            p=p,
            use_width_scaling=use_scaling,
            log_interval=1  # Log every step for testing
        )

        # Test one optimization step
        output = model(x)
        loss = torch.nn.CrossEntropyLoss()(output, y)

        print(f"Initial loss: {loss.item():.4f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Forward pass again
        output = model(x)
        new_loss = torch.nn.CrossEntropyLoss()(output, y)
        print(f"After step loss: {new_loss.item():.4f}")
        print(f"Loss change: {new_loss.item() - loss.item():.4f}")

    print("\n" + "="*80)
    print(" All tests completed!")
    print("="*80)


def create_rownorm_pnormalize_optimizer_with_init(
    model: torch.nn.Module,
    lr: float = 0.1,
    beta_1: float = 0.9,
    beta_2: float = 0.9,
    p: float = 2.0,
    use_width_scaling: bool = True,
    weight_decay: float = 0.0,
    use_row_normalized_init: bool = True,
    momentum: Optional[float] = None,  # Backward compatibility
    **kwargs
) -> RowNormPNormalize:
    """
    Create RowNormPNormalize optimizer with optional initialization

    Args:
        model: PyTorch model
        lr: Learning rate
        beta_1: Momentum coefficient for normalization buffer
        beta_2: Momentum coefficient for main buffer
        p: Power parameter (>= 2)
        use_width_scaling: Whether to use fan_in^{-1/p} scaling
        weight_decay: Weight decay
        use_row_normalized_init: Whether to use row-normalized initialization
        momentum: Legacy parameter (for backward compatibility)
        **kwargs: Other parameters

    Returns:
        RowNormPNormalize optimizer instance
    """
    # Apply row-normalized initialization (using the corresponding p value)
    if use_row_normalized_init:
        model.apply(lambda m: init_rownorm_weights(m, p=p))

    named_params = list(model.named_parameters())
    param_list = [param for name, param in named_params if param.requires_grad]

    return RowNormPNormalize(
        param_list,
        lr=lr,
        beta_1=beta_1,
        beta_2=beta_2,
        p=p,
        use_width_scaling=use_width_scaling,
        weight_decay=weight_decay,
        named_parameters=named_params,
        momentum=momentum,  # For backward compatibility
        **kwargs
    )
