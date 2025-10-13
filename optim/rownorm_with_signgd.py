import math
import logging
from typing import Iterable, List, Optional

import torch


def _fan_in_of_weight(param):
    """
    Linear: [out, in] -> fan_in = in
    Conv2d: [out_c, in_c, kH, kW] -> fan_in = in_c * kH * kW
    Others: return None (row normalization not supported)
    """
    if param.ndim == 2:
        return param.shape[1]
    elif param.ndim == 4:
        return param.shape[1] * param.shape[2] * param.shape[3]
    return None



def _rows_l2_norms(t):
    """
    Compute the L2 norms of rows of a tensor reshaped to [num_rows, row_dim].
    """
    return t.norm(p=2, dim=1)  # shape: [num_rows]


class RowNormSGDWithSignGD(torch.optim.Optimizer):
    """
    Enhanced RowNormSGD that uses SignGD for 1-dimensional gradients.

    For different gradient dimensions:
    - grad.ndim == 1: Apply SignGD (sign-based gradient descent)
    - grad.ndim == 2: Apply row normalization with scaling
    - grad.ndim == 4: Apply row normalization (reshape to 2D first) with scaling
    - Other dimensions: Use standard SGD

    Row normalization scaling methods:
    - 'none': g[i, :] <- g[i, :] / ||g[i, :]||_2
    - 'mean_scale': g[i, :] <- (g[i, :] / ||g[i, :]||_2) * (sum(||g[i, :]||_2) / sqrt(n))
    - 'rms_scale': g[i, :] <- (g[i, :] / ||g[i, :]||_2) * sqrt(mean(||g[i, :]||_2^2))
    - 'dim_scale': g[i, :] <- (g[i, :] / ||g[i, :]||_2) * (1 / sqrt(n))
    - 'opnorm_constraint': g[i, :] <- (g[i, :] / ||g[i, :]||_2) * (1 / sqrt(fan_in))
    - 'opnorm_reg': g[i, :] <- (g[i, :] / ||g[i, :]||_2) * (sum(||g[i, :]||_2) / (2 * fan_in))

    For 1D gradients (BatchNorm weights/biases, Linear biases):
    - Apply SignGD: g <- sign(g) * ||g||_2 / sqrt(d) where d is the dimension

    Decoupled weight decay is applied (AdamW-style): p <- p * (1 - lr * weight_decay).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        eps: float = 1e-8,
        max_grad_norm: float = 1.0,
        scale_type: str = "none",
        signgd_1d: bool = True,  # Whether to use SignGD for 1D gradients
        signgd_scaling: str = "none",  # SignGD scaling: "none", "numel", "fan_in"
        log_interval: int = 50,
        named_parameters: Optional[List] = None,  # 添加named_parameters参数
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
        if scale_type not in ["none", "mean_scale", "rms_scale", "dim_scale", "opnorm_constraint", "opnorm_reg"]:
            raise ValueError(
                f"Invalid scale_type: {scale_type}. Must be one of: none, mean_scale, rms_scale, dim_scale, opnorm_constraint, opnorm_reg")
        if signgd_scaling not in ["none", "numel", "fan_in"]:
            raise ValueError(
                f"Invalid signgd_scaling: {signgd_scaling}. Must be one of: none, numel, fan_in")

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        nesterov=nesterov, eps=eps, max_grad_norm=max_grad_norm,
                        scale_type=scale_type, signgd_1d=signgd_1d,
                        signgd_scaling=signgd_scaling, log_interval=log_interval)
        super().__init__(params, defaults)

        # Initialize logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self._step = 0

        # Build parameter to fan_in mapping
        self.param_fan_in_map = {}
        self.param_names = {}

        if named_parameters:
            self.param_fan_in_map = self._build_param_fan_in_map(
                named_parameters)
            for name, param in named_parameters:
                self.param_names[id(param)] = name

            # Print mapping information for debugging
            self.logger.info("=" * 80)
            self.logger.info("Parameter fan_in mapping:")
            for name, param in named_parameters:
                param_id = id(param)
                fan_in = self.param_fan_in_map.get(param_id, None)
                self.logger.info(
                    f"  {name}: shape={tuple(param.shape)}, fan_in={fan_in}")
            self.logger.info("=" * 80)

    def _build_param_fan_in_map(self, named_parameters: List):
        """
        Build fan_in mapping for all parameters (including bias).

        Strategy:
        1. For weight parameters, directly compute fan_in
        2. For bias parameters, try to find corresponding weight parameters and use their fan_in

        Args:
            named_parameters: List of (name, parameter) tuples

        Returns:
            Dict mapping param id to fan_in value
        """
        param_fan_in_map = {}

        # First pass: collect all weight information
        weight_info = {}  # layer_prefix -> (param, fan_in)

        for name, param in named_parameters:
            if 'weight' in name and param.ndim >= 2:
                fan_in = _fan_in_of_weight(param)
                if fan_in is not None:
                    # 提取层名称前缀（去掉.weight部分）
                    layer_prefix = name.rsplit('.weight', 1)[0]
                    weight_info[layer_prefix] = (param, fan_in)
                    param_fan_in_map[id(param)] = fan_in

        # Second pass: match bias parameters to corresponding weight fan_in
        for name, param in named_parameters:
            if 'bias' in name:
                # Extract layer name prefix (remove .bias part)
                layer_prefix = name.rsplit('.bias', 1)[0]

                if layer_prefix in weight_info:
                    # Found corresponding weight
                    _, fan_in = weight_info[layer_prefix]
                    param_fan_in_map[id(param)] = fan_in
                else:
                    # No corresponding weight found, use bias dimension as fallback
                    param_fan_in_map[id(param)] = param.numel()
                    self.logger.warning(
                        f"Cannot find corresponding weight for bias parameter {name}, "
                        f"using bias dimension {param.numel()} as fan_in"
                    )

        # Third pass: process other parameters (LayerNorm, etc.)
        for name, param in named_parameters:
            if id(param) not in param_fan_in_map:
                # For LayerNorm, etc., use parameter dimension
                param_fan_in_map[id(param)] = param.numel()

        return param_fan_in_map

    def _signgd_1d(self, grad: torch.Tensor, eps: float, param: torch.nn.Parameter, scaling: str = "none", name: str = "") -> torch.Tensor:
        """Apply SignGD to 1-dimensional gradients.

        Three variants:
        - "none": g <- sign(g)  # Standard SignGD, no scaling
        - "numel": g <- sign(g) * ||g||_2 / sqrt(numel(g))  # Scale by gradient dimension
        - "fan_in": g <- sign(g) * ||g||_2 / sqrt(fan_in)  # Scale by layer input dimension
        """
        if grad is None or grad.ndim != 1:
            return grad

        grad_float = grad.float()
        grad_sign = torch.sign(grad_float)
        grad_norm = torch.norm(grad_float)  # Always compute grad_norm for logging

        if scaling == "none":
            # Standard SignGD: g <- sign(g)
            normalized = grad_sign
            scale_factor = 1.0
            scale_denominator = 1
            method_used = "standard"
        else:
            # Avoid division by zero
            if grad_norm < eps:
                return grad

            if scaling == "numel":
                # Scale by gradient dimension: g <- sign(g) * ||g||_2 / sqrt(numel(g))
                scale_denominator = grad.numel()
                method_used = "numel"
            elif scaling == "fan_in":
                # Scale by fan_in: g <- sign(g) * ||g||_2 / sqrt(fan_in)
                param_id = id(param) if param is not None else None
                fan_in = None
                if param_id is not None and hasattr(self, 'param_fan_in_map'):
                    fan_in = self.param_fan_in_map.get(param_id, None)

                if fan_in is None:
                    # Fallback to numel if fan_in not available
                    scale_denominator = grad.numel()
                    method_used = "numel_fallback"
                else:
                    scale_denominator = fan_in
                    method_used = "fan_in"
            else:
                raise ValueError(f"Unknown scaling method: {scaling}")

            scale_factor = grad_norm / math.sqrt(scale_denominator)
            normalized = grad_sign * scale_factor

        grad.copy_(normalized.to(grad.dtype))

        # Log
        if self._step % self.defaults.get('log_interval', 50) == 0:
            self.logger.info(
                f"[SignGD] name={name} type=1D method={method_used} "
                f"dim={grad.numel()} scale_denom={scale_denominator} "
                f"grad_norm={grad_norm:.6f} scale_factor={scale_factor:.6f}")

        return grad

    def _rownorm_inplace(self, grad: torch.Tensor, eps: float, scale_type: str, param: torch.nn.Parameter = None, name: str = "") -> torch.Tensor:
        """Row-normalize gradient for 2D or 4D tensors with different scaling methods.

        - If grad.ndim == 2: normalize rows.
        - If grad.ndim == 4: treat as (out_channels, in_channels * kH * kW).
        - Else: return grad unchanged.

        Args:
            scale_type: One of 'none', 'mean_scale', 'rms_scale', 'dim_scale', 'opnorm_constraint', 'opnorm_reg'
        """
        if grad is None:
            return grad

        if grad.ndim == 2:
            g2d_float = grad.float()
            norms = g2d_float.norm(dim=1, keepdim=True)
            # Use more stable normalization with adaptive epsilon
            adaptive_eps = eps * torch.ones_like(norms)
            # Scale eps with norm magnitude
            adaptive_eps = torch.max(adaptive_eps, norms * 1e-6)
            norms = norms.clamp_min(adaptive_eps)

            # Basic row normalization
            normalized = g2d_float / norms

            # Apply scaling based on scale_type
            n = grad.size(0)  # number of rows
            if scale_type == "mean_scale":
                # mean scale: (sum of row norms) / sqrt(n)
                mean_scale = norms.sum() / math.sqrt(n)
                normalized = normalized * mean_scale
            elif scale_type == "rms_scale":
                # RMS scale: sqrt(mean(norms^2))
                rms_scale = torch.sqrt(torch.mean(norms**2))
                normalized = normalized * rms_scale
            elif scale_type == "dim_scale":
                # dimension scale: 1 / sqrt(n)
                dim_scale = 1.0 / math.sqrt(n)
                normalized = normalized * dim_scale
            elif scale_type == "opnorm_constraint":
                # opnorm constraint: 1/sqrt(fan_in)
                fan_in = _fan_in_of_weight(
                    param) if param is not None else None
                if fan_in is not None:
                    opnorm_scale = 1.0 / math.sqrt(fan_in)
                    normalized = normalized * opnorm_scale
                    # 记录日志
                    if self._step % self.defaults.get('log_interval', 50) == 0:
                        self.logger.info(
                            f"[RowNorm] name={name} type=2D "
                            f"scale_type={scale_type} fan_in={fan_in} used_scale=1/sqrt({fan_in})={opnorm_scale:.6f} "
                            f"num_rows={n}")
                else:
                    self.logger.warning(
                        f"Cannot compute fan_in for parameter {name}, using no scaling")
            elif scale_type == "opnorm_reg":
                # opnorm regularization: t* = (sum_i ||g_i||_2) / (2 * fan_in)
                fan_in = _fan_in_of_weight(
                    param) if param is not None else None
                if fan_in is not None:
                    t_star = norms.sum() / (2.0 * float(fan_in))
                    normalized = normalized * t_star
                    # 记录日志
                    if self._step % self.defaults.get('log_interval', 50) == 0:
                        self.logger.info(
                            f"[RowNorm] name={name} type=2D "
                            f"scale_type={scale_type} fan_in={fan_in} used_scale=t_star={t_star:.6f} "
                            f"num_rows={n}")
                else:
                    self.logger.warning(
                        f"Cannot compute fan_in for parameter {name}, using no scaling")
            # scale_type == "none" - no additional scaling

            grad.copy_(normalized.to(grad.dtype))
            return grad

        elif grad.ndim == 4:
            out_channels = grad.size(0)
            g2d_float = grad.reshape(out_channels, -1).float()
            norms = g2d_float.norm(dim=1, keepdim=True)
            # Use more stable normalization with adaptive epsilon
            adaptive_eps = eps * torch.ones_like(norms)
            # Scale eps with norm magnitude
            adaptive_eps = torch.max(adaptive_eps, norms * 1e-6)
            norms = norms.clamp_min(adaptive_eps)

            # Basic row normalization
            normalized = g2d_float / norms

            # Apply scaling based on scale_type
            n = out_channels  # number of rows (output channels)
            if scale_type == "mean_scale":
                # mean scale: (sum of row norms) / sqrt(n)
                mean_scale = norms.sum() / math.sqrt(n)
                normalized = normalized * mean_scale
            elif scale_type == "rms_scale":
                # RMS scale: sqrt(mean(norms^2))
                rms_scale = torch.sqrt(torch.mean(norms**2))
                normalized = normalized * rms_scale
            elif scale_type == "dim_scale":
                # dimension scale: 1 / sqrt(n)
                dim_scale = 1.0 / math.sqrt(n)
                normalized = normalized * dim_scale
            elif scale_type == "opnorm_constraint":
                # opnorm constraint: 1/sqrt(fan_in)
                fan_in = _fan_in_of_weight(
                    param) if param is not None else None
                if fan_in is not None:
                    opnorm_scale = 1.0 / math.sqrt(fan_in)
                    normalized = normalized * opnorm_scale
                    # 记录日志
                    if self._step % self.defaults.get('log_interval', 50) == 0:
                        self.logger.info(
                            f"[RowNorm] name={name} type=4D "
                            f"scale_type={scale_type} fan_in={fan_in} used_scale=1/sqrt({fan_in})={opnorm_scale:.6f} "
                            f"num_rows={n}")
                else:
                    self.logger.warning(
                        f"Cannot compute fan_in for parameter {name}, using no scaling")
            elif scale_type == "opnorm_reg":
                # opnorm regularization: t* = (sum_i ||g_i||_2) / (2 * fan_in)
                fan_in = _fan_in_of_weight(
                    param) if param is not None else None
                if fan_in is not None:
                    t_star = norms.sum() / (2.0 * float(fan_in))
                    normalized = normalized * t_star
                    # 记录日志
                    if self._step % self.defaults.get('log_interval', 50) == 0:
                        self.logger.info(
                            f"[RowNorm] name={name} type=4D "
                            f"scale_type={scale_type} fan_in={fan_in} used_scale=t_star={t_star:.6f} "
                            f"num_rows={n}")
                else:
                    self.logger.warning(
                        f"Cannot compute fan_in for parameter {name}, using no scaling")
            # scale_type == "none" - no additional scaling

            normalized = normalized.reshape_as(grad).to(grad.dtype)
            grad.copy_(normalized)
            return grad

        return grad

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            momentum: float = group["momentum"]
            weight_decay: float = group["weight_decay"]
            nesterov: bool = group["nesterov"]
            eps: float = group["eps"]
            max_grad_norm: float = group["max_grad_norm"]
            scale_type: str = group["scale_type"]
            signgd_1d: bool = group["signgd_1d"]
            signgd_scaling: str = group["signgd_scaling"]

            # Collect all gradients for this group for global gradient clipping
            gradients = []
            for p in group["params"]:
                if p.grad is not None:
                    gradients.append(p.grad)

            # Global gradient clipping before normalization
            if gradients and max_grad_norm > 0:
                total_norm = torch.norm(torch.stack(
                    [torch.norm(g.detach()) for g in gradients]))
                clip_coef = max_grad_norm / (total_norm + 1e-6)
                if clip_coef < 1:
                    for g in gradients:
                        g.mul_(clip_coef)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                param_name = self.param_names.get(id(p), f"param_{id(p)}")

                # Apply different processing based on gradient dimension
                if grad.ndim == 1 and signgd_1d:
                    # Use SignGD for 1D gradients (BatchNorm weights/biases, Linear biases)
                    self._signgd_1d(
                        grad, eps, p, signgd_scaling, param_name)
                elif grad.ndim in (2, 4):
                    # Row-wise L2 normalization for 2D/4D params (weights)
                    self._rownorm_inplace(grad, eps, scale_type, p, param_name)
                # For other dimensions, use gradient as-is (standard SGD)

                state = self.state[p]
                if len(state) == 0 and momentum > 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                if momentum > 0:
                    buf: torch.Tensor = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)
                    d_p = grad.add(buf, alpha=momentum) if nesterov else buf
                else:
                    d_p = grad

                # Decoupled weight decay
                if weight_decay != 0:
                    p.mul_(1.0 - lr * weight_decay)

                p.add_(d_p, alpha=-lr)

        self._step += 1
        return loss
