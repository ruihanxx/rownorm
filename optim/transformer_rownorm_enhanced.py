import math
import logging
from typing import Iterable, List, Optional

import torch


def _fan_in_of_weight(param):
    """
    Compute the fan_in value of a weight parameter.

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


def _is_transformer_linear_weight_name(name, param):
    """
    Check if the parameter is a linear weight of a Transformer.
    """
    return (param.ndim == 2 and
            ('self_attn.in_proj_weight' in name or
             'self_attn.out_proj.weight' in name or
             'linear1.weight' in name or
             'linear2.weight' in name))


class TransformerRowNormMeanScaleSignGD(torch.optim.Optimizer):
    """
    Conservative + MeanScale(2D) + SignGD(1D) optimizer for Transformers.

    This optimizer combines the best strategies from CIFAR experiments:
    - 2D Linear Weights: RowNorm with configurable scale_type
    - 1D Parameters: SignGD (from CIFAR best results)

    Parameter classification (Conservative):
    - 2D weights: self_attn.{in_proj_weight, out_proj.weight}, linear{1,2}.weight
    - 1D params: all biases, LayerNorm weights/biases
    - Embedding: standard SGD (Conservative)
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
        scale_type: str = "mean_scale",  # 新增scale_type参数
        log_interval: int = 50,
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
        if scale_type not in ["mean_scale", "rms_scale", "dim_scale", "opnorm_constraint", "opnorm_reg"]:
            raise ValueError(
                f"Invalid scale_type: {scale_type}. Must be one of: mean_scale, rms_scale, dim_scale, opnorm_constraint, opnorm_reg")

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        nesterov=nesterov, eps=eps, max_grad_norm=max_grad_norm,
                        scale_type=scale_type, log_interval=log_interval)
        super().__init__(params, defaults)

        # Store named parameters for classification
        self.named_parameters = named_parameters or []

        # Initialize logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self._step = 0

        # Pre-classify parameters (independent of grad is None)
        self._classify_parameters()

    def _classify_parameters(self):
        """Classify parameters into different types for targeted optimization."""
        if not self.named_parameters:
            return

        # 先为所有参数设置默认类型
        for group in self.param_groups:
            for param in group['params']:
                param._optimizer_param_type = 'unknown'

        # 按名字和维度分类参数（不依赖grad是否为None）
        for name, param in self.named_parameters:
            # Conservative 2D classification (attention + FFN weights only)
            if _is_transformer_linear_weight_name(name, param):
                param._optimizer_param_type = 'linear_2d'

            # 1D parameters (biases, LayerNorm)
            elif ('bias' in name or
                  'norm' in name.lower() or
                  'layernorm' in name.lower()) and param.ndim == 1:
                param._optimizer_param_type = 'bias_1d'

            # Embedding (Conservative: use standard SGD)
            elif 'embed' in name.lower():
                param._optimizer_param_type = 'embedding'
            else:
                param._optimizer_param_type = 'other'

    def _signgd_1d(self, grad: torch.Tensor, eps: float, name: str = "") -> torch.Tensor:
        """Apply SignGD to 1-dimensional gradients."""
        if grad is None or grad.ndim != 1:
            return grad

        grad_float = grad.float()
        grad_norm = torch.norm(grad_float)
        grad_sign = torch.sign(grad_float)

        if grad_norm < eps:
            return grad

        # SignGD: sign(g) * ||g||_2 / sqrt(d)
        d = grad.numel()
        scale_factor = grad_norm / math.sqrt(d)

        normalized = grad_sign * scale_factor
        grad.copy_(normalized.to(grad.dtype))

        # Log
        if self._step % self.defaults.get('log_interval', 50) == 0:
            self.logger.info(
                f"[SignGD] name={name} type=1D "
                f"dim={d} grad_norm={grad_norm:.6f} used_scale={scale_factor:.6f}")

        return grad

    def _rownorm_inplace(self, grad: torch.Tensor, eps: float, scale_type: str, param: torch.nn.Parameter = None, name: str = "") -> torch.Tensor:
        """Row-normalize gradient for 2D tensors with configurable scaling."""
        if grad is None or grad.ndim != 2:
            return grad

        g2d_float = grad.float()
        norms = g2d_float.norm(dim=1, keepdim=True)

        # Use more stable normalization with adaptive epsilon
        adaptive_eps = eps * torch.ones_like(norms)
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
            fan_in = _fan_in_of_weight(param) if param is not None else None
            if fan_in is not None:
                opnorm_scale = 1.0 / math.sqrt(fan_in)
                normalized = normalized * opnorm_scale
                # Log
                if self._step % self.defaults.get('log_interval', 50) == 0:
                    self.logger.info(
                        f"[RowNorm] name={name} type=linear_2d "
                        f"scale_type={scale_type} fan_in={fan_in} used_scale=1/sqrt({fan_in})={opnorm_scale:.6f} "
                        f"num_rows={n}")
            else:
                self.logger.warning(
                    f"Cannot compute fan_in for parameter {name}, using no scaling")
        elif scale_type == "opnorm_reg":
            # opnorm regularization: t* = (sum_i ||g_i||_2) / (2 * fan_in)
            fan_in = _fan_in_of_weight(param) if param is not None else None
            if fan_in is not None:
                t_star = norms.sum() / (2.0 * float(fan_in))
                normalized = normalized * t_star
                # Log
                if self._step % self.defaults.get('log_interval', 50) == 0:
                    self.logger.info(
                        f"[RowNorm] name={name} type=linear_2d "
                        f"scale_type={scale_type} fan_in={fan_in} used_scale=t_star={t_star:.6f} "
                        f"num_rows={n}")
            else:
                self.logger.warning(
                    f"Cannot compute fan_in for parameter {name}, using no scaling")

        grad.copy_(normalized.to(grad.dtype))
        return grad

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # No need to reclassify, already done in initialization
        # self._classify_parameters()

        for group in self.param_groups:
            weight_decay: float = group["weight_decay"]
            momentum: float = group["momentum"]
            nesterov: bool = group["nesterov"]
            lr: float = group["lr"]
            eps: float = group["eps"]
            max_grad_norm: float = group["max_grad_norm"]
            scale_type: str = group["scale_type"]

            # Collect all gradients for global gradient clipping
            gradients = []
            for param in group["params"]:
                if param.grad is not None:
                    gradients.append(param.grad)

            # Global gradient clipping
            if gradients and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(gradients, max_grad_norm)

            # 创建参数名称映射
            param_names = {}
            if self.named_parameters:
                for name, param in self.named_parameters:
                    param_names[id(param)] = name

            # Apply optimizations based on parameter type
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                param_type = getattr(param, '_optimizer_param_type', 'unknown')
                param_name = param_names.get(id(param), f"param_{id(param)}")

                # Apply targeted optimization strategies
                if param_type == 'linear_2d' and grad.ndim == 2:
                    # Conservative 2D: RowNorm with configurable scale_type
                    self._rownorm_inplace(
                        grad, eps, scale_type, param, param_name)
                elif param_type == 'bias_1d' and grad.ndim == 1:
                    # 1D: SignGD (from CIFAR best results)
                    self._signgd_1d(grad, eps, param_name)
                # For embedding and other: use standard SGD (no modification)

                # Apply weight decay (AdamW-style)
                if weight_decay != 0:
                    param.mul_(1 - lr * weight_decay)

                # Get momentum buffer
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


class TransformerRowNormDimScale(torch.optim.Optimizer):
    """
    Conservative + DimScale(2D) optimizer for Transformers.

    This optimizer uses:
    - 2D Linear Weights: RowNorm with configurable scale_type
    - 1D Parameters: standard SGD
    - Embedding: standard SGD (Conservative)
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
        scale_type: str = "dim_scale",  # 新增scale_type参数
        log_interval: int = 50,
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
        if scale_type not in ["mean_scale", "rms_scale", "dim_scale", "opnorm_constraint", "opnorm_reg"]:
            raise ValueError(
                f"Invalid scale_type: {scale_type}. Must be one of: mean_scale, rms_scale, dim_scale, opnorm_constraint, opnorm_reg")

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        nesterov=nesterov, eps=eps, max_grad_norm=max_grad_norm,
                        scale_type=scale_type, log_interval=log_interval)
        super().__init__(params, defaults)

        # Store named parameters for classification
        self.named_parameters = named_parameters or []

        # Initialize logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self._step = 0

        # Pre-classify parameters (independent of grad is None)
        self._classify_parameters()

    def _classify_parameters(self):
        """Classify parameters into different types for targeted optimization."""
        if not self.named_parameters:
            return

        # Set default type for all parameters
        for group in self.param_groups:
            for param in group['params']:
                param._optimizer_param_type = 'unknown'

        # Classify parameters by name and dimension (independent of grad is None)
        for name, param in self.named_parameters:
            # Conservative 2D classification (attention + FFN weights only)
            if _is_transformer_linear_weight_name(name, param):
                param._optimizer_param_type = 'linear_2d'

            # All other parameters: standard SGD
            else:
                param._optimizer_param_type = 'other'

    def _rownorm_inplace(self, grad: torch.Tensor, eps: float, scale_type: str, param: torch.nn.Parameter = None, name: str = "") -> torch.Tensor:
        """Row-normalize gradient for 2D tensors with configurable scaling."""
        if grad is None or grad.ndim != 2:
            return grad

        g2d_float = grad.float()
        norms = g2d_float.norm(dim=1, keepdim=True)

        # Use more stable normalization with adaptive epsilon
        adaptive_eps = eps * torch.ones_like(norms)
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
            fan_in = _fan_in_of_weight(param) if param is not None else None
            if fan_in is not None:
                opnorm_scale = 1.0 / math.sqrt(fan_in)
                normalized = normalized * opnorm_scale
                # Log
                if self._step % self.defaults.get('log_interval', 50) == 0:
                    self.logger.info(
                        f"[RowNorm] name={name} type=linear_2d "
                        f"scale_type={scale_type} fan_in={fan_in} used_scale=1/sqrt({fan_in})={opnorm_scale:.6f} "
                        f"num_rows={n}")
            else:
                self.logger.warning(
                    f"Cannot compute fan_in for parameter {name}, using no scaling")
        elif scale_type == "opnorm_reg":
            # opnorm regularization: t* = (sum_i ||g_i||_2) / (2 * fan_in)
            fan_in = _fan_in_of_weight(param) if param is not None else None
            if fan_in is not None:
                t_star = norms.sum() / (2.0 * float(fan_in))
                normalized = normalized * t_star
                # Log
                if self._step % self.defaults.get('log_interval', 50) == 0:
                    self.logger.info(
                        f"[RowNorm] name={name} type=linear_2d "
                        f"scale_type={scale_type} fan_in={fan_in} used_scale=t_star={t_star:.6f} "
                        f"num_rows={n}")
            else:
                self.logger.warning(
                    f"Cannot compute fan_in for parameter {name}, using no scaling")

        grad.copy_(normalized.to(grad.dtype))
        return grad

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 不需要重新分类，已在初始化时完成
        # self._classify_parameters()

        for group in self.param_groups:
            weight_decay: float = group["weight_decay"]
            momentum: float = group["momentum"]
            nesterov: bool = group["nesterov"]
            lr: float = group["lr"]
            eps: float = group["eps"]
            max_grad_norm: float = group["max_grad_norm"]
            scale_type: str = group["scale_type"]

            # Collect all gradients for global gradient clipping
            gradients = []
            for param in group["params"]:
                if param.grad is not None:
                    gradients.append(param.grad)

            # Global gradient clipping
            if gradients and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(gradients, max_grad_norm)

            # 创建参数名称映射
            param_names = {}
            if self.named_parameters:
                for name, param in self.named_parameters:
                    param_names[id(param)] = name

            # Apply optimizations based on parameter type
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                param_type = getattr(param, '_optimizer_param_type', 'unknown')
                param_name = param_names.get(id(param), f"param_{id(param)}")

                # Apply targeted optimization strategies
                if param_type == 'linear_2d' and grad.ndim == 2:
                    # Conservative 2D: RowNorm with configurable scale_type
                    self._rownorm_inplace(
                        grad, eps, scale_type, param, param_name)
                # For all other parameters: use standard SGD (no modification)

                # Apply weight decay (AdamW-style)
                if weight_decay != 0:
                    param.mul_(1 - lr * weight_decay)

                # Get momentum buffer
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
