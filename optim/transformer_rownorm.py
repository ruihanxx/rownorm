import math
from typing import Iterable, List, Optional

import torch


class TransformerRowNorm(torch.optim.Optimizer):
    """
    Transformer-optimized RowNorm based on ResNet experimental results.

    This optimizer applies different strategies to different parameter types:

    1. 2D Linear Weights (Attention, Feed-forward, Classifier):
       - Apply RowNorm with scale_type='mean_scale' (best performer from ResNet experiments)
       - Includes: self_attn.{in_proj_weight, out_proj.weight}, linear{1,2}.weight

    2. 1D Parameters (Biases, LayerNorm):
       - Use standard SGD (SignGD showed inconsistent results on ResNet)
       - Includes: all biases, LayerNorm weights/biases

    3. Embedding Matrix:
       - Configurable: either RowNorm or standard SGD
       - Each row represents a word embedding vector

    Based on ResNet experiments:
    - RowNorm + mean_scale: most stable high performance
    - SignGD for 1D: inconsistent (+0.78% best, -0.42% average)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.01,  # Lower default LR for Transformers
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        eps: float = 1e-8,
        max_grad_norm: float = 1.0,
        scale_type: str = "mean_scale",  # Force best performer from ResNet
        apply_rownorm_to_embedding: bool = False,  # Conservative default
        use_signgd_1d: bool = False,  # Conservative default based on ResNet results
        # For parameter classification
        named_parameters: Optional[List] = None,
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
        if scale_type not in ["none", "mean_scale", "rms_scale", "dim_scale"]:
            raise ValueError(
                f"Invalid scale_type: {scale_type}. Must be one of: none, mean_scale, rms_scale, dim_scale")

        defaults = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov,
            eps=eps, max_grad_norm=max_grad_norm, scale_type=scale_type,
            apply_rownorm_to_embedding=apply_rownorm_to_embedding,
            use_signgd_1d=use_signgd_1d
        )

        # Store parameter names for classification
        self.param_name_map = {}
        if named_parameters is not None:
            for name, param in named_parameters:
                self.param_name_map[id(param)] = name

        super().__init__(params, defaults)

    def _classify_parameter(self, param: torch.Tensor) -> str:
        """
        Classify parameter type for optimization strategy.

        Returns:
            'embedding': Embedding matrix
            'linear_2d': 2D linear weight (attention, feed-forward, classifier)
            'bias_1d': 1D bias or LayerNorm parameter
            'other': Other parameter types
        """
        param_id = id(param)
        param_name = self.param_name_map.get(param_id, "unknown").lower()

        # Check if it's an embedding parameter
        if 'embedding' in param_name and param.ndim == 2:
            return 'embedding'

        # Check for 2D linear weights (attention, feed-forward, classifier)
        if param.ndim == 2:
            # Attention weights
            if any(x in param_name for x in ['self_attn', 'multihead', 'attention']):
                if 'weight' in param_name:
                    return 'linear_2d'
            # Feed-forward and classifier weights
            elif any(x in param_name for x in ['linear', 'fc', 'feedforward', 'classifier']):
                if 'weight' in param_name:
                    return 'linear_2d'
            # Other 2D weights (fallback)
            elif 'weight' in param_name:
                return 'linear_2d'

        # 1D parameters (biases, LayerNorm)
        elif param.ndim == 1:
            return 'bias_1d'

        return 'other'

    def _signgd_1d(self, grad: torch.Tensor, eps: float) -> torch.Tensor:
        """Apply SignGD to 1-dimensional gradients."""
        if grad is None or grad.ndim != 1:
            return grad

        grad_float = grad.float()
        grad_norm = torch.norm(grad_float)
        grad_sign = torch.sign(grad_float)

        # Avoid division by zero
        if grad_norm < eps:
            return grad

        # SignGD: sign(g) * ||g||_2 / sqrt(d)
        d = grad.numel()
        scale_factor = grad_norm / math.sqrt(d)

        normalized = grad_sign * scale_factor
        grad.copy_(normalized.to(grad.dtype))
        return grad

    def _rownorm_inplace(self, grad: torch.Tensor, eps: float, scale_type: str) -> torch.Tensor:
        """Row-normalize gradient for 2D tensors with scaling."""
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
        # scale_type == "none" - no additional scaling

        grad.copy_(normalized.to(grad.dtype))
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
            apply_rownorm_to_embedding: bool = group["apply_rownorm_to_embedding"]
            use_signgd_1d: bool = group["use_signgd_1d"]

            # Collect all gradients for global gradient clipping
            gradients = []
            for p in group["params"]:
                if p.grad is not None:
                    gradients.append(p.grad)

            # Global gradient clipping before processing
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
                param_type = self._classify_parameter(p)

                # Apply different processing based on parameter type
                if param_type == 'embedding':
                    if apply_rownorm_to_embedding:
                        # Apply RowNorm to embedding (each row is a word vector)
                        self._rownorm_inplace(grad, eps, scale_type)
                    # Else: use gradient as-is (standard SGD)

                elif param_type == 'linear_2d':
                    # Apply RowNorm to all 2D linear weights
                    self._rownorm_inplace(grad, eps, scale_type)

                elif param_type == 'bias_1d':
                    if use_signgd_1d:
                        # Apply SignGD to 1D parameters (experimental)
                        self._signgd_1d(grad, eps)
                    # Else: use gradient as-is (standard SGD)

                # For 'other' or unclassified parameters: use gradient as-is

                # Standard momentum SGD update
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

        return loss


def create_transformer_rownorm_optimizer(
    model,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
    nesterov: bool = True,
    scale_type: str = "mean_scale",
    apply_rownorm_to_embedding: bool = False,
    use_signgd_1d: bool = False,
    max_grad_norm: float = 1.0
):
    """
    Convenient function to create TransformerRowNorm optimizer with parameter groups.

    Args:
        model: The Transformer model
        lr: Learning rate
        weight_decay: Weight decay coefficient
        momentum: Momentum coefficient
        nesterov: Whether to use Nesterov momentum
        scale_type: RowNorm scaling type ('mean_scale' recommended)
        apply_rownorm_to_embedding: Whether to apply RowNorm to embedding matrix
        use_signgd_1d: Whether to use SignGD for 1D parameters (experimental)
        max_grad_norm: Maximum gradient norm for clipping

    Returns:
        TransformerRowNorm optimizer
    """
    # Collect named parameters for classification
    named_params = list(model.named_parameters())
    param_list = [param for name, param in named_params if param.requires_grad]

    # Create single parameter group (could extend to multiple groups later)
    param_groups = [{"params": param_list, "weight_decay": weight_decay}]

    return TransformerRowNorm(
        param_groups,
        lr=lr,
        momentum=momentum,
        weight_decay=0.0,  # Handled in parameter groups
        nesterov=nesterov,
        scale_type=scale_type,
        apply_rownorm_to_embedding=apply_rownorm_to_embedding,
        use_signgd_1d=use_signgd_1d,
        max_grad_norm=max_grad_norm,
        named_parameters=named_params
    )

