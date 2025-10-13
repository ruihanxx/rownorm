import torch
from typing import Iterable, Optional


class NormMomentum(torch.optim.Optimizer):
    """
    NormMomentum optimizer implementing a momentum on raw gradients followed by
    normalized updates. Supports an optional parameter shrink factor gamma.
    Update rule per parameter tensor p:
        d_t = (1 - alpha) * d_{t-1} + alpha * grad
        p    <- (1 - lr * weight_decay) * p   # decoupled weight decay (if provided)
        p    <- (1 - gamma) * p               # optional parameter shrink
        p    <- p - lr * normalize(d_t)       # normalized step (global L2 per tensor)

    Args:
        params: iterable of parameters or dicts defining parameter groups
        lr: learning rate
        alpha: momentum blend factor in [0, 1]
        gamma: parameter shrink factor in [0, 1]
        eps: small constant to avoid division by zero in normalization
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.1,
        alpha: float = 0.1,
        gamma: float = 0.0,
        eps: float = 1e-12,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"Invalid alpha (should be in [0,1]): {alpha}")
        if gamma < 0.0:
            raise ValueError(f"Invalid gamma (should be >= 0): {gamma}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")

        defaults = dict(lr=lr, alpha=alpha, gamma=gamma, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            alpha: float = group["alpha"]
            gamma: float = group["gamma"]
            eps: float = group["eps"]
            weight_decay: float = group.get("weight_decay", 0.0)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.detach()
                if grad.is_sparse:
                    raise RuntimeError(
                        "NormMomentum does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["direction"] = torch.zeros_like(p)

                d: torch.Tensor = state["direction"]
                # Exponential moving average of gradients
                d.mul_(1.0 - alpha).add_(grad, alpha=alpha)

                # Global L2 normalization per tensor
                denom = d.norm().clamp_min(eps)
                step_dir = d / denom

                # Combined weight decay and parameter shrink for efficiency
                if weight_decay != 0.0 or gamma != 0.0:
                    decay_factor = lr * weight_decay if weight_decay != 0.0 else 0.0
                    shrink_factor = gamma if gamma != 0.0 else 0.0
                    combined_factor = (1.0 - shrink_factor) * \
                        (1.0 - decay_factor)
                    p.mul_(combined_factor)

                # Apply normalized descent step
                p.add_(step_dir, alpha=-lr)

        return loss
