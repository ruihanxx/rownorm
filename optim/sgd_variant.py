import torch
from typing import Iterable, Optional


class SGDVariant(torch.optim.Optimizer):
    """
    Let G_t be gradient, M_t be buffer, α be momentum, η be learning rate.
    M_t ← α * M_{t-1} + (1-α) * G_t
    G̃_t ← (1-α) * G_t + α * M_t  
    W_t ← W_{t-1} - η * normalize(G̃_t)

    Args:
        params: iterable of parameters or dicts defining parameter groups
        lr: learning rate (η)
        momentum: momentum coefficient (α) in [0, 1]
        eps: small constant to avoid division by zero in normalization
        weight_decay: L2 penalty (applied as decoupled weight decay)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.05,
        momentum: float = 0.9,
        eps: float = 1e-12,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= momentum <= 1.0):
            raise ValueError(
                f"Invalid momentum (should be in [0,1]): {momentum}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, momentum=momentum, eps=eps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            momentum: float = group["momentum"]
            eps: float = group["eps"]
            weight_decay: float = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.detach()
                if grad.is_sparse:
                    raise RuntimeError(
                        "SGDVariant does not support sparse gradients")

                # Decoupled weight decay
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf: torch.Tensor = state["momentum_buffer"]

                # M_t ← α * M_{t-1} + (1-α) * G_t
                buf.mul_(momentum).add_(grad, alpha=1.0 - momentum)

                # G̃_t ← (1-α) * G_t + α * M_t
                tilde_grad = (1.0 - momentum) * grad + momentum * buf

                # Global L2 normalization per tensor
                denom = tilde_grad.norm().clamp_min(eps)
                normalized_grad = tilde_grad / denom

                # W_t ← W_{t-1} - η * normalize(G̃_t)
                p.add_(normalized_grad, alpha=-lr)

        return loss

