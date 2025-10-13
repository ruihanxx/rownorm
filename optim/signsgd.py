from typing import Iterable, Optional

import torch


class SignSGD(torch.optim.Optimizer):
    """
    SignSGD optimizer with optional momentum and decoupled weight decay.
    Update rule:
        if momentum == 0:
            p <- p - lr * sign(grad)
        else:
            v <- momentum * v + (1 - momentum) * grad
            p <- p - lr * sign(v)
    Weight decay is decoupled (AdamW-style): p <- p * (1 - lr * weight_decay).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        eps: float = 1e-12,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")

        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay, eps=eps)
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
            weight_decay: float = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Decoupled weight decay
                if weight_decay != 0:
                    p.mul_(1.0 - lr * weight_decay)

                if momentum > 0.0:
                    state = self.state[p]
                    if len(state) == 0:
                        state["velocity"] = torch.zeros_like(p)
                    v: torch.Tensor = state["velocity"]
                    # Exponential moving average of gradients
                    v.mul_(momentum).add_(grad, alpha=1.0 - momentum)
                    direction = torch.sign(v)
                else:
                    direction = torch.sign(grad)

                # Avoid NaNs from exact zeros
                direction = torch.where(
                    direction == 0, torch.zeros_like(direction), direction)
                p.add_(direction, alpha=-lr)

        return loss

