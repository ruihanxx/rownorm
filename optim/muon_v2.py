import torch
from typing import Iterable, Optional


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
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


class MuonV2(torch.optim.Optimizer):
    """
    MuonV2 optimizer implementing the modified update rule:
    d <- alpha * nabla f(x) + (1-alpha) * d
    x <- (1-gamma) * x + lr * normalize(d)

    This version uses the same Newton-Schulz orthogonalization as Muon_wrap
    but with a modified parameter update rule that includes weight decay.

    Args:
        params: iterable of parameters or dicts defining parameter groups
        lr: learning rate
        beta: momentum factor (equivalent to 1-alpha)
        gamma: parameter shrink factor in [0, 1]
        ns_steps: number of Newton-Schulz steps for orthogonalization
        nesterov: whether to use Nesterov momentum
        weight_decay: L2 penalty (applied as decoupled weight decay)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.05,
        beta: float = 0.95,
        gamma: float = 0.0,
        ns_steps: int = 5,
        nesterov: bool = True,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= beta <= 1.0):
            raise ValueError(f"Invalid beta (should be in [0,1]): {beta}")
        if gamma < 0.0:
            raise ValueError(f"Invalid gamma (should be >= 0): {gamma}")
        if ns_steps <= 0:
            raise ValueError(f"Invalid ns_steps: {ns_steps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, beta=beta, gamma=gamma,
                        ns_steps=ns_steps, nesterov=nesterov, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            beta: float = group["beta"]
            gamma: float = group["gamma"]
            ns_steps: int = group["ns_steps"]
            nesterov: bool = group["nesterov"]
            weight_decay: float = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "MuonV2 does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                # Use Muon update with Newton-Schulz orthogonalization for 2D+ tensors
                if grad.ndim >= 2:
                    update = muon_update(grad, state["momentum_buffer"], 
                                       beta=beta, ns_steps=ns_steps, nesterov=nesterov)
                else:
                    # For 1D parameters (like bias), use standard momentum without orthogonalization
                    momentum_buffer = state["momentum_buffer"]
                    momentum_buffer.lerp_(grad, 1 - beta)
                    update = grad.lerp_(momentum_buffer, beta) if nesterov else momentum_buffer

                # Apply weight decay first (decoupled)
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                # Modified update rule: x <- (1-gamma) * x + lr * normalize(d)
                # This combines parameter shrink with the normalized step
                p.mul_(1.0 - gamma).add_(update.reshape(p.shape), alpha=-lr)

        return loss
