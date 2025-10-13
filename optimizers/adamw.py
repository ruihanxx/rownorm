"""
AdamW optimizer implementation.

AdamW applies weight decay directly to parameters (decoupled weight decay):
    x_{t+1} = x_t - α*λ*x_t - α*adam_update

This is different from Adam with L2 regularization, where weight decay is added to gradients.

"""

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    """
    AdamW optimizer with decoupled weight decay.

    Args:
        params: iterable of parameters to optimize
        lr: learning rate (default: 1e-3)
        betas: coefficients for computing running averages (default: (0.9, 0.999))
        eps: term for numerical stability (default: 1e-8)
        weight_decay: weight decay coefficient (default: 0.01)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            weight_decay = group['weight_decay']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Compute bias-corrected estimates
                step_size = lr / bias_correction1
                bias_correction2_sqrt = bias_correction2 ** 0.5

                # Compute Adam update
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
                adam_update = exp_avg / denom

                # Apply weight decay (decoupled)
                # x_{t+1} = x_t - α*λ*x_t - α*adam_update
                p.mul_(1 - lr * weight_decay)

                # Apply Adam update
                p.add_(adam_update, alpha=-step_size)

        return loss


def create_adamw_optimizer(model, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
    """
    Create AdamW optimizer for a model.

    Args:
        model: PyTorch model
        lr: learning rate
        betas: Adam beta parameters
        eps: epsilon for numerical stability
        weight_decay: weight decay coefficient

    Returns:
        AdamW optimizer
    """
    return AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
