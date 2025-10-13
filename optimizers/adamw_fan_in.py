"""
AdamW with fan-in learning rate scaling.

This optimizer applies AdamW with per-parameter learning rate scaling based on fan-in:
    x_{t+1} = x_t - α*λ*x_t - (α/fan_in)*adam_update

The fan-in scaling helps normalize the learning rate across layers with different sizes.

Reference:
    - AdamW: Decoupled Weight Decay Regularization (Loshchilov & Hutter, 2019)
    - Fan-in scaling: Modern parameter initialization techniques
"""

import torch
from torch.optim import Optimizer
import math


def compute_fan_in(tensor):
    """
    Compute the fan-in for a parameter tensor.

    Args:
        tensor: Parameter tensor

    Returns:
        fan_in: Number of input units
    """
    dimensions = tensor.dim()

    if dimensions < 2:
        # For 1D parameters (biases), fan_in = 1
        return 1

    if dimensions == 2:
        # For 2D weight matrices: [out_features, in_features]
        fan_in = tensor.size(1)
    else:
        # For conv layers or higher dimensional tensors
        # tensor.shape is typically [out_channels, in_channels, *kernel_size]
        num_input_fmaps = tensor.size(1)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0, 0].numel()
        fan_in = num_input_fmaps * receptive_field_size

    return max(fan_in, 1)  # Ensure fan_in >= 1


class AdamWFanIn(Optimizer):
    """
    AdamW optimizer with fan-in learning rate scaling.

    The effective learning rate for each parameter is lr/fan_in, where fan_in
    is the number of input connections to that parameter.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups
        lr: base learning rate (default: 1e-3)
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
        super(AdamWFanIn, self).__init__(params, defaults)

        # Compute and store fan_in for each parameter
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['fan_in'] = compute_fan_in(p.data)

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
            base_lr = group['lr']
            weight_decay = group['weight_decay']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]

                # State initialization
                if 'step' not in state:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                fan_in = state['fan_in']
                state['step'] += 1

                # Compute effective learning rate with fan-in scaling
                lr = base_lr / fan_in

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

                # Apply weight decay (decoupled, also scaled by fan-in)
                # x_{t+1} = x_t - α*λ*x_t - (α/fan_in)*adam_update
                p.mul_(1 - lr * weight_decay)

                # Apply Adam update
                p.add_(adam_update, alpha=-step_size)

        return loss


def create_adamw_fan_in_optimizer(model, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
    """
    Create AdamWFanIn optimizer for a model.

    Args:
        model: PyTorch model
        lr: base learning rate (will be scaled by 1/fan_in for each parameter)
        betas: Adam beta parameters
        eps: epsilon for numerical stability
        weight_decay: weight decay coefficient

    Returns:
        AdamWFanIn optimizer
    """
    return AdamWFanIn(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
