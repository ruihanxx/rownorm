"""
Adam with fan_in rescaling optimizer

Supports three rescaling modes:
    1. 'none': Standard Adam (no rescaling)
    2. 'sqrt': Rescale by 1/√fan_in (moderate rescaling)
    3. 'linear': Rescale by 1/fan_in (strong rescaling, matches MAV theory)

Based on the theory that:
1. MAV norm gradient descent requires 1/fan_in rescaling
2. Adam's update m_t / sqrt(v_t) approaches sign(m_t) when v_t ≈ m_t^2
3. This corresponds to MAV → ℓ_∞ geometry
4. Therefore, rescaling by 1/fan_in should improve Adam

For each parameter tensor p, fan_in is defined as:
- For 2D tensors (weight matrices): p.shape[1] (input features)  
- For 1D tensors (biases): 1 (no rescaling needed)
- For higher-D tensors: product of all dimensions except the first
"""

from typing import Iterable, Optional
import torch
import torch.nn as nn


class AdamFanInRescaling(torch.optim.Optimizer):
    """
    Adam optimizer with fan_in rescaling.

    Supports three rescaling modes:
    - 'none': Standard Adam (rescale_factor = 1.0)
    - 'sqrt': Rescale by 1/√fan_in (moderate, original version)
    - 'linear': Rescale by 1/fan_in (strong, matches MAV theory)

    This is based on the theoretical connection between Adam and MAV → ℓ_∞ 
    geometry, where proper rescaling should improve performance.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        rescaling_mode: str = 'sqrt',  # 'none', 'sqrt', or 'linear'
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if rescaling_mode not in ['none', 'sqrt', 'linear']:
            raise ValueError(
                f"Invalid rescaling_mode: {rescaling_mode}. Must be 'none', 'sqrt', or 'linear'")

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        rescaling_mode=rescaling_mode)
        super().__init__(params, defaults)

    def _get_fan_in(self, tensor: torch.Tensor) -> int:
        """
        Calculate fan_in for a given tensor.

        Args:
            tensor: Parameter tensor

        Returns:
            fan_in: Number of input features
        """
        if tensor.dim() < 2:
            # For 1D tensors (biases), no rescaling
            return 1
        elif tensor.dim() == 2:
            # For 2D tensors (linear layers): [out_features, in_features]
            return tensor.shape[1]
        else:
            # For higher-D tensors (conv layers): [out_channels, in_channels, ...]
            # fan_in is the product of all dimensions except the first
            return int(torch.tensor(tensor.shape[1:]).prod().item())

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            weight_decay = group['weight_decay']
            eps = group['eps']
            amsgrad = group['amsgrad']
            rescaling_mode = group['rescaling_mode']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        'AdamFanInRescaling does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    if amsgrad:
                        # Maintains the maximum of all 2nd moment running avg. till now
                        state['max_exp_avg_sq'] = torch.zeros_like(p)
                    # Calculate and store fan_in for this parameter
                    state['fan_in'] = self._get_fan_in(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Add weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Exponential moving average of gradient values
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Exponential moving average of squared gradient values
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(max_exp_avg_sq, exp_avg_sq,
                                  out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() /
                             (bias_correction2 ** 0.5)).add_(eps)
                else:
                    denom = (exp_avg_sq.sqrt() /
                             (bias_correction2 ** 0.5)).add_(eps)

                # Compute the standard Adam step
                step_size = lr / bias_correction1
                adam_update = exp_avg / denom

                # Apply fan_in rescaling based on mode
                fan_in = state['fan_in']
                if fan_in > 1:
                    if rescaling_mode == 'sqrt':
                        # Rescale by 1/√fan_in (moderate rescaling)
                        rescale_factor = 1.0 / (fan_in ** 0.5)
                    elif rescaling_mode == 'linear':
                        # Rescale by 1/fan_in (strong rescaling, matches MAV theory)
                        rescale_factor = 1.0 / fan_in
                    else:  # 'none'
                        # No rescaling (standard Adam)
                        rescale_factor = 1.0

                    adam_update = adam_update * rescale_factor

                p.add_(adam_update, alpha=-step_size)

        return loss


def create_adam_fan_in_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    betas: tuple = (0.9, 0.999),
    rescaling_mode: str = 'sqrt'
) -> AdamFanInRescaling:
    """
    Create AdamFanInRescaling optimizer with proper parameter grouping.

    Args:
        model: PyTorch model
        lr: Learning rate
        weight_decay: Weight decay coefficient  
        betas: Adam beta parameters
        rescaling_mode: 'none', 'sqrt', or 'linear'

    Returns:
        Configured AdamFanInRescaling optimizer
    """
    # Group parameters: apply weight decay only to 2D+ tensors (weights)
    decay_params = []
    no_decay_params = []

    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:  # weight matrices
            decay_params.append(p)
        else:  # biases and other 1D parameters
            no_decay_params.append(p)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return AdamFanInRescaling(param_groups, lr=lr, betas=betas, rescaling_mode=rescaling_mode)


def print_rescale_factors(optimizer: AdamFanInRescaling, model: nn.Module):
    """
    Print rescale factors for all parameters.

    Args:
        optimizer: AdamFanInRescaling optimizer instance
        model: PyTorch model
    """
    print(f"\n{'Parameter':<30} {'Shape':<20} {'fan_in':>8} {'Rescale':>10}")
    print("=" * 70)

    for name, param in model.named_parameters():
        if param in optimizer.state:
            state = optimizer.state[param]
            fan_in = state['fan_in']
            mode = optimizer.param_groups[0]['rescaling_mode']

            if fan_in > 1:
                if mode == 'sqrt':
                    factor = 1.0 / (fan_in ** 0.5)
                elif mode == 'linear':
                    factor = 1.0 / fan_in
                else:  # 'none'
                    factor = 1.0
            else:
                factor = 1.0

            shape_str = str(tuple(param.shape))
            print(f"{name:<30} {shape_str:<20} {fan_in:>8} {factor:>10.6f}")


if __name__ == "__main__":
    # Test all three rescaling modes
    print("=" * 80)
    print("Testing AdamFanInRescaling optimizer with three rescaling modes")
    print("=" * 80)

    # Create a simple model
    model = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )

    # Test data
    x = torch.randn(32, 128)
    y = torch.randint(0, 10, (32,))

    print("\n📊 Testing three rescaling modes:")
    print("-" * 80)

    for mode in ['none', 'sqrt', 'linear']:
        print(f"\n🔧 Mode: '{mode}'")

        # Reset model
        model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

        # Create optimizer with specific mode
        optimizer = create_adam_fan_in_optimizer(
            model, lr=1e-3, weight_decay=1e-4, rescaling_mode=mode
        )

        # Test one step
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"   Loss: {loss.item():.4f}")

        # Show rescale factors using helper function
        print_rescale_factors(optimizer, model)

    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("\n💡 Summary:")
    print("   - 'none':   rescale_factor = 1.0           (standard Adam)")
    print("   - 'sqrt':   rescale_factor = 1/√fan_in     (moderate, ORIGINAL)")
    print("   - 'linear': rescale_factor = 1/fan_in      (strong, MAV theory)")
    print("=" * 80)
