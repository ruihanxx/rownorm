# adamw_custom.py
import math
import torch
from torch.optim import Optimizer

class AdamWScale(Optimizer):
    r"""
    AdamW optimizer (decoupled weight decay).
    参考公式：
      m_t = β1 m_{t-1} + (1-β1) g_t
      v_t = β2 v_{t-1} + (1-β2) g_t^2
      \hat m_t = m_t / (1-β1^t), \hat v_t = v_t / (1-β2^t)
      θ ← θ * (1 - lr * wd) - lr * \hat m_t / (sqrt(\hat v_t) + eps)
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        p_exp: float = 1.0,  
        q_exp: float = torch.inf
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        b1, b2 = betas
        if not 0.0 <= b1 < 1.0:
            raise ValueError(f"Invalid beta1: {b1}")
        if not 0.0 <= b2 < 1.0:
            raise ValueError(f"Invalid beta2: {b2}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if p_exp < 1.0:                    
            raise ValueError("p must be >= 1")
        if p_exp > q_exp:
            raise ValueError("q must be > p")

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,p_exp=p_exp, q_exp=q_exp,  amsgrad=amsgrad)
        super().__init__(params, defaults)

    @staticmethod
    def _fans_by_rule(t: torch.Tensor) -> tuple[int, int]:
        fout_override = getattr(t, "fan_out_override", None)
        fin_override  = getattr(t, "fan_in_override", None)

        if t.ndim == 2:
            fin = t.size(1)
            fout = t.size(0)
            if isinstance(fin_override, int):  fin = fin_override
            if isinstance(fout_override, int): fout = fout_override
            return int(fin), int(fout)

        if t.ndim == 4:
            fout = t.size(0) if not isinstance(fout_override, int) else int(fout_override)
            if not isinstance(fin_override, int):
                raise ValueError(
                    "For 4D tensors, fan_in must be provided externally via "
                    "`param.fan_in_override = <int>`."
                )
            fin = int(fin_override)
            return fin, fout
        raise ValueError(
            f"Tensor with ndim={t.ndim} not supported by current fan rule. "
            "Use 2D/4D or disable fan scaling."
        )

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            amsgrad = group["amsgrad"]
            p_exp: float = group["p_exp"]            
            q_exp: float = group["q_exp"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamWCustom does not support sparse gradients")

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                # 1) Decoupled weight decay
                if wd != 0.0:
                    p.mul_(1 - lr * wd)

                # 2) Update first/second moments
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 3) Bias correction
                bias_c1 = 1.0 - beta1 ** state["step"]
                bias_c2 = 1.0 - beta2 ** state["step"]

                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(eps)
                else:
                    denom = exp_avg_sq.sqrt().add_(eps)

                step_size = lr * math.sqrt(bias_c2) / bias_c1
                
                if p.ndim != 1:
                    fin, fout = self._fans_by_rule(p)
                    # p_exp > 1 guaranteed in __init__
                    
                    # scale = fan_out^{1/q} / fan_in^{1/p}
                    scale = (float(fout) ** (1.0 / q_exp)) / (float(fin) ** (1.0 / p_exp))
                    step_size *= scale

                # 4) Parameter update
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
