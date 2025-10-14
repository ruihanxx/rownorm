import math
from typing import Iterable, List, Optional

import torch


class RowNormSGD(torch.optim.Optimizer):
    """
    Row normalization is performed as:
        g[i, :] <- g[i, :] / (||g[i, :]||_2 + eps)

    For convolutional weights with shape (out_channels, in_channels, kH, kW), we reshape
    to (out_channels, in_channels * kH * kW) and normalize rows.

    Decoupled weight decay is applied (AdamW-style): p <- p * (1 - lr * weight_decay).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        nesterov_mom: float = 0.0,
        eps: float = 1e-8,
        max_grad_norm: float = 1.0,
        p_exp: float = 2.0,  
        q_exp: float = torch.inf,          
        use_fan_scaling: bool = True,    
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov_mom < 0 and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum > 0")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if max_grad_norm <= 0.0:
            raise ValueError(f"Invalid max_grad_norm: {max_grad_norm}")
        if p_exp < 1.0:                    
            raise ValueError("p must be >= 1")
        if p_exp > q_exp:
            raise ValueError("q must be > p")

        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay, nesterov_mom=nesterov_mom, 
                        eps=eps, max_grad_norm=max_grad_norm, p_exp=p_exp, q_exp=q_exp, use_fan_scaling=use_fan_scaling,)
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
    @staticmethod
    def _rownorm_inplace(grad: torch.Tensor, eps: float, p_exp: float=2, q_exp: float = torch.inf) -> torch.Tensor:
        """Row-normalize gradient for 2D or 4D tensors. Returns the normalized grad view.

        - If grad.ndim == 2: normalize rows.
        - If grad.ndim == 4: treat as (out_channels, in_channels * kH * kW).
        - Else: return grad unchanged.
        
        Uses more stable normalization to prevent gradient explosion.
        """
        if grad is None:
            return grad
        if grad.ndim == 2:
            g2d = grad
            g2d_float = g2d.float()
            if q_exp > 999999999:
                norms = g2d_float.norm(p=p_exp, dim=1, keepdim=True).pow_(p_exp-1)
                # Use more stable normalization with adaptive epsilon
                adaptive_eps = eps * torch.ones_like(norms)
                adaptive_eps = torch.max(adaptive_eps, norms * 1e-6)  # Scale eps with norm magnitude
                norms = norms.clamp_min(adaptive_eps)
                g2d_float = g2d_float.abs().pow(p_exp-1) * g2d_float.sign()
                normalized = (g2d_float / norms).to(grad.dtype)
                g2d.copy_(normalized)
                return grad
            elif p_exp == 1:
                q_exp_star = 1/(1-1/q_exp)
                norms = g2d_float.norm(p=q_exp_star, dim=0, keepdim=True).pow_(q_exp_star-1)
                # Use more stable normalization with adaptive epsilon
                adaptive_eps = eps * torch.ones_like(norms)
                adaptive_eps = torch.max(adaptive_eps, norms * 1e-6)  # Scale eps with norm magnitude
                norms = norms.clamp_min(adaptive_eps)
                g2d_float = g2d_float.abs().pow(q_exp_star-1) * g2d_float.sign()
                normalized = (g2d_float / norms).to(grad.dtype)
                g2d.copy_(normalized)
                return grad
        if grad.ndim == 4:
            
            if q_exp > 999999999:
                out_channels = grad.size(0)
                g2d_float = grad.reshape(out_channels, -1).float()
                norms = g2d_float.norm(p=p_exp, dim=1, keepdim=True).pow_(p_exp-1)
                # Use more stable normalization with adaptive epsilon
                adaptive_eps = eps * torch.ones_like(norms)
                adaptive_eps = torch.max(adaptive_eps, norms * 1e-6)  # Scale eps with norm magnitude
                norms = norms.clamp_min(adaptive_eps)
                g2d_float = g2d_float.abs().pow(p_exp-1) * g2d_float.sign() / norms
                normalized = g2d_float.reshape_as(grad).to(grad.dtype)
                grad.copy_(normalized)
                return grad
            elif p_exp == 1:
                out_channels = grad.size(0)*grad.size(1)
                g2d_float = grad.reshape(out_channels, -1).float()
                q_exp_star = 1/(1-1/q_exp)
                norms = g2d_float.norm(p=q_exp_star, dim=0, keepdim=True).pow_(q_exp_star-1)
                # Use more stable normalization with adaptive epsilon
                adaptive_eps = eps * torch.ones_like(norms)
                adaptive_eps = torch.max(adaptive_eps, norms * 1e-6)  # Scale eps with norm magnitude
                norms = norms.clamp_min(adaptive_eps)
                g2d_float = g2d_float.abs().pow(q_exp_star-1) * g2d_float.sign() / norms
                normalized = g2d_float.reshape_as(grad).to(grad.dtype)
                grad.copy_(normalized)
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
            nesterov_mom: float = group["nesterov_mom"]
            eps: float = group["eps"]
            max_grad_norm: float = group["max_grad_norm"]
            p_exp: float = group["p_exp"]            
            q_exp: float = group["q_exp"]      
            use_fan_scaling: bool = group["use_fan_scaling"]  

            # Collect all gradients for this group for global gradient clipping
            gradients = []
            for p in group["params"]:
                if p.grad is not None:
                    gradients.append(p.grad)
            
            # Global gradient clipping before row normalization
            if gradients and max_grad_norm > 0:
                total_norm = torch.norm(torch.stack([torch.norm(g.detach()) for g in gradients]))
                clip_coef = max_grad_norm / (total_norm + 1e-6)
                if clip_coef < 1:
                    for g in gradients:
                        g.mul_(clip_coef)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # momentum and Nesterov part
                state = self.state[p]
                if len(state) == 0 and momentum > 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                
                if momentum > 0:
                    buf: torch.Tensor = state["momentum_buffer"]
                    if nesterov_mom > 0:
                        d_p = grad.mul(1-nesterov_mom).add(buf, alpha=nesterov_mom) 
                        buf.mul_(momentum).add_(grad, alpha=1-momentum)
                    else:
                        buf.mul_(momentum).add_(grad, alpha=1-momentum)
                        d_p = buf  
                else:
                    d_p = grad
                
                # Row-wise L2 normalization for 2D/4D params
                if d_p.ndim in (2, 4):
                    # print(torch.count_nonzero(d_p),'1')
                    if p_exp < 99999:
                        
                        self._rownorm_inplace(d_p, eps, p_exp, q_exp)
                    else:
                        if d_p.ndim == 4:
                            G = d_p.reshape(d_p.shape[0], -1)
                            m, n = G.shape
                            _, jstar = (G.abs()).max(dim=1) 
                            X = torch.zeros_like(G)
                            rows = torch.arange(m, device=G.device)
                            X[rows, jstar] = G[rows, jstar].sign()
                            d_p = X.reshape_as(d_p)
                        else:
                            G = d_p
                            m, n = G.shape
                            _, jstar = (G.abs()).max(dim=1) 
                            X = torch.zeros_like(G)
                            rows = torch.arange(m, device=G.device)
                            X[rows, jstar] = G[rows, jstar].sign()
                            d_p = X
                        
                    # print(torch.count_nonzero(d_p))
                elif d_p.ndim == 1:
                    d_p = torch.sign(d_p)
                
                # Decoupled weight decay
                if weight_decay != 0:
                    p.mul_(1.0 - lr * weight_decay)

                alpha = -lr
                if use_fan_scaling and p.ndim != 1 and p_exp < 99999:
                    fin, fout = self._fans_by_rule(p)
                    
                    # scale = fan_out^{1/q} / fan_in^{1/p}
                    scale = (float(fout) ** (1.0 / q_exp)) / (float(fin) ** (1.0 / p_exp))
                    alpha *= scale

                p.add_(d_p, alpha=alpha)

        return loss
