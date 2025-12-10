
import math
from typing import Iterable, List, Optional

import torch


@torch.no_grad()
def row_topk_signed(G: torch.Tensor, k: int) -> torch.Tensor:
    """
    输入:
        G: 形状 (n, d) 的浮点 tensor（建议已在 GPU 上）
        k: 每行选取的元素个数
    输出:
        X: 形状 (n, d)，每行只有 |G| 最大的 k 个位置非零，
           值为 (1/k) * sign(G_ij)
    """
    assert G.dim() == 2, "G 必须是二维张量"
    n, d = G.shape
    if k <= 0:
        return torch.zeros_like(G)
    k = min(k, d)

    # 取每行 |G| 的 top-k 下标
    topk_idx = torch.topk(G.abs(), k, dim=1, largest=True, sorted=False).indices  # (n, k)

    # 取对应位置的符号（-1/0/1），并缩放到 1/k
    signed_vals = G.gather(1, topk_idx).sign() * (1.0 / k)                       # (n, k)

    # 按行把值 scatter 回原形状；其余位置保持 0
    X = torch.zeros_like(G)
    X.scatter_(dim=1, index=topk_idx, src=signed_vals)
    return X

def calc_row_max(G: torch.Tensor):
    
    return (G.norm(p=2, dim=1, keepdim=True)).max()

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
    try:
        G.dim
    except AttributeError:

        G = np.asarray(G)
    
    assert G.ndim >= 2  
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

class RowmixSGD(torch.optim.Optimizer):
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
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov_mom < 0 and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum > 0")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if max_grad_norm < 0.0:
            raise ValueError(f"Invalid max_grad_norm: {max_grad_norm}")
        

        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay, nesterov_mom=nesterov_mom, 
                        eps=eps, max_grad_norm=max_grad_norm)
        super().__init__(params, defaults)

    @staticmethod
    def _fans_by_rule(t: torch.Tensor) -> tuple[int, int]:
        fout_override = getattr(t, "fan_out_override", None)
        fin_override  = getattr(t, "fan_in_override", None)
        return fin_override, fout_override

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
                  
             

            # Collect all gradients for this group for global gradient clipping
            gradients = []
            for p in group["params"]:
                if p.grad is not None:
                    gradients.append(p.grad)
            
            # Global gradient clipping before row normalization
            # if gradients and max_grad_norm > 0:
            #     total_norm = torch.norm(torch.stack([torch.norm(g.detach()) for g in gradients]))
            #     clip_coef = max_grad_norm / (total_norm + 1e-6)
            #     if clip_coef < 1:
            #         for g in gradients:
            #             g.mul_(clip_coef)

            for p in group["params"]:
                if p.grad is None:
                    continue

                gggggg = p.grad
                grad = gggggg.detach().clone().to(gggggg.dtype).contiguous()


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
                p_exp = 1
                # # Row-wise L2 normalization for 2D/4D params
                if d_p.ndim == 2:
                    if p_exp < 99999:
                        p_is_ebd = getattr(p, "is_ebd", 0)
                        p_is_conv = getattr(p, "is_conv", 0)
                        fin, fout = self._fans_by_rule(p)
                        if p_is_conv:
                            
                            p_is_qkv = getattr(p, "is_qkv", 0)
                            if not p_is_qkv:
                                fin, fout = self._fans_by_rule(p)
                                linf_update = row_topk_signed(d_p.T,1).T

                                d_p = torch.sign(d_p).mul_(1/(float(fin) ** (1.0 / p_exp))).add_(linf_update)
                                
                                # d_p = torch.sign(d_p).mul(1/(float(fin)))
                            elif (not p_is_ebd) and p_is_qkv:
                                fin, fout = self._fans_by_rule(p)
                                gq, gk, gv = torch.chunk(d_p, 3, dim=1)
                                gq2_linf = row_topk_signed(gq.T,1).T
                                gq2 = torch.sign(gq).mul_(1/(float(fin) ** (1.0 / p_exp))).add_(gq2_linf)
                                gk2_linf = row_topk_signed(gk.T,1).T
                                gk2 = torch.sign(gk).mul_(1/(float(fin) ** (1.0 / p_exp))).add_(gk2_linf)
                                gv2_linf = row_topk_signed(gv.T,1).T
                                gv2 = torch.sign(gv).mul_(1/(float(fin) ** (1.0 / p_exp))).add_(gv2_linf)
                                
                                d_p = torch.cat([gq2, gk2, gv2], dim=1)
                                
                                # d_p = torch.sign(d_p).mul(1/(float(fin)))
                        elif p_is_ebd:
                            
                            fin, fout = self._fans_by_rule(p)
                            linf_update = row_topk_signed(d_p.T,1).T
                            # d_p = torch.sign(d_p).mul_(1/(float(fin) ** (1.0 / p_exp))).add_(linf_update)
                            d_p = torch.sign(d_p)
                            
                           
                        else:
                            fin, fout = self._fans_by_rule(p)
                            linf_update = row_topk_signed(d_p,1)
                            # d_p = torch.sign(d_p).mul_(1/(float(fin) ** (1.0 / p_exp))).add_(linf_update)
                            d_p = torch.sign(d_p)
                           
                    # d_p = self._rownorm_inplace(d_p, eps, p_exp, q_exp)
                    # print(torch.count_nonzero(d_p))
                elif d_p.ndim == 1:
                    d_p = torch.sign(d_p)
                    
                
                # Decoupled weight decay
                if weight_decay != 0 :
                    
                    if p.ndim != 1:
                        p.mul_(1.0 - lr * weight_decay)

                alpha = -lr
                
                # if use_fan_scaling and p_exp < 99999:
                #     if p.ndim != 1:
                #         fin, fout = self._fans_by_rule(p)
                #         # fout, fin = self._fans_by_rule(p)
                #         # print(fin,fout)
                #         # scale = fan_out^{1/q} / fan_in^{1/p}
                        
                #         scale = (float(fout) ** (1.0 / q_exp)) / (float(fin) ** (1.0 / p_exp))
                #         # print(alpha,scale)
                #         alpha *= scale
                #     # elif False:
                #     #     p_is_ln = getattr(p, "is_ln", 0)
                #     #     if p_is_ln:
                #     #         fin = len(p)
                #     #         fout = len(p)
                #     #         scale = (float(fout) ** (1.0 / q_exp)) / (float(fin) ** (1.0 / p_exp))
                            
                #     #         # alpha *= scale
                #     #         alpha *= 10


                p.add_(d_p, alpha=alpha)

        return loss
