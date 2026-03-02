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
        p_exp: float = 1.0,  
        q_exp: float = torch.inf,          
        use_fan_scaling: bool = True,    
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if max_grad_norm < 0.0:
            raise ValueError(f"Invalid max_grad_norm: {max_grad_norm}")
        if p_exp < 1.0:                    
            raise ValueError("p must be >= 1")
        if p_exp > q_exp:
            raise ValueError("q must be > p")
        print(p_exp)
        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay, nesterov_mom=nesterov_mom, 
                        eps=eps, max_grad_norm=max_grad_norm, p_exp=p_exp, q_exp=q_exp, use_fan_scaling=use_fan_scaling,)
        super().__init__(params, defaults)

    @staticmethod
    def _fans_by_rule(t: torch.Tensor) -> tuple[int, int]:
        fout_override = getattr(t, "fan_out_override", None)
        fin_override  = getattr(t, "fan_in_override", None)
        return fin_override, fout_override
        # if t.ndim == 2:
        #     fin = t.size(1)
        #     fout = t.size(0)
        #     if isinstance(fin_override, int):  
        #         fin = fin_override
        #     if isinstance(fout_override, int): fout = fout_override
        #     return int(fin), int(fout)

        # if t.ndim == 4:
        #     fout = t.size(0) if not isinstance(fout_override, int) else int(fout_override)
        #     if not isinstance(fin_override, int):
        #         raise ValueError(
        #             "For 4D tensors, fan_in must be provided externally via "
        #             "`param.fan_in_override = <int>`."
        #         )
        #     fin = int(fin_override)
        #     return fin, fout
        
        raise ValueError(
            f"Tensor with ndim={t.ndim} not supported by current fan rule. "
            "Use 2D/4D or disable fan scaling."
        )
    @staticmethod
    def _rownorm_inplace(gggrad: torch.Tensor, eps: float, p_exp: float=1, q_exp: float = torch.inf) -> torch.Tensor:
        """Row-normalize gradient for 2D or 4D tensors. Returns the normalized grad view.

        - If grad.ndim == 2: normalize rows.
        - If grad.ndim == 4: treat as (out_channels, in_channels * kH * kW).
        - Else: return grad unchanged.
        
        Uses more stable normalization to prevent gradient explosion.
        """
        if gggrad is None:
            return gggrad
        if gggrad.ndim == 2:
            
            g2d = gggrad.detach().clone()
            g2d_float = g2d.float()
            # print(g2d.shape)
            if q_exp > 999999999:
                norms = g2d_float.norm(p=p_exp, dim=1, keepdim=True).pow_(p_exp-1)
                # Use more stable normalization with adaptive epsilon
                adaptive_eps = eps * torch.ones_like(norms)
                adaptive_eps = torch.max(adaptive_eps, norms)  # Scale eps with norm magnitude
                norms = norms.clamp_min(adaptive_eps)
                g2d_float = g2d_float.abs().pow(p_exp-1) * g2d_float.sign()
                normalized = (g2d_float / norms).to(gggrad.dtype)
                
                g2d.copy_(normalized)
                # print(grad)
                return g2d
            elif p_exp == 1:
                print('true')
                q_exp_star = 1/(1-1/q_exp)
                norms = g2d_float.norm(p=q_exp_star, dim=0, keepdim=True).pow_(q_exp_star-1)
                # Use more stable normalization with adaptive epsilon
                adaptive_eps = eps * torch.ones_like(norms)
                adaptive_eps = torch.max(adaptive_eps, norms)  # Scale eps with norm magnitude
                norms = norms.clamp_min(adaptive_eps)
                g2d_float = g2d_float.abs().pow(q_exp_star-1) * g2d_float.sign()
                normalized = (g2d_float / norms).to(gggrad.dtype)
                gggrad.copy_(normalized)
                
                return gggrad
        if gggrad.ndim == 4:
            
            if q_exp > 999999999:
                out_channels = gggrad.size(0)
                g2d_float = gggrad.reshape(out_channels, -1).float()
                norms = g2d_float.norm(p=p_exp, dim=1, keepdim=True).pow_(p_exp-1)
                # Use more stable normalization with adaptive epsilon
                # adaptive_eps = eps * torch.ones_like(norms)
                # adaptive_eps = torch.max(adaptive_eps, norms)  # Scale eps with norm magnitude
                # norms = norms.clamp_min(adaptive_eps)
                g2d_float = g2d_float.abs().pow(p_exp-1) * g2d_float.sign() / norms
                normalized = g2d_float.reshape_as(gggrad).to(gggrad.dtype)
                gggrad.copy_(normalized)
                return gggrad
            elif p_exp == 1:
                out_channels = gggrad.size(0)*gggrad.size(1)
                g2d_float = gggrad.reshape(out_channels, -1).float()
                q_exp_star = 1/(1-1/q_exp)
                norms = g2d_float.norm(p=q_exp_star, dim=0, keepdim=True).pow_(q_exp_star-1)
                # Use more stable normalization with adaptive epsilon
                adaptive_eps = eps * torch.ones_like(norms)
                adaptive_eps = torch.max(adaptive_eps, norms)  # Scale eps with norm magnitude
                norms = norms.clamp_min(adaptive_eps)
                g2d_float = g2d_float.abs().pow(q_exp_star-1) * g2d_float.sign() / norms
                normalized = g2d_float.reshape_as(gggrad).to(gggrad.dtype)
                gggrad.copy_(normalized)
        return gggrad

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

                if len(state) == 0 and momentum != 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                p_is_head = getattr(p, "is_head", 0)
                if momentum != 0 and True:
                    # print(grad.ndim)
                    buf: torch.Tensor = state["momentum_buffer"]
                    # buf.mul_(momentum).add_(grad, alpha=1-momentum)
                    buf.mul_(momentum).add_(grad, alpha=1)
                    if nesterov_mom > 0:
                        d_p = grad.add(buf, alpha=nesterov_mom) 
                        
                    else:
                        
                        d_p = buf  
                else:
                    d_p = grad
                
                # # Row-wise L2 normalization for 2D/4D params
                if d_p.ndim == 2:
                    if p_exp < 99999:
                        p_is_ebd = getattr(p, "is_ebd", 0)
                        p_is_linear = getattr(p, "is_linear", 0)
                        p_is_qkv = getattr(p, "is_qkv", 0)
                        p_is_proj = getattr(p, "is_proj", 0)
                        fin, fout = self._fans_by_rule(p)
                        # if (fin > 50000) or (fout > 50000):
                        #     print(fin,fout,p_is_ebd,p_is_conv)
                        if p_is_linear:
                            
                            if p_is_proj:
                                # print('1')
                                fin, fout = self._fans_by_rule(p)
                                temp = self._rownorm_inplace(d_p, eps, p_exp, q_exp)
                                # temp = torch.sign(d_p)
                                update = temp.detach().clone()
                                # d_p = self._rownorm_inplace(d_p, eps, p_exp, q_exp)
                                if use_fan_scaling:
                                    
                                    if p_exp < 3:
                                        update.mul_(1/(float(fin) ** (1.0 / p_exp))).mul_(50)
                                    elif p_exp == 3:
                                        update.mul_(1/(float(fin) ** (1.0 / p_exp))).mul_(2)

                                # d_p = torch.sign(d_p).mul(1/(float(fin)))
                            elif p_is_qkv:
                                
                                fin, fout = self._fans_by_rule(p)
                                
                                # gq, gk, gv = torch.chunk(d_p, 3, dim=1)
                                # gq2 = self._rownorm_inplace(gq.T, eps, p_exp, q_exp).T
                                # gk2 = self._rownorm_inplace(gk.T, eps, p_exp, q_exp).T
                                # gv2 = self._rownorm_inplace(gv.T, eps, p_exp, q_exp).T

                                gq, gk, gv = torch.chunk(d_p, 3, dim=0)
                                gq2 = self._rownorm_inplace(gq, eps, p_exp, q_exp)
                                gk2 = self._rownorm_inplace(gk, eps, p_exp, q_exp)
                                gv2 = self._rownorm_inplace(gv, eps, p_exp, q_exp)
                                
                                temp = torch.cat([gq2, gk2, gv2], dim=0)
                                # temp = torch.sign(d_p)
                                update = temp.detach().clone()
                                # d_p = self._rownorm_inplace(d_p, eps, p_exp, q_exp)
                                
                                
                                
                                if use_fan_scaling:
                                    # print('true!!')
                                    
                                    if p_exp < 3:
                                        update.mul_(1/(float(fin) ** (1.0 / p_exp))).mul_(50)
                                    elif p_exp == 3:
                                        update.mul_(1/(float(fin) ** (1.0 / p_exp))).mul_(2)
                                    # update.mul_(1/(float(fin)))
                                # d_p = torch.sign(d_p).mul(1/(float(fin)))
                            else:
                                
                                fin, fout = self._fans_by_rule(p)
                                # print('head',fin,fout)
                                # temp1 = torch.sign(d_p)
                                temp = self._rownorm_inplace(d_p, eps, p_exp, q_exp)
                                # print((temp-temp1).abs().sum())
                                # temp = torch.sign(d_p)
                                update = temp.detach().clone()
                                # print(use_fan_scaling)
                                if use_fan_scaling and False:
                                    update.mul_(1/(float(fin) ** (1.0 / p_exp)))
                                    update.mul_(1/(float(fin)))
                        elif p_is_ebd:
                            # print('3')
                            
                            fin, fout = self._fans_by_rule(p)
                            # print('ebd',fin,fout)
                            # d_p = zeropower_via_newtonschulz5(d_p, 5)
                            
                            # d_p.mul_(fout**0.5)
                            # d_p = torch.sign(d_p)
                            # temp = self._rownorm_inplace(d_p, eps, p_exp, q_exp)
                            
                            # temp = self._rownorm_inplace(d_p.T, eps, p_exp, q_exp).T
                            # temp = torch.sign(d_p)
                            temp = torch.sign(d_p)
                            update = temp.detach().clone()
                            # update.mul_((float(fout) ** (1.0 / p_exp)))
                            
                            if use_fan_scaling and False:
                                update.mul_(1/(float(fin) ** (1.0 / p_exp)))
                            # d_p.mul_(fin**0.5)
                            
                            # d_p
                        
                    # d_p = self._rownorm_inplace(d_p, eps, p_exp, q_exp)
                    # print(torch.count_nonzero(d_p))
                elif d_p.ndim == 1:
                    d_p = torch.sign(d_p)
                    update = d_p.detach().clone()
                    
                
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

                
                p.add_(update, alpha=alpha)

        return loss
