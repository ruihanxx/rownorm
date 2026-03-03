from typing import Iterable, Optional

import torch


class MOGASGD(torch.optim.Optimizer):
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
        base_factor: float = 20.0,
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

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov_mom=nesterov_mom,
            eps=eps,
            max_grad_norm=max_grad_norm,
            p_exp=p_exp,
            q_exp=q_exp,
            use_fan_scaling=use_fan_scaling,
            base_factor=base_factor,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _fans_by_rule(t: torch.Tensor) -> tuple[int, int]:
        """Return (fan_in, fan_out) for a parameter tensor.

        Supports:
        - 2D (linear): (out_features, in_features)
        - 4D (conv):  (out_channels, in_channels, kH, kW)

        Optional overrides:
        - param.fan_in_override: int
        - param.fan_out_override: int
        """
        fin_override = getattr(t, "fan_in_override", None)
        fout_override = getattr(t, "fan_out_override", None)

        if t.ndim == 2:
            fan_in = int(fin_override) if isinstance(fin_override, int) else int(t.size(1))
            fan_out = int(fout_override) if isinstance(fout_override, int) else int(t.size(0))
            return fan_in, fan_out

        if t.ndim == 4:
            # Standard conv fan definition
            kernel_area = int(t.size(2) * t.size(3))
            fan_in = int(fin_override) if isinstance(fin_override, int) else int(t.size(1) * kernel_area)
            fan_out = int(fout_override) if isinstance(fout_override, int) else int(t.size(0) * kernel_area)
            return fan_in, fan_out

        raise ValueError(
            f"Tensor with ndim={t.ndim} is not supported for fan scaling. "
            "Use 2D/4D parameters or disable fan scaling."
        )

    @staticmethod
    def _moga_inplace(gggrad: torch.Tensor, eps: float, p_exp: float = 1.0, q_exp: float = torch.inf) -> torch.Tensor:
        """Row-normalize gradients for 2D or 4D tensors.

        This implements a stabilized normalization that is compatible with general p/q exponents:
        - If q_exp is effectively infinity, we apply an Lp-based row normalization.
        - If p_exp == 1 and q_exp is finite, we use the conjugate exponent q*.

        The function returns a tensor containing the normalized gradients (same dtype as input).
        """
        if gggrad is None:
            return gggrad

        grad = gggrad.detach().clone()
        grad_float = grad.float()

        if gggrad.ndim == 2:
            if q_exp > 999999999:
                row_norms = grad_float.norm(p=p_exp, dim=1, keepdim=True).pow_(p_exp - 1.0)
                row_norms = row_norms.clamp_min(eps)
                grad_float = grad_float.abs().pow(p_exp - 1.0) * grad_float.sign()
                normalized = (grad_float / row_norms).to(gggrad.dtype)
                grad.copy_(normalized)
                return grad

            if p_exp == 1:
                q_star = 1.0 / (1.0 - 1.0 / q_exp)
                col_norms = grad_float.norm(p=q_star, dim=0, keepdim=True).pow_(q_star - 1.0)
                col_norms = col_norms.clamp_min(eps)
                grad_float = grad_float.abs().pow(q_star - 1.0) * grad_float.sign()
                normalized = (grad_float / col_norms).to(gggrad.dtype)
                gggrad.copy_(normalized)
                return gggrad

            return gggrad

        # 4D conv: reshape to 2D then apply the same logic
        if gggrad.ndim == 4:
            if q_exp > 999999999:
                out_channels = int(gggrad.size(0))
                grad_2d = gggrad.reshape(out_channels, -1).float()
                row_norms = grad_2d.norm(p=p_exp, dim=1, keepdim=True).pow_(p_exp - 1.0).clamp_min(eps)
                grad_2d = grad_2d.abs().pow(p_exp - 1.0) * grad_2d.sign() / row_norms
                gggrad.copy_(grad_2d.reshape_as(gggrad).to(gggrad.dtype))
                return gggrad

            if p_exp == 1:
                rows = int(gggrad.size(0) * gggrad.size(1))
                grad_2d = gggrad.reshape(rows, -1).float()
                q_star = 1.0 / (1.0 - 1.0 / q_exp)
                col_norms = grad_2d.norm(p=q_star, dim=0, keepdim=True).pow_(q_star - 1.0).clamp_min(eps)
                grad_2d = grad_2d.abs().pow(q_star - 1.0) * grad_2d.sign() / col_norms
                gggrad.copy_(grad_2d.reshape_as(gggrad).to(gggrad.dtype))
                return gggrad

            return gggrad

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
            base_factor: float = group["base_factor"]

            gradients = []
            for p in group["params"]:
                if p.grad is not None:
                    gradients.append(p.grad)

            # Global gradient clipping before row normalization

            for p in group["params"]:
                if p.grad is None:
                    continue

                raw_grad = p.grad
                grad = raw_grad.detach().clone().to(raw_grad.dtype).contiguous()

                # momentum and Nesterov part
                state = self.state[p]

                if len(state) == 0 and momentum != 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                if momentum != 0 and True:
                    buf: torch.Tensor = state["momentum_buffer"]
                    if nesterov_mom > 0:
                        d_p = grad.mul(1 - nesterov_mom).add(buf, alpha=nesterov_mom)
                        buf.mul_(momentum).add_(grad, alpha=1 - momentum)
                    else:
                        buf.mul_(momentum).add_(grad, alpha=1 - momentum)
                        d_p = buf
                else:
                    d_p = grad

                if d_p.ndim == 2:
                    if p_exp < 99999:
                        p_is_ebd = getattr(p, "is_ebd", 0)
                        p_is_conv = getattr(p, "is_conv", 0)
                        fin, fout = self._fans_by_rule(p)
                        if p_is_conv:
                            p_is_qkv = getattr(p, "is_qkv", 0)
                            if not p_is_qkv:
                                fin, fout = self._fans_by_rule(p)
                                temp = self._moga_inplace(d_p.T, eps, p_exp, q_exp).T
                                update = temp.detach().clone()
                                if use_fan_scaling:
                                    update.mul_(1 / (float(fin) ** (1.0 / p_exp))).mul_(base_factor)
                            elif (not p_is_ebd) and p_is_qkv:
                                fin, fout = self._fans_by_rule(p)
                                gq, gk, gv = torch.chunk(d_p, 3, dim=1)
                                gq2 = self._moga_inplace(gq.T, eps, p_exp, q_exp).T
                                gk2 = self._moga_inplace(gk.T, eps, p_exp, q_exp).T
                                gv2 = self._moga_inplace(gv.T, eps, p_exp, q_exp).T
                                temp = torch.cat([gq2, gk2, gv2], dim=1)
                                update = temp.detach().clone()
                                if use_fan_scaling:
                                    update.mul_(1 / (float(fin) ** (1.0 / p_exp))).mul_(base_factor)
                        elif p_is_ebd:
                            fin, fout = self._fans_by_rule(p)
                            temp = torch.sign(d_p)
                            update = temp.detach().clone()
                        else:
                            fin, fout = self._fans_by_rule(p)
                            temp = self._moga_inplace(d_p, eps, p_exp, q_exp)
                            update = temp.detach().clone()
                elif d_p.ndim == 1:
                    d_p = torch.sign(d_p)
                    update = d_p.detach().clone()

                # Decoupled weight decay
                if weight_decay != 0:
                    if p.ndim != 1:
                        p.mul_(1.0 - lr * weight_decay)

                alpha = -lr
                p.add_(update, alpha=alpha)

        return loss

