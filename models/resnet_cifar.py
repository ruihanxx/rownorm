import torch
import torch.nn as nn
import torch.nn.functional as F
import math
EPS = 1e-7  # numerical stability

def _normalize_rows_lmo(A: torch.Tensor, p: float) -> torch.Tensor:
    """
    Row-wise: row <- sign(row) * |row|^{p-1} / ||row||_p^{p-1} / din^{1/p}
    A: (m, n)
    """
    assert A.dim() == 2, "Row normalization expects a 2D tensor"
    m, n = A.shape
    p = float(p)
    if p > 99999:
        # with torch.no_grad():
        #     out = torch.zeros_like(A)
        #     absA = A.abs()
        #     # argmax over columns for each row -> (m,)
        #     idx = torch.argmax(absA, dim=1)
        #     rows = torch.arange(m, device=A.device)
        #     # pick original signs at argmax positions
        #     x = A[rows, idx]
        #     out[rows, idx] = torch.sign(x)  # sign(0)=0 -> stays 0 (as spec doesn’t override)
        # return out
        with torch.no_grad():
            norms = A.norm(p=1, dim=1, keepdim=True)
            # Use more stable normalization with adaptive epsilon
            adaptive_eps = EPS * torch.ones_like(norms)
            adaptive_eps = torch.max(adaptive_eps, norms * 1e-6)  # Scale eps with norm magnitude
            norms = norms.clamp_min(adaptive_eps)
            
            normalized = (A / norms).to(A.dtype)
        return normalized

    # sign(x)*|x|^{p-1}
    V = torch.sign(A) * A.abs().clamp_min(EPS).pow(max(p - 1.0, 0.0))
    # row Lp norms: (m, 1)
    lp = A.abs().clamp_min(EPS).pow(p).sum(dim=1, keepdim=True).pow(1.0 / p)
    scale = lp.clamp_min(EPS).pow(max(p - 1.0, 0.0)) * (n ** (1.0 / p))
    return V / scale
    

def _normalize_cols_lmo(A: torch.Tensor, q: float) -> torch.Tensor:
    """
    Column-wise: col <- sign(col) * |col|^{q*-1} / ||col||_{q*}^{q*-1} * (dout^{1/q} / din)
    A: (dout=m, din=n)
    """
    assert A.dim() == 2, "Column normalization expects a 2D tensor"
    m, n = A.shape
    q = float(q)
    if not (q > 1.0 and math.isfinite(q)):
        raise ValueError("For p=1 branch, q must be finite and > 1.")
    q_star = q / (q - 1.0)

    V = torch.sign(A) * A.abs().clamp_min(EPS).pow(max(q_star - 1.0, 0.0))
    # column L_{q*} norms: (1, n)
    lqstar = A.abs().clamp_min(EPS).pow(q_star).sum(dim=0, keepdim=True).pow(1.0 / q_star)
    scale = lqstar.clamp_min(EPS).pow(max(q_star - 1.0, 0.0))
    res = V / scale
    res = res * ((m ** (1.0 / q)) / float(n))  # *dout^{1/q} / din
    return res

def _lmo_init_tensor_(t: torch.Tensor, p: float | None = 1, q: float | str | None = math.inf):
    """
    In-place LMO-style init for a single parameter tensor t, following:
    - if q == inf     : row-normalize with p
    - elif p == 1     : column-normalize with q and its conjugate q*
    Shapes:
      * 2D (dout, din): as-is per rule
      * 4D (fout, fin, H, W):
           - q=inf -> reshape to (fout, fin*H*W) then row rule
           - p=1   -> reshape to (fout*fin, H*W) then col rule
      * 1D bias: treat as (N, 1), i.e., column vector
    """
    use_q_inf = (isinstance(q, str) and q.lower() == "inf") or (isinstance(q, float) and math.isinf(q))
    use_p_one = (p == 1)
    

    if (not use_q_inf) and (not use_p_one):
        raise ValueError("You must set exactly one of: q=inf  OR  p=1 (exclusive).")

    with torch.no_grad():
        if t.dim() == 4:
            fout, fin, H, W = t.shape
            if use_q_inf:
                A = t.view(fout, fin * H * W)
                A = _normalize_rows_lmo(A, float(p))
                t.copy_(A.view_as(t))
            else:  # p == 1
                A = t.view(fout * fin, H * W)
                A = _normalize_cols_lmo(A, float(q))
                t.copy_(A.view_as(t))

        elif t.dim() == 2:
            if use_q_inf:
                t.copy_(_normalize_rows_lmo(t, float(p)))
            else:  # p == 1
                t.copy_(_normalize_cols_lmo(t, float(q)))

        elif t.dim() == 1:
            # treat bias (or any 1D) as a (N, 1) column
            A = t.view(-1, 1)
            if use_q_inf:
                A = _normalize_rows_lmo(A, float(p))
            else:
                A = _normalize_cols_lmo(A, float(q))
            t.copy_(A.view_as(t))
        else:
            # other shapes are rare; leave untouched
            pass

def lmo_init_model_(
    model: nn.Module,
    *,
    p: float | None = None,
    q: float | str | None = None,
    include_bias: bool = True,
    touch_bn: bool = False,
):
    """
    Apply LMO-style init to Conv/Linear (and optionally BN) parameters in-place.
    - keep BN parameters by default (touch_bn=False).
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if m.weight is not None:
                _lmo_init_tensor_(m.weight, p, q)
            if include_bias and getattr(m, "bias", None) is not None:
                _lmo_init_tensor_(m.bias, p, q)
        elif touch_bn and isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if m.weight is not None:
                _lmo_init_tensor_(m.weight, p, q)
            if getattr(m, "bias", None) is not None:
                _lmo_init_tensor_(m.bias, p, q)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, 
                 num_blocks, 
                 num_classes=10, 
                 base_width: int = 64,
                 lmo_p: float | None = 1,
                 lmo_q: float | str | None = math.inf,
                 lmo_enable: bool = False,
                 lmo_include_bias: bool = True,
                 lmo_touch_bn: bool = False,):
        super().__init__()
        self.in_planes = int(base_width)
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, int(
            base_width * 1), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(
            base_width * 2), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(
            base_width * 4), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, int(
            base_width * 8), num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_width* 8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # >>> LMO-style init on top (optional) <<<
        if lmo_enable:
            if (lmo_p == 1 or lmo_q == math.inf):
                lmo_init_model_(self, p=lmo_p, q=lmo_q, include_bias=lmo_include_bias, touch_bn=lmo_touch_bn)    
            else:
                raise(ValueError("You must set exactly one of: q=inf  OR  p=1 (exclusive)."))
            
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet18_cifar(
    num_classes: int = 10,
    base_width: int = 64,
    # >>> pass-through for LMO init <<<
    lmo_p: float | None = 1,
    lmo_q: float | str | None = math.inf,
    lmo_enable: bool = False,
    lmo_include_bias: bool = True,
    lmo_touch_bn: bool = False,
) -> ResNet:
    return ResNet(
        BasicBlock, [2, 2, 2, 2],
        num_classes=num_classes,
        base_width=base_width,
        lmo_p=lmo_p, lmo_q=lmo_q, lmo_enable=lmo_enable,
        lmo_include_bias=lmo_include_bias, lmo_touch_bn=lmo_touch_bn,
    )
