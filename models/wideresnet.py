"""
Wide ResNet implementation for CIFAR-10
Based on the paper "Wide Residual Networks" by Zagoruyko & Komodakis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BasicBlock(nn.Module):
    """Basic residual block for Wide ResNet"""

    def __init__(self, in_planes, out_planes, stride, dropout_rate=0.0):
        super(BasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.dropout_rate = dropout_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=stride,
            padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))

        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))

        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)

        out = self.conv2(out)

        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    """A block consisting of multiple BasicBlocks"""

    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropout_rate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, dropout_rate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropout_rate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    """Wide ResNet model"""

    def __init__(self, depth=28, width=10, num_classes=10, dropout_rate=0.3):
        super(WideResNet, self).__init__()

        # Wide ResNet 구조: depth = 6n + 4
        # 예: depth=28 -> n=4, 각 그룹에 4개의 블록
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6

        # 채널 수 계산
        nChannels = [16, 16 * width, 32 * width, 64 * width]

        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)

        # 3개의 residual block groups
        self.block1 = NetworkBlock(
            n, nChannels[0], nChannels[1], BasicBlock, 1, dropout_rate)
        self.block2 = NetworkBlock(
            n, nChannels[1], nChannels[2], BasicBlock, 2, dropout_rate)
        self.block3 = NetworkBlock(
            n, nChannels[2], nChannels[3], BasicBlock, 2, dropout_rate)

        # 최종 분류층
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        # 가중치 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


def create_wideresnet(depth=28, width=10, num_classes=10, dropout_rate=0.3):
    """
    Create a Wide ResNet model

    Args:
        depth: Network depth (should be 6n+4)
        width: Width multiplier
        num_classes: Number of output classes
        dropout_rate: Dropout rate

    Returns:
        WideResNet model
    """
    return WideResNet(depth=depth, width=width, num_classes=num_classes, dropout_rate=dropout_rate)


# 预定义的常用配置
def wideresnet28_1(num_classes=10):
    """WRN-28-1"""
    return create_wideresnet(depth=28, width=1, num_classes=num_classes)


def wideresnet28_2(num_classes=10):
    """WRN-28-2"""
    return create_wideresnet(depth=28, width=2, num_classes=num_classes)


def wideresnet28_4(num_classes=10):
    """WRN-28-4"""
    return create_wideresnet(depth=28, width=4, num_classes=num_classes)


def wideresnet28_8(num_classes=10):
    """WRN-28-8"""
    return create_wideresnet(depth=28, width=8, num_classes=num_classes)


def wideresnet28_10(num_classes=10):
    """WRN-28-10 (standard)"""
    return create_wideresnet(depth=28, width=10, num_classes=num_classes)


if __name__ == "__main__":
    # 测试模型
    import torch

    models = [
        ("WRN-28-1", wideresnet28_1()),
        ("WRN-28-2", wideresnet28_2()),
        ("WRN-28-4", wideresnet28_4()),
        ("WRN-28-8", wideresnet28_8()),
    ]

    x = torch.randn(1, 3, 32, 32)

    for name, model in models:
        try:
            y = model(x)
            params = sum(p.numel() for p in model.parameters())
            print(
                f"{name}: {params:,} parameters ({params/1e6:.1f}M), output shape: {y.shape}")
        except Exception as e:
            print(f"{name}: Error - {e}")







