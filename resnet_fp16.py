"""
ResNet Implementation in Pure FP16 (Half Precision) PyTorch.
Berdasarkan pendekatan repo YUNBLAK/standalone_16bits_nn.

Fitur Utama:
1. FP16BatchNorm2d: Mengakumulasi mean & variance batch dalam FP32 untuk mencegah overflow/underflow,
   lalu mengalikan dengan weight & bias FP16. (Ekuivalen PyTorch dari BatchNormalization16.py di repo YUNBLAK).
2. Seluruh bobot konvolusi, linier, dan masukan beroperasi murni dalam torch.float16.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'FP16BatchNorm2d',
    'ResNetFP16',
    'resnet18_fp16',
    'resnet32_fp16',
    'resnet56_fp16',
]


class FP16BatchNorm2d(nn.Module):
    """
    BatchNorm2d khusus Pure FP16 (Ekuivalen PyTorch dari BatchNormalization16.py di repo YUNBLAK).
    Mengakumulasi statistik mean dan varians dalam presisi float32 untuk mencegah overflow
    saat kuadrat selisih diakumulasikan pada resolusi tinggi/banyak channel.
    """
    def __init__(self, num_features: int, eps: float = 1e-4, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1, dtype=torch.float16))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1, dtype=torch.float16))
        self.register_buffer('running_mean', torch.zeros(1, num_features, 1, 1, dtype=torch.float32))
        self.register_buffer('running_var', torch.ones(1, num_features, 1, 1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            x_float = x.float()
            # Hitung mean & variance dalam float32 melintasi (Batch, Height, Width)
            mean = x_float.mean(dim=(0, 2, 3), keepdim=True)
            var = ((x_float - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
            var = torch.clamp(var, min=0.0)

            # Perbarui running stats dalam float32
            with torch.no_grad():
                self.running_mean.mul_(1.0 - self.momentum).add_(mean * self.momentum)
                self.running_var.mul_(1.0 - self.momentum).add_(var * self.momentum)

            x_norm = (x_float - mean) / torch.sqrt(var + self.eps)
        else:
            x_float = x.float()
            x_norm = (x_float - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        return (x_norm.to(x.dtype) * self.weight) + self.bias


class BasicBlockFP16(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = FP16BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = FP16BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                FP16BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetFP16(nn.Module):
    def __init__(self, block, num_blocks: list, num_classes: int = 1000, is_cifar: bool = False):
        super().__init__()
        self.in_planes = 64 if not is_cifar else 16
        self.is_cifar = is_cifar

        if is_cifar:
            # ResNet untuk CIFAR (32x32)
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = FP16BatchNorm2d(16)
            self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
            self.linear = nn.Linear(64 * block.expansion, num_classes)
        else:
            # ResNet untuk ImageNet / ImageNet-100 (224x224)
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = FP16BatchNorm2d(64)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            self.linear = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()
        self.to(torch.float16)

    def _make_layer(self, block, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float16:
            x = x.to(torch.float16)

        if self.is_cifar:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            out = self.linear(out)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.maxpool(out)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            out = self.linear(out)

        return out


def resnet18_fp16(num_classes: int = 1000, is_cifar: bool = False, **kwargs) -> ResNetFP16:
    """ResNet-18 FP16 Murni."""
    return ResNetFP16(BasicBlockFP16, [2, 2, 2, 2], num_classes=num_classes, is_cifar=is_cifar)


def resnet32_fp16(num_classes: int = 1000, is_cifar: bool = False, **kwargs) -> ResNetFP16:
    """ResNet-32 FP16 Murni (seperti pada repo YUNBLAK/standalone_16bits_nn)."""
    return ResNetFP16(BasicBlockFP16, [5, 5, 5, 5] if not is_cifar else [5, 5, 5], num_classes=num_classes, is_cifar=is_cifar)


def resnet56_fp16(num_classes: int = 1000, is_cifar: bool = False, **kwargs) -> ResNetFP16:
    """ResNet-56 FP16 Murni (seperti pada repo YUNBLAK/standalone_16bits_nn)."""
    return ResNetFP16(BasicBlockFP16, [9, 9, 9, 9] if not is_cifar else [9, 9, 9], num_classes=num_classes, is_cifar=is_cifar)
