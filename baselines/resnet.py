"""ResNet-50 as used by the convolutional comparison models.

PSPNet, DeepLabV3+, K-Net and Mask2Former use output stride 8
(``r50-d8`` in the MMSegmentation configs). FastFCN uses output stride 32
and restores resolution in the joint pyramid upsampling head.

The comparison runs loaded ImageNet weights through MMSegmentation.
``load_imagenet`` reads a standard ResNet-50 ``state_dict`` when a local
checkpoint is passed to the training script later; this module does not
download weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, dilation: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class ResNet50(nn.Module):
    def __init__(self, output_stride: int = 8) -> None:
        super().__init__()
        if output_stride == 8:
            strides = (1, 2, 1, 1)
            dilations = (1, 1, 2, 4)
        elif output_stride == 32:
            strides = (1, 2, 2, 2)
            dilations = (1, 1, 1, 1)
        else:
            raise ValueError("output_stride must be 8 or 32")
        self.output_stride = output_stride
        self.inplanes = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 3, strides[0], dilations[0])
        self.layer2 = self._make_layer(128, 4, strides[1], dilations[1])
        self.layer3 = self._make_layer(256, 6, strides[2], dilations[2])
        self.layer4 = self._make_layer(512, 3, strides[3], dilations[3])
        self.out_channels = (256, 512, 1024, 2048)

    def _make_layer(self, planes: int, blocks: int, stride: int, dilation: int) -> nn.Sequential:
        outplanes = planes * Bottleneck.expansion
        downsample = None
        if stride != 1 or self.inplanes != outplanes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outplanes),
            )
        layers = [Bottleneck(self.inplanes, planes, stride, dilation, downsample)]
        self.inplanes = outplanes
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return c1, c2, c3, c4
