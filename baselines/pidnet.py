"""PIDNet-S reference.

Xu et al., PIDNet: A Real-time Semantic Segmentation Network, CVPR 2023.
Proportional, integral, and derivative branches meet at a boundary gate.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines._blocks import ConvBNReLU


class PIDNetS(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(ConvBNReLU(3, 32, stride=2), ConvBNReLU(32, 64, stride=2))
        self.p = nn.Sequential(ConvBNReLU(64, 64), ConvBNReLU(64, 128, stride=2))
        self.i = nn.Sequential(ConvBNReLU(128, 128), ConvBNReLU(128, 128))
        self.d = nn.Sequential(ConvBNReLU(64, 64), ConvBNReLU(64, 128, stride=2))
        self.gate = nn.Sequential(nn.Conv2d(128, 1, 1), nn.Sigmoid())
        self.head = nn.Conv2d(128, num_classes, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.stem(image)
        detail = self.d(x)
        context = self.i(self.p(x))
        if context.shape[-2:] != detail.shape[-2:]:
            context = F.interpolate(context, size=detail.shape[-2:], mode="bilinear", align_corners=False)
        fused = detail + context * self.gate(detail)
        logits = self.head(fused)
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
