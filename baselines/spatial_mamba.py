"""Spatial-Mamba reference.

Xiao et al., Spatial-Mamba: Effective Visual State Space Models via Structure-aware State Fusion, ICLR 2025.
A depthwise scan is fused with a local structure residual.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines._blocks import ConvBNReLU


class SpatialMamba(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(ConvBNReLU(3, 64, stride=2), ConvBNReLU(64, 128, stride=2))
        self.local = nn.Conv2d(128, 128, 3, padding=1, groups=128, bias=False)
        self.scan_h = nn.Conv2d(128, 128, kernel_size=(1, 9), padding=(0, 4), groups=128, bias=False)
        self.scan_w = nn.Conv2d(128, 128, kernel_size=(9, 1), padding=(4, 0), groups=128, bias=False)
        self.fuse = ConvBNReLU(128 * 3, 128, 1)
        self.head = nn.Conv2d(128, num_classes, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feat = self.stem(image)
        mixed = self.fuse(torch.cat([self.local(feat), self.scan_h(feat), self.scan_w(feat)], dim=1))
        logits = self.head(feat + mixed)
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
