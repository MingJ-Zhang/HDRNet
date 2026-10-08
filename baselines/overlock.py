"""OverLoCK-T.

Paper
    Lou et al., OverLoCK: An Overview-first-Look-Closely-next ConvNet with
    Context-Mixing Dynamic Kernels, CVPR 2025.
    https://arxiv.org/abs/2502.20087

Official code
    https://github.com/LMMMEng/OverLoCK
    The comparison config is
    ``configs/floodnet/upernet_overlock-t_2xb8-80k_floodnet-crop1024.py``
    (UPerNet head on OverLoCK-T).

This file follows the paper's two-step path: a downsampled overview produces
a spatial gate, and the higher-resolution features are convolved under that
gate. It does not include the official dynamic-kernel CUDA kernels.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines._blocks import ConvBNReLU


class OverLoCK(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.base = nn.Sequential(ConvBNReLU(3, 48, stride=2), ConvBNReLU(48, 96, stride=2), ConvBNReLU(96, 160, stride=2))
        self.overview = nn.Sequential(ConvBNReLU(160, 160, stride=2), nn.Conv2d(160, 160, 1), nn.Sigmoid())
        self.focus = nn.Sequential(ConvBNReLU(160, 160), ConvBNReLU(160, 160))
        self.head = nn.Conv2d(160, num_classes, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feat = self.base(image)
        gate = self.overview(feat)
        gate = F.interpolate(gate, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        refined = self.focus(feat * gate)
        logits = self.head(refined)
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
