"""RS3Mamba reference.

Ma et al., RS3Mamba: Visual State Space Model for Remote Sensing Image Semantic Segmentation, 2024.
A convolutional branch and a state-space branch are fused before the decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines._blocks import ConvBNReLU


class RS3Mamba(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv = nn.Sequential(ConvBNReLU(3, 64, stride=2), ConvBNReLU(64, 128, stride=2), ConvBNReLU(128, 256, stride=2))
        self.ssm = nn.Sequential(
            ConvBNReLU(3, 64, stride=2),
            ConvBNReLU(64, 128, stride=2),
            nn.Conv2d(128, 256, kernel_size=(1, 7), padding=(0, 3), groups=128, bias=False),
            nn.Conv2d(256, 256, 1, bias=False),
            nn.GELU(),
        )
        self.fuse = ConvBNReLU(512, 256, 1)
        self.head = nn.Conv2d(256, num_classes, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        local = self.conv(image)
        state = self.ssm(image)
        if state.shape[-2:] != local.shape[-2:]:
            state = F.interpolate(state, size=local.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.head(self.fuse(torch.cat([local, state], dim=1)))
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
