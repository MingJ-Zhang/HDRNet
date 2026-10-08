"""FastFCN reference.

Wu et al., FastFCN: Rethinking Dilated Convolution in the Backbone, 2019.
The head replaces dense dilated context with joint pyramid upsampling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines._blocks import ConvBNReLU, Stem


class FastFCN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.backbone = Stem()
        channels = self.backbone.out_channels[-1]
        self.reduce = ConvBNReLU(channels, 256, 1)
        self.joint = nn.Sequential(
            nn.Conv2d(256, 256 * 4, 1, bias=False),
            nn.PixelShuffle(2),
            ConvBNReLU(256, 256),
        )
        self.head = nn.Conv2d(256, num_classes, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feat = self.reduce(self.backbone(image)[-1])
        up = self.joint(feat)
        logits = self.head(up)
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
