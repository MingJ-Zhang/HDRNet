"""PSPNet reference.

Zhao et al., Pyramid Scene Parsing Network, CVPR 2017.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines._blocks import ConvBNReLU, Stem


class PSPNet(nn.Module):
    def __init__(self, num_classes: int = 10, pools: tuple[int, ...] = (1, 2, 3, 6)) -> None:
        super().__init__()
        self.backbone = Stem()
        channels = self.backbone.out_channels[-1]
        width = channels // len(pools)
        self.pools = pools
        self.branches = nn.ModuleList(nn.Sequential(nn.AdaptiveAvgPool2d(size), ConvBNReLU(channels, width, 1)) for size in pools)
        self.head = nn.Sequential(ConvBNReLU(channels + width * len(pools), 256, 3), nn.Conv2d(256, num_classes, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(image)[-1]
        parts = [feat]
        for branch in self.branches:
            pooled = branch(feat)
            parts.append(F.interpolate(pooled, size=feat.shape[-2:], mode="bilinear", align_corners=False))
        logits = self.head(torch.cat(parts, dim=1))
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
