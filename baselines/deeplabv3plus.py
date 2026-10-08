"""DeepLabV3+ reference.

Chen et al., Encoder-Decoder with Atrous Separable Convolution, ECCV 2018.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines._blocks import ConvBNReLU, Stem


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes: int = 10, dilations: tuple[int, ...] = (1, 6, 12, 18)) -> None:
        super().__init__()
        self.backbone = Stem()
        low_ch, _, _, high_ch = self.backbone.out_channels
        self.aspp = nn.ModuleList(ConvBNReLU(high_ch, 256, 3, dilation=d) if d else ConvBNReLU(high_ch, 256, 1) for d in dilations)
        self.image_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), ConvBNReLU(high_ch, 256, 1))
        self.project = ConvBNReLU(256 * (len(dilations) + 1), 256, 1)
        self.low = ConvBNReLU(low_ch, 48, 1)
        self.head = nn.Sequential(ConvBNReLU(256 + 48, 256), nn.Conv2d(256, num_classes, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        c1, _, _, c4 = self.backbone(image)
        pooled = F.interpolate(self.image_pool(c4), size=c4.shape[-2:], mode="bilinear", align_corners=False)
        context = self.project(torch.cat([*(branch(c4) for branch in self.aspp), pooled], dim=1))
        context = F.interpolate(context, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.head(torch.cat([context, self.low(c1)], dim=1))
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
