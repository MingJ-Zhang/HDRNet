"""SegMAN-T.

Paper
    Fu et al., SegMAN: Omni-scale Context Modeling with State Space Models
    and Local Attention for Semantic Segmentation, CVPR 2025.
    https://arxiv.org/abs/2412.11890

Official code
    https://github.com/yunxiangfu2001/SegMAN

Setting used for the tables
    SegMAN-Tiny, trained with the same 512 crop / 80k / AdamW schedule as
    SegFormer. The official model is an omni-scale state-space encoder. This
    file keeps that split: a local convolution branch beside a directional
    state mix, then a joint head. It is not a weight-compatible copy of the
    official SegMAN-T checkpoint.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines._blocks import ConvBNReLU


class _StateMix(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.in_proj = nn.Conv2d(channels, channels * 2, 1)
        self.scan = nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels, bias=False)
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.in_proj(self.norm(x)).chunk(2, dim=1)
        mixed = self.scan(a.sigmoid() * b)
        return x + self.out(mixed)


class SegMANT(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(ConvBNReLU(3, 32, stride=2), ConvBNReLU(32, 64, stride=2))
        self.local = nn.Sequential(ConvBNReLU(64, 96), ConvBNReLU(96, 96))
        self.global_down = ConvBNReLU(64, 96, stride=2)
        self.mixer = _StateMix(96)
        self.head = nn.Sequential(ConvBNReLU(192, 128), nn.Conv2d(128, num_classes, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.stem(image)
        local = self.local(x)
        glob = self.mixer(self.global_down(x))
        glob = F.interpolate(glob, size=local.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.head(torch.cat([local, glob], dim=1))
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
