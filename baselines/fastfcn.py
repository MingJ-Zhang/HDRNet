"""FastFCN with joint pyramid upsampling.

Paper
    Wu et al., FastFCN: Rethinking Dilated Convolution in the Backbone for
    Semantic Segmentation, arXiv 2019.
    https://arxiv.org/abs/1903.11816

Official code
    https://github.com/wuhuikai/FastFCN

Setting used for the tables
    ``configs/floodnet/fastfcn_r50-d32_jpu_psp_2xb8-80k_floodnet-crop1024.py``.
    ResNet-50 keeps output stride 32. JPU gathers layer2/3/4, upsamples them
    to the stride-8 map, and applies dilated depthwise convolutions before the
    PSP head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.pspnet import PyramidPooling
from baselines.resnet import ResNet50


class JointPyramidUpsampling(nn.Module):
    def __init__(self, channels: tuple[int, int, int] = (512, 1024, 2048), width: int = 512) -> None:
        super().__init__()
        self.reduce = nn.ModuleList(
            nn.Sequential(nn.Conv2d(c, width, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True)) for c in channels
        )
        self.dilations = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(width, width, 3, padding=d, dilation=d, groups=width, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
            )
            for d in (1, 2, 4, 8)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(width * 4, width, 1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        target = features[0].shape[-2:]
        reduced = []
        for projection, feat in zip(self.reduce, features):
            x = projection(feat)
            if x.shape[-2:] != target:
                x = F.interpolate(x, size=target, mode="bilinear", align_corners=False)
            reduced.append(x)
        mixed = sum(reduced)
        return self.fuse(torch.cat([branch(mixed) for branch in self.dilations], dim=1))


class FastFCN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.backbone = ResNet50(output_stride=32)
        self.jpu = JointPyramidUpsampling()
        self.ppm = PyramidPooling(512)
        self.cls = nn.Conv2d(512, num_classes, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        _, c2, c3, c4 = self.backbone(image)
        feat = self.jpu((c2, c3, c4))
        logits = self.cls(self.ppm(feat))
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
