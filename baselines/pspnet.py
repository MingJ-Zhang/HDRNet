"""PSPNet, ResNet-50, output stride 8.

Paper
    Zhao et al., Pyramid Scene Parsing Network, CVPR 2017.
    https://arxiv.org/abs/1612.01105

Official code
    https://github.com/hszhao/PSPNet

Setting used for the tables
    MMSegmentation ``configs/floodnet/pspnet_r50-d8_2xb8-80k_floodnet-crop1024.py``
    (the same schedule on RescueNet and FWISD). ResNet-50-d8, pyramid bins
    1/2/3/6, 512 crop, 80k iterations, AdamW 6e-5. This file is a standalone
    rendering of that setting so the repository can train without MMSegmentation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.resnet import ResNet50


class PyramidPooling(nn.Module):
    def __init__(self, channels: int, bins: tuple[int, ...] = (1, 2, 3, 6)) -> None:
        super().__init__()
        width = channels // len(bins)
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(channels, width, 1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
            )
            for size in bins
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels + width * len(bins), 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        height, width = feat.shape[-2:]
        parts = [feat]
        for branch in self.branches:
            pooled = branch(feat)
            parts.append(F.interpolate(pooled, size=(height, width), mode="bilinear", align_corners=False))
        return self.bottleneck(torch.cat(parts, dim=1))


class PSPNet(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.backbone = ResNet50(output_stride=8)
        self.ppm = PyramidPooling(self.backbone.out_channels[-1])
        self.cls = nn.Conv2d(512, num_classes, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        _, _, _, c4 = self.backbone(image)
        logits = self.cls(self.ppm(c4))
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
