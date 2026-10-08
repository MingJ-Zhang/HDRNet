"""OCRNet reference.

Yuan et al., Object-Contextual Representations, ECCV 2020.
A coarse map forms object regions, which are then attended by every pixel.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines._blocks import ConvBNReLU, Stem


class OCRNet(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.backbone = Stem()
        channels = self.backbone.out_channels[-1]
        self.pixel = ConvBNReLU(channels, 256, 1)
        self.soft = nn.Conv2d(256, num_classes, 1)
        self.query = nn.Conv2d(256, 256, 1, bias=False)
        self.key = nn.Linear(256, 256, bias=False)
        self.head = nn.Sequential(ConvBNReLU(512, 256), nn.Conv2d(256, num_classes, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feat = self.pixel(self.backbone(image)[-1])
        coarse = self.soft(feat).softmax(dim=1)
        n, c, h, w = feat.shape
        regions = torch.einsum("nkhw,nchw->nkc", coarse, feat)
        regions = regions / coarse.flatten(2).sum(-1).unsqueeze(-1).clamp_min(1e-6)
        query = self.query(feat).flatten(2).transpose(1, 2)
        attn = (query @ self.key(regions).transpose(1, 2) / (c ** 0.5)).softmax(dim=-1)
        context = (attn @ regions).transpose(1, 2).reshape(n, c, h, w)
        logits = self.head(torch.cat([feat, context], dim=1))
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
