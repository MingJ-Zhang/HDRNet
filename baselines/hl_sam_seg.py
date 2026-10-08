"""HL-SAM-Seg reference.

A hierarchical prompt-free segmentation head in the style of SAM adapters used
for remote-sensing damage mapping. The image encoder here is a lightweight
stand-in for the frozen SAM backbone used in the comparison.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines._blocks import ConvBNReLU


class HLSAMSeg(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBNReLU(3, 64, stride=2),
            ConvBNReLU(64, 128, stride=2),
            ConvBNReLU(128, 256, stride=2),
            ConvBNReLU(256, 256, stride=2),
        )
        self.prompts = nn.Embedding(num_classes, 256)
        self.adapter = nn.Sequential(nn.Conv2d(256, 256, 1), nn.GELU(), nn.Conv2d(256, 256, 1))
        self.mask = nn.Conv2d(256, 256, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feat = self.adapter(self.encoder(image))
        tokens = self.prompts.weight
        masks = torch.einsum("kc,nchw->nkhw", tokens, self.mask(feat))
        return F.interpolate(masks, size=image.shape[-2:], mode="bilinear", align_corners=False)
