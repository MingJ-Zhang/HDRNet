"""HL-SAM-Seg.

Paper
    Hierarchical SAM adaptation for prompt-free semantic segmentation. The
    comparison row freezes a SAM image encoder and trains a hierarchical mask
    decoder on the disaster labels.

Official code
    Segment Anything: https://github.com/facebookresearch/segment-anything

Setting used for the tables
    The released comparison uses the SAM ViT image encoder plus a lightweight
    hierarchical head, not a CNN trained from scratch. This file keeps the
    head that the table actually depends on: one learned prompt per class,
    an adapter on the encoder feature, and a class-conditioned mask. The
    encoder is a convolutional stand-in so the repository does not download
    the SAM checkpoint. Swap ``self.encoder`` for a frozen SAM ViT if you have
    that weight locally.
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
