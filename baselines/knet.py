"""K-Net reference.

Zhang et al., K-Net: Towards Unified Image Segmentation, NeurIPS 2021.
A fixed set of kernels is updated by grouped feature gathering.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines._blocks import ConvBNReLU, Stem


class KNet(nn.Module):
    def __init__(self, num_classes: int = 10, iters: int = 3) -> None:
        super().__init__()
        self.backbone = Stem()
        channels = self.backbone.out_channels[-1]
        self.proj = ConvBNReLU(channels, 256, 1)
        self.kernels = nn.Embedding(num_classes, 256)
        self.updates = nn.ModuleList(nn.Linear(256, 256) for _ in range(iters))
        self.head = nn.Linear(256, num_classes)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feat = self.proj(self.backbone(image)[-1])
        n, c, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        kernels = self.kernels.weight.unsqueeze(0).expand(n, -1, -1)
        for update in self.updates:
            masks = (kernels @ tokens.transpose(1, 2)).softmax(dim=-1)
            gathered = masks @ tokens
            kernels = kernels + update(gathered)
        logits = self.head(kernels).transpose(1, 2)
        logits = logits.reshape(n, -1, 1, 1).expand(n, -1, h, w)
        # The kernel masks already carry spatial layout; mix them back in.
        spatial = (kernels @ tokens.transpose(1, 2)).transpose(1, 2).reshape(n, -1, h, w)
        mixed = spatial + logits
        return F.interpolate(mixed, size=image.shape[-2:], mode="bilinear", align_corners=False)
