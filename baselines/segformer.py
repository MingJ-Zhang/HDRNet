"""SegFormer-B2.

Paper
    Xie et al., SegFormer: Simple and Efficient Design for Semantic Image
    Segmentation with Transformers, NeurIPS 2021.
    https://arxiv.org/abs/2105.15203

Official code
    https://github.com/NVlabs/SegFormer
    MMSegmentation: ``mmseg/models/backbones/mit.py`` and
    ``mmseg/models/decode_heads/segformer_head.py``.

Setting used for the tables
    ``configs/floodnet/segformer_mit-b2_2xb8-80k_floodnet-crop1024.py``.
    MiT-B2, ImageNet init ``mit_b2_20220624-66e8bf70.pth``, MLP decoder at
    256 channels, cross-entropy plus Dice (weight 3), 512 crop, 80k iterations,
    AdamW 6e-5. HDRNet uses this same encoder. The decoder here is only the
    SegFormer MLP head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hdrnet.mit import MixVisionTransformer


class SegFormer(nn.Module):
    def __init__(self, num_classes: int = 10, embed_dim: int = 256) -> None:
        super().__init__()
        self.encoder = MixVisionTransformer()
        self.projections = nn.ModuleList(nn.Conv2d(ch, embed_dim, 1) for ch in self.encoder.out_channels)
        self.fuse = nn.Conv2d(embed_dim * 4, embed_dim, 1)
        self.head = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.encoder(image)
        size = features[0].shape[-2:]
        aligned = [
            F.interpolate(proj(feat), size=size, mode="bilinear", align_corners=False)
            for proj, feat in zip(self.projections, features)
        ]
        fused = self.fuse(torch.cat(aligned, dim=1))
        logits = self.head(fused)
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
