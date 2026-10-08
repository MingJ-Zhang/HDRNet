"""Mask2Former reference.

Cheng et al., Masked-attention Mask Transformer, CVPR 2022.
A small set of queries reads multi-scale features through masked attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines._blocks import ConvBNReLU, Stem


class Mask2Former(nn.Module):
    def __init__(self, num_classes: int = 10, num_queries: int = 32) -> None:
        super().__init__()
        self.backbone = Stem()
        self.lateral = nn.ModuleList(ConvBNReLU(ch, 256, 1) for ch in self.backbone.out_channels)
        self.queries = nn.Embedding(num_queries, 256)
        self.cross = nn.MultiheadAttention(256, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(256)
        self.class_head = nn.Linear(256, num_classes)
        self.mask_head = nn.Linear(256, 256)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feats = [lat(feat) for lat, feat in zip(self.lateral, self.backbone(image))]
        query = self.queries.weight.unsqueeze(0).expand(image.shape[0], -1, -1)
        finest = feats[0]
        for feat in feats:
            tokens = feat.flatten(2).transpose(1, 2)
            attended, _ = self.cross(query, tokens, tokens)
            query = self.norm(query + attended)
        n, _, h, w = finest.shape
        masks = torch.einsum("nqc,nchw->nqhw", self.mask_head(query), finest)
        classes = self.class_head(query).softmax(dim=-1)
        logits = torch.einsum("nqc,nqhw->nchw", classes, masks.sigmoid())
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
