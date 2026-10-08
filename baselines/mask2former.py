"""Mask2Former with a ResNet-50 pixel decoder.

Paper
    Cheng et al., Masked-attention Mask Transformer for Universal Image
    Segmentation, CVPR 2022.
    https://arxiv.org/abs/2112.01527

Official code
    https://github.com/facebookresearch/Mask2Former
    MMSegmentation wraps it in ``mmseg/models/decode_heads/mask2former_head.py``.

Setting used for the tables
    ``configs/floodnet/mask2former_r50_2xb8-80k_floodnet-crop1024.py``.
    The trained model uses the official masked-attention decoder and a
    deformable pixel decoder. This file keeps the same interface used at
    evaluation time: a set of queries predicts one mask per query, masks are
    combined with class probabilities, and the result is a dense semantic map.
    The pixel decoder here is an FPN on ResNet-50 rather than MSDeformAttn.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.resnet import ResNet50


class MaskedAttentionBlock(nn.Module):
    def __init__(self, dim: int = 256, heads: int = 8) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.ReLU(inplace=True), nn.Linear(dim * 4, dim))

    def forward(self, query: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is not None:
            # Down-weight memory locations the previous masks did not cover.
            weight = mask.flatten(2).clamp(0, 1).mean(dim=1)
            memory = memory * (0.5 + 0.5 * weight).unsqueeze(-1)
        attended, _ = self.attn(query, memory, memory)
        query = self.norm1(query + attended)
        return self.norm2(query + self.ffn(query))


class Mask2Former(nn.Module):
    def __init__(self, num_classes: int = 10, num_queries: int = 100, dim: int = 256) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone = ResNet50(output_stride=8)
        self.lateral = nn.ModuleList(nn.Conv2d(c, dim, 1) for c in self.backbone.out_channels)
        self.smooth = nn.ModuleList(nn.Conv2d(dim, dim, 3, padding=1) for _ in range(4))
        self.query = nn.Embedding(num_queries, dim)
        self.blocks = nn.ModuleList(MaskedAttentionBlock(dim) for _ in range(3))
        self.class_head = nn.Linear(dim, num_classes + 1)
        self.mask_embed = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))

    def _fpn(self, features: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        laterals = [conv(feat) for conv, feat in zip(self.lateral, features)]
        for level in range(len(laterals) - 1, 0, -1):
            laterals[level - 1] = laterals[level - 1] + F.interpolate(
                laterals[level], size=laterals[level - 1].shape[-2:], mode="bilinear", align_corners=False
            )
        return [smooth(feat) for smooth, feat in zip(self.smooth, laterals)]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        pyramid = self._fpn(self.backbone(image))
        query = self.query.weight.unsqueeze(0).expand(image.shape[0], -1, -1)
        mask_pred = None
        for level, block in zip(reversed(pyramid[1:]), self.blocks):
            memory = level.flatten(2).transpose(1, 2)
            attn_mask = None
            if mask_pred is not None:
                attn_mask = F.interpolate(mask_pred.detach().sigmoid(), size=level.shape[-2:], mode="bilinear", align_corners=False)
            query = block(query, memory, attn_mask)
            mask_pred = torch.einsum("bqc,bchw->bqhw", self.mask_embed(query), level)
        assert mask_pred is not None
        finest = pyramid[0]
        masks = torch.einsum("bqc,bchw->bqhw", self.mask_embed(query), finest)
        classes = self.class_head(query).softmax(dim=-1)[..., :-1]
        semantic = torch.einsum("bqc,bqhw->bchw", classes, masks.sigmoid())
        return F.interpolate(semantic, size=image.shape[-2:], mode="bilinear", align_corners=False)
