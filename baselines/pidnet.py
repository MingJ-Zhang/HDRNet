"""PIDNet-S, the small three-branch real-time model.

Paper
    Xu, Xiong, Bhattacharyya, PIDNet: A Real-time Semantic Segmentation
    Network Inspired by PID Controllers, CVPR 2023.
    https://arxiv.org/abs/2206.02066

Official code
    https://github.com/XuJiacong/PIDNet
    MMSegmentation: ``mmseg/models/backbones/pidnet.py``.

Setting used for the tables
    ``configs/floodnet/pidnet-s_2xb8-80k_floodnet-crop1024.py``.
    P carries context (integral), I refines it, and D keeps high-resolution
    detail (derivative). A boundary head gates the residual between the detail
    branch and the context branch, which is the Bag fusion in the paper.

Channel widths follow PIDNet-S (32 / 64 / 128) rather than the larger M/L models.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class PIDNetS(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(_block(3, 32, stride=2), _block(32, 64, stride=2))
        self.p = nn.Sequential(_block(64, 64), _block(64, 128, stride=2))
        self.i = nn.Sequential(_block(128, 128), _block(128, 128))
        self.d = nn.Sequential(_block(64, 64), _block(64, 64))
        self.detail_down = _block(64, 128, stride=2)
        self.boundary = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 1, 1))
        self.compress = nn.Conv2d(128, 128, 1, bias=False)
        self.head = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        stem = self.stem(image)
        detail = self.detail_down(self.d(stem))
        context = self.i(self.p(stem))
        if context.shape[-2:] != detail.shape[-2:]:
            context = F.interpolate(context, size=detail.shape[-2:], mode="bilinear", align_corners=False)
        boundary = self.boundary(detail).sigmoid()
        # Bag: trust detail where the boundary gate is high, otherwise context.
        fused = self.compress(detail) * boundary + context * (1.0 - boundary)
        logits = self.head(fused)
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
