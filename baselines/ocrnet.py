"""OCRNet with an HRNet-style multi-resolution backbone.

Paper
    Yuan, Chen, Wang, Object-Contextual Representations for Semantic
    Segmentation, ECCV 2020.
    https://arxiv.org/abs/1909.11065

Official code
    https://github.com/HRNet/HRNet-Semantic-Segmentation
    MMSegmentation: ``mmseg/models/decode_heads/ocr_head.py`` and
    ``mmseg/models/backbones/hrnet.py``.

Setting used for the tables
    ``configs/floodnet/ocrnet_hr48_2xb8-80k_floodnet-crop1024.py``, HRNet-W48.
    A coarse softmax forms object regions. Every pixel then attends to those
    regions, and the pixel feature is concatenated with that context.

The backbone below follows HRNet's repeated multi-resolution fusion at a
narrower width than W48, so the file stays readable and does not require the
MMSegmentation package. It is not a weight-compatible port of HRNet-W48.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class Fusion(nn.Module):
    """Exchange information across HRNet branches, then concatenate."""

    def __init__(self, channels: tuple[int, ...]) -> None:
        super().__init__()
        self.proj = nn.ModuleList(_conv(c, channels[-1], stride=1) for c in channels)

    def forward(self, branches: list[torch.Tensor]) -> torch.Tensor:
        target = branches[0].shape[-2:]
        aligned = []
        for feat in branches:
            if feat.shape[-2:] != target:
                feat = F.interpolate(feat, size=target, mode="bilinear", align_corners=False)
            aligned.append(feat)
        return torch.cat(aligned, dim=1)


class HRBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(_conv(3, 64, 2), _conv(64, 64, 2))
        self.stage1 = nn.Sequential(_conv(64, 48), _conv(48, 48))
        self.down2 = _conv(48, 96, 2)
        self.stage2 = nn.ModuleList([nn.Sequential(_conv(48, 48), _conv(48, 48)), nn.Sequential(_conv(96, 96), _conv(96, 96))])
        self.down3 = _conv(96, 192, 2)
        self.stage3 = nn.ModuleList(
            [
                nn.Sequential(_conv(48, 48), _conv(48, 48)),
                nn.Sequential(_conv(96, 96), _conv(96, 96)),
                nn.Sequential(_conv(192, 192), _conv(192, 192)),
            ]
        )
        self.fuse = Fusion((48, 96, 192))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.stage1(self.stem(image))
        b2 = self.down2(x)
        b1, b2 = self.stage2[0](x), self.stage2[1](b2)
        b3 = self.down3(b2)
        branches = [block(feat) for block, feat in zip(self.stage3, (b1, b2, b3))]
        return self.fuse(branches)


class OCRNet(nn.Module):
    def __init__(self, num_classes: int = 10, channels: int = 512) -> None:
        super().__init__()
        self.backbone = HRBackbone()
        in_ch = 48 + 96 + 192
        self.pixel = nn.Sequential(_conv(in_ch, channels), _conv(channels, channels))
        self.soft = nn.Conv2d(channels, num_classes, 1)
        self.query = nn.Conv2d(channels, channels, 1, bias=False)
        self.key = nn.Linear(channels, channels, bias=False)
        self.context = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(channels, num_classes, 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feat = self.pixel(self.backbone(image))
        coarse = self.soft(feat).softmax(dim=1)
        regions = torch.einsum("bkhw,bchw->bkc", coarse, feat)
        mass = coarse.flatten(2).sum(-1).unsqueeze(-1).clamp_min(1e-6)
        regions = regions / mass
        query = self.query(feat).flatten(2).transpose(1, 2)
        attn = (query @ self.key(regions).transpose(1, 2) / (feat.shape[1] ** 0.5)).softmax(dim=-1)
        context = (attn @ regions).transpose(1, 2).reshape_as(feat)
        logits = self.context(torch.cat([feat, context], dim=1))
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
