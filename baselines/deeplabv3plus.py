"""DeepLabV3+, ResNet-50, output stride 8.

Paper
    Chen et al., Encoder-Decoder with Atrous Separable Convolution for
    Semantic Image Segmentation, ECCV 2018.
    https://arxiv.org/abs/1802.02611

Official code
    https://github.com/tensorflow/models/tree/master/research/deeplab
    MMSegmentation: ``mmseg/models/decode_heads/sep_aspp_head.py``

Setting used for the tables
    ``configs/floodnet/deeplabv3plus_r50-d8_4xb4-80k_floodnet-crop1024_v3.py``.
    Atrous rates 1, 12, 24, 36 on the stride-8 map, low-level features from
    ResNet layer1, 512 crop, 80k iterations. The file below keeps the ASPP
    decoder structure and uses standard 3x3 convolutions in place of the
    depthwise separable projection inside MMSegmentation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.resnet import ResNet50


class ASPP(nn.Module):
    def __init__(self, channels: int, rates: tuple[int, ...] = (1, 12, 24, 36), width: int = 256) -> None:
        super().__init__()
        self.branches = nn.ModuleList()
        for rate in rates:
            kernel = 1 if rate == 1 else 3
            padding = 0 if rate == 1 else rate
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(channels, width, kernel, padding=padding, dilation=rate if rate > 1 else 1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.ReLU(inplace=True),
                )
            )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, width, 1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(width * (len(rates) + 1), width, 1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        size = feat.shape[-2:]
        parts = [branch(feat) for branch in self.branches]
        parts.append(F.interpolate(self.pool(feat), size=size, mode="bilinear", align_corners=False))
        return self.project(torch.cat(parts, dim=1))


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.backbone = ResNet50(output_stride=8)
        self.aspp = ASPP(self.backbone.out_channels[-1])
        self.low = nn.Sequential(
            nn.Conv2d(self.backbone.out_channels[0], 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        c1, _, _, c4 = self.backbone(image)
        high = self.aspp(c4)
        high = F.interpolate(high, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.decoder(torch.cat([high, self.low(c1)], dim=1))
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
