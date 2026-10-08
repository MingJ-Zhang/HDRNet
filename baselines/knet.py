"""K-Net with a PSP kernel head on ResNet-50-d8.

Paper
    Zhang et al., K-Net: Towards Unified Image Segmentation, NeurIPS 2021.
    https://arxiv.org/abs/2106.14855

Official code
    https://github.com/ZwwWayne/K-Net

Setting used for the tables
    ``configs/floodnet/knet-s3_r50-d8_pspnet_2xb8-adamw-80k_floodnet-crop1024.py``.
    Three kernel-update stages refine one kernel per class. Masks are produced
    by multiplying kernels with the PSP feature, then gathered features update
    the kernels through a grouped linear layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.pspnet import PyramidPooling
from baselines.resnet import ResNet50


class KernelUpdate(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, channels * 2), nn.ReLU(inplace=True), nn.Linear(channels * 2, channels))

    def forward(self, kernels: torch.Tensor, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masks = torch.einsum("bkc,bchw->bkhw", kernels, feat)
        flat = feat.flatten(2)
        weights = masks.flatten(2).softmax(dim=-1)
        gathered = torch.einsum("bkn,bcn->bkc", weights, flat)
        updated = self.norm(kernels + self.gate(gathered))
        updated = updated + self.ffn(updated)
        return updated, masks


class KNet(nn.Module):
    def __init__(self, num_classes: int = 10, stages: int = 3) -> None:
        super().__init__()
        self.backbone = ResNet50(output_stride=8)
        self.ppm = PyramidPooling(self.backbone.out_channels[-1])
        self.kernels = nn.Embedding(num_classes, 512)
        self.updates = nn.ModuleList(KernelUpdate(512) for _ in range(stages))
        self.mask_embed = nn.Linear(512, 512)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        _, _, _, c4 = self.backbone(image)
        feat = self.ppm(c4)
        kernels = self.kernels.weight.unsqueeze(0).expand(image.shape[0], -1, -1)
        masks = None
        for update in self.updates:
            kernels, masks = update(self.mask_embed(kernels), feat)
        assert masks is not None
        return F.interpolate(masks, size=image.shape[-2:], mode="bilinear", align_corners=False)
