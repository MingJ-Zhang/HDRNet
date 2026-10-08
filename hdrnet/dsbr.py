"""Damage-Sensitive Structural Boundary Refinement.

Boundary targets are morphological gradients of the selected disaster classes.
Directional kernels start as finite differences and stay trainable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _finite_difference(kind: str) -> torch.Tensor:
    kernels = {
        "horizontal": torch.tensor([[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]),
        "vertical": torch.tensor([[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "diagonal": torch.tensor([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        "anti_diagonal": torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    }
    return kernels[kind]


class DSBR(nn.Module):
    def __init__(self, c1: int = 64, c2: int = 128, dec_channels: int = 256, groups: int = 8) -> None:
        super().__init__()
        self.proj1 = nn.Conv2d(c1, dec_channels, 1, bias=False)
        self.proj2 = nn.Conv2d(c2, dec_channels, 1, bias=False)
        self.fuse = nn.Conv2d(dec_channels * 2, dec_channels, 1, bias=False)
        self.directions = nn.ModuleList(
            [nn.Conv2d(dec_channels, dec_channels, 3, padding=1, groups=groups, bias=False) for _ in range(4)]
        )
        self._init_directions()
        self.boundary = nn.Conv2d(dec_channels * 4, 1, 1)
        self.gate = nn.Conv2d(dec_channels * 2 + 1, dec_channels, 1)
        self.out = nn.Conv2d(dec_channels, dec_channels, 3, padding=1, bias=False)

    def _init_directions(self) -> None:
        kinds = ("horizontal", "vertical", "diagonal", "anti_diagonal")
        for conv, kind in zip(self.directions, kinds):
            weight = conv.weight.data
            base = _finite_difference(kind)
            weight.zero_()
            for channel in range(weight.shape[0]):
                weight[channel, 0].copy_(base)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor, decoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        size = decoded.shape[-2:]
        low = torch.cat(
            [
                self.proj1(F.interpolate(f1, size=size, mode="bilinear", align_corners=False)),
                self.proj2(F.interpolate(f2, size=size, mode="bilinear", align_corners=False)),
            ],
            dim=1,
        )
        structural = self.fuse(low)
        responses = [conv(structural) for conv in self.directions]
        stacked = torch.cat(responses, dim=1)
        boundary = self.boundary(stacked).sigmoid()
        gate = self.gate(torch.cat([decoded, structural, boundary], dim=1)).sigmoid()
        refined = self.out(decoded + decoded * gate)
        return refined, boundary


def morphological_boundary(label: torch.Tensor, class_ids: tuple[int, ...], ignore_index: int = 255) -> torch.Tensor:
    """3x3 morphological gradient, merged with a logical OR over C_d."""
    target = torch.zeros(label.shape, dtype=torch.float32, device=label.device)
    valid = label != ignore_index
    for class_id in class_ids:
        mask = ((label == class_id) & valid).float().unsqueeze(1)
        dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
        target = torch.maximum(target, (dilated - eroded).squeeze(1))
    return (target > 0).float()
