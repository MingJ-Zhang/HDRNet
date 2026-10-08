"""Losses for the comparison models.

SegFormer and the CNN baselines use cross-entropy plus Dice with weight 3,
matching ``segformer_mit-b2_2xb8-80k_*``. DA-SegFormer uses OHEM plus Dice
with equal weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hdrnet.engine.infer import logits_of


def _dice(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    num_classes = logits.shape[1]
    valid = target != ignore_index
    safe = target.clamp(0, num_classes - 1)
    one_hot = F.one_hot(safe, num_classes).permute(0, 3, 1, 2).float() * valid.unsqueeze(1)
    prob = logits.softmax(dim=1) * valid.unsqueeze(1)
    dims = (0, 2, 3)
    intersection = (prob * one_hot).sum(dims)
    union = prob.sum(dims) + one_hot.sum(dims)
    return 1.0 - ((2 * intersection + 1.0) / (union + 1.0)).mean()


class SegLoss(nn.Module):
    def __init__(self, dice_weight: float = 3.0, ignore_index: int = 255) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, output, label: torch.Tensor) -> torch.Tensor:
        logits = logits_of(output)
        if logits.shape[-2:] != label.shape[-2:]:
            logits = F.interpolate(logits, size=label.shape[-2:], mode="bilinear", align_corners=False)
        return self.ce(logits, label) + self.dice_weight * _dice(logits, label, self.ignore_index)


class OhemDiceLoss(nn.Module):
    """DA-SegFormer objective. ``min_kept`` follows the 512-crop config (209715)."""

    def __init__(self, min_kept: int = 209715, ignore_index: int = 255) -> None:
        super().__init__()
        self.min_kept = min_kept
        self.ignore_index = ignore_index

    def forward(self, output, label: torch.Tensor) -> torch.Tensor:
        logits = logits_of(output)
        if logits.shape[-2:] != label.shape[-2:]:
            logits = F.interpolate(logits, size=label.shape[-2:], mode="bilinear", align_corners=False)
        pixel = F.cross_entropy(logits, label, ignore_index=self.ignore_index, reduction="none")
        valid = label != self.ignore_index
        losses = pixel[valid]
        if losses.numel() == 0:
            return logits.sum() * 0.0
        keep = max(1, min(self.min_kept, losses.numel()))
        threshold = torch.topk(losses, keep).values[-1]
        kept = losses[losses >= threshold]
        return kept.mean() + _dice(logits, label, self.ignore_index)
