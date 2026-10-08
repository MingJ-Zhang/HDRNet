"""DA-SegFormer reference.

Zhang et al. The comparison in this paper uses the same MiT-B2 graph as SegFormer.
The difference is the training recipe: class-aware sampling and OHEM + Dice.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from baselines.segformer import SegFormer


class DASegFormer(SegFormer):
    """MiT-B2 SegFormer. Class-aware crop and OHEM+Dice belong to the trainer."""

    def loss(self, logits: torch.Tensor, label: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
        ce = F.cross_entropy(logits, label, ignore_index=ignore_index, reduction="none")
        valid = label != ignore_index
        hard = ce[valid]
        if hard.numel() == 0:
            return ce.sum() * 0.0
        keep = hard > hard.median()
        ohem = hard[keep].mean() if keep.any() else hard.mean()
        prob = logits.softmax(dim=1)
        one_hot = F.one_hot(label.clamp(min=0), prob.shape[1]).permute(0, 3, 1, 2).float()
        one_hot = one_hot * valid.unsqueeze(1)
        dims = (0, 2, 3)
        dice = 1.0 - ((2 * (prob * one_hot).sum(dims) + 1.0) / (prob.sum(dims) + one_hot.sum(dims) + 1.0)).mean()
        return ohem + dice
