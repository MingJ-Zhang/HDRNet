"""Confusion-matrix IoU and F-score.

The manuscript also prints an IoU-derived F1, ``200 * IoU / (100 + IoU)`` with
IoU in percent. ``paper_f1`` reports that conversion. ``f1`` is the usual
per-class harmonic mean of precision and recall.
"""

from __future__ import annotations

import torch


class ConfusionMeter:
    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred = pred.detach().cpu().reshape(-1)
        target = target.detach().cpu().reshape(-1)
        valid = target != self.ignore_index
        pred = pred[valid]
        target = target[valid]
        keep = (target >= 0) & (target < self.num_classes)
        pred = pred[keep].clamp(0, self.num_classes - 1)
        target = target[keep]
        index = target * self.num_classes + pred
        self.matrix += torch.bincount(index, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, float]:
        matrix = self.matrix.float()
        true_positive = matrix.diag()
        support = matrix.sum(dim=1)
        predicted = matrix.sum(dim=0)
        union = support + predicted - true_positive
        iou = true_positive / union.clamp_min(1)
        precision = true_positive / predicted.clamp_min(1)
        recall = true_positive / support.clamp_min(1)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-6)
        present = support > 0
        mean_iou = iou[present].mean().item() * 100
        paper_f1 = 200 * (mean_iou) / (100 + mean_iou) if present.any() else 0.0
        # Per-class paper F1, then the unweighted mean. IoU here is in percent.
        per_paper = 200 * (iou * 100) / (100 + iou * 100)
        return {
            "mIoU": mean_iou,
            "mF1": f1[present].mean().item() * 100,
            "paper_mF1": per_paper[present].mean().item(),
            "paper_f1_from_miou": paper_f1,
        }
