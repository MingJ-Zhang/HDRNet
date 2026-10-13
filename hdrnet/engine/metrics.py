"""Exact per-class IoU and F1 from one accumulated confusion matrix."""

from __future__ import annotations

import torch


class ConfusionMeter:
    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        if pred.shape != target.shape:
            raise ValueError("prediction and target shapes must match")
        if pred.is_floating_point() or target.is_floating_point():
            raise ValueError("prediction and target must contain integer class ids")
        pred = pred.detach().cpu().long().reshape(-1)
        target = target.detach().cpu().long().reshape(-1)
        valid = target != self.ignore_index
        pred = pred[valid]
        target = target[valid]
        if ((target < 0) | (target >= self.num_classes) | (pred < 0) | (pred >= self.num_classes)).any():
            raise ValueError("class id outside configured ontology")
        index = target * self.num_classes + pred
        self.matrix += torch.bincount(index, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict:
        matrix = self.matrix.double()
        true_positive = matrix.diag()
        support = matrix.sum(dim=1)
        predicted = matrix.sum(dim=0)
        union = support + predicted - true_positive
        iou = true_positive / union.clamp_min(1)
        precision = true_positive / predicted.clamp_min(1)
        recall = true_positive / support.clamp_min(1)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-6)
        present = union > 0
        mean_iou = iou[present].mean().item() * 100 if present.any() else 0.0
        per_paper = 200 * (iou * 100) / (100 + iou * 100)
        return {
            "mIoU": mean_iou,
            "mF1": f1[present].mean().item() * 100 if present.any() else 0.0,
            "paper_mF1": per_paper[present].mean().item() if present.any() else 0.0,
            "evaluated_pixels": int(self.matrix.sum()),
            "per_class": [{"class_id": i, "support": int(support[i]),
                           "IoU": float(iou[i] * 100) if present[i] else None,
                           "F1": float(f1[i] * 100) if present[i] else None}
                          for i in range(self.num_classes)],
            "confusion_matrix": self.matrix.tolist(),
        }
