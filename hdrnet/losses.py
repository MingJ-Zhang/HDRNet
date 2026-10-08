"""Loss terms used by HDRNet.

Semantic segmentation uses cross-entropy plus Dice. The auxiliary terms are
the coarse composition loss, factor losses, hierarchical consistency, and the
boundary BCE + Dice loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hdrnet.dsbr import morphological_boundary
from hdrnet.hdad import hierarchical_kl


def _dice(logits: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255) -> torch.Tensor:
    valid = target != ignore_index
    safe = target.clamp(0, num_classes - 1)
    one_hot = F.one_hot(safe, num_classes).permute(0, 3, 1, 2).float()
    one_hot = one_hot * valid.unsqueeze(1)
    prob = logits.softmax(dim=1) * valid.unsqueeze(1)
    dims = (0, 2, 3)
    intersection = (prob * one_hot).sum(dims)
    union = prob.sum(dims) + one_hot.sum(dims)
    return 1.0 - ((2 * intersection + 1.0) / (union + 1.0)).mean()


class HDRNetLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        boundary_classes: tuple[int, ...],
        obj_matrix: torch.Tensor,
        state_matrix: torch.Tensor,
        weights: dict[str, float] | None = None,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.boundary_classes = boundary_classes
        self.ignore_index = ignore_index
        self.register_buffer("obj_matrix", obj_matrix.float())
        self.register_buffer("state_matrix", state_matrix.float())
        defaults = dict(coarse=0.1, obj=0.4, state=0.6, hier=0.3, boundary=0.4)
        self.weights = {**defaults, **(weights or {})}
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, outputs: dict[str, torch.Tensor], label: torch.Tensor) -> dict[str, torch.Tensor]:
        semantic = outputs["semantic"]
        if semantic.shape[-2:] != label.shape[-2:]:
            semantic = F.interpolate(semantic, size=label.shape[-2:], mode="bilinear", align_corners=False)
        semantic = self._resize(semantic, label)
        obj_logits = self._resize(outputs["object"], label)
        state_logits = self._resize(outputs["state"], label)
        seg = self.ce(semantic, label) + _dice(semantic, label, self.num_classes, self.ignore_index)
        coarse = F.binary_cross_entropy(outputs["coarse"], self._image_composition(label))
        obj_target = self._project(label, self.obj_matrix)
        state_target = self._project(label, self.state_matrix)
        obj = self.ce(obj_logits, obj_target)
        state = self.ce(state_logits, state_target)
        hier = hierarchical_kl(semantic, obj_logits, self.obj_matrix)
        boundary_logit = outputs["boundary"].squeeze(1)
        boundary_target = morphological_boundary(label, self.boundary_classes, self.ignore_index)
        if boundary_logit.shape[-2:] != boundary_target.shape[-2:]:
            boundary_logit = F.interpolate(
                boundary_logit.unsqueeze(1), size=boundary_target.shape[-2:], mode="bilinear", align_corners=False
            ).squeeze(1)
        bce = F.binary_cross_entropy_with_logits(boundary_logit, boundary_target)
        prob = boundary_logit.sigmoid()
        dice = 1.0 - (2 * (prob * boundary_target).sum() + 1.0) / (prob.sum() + boundary_target.sum() + 1.0)
        total = (
            seg
            + self.weights["coarse"] * coarse
            + self.weights["obj"] * obj
            + self.weights["state"] * state
            + self.weights["hier"] * hier
            + self.weights["boundary"] * (bce + dice)
        )
        return {"loss": total, "seg": seg.detach(), "boundary": (bce + dice).detach()}

    def _image_composition(self, label: torch.Tensor) -> torch.Tensor:
        valid = label != self.ignore_index
        safe = label.clamp(0, self.num_classes - 1)
        one_hot = F.one_hot(safe, self.num_classes).float()
        counts = (one_hot * valid.unsqueeze(-1)).sum(dim=(1, 2))
        return counts / counts.sum(dim=-1, keepdim=True).clamp_min(1.0)

    def _resize(self, logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        if logits.shape[-2:] != label.shape[-2:]:
            logits = F.interpolate(logits, size=label.shape[-2:], mode="bilinear", align_corners=False)
        return logits

    def _project(self, label: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        index = matrix.argmax(dim=0)
        index = index.masked_fill(matrix.sum(dim=0) <= 0, self.ignore_index)
        mapped = index[label.clamp(0, index.numel() - 1)]
        return mapped.masked_fill(label == self.ignore_index, self.ignore_index)
