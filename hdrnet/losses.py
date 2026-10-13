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
        state_groups: dict | None = None,
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
        self.groups = state_groups or {"0": {"index": tuple(
            int(state_matrix[:, i].argmax()) if state_matrix[:, i].sum() else -1
            for i in range(num_classes)), "num_states": state_matrix.shape[0], "ordered": False}}
        for name, group in self.groups.items():
            mapping = torch.zeros(group["num_states"], num_classes)
            for class_id, state in enumerate(group["index"]):
                if state >= 0:
                    mapping[state, class_id] = 1
            self.register_buffer(f"state_matrix_{name}", mapping)

    def forward(self, outputs: dict[str, torch.Tensor], label: torch.Tensor) -> dict[str, torch.Tensor]:
        semantic = outputs["semantic"]
        if semantic.shape[-2:] != label.shape[-2:]:
            semantic = F.interpolate(semantic, size=label.shape[-2:], mode="bilinear", align_corners=False)
        semantic = self._resize(semantic, label)
        obj_logits = self._resize(outputs["object"], label)
        valid = label != self.ignore_index
        seg = self._ce(semantic, label) + _dice(semantic, label, self.num_classes, self.ignore_index)
        coarse = -(self._image_composition(label) * outputs["coarse"].clamp_min(1e-7).log()).sum(1).mean()
        obj_target = self._project(label, self.obj_matrix)
        obj = self._ce(obj_logits, obj_target)
        categorical_sum = semantic.sum() * 0
        ordinal_sum = semantic.sum() * 0
        state_hier_sum = semantic.sum() * 0
        categorical_count = ordinal_count = state_count = 0
        for name, group in self.groups.items():
            matrix = getattr(self, f"state_matrix_{name}")
            state_target = self._project(label, matrix)
            state_valid = state_target != self.ignore_index
            state_logits = self._resize(outputs["state"][name], label)
            count = state_valid.sum()
            if group["ordered"]:
                cumulative = self._resize(outputs["ordinal"][name], label)
                thresholds = torch.arange(1, group["num_states"], device=label.device).view(1, -1, 1, 1)
                binary = (state_target.unsqueeze(1) >= thresholds).float()
                loss = F.binary_cross_entropy_with_logits(cumulative, binary, reduction="none")
                ordinal_sum = ordinal_sum + (loss * state_valid.unsqueeze(1)).sum()
                ordinal_count = ordinal_count + count * (group["num_states"] - 1)
            else:
                loss = F.cross_entropy(state_logits, state_target, ignore_index=self.ignore_index, reduction="none")
                categorical_sum = categorical_sum + loss.sum()
                categorical_count = categorical_count + count
            state_hier_sum = state_hier_sum + hierarchical_kl(semantic, state_logits, matrix, state_valid) * count
            state_count = state_count + count
        state = categorical_sum / max(1, categorical_count) + ordinal_sum / max(1, ordinal_count)
        hier = hierarchical_kl(semantic, obj_logits, self.obj_matrix, valid) + state_hier_sum / max(1, state_count)
        boundary_logit = outputs["boundary"].squeeze(1)
        boundary_target = morphological_boundary(label, self.boundary_classes, self.ignore_index)
        if boundary_logit.shape[-2:] != boundary_target.shape[-2:]:
            boundary_logit = F.interpolate(
                boundary_logit.unsqueeze(1), size=boundary_target.shape[-2:], mode="bilinear", align_corners=False
            ).squeeze(1)
        bce = (F.binary_cross_entropy_with_logits(boundary_logit, boundary_target, reduction="none") * valid).sum()
        bce = bce / valid.sum().clamp_min(1)
        prob = boundary_logit.sigmoid() * valid
        dice = 1.0 - (2 * (prob * boundary_target).sum() + 1.0) / (prob.sum() + boundary_target.sum() + 1.0)
        total = (
            seg
            + self.weights["coarse"] * coarse
            + self.weights["obj"] * obj
            + self.weights["state"] * state
            + self.weights["hier"] * hier
            + self.weights["boundary"] * (bce + dice)
        )
        return {"loss": total, "seg": seg.detach(), "coarse": coarse.detach(), "object": obj.detach(),
                "state": state.detach(), "hierarchy": hier.detach(), "boundary": (bce + dice).detach()}

    def _ce(self, logits, target):
        if not (target != self.ignore_index).any():
            return logits.sum() * 0
        return self.ce(logits, target)

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
