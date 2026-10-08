"""Hierarchical Damage-Aware Decoding.

Composite labels are factorized into an object category and, when the object
carries one, a disaster state. RescueNet buildings use an ordered head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HDAD(nn.Module):
    def __init__(
        self,
        channels: int,
        num_classes: int,
        num_objects: int,
        num_states: int = 2,
        ordered: bool = False,
    ) -> None:
        super().__init__()
        self.ordered = ordered
        self.num_states = num_states
        self.object_feat = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True))
        self.state_feat = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True))
        self.object_cls = nn.Conv2d(channels, num_objects, 1)
        if ordered and num_states > 1:
            self.state_score = nn.Conv2d(channels, 1, 1)
            self.thresholds = nn.Parameter(torch.linspace(-1.0, 1.0, num_states - 1))
        else:
            self.state_cls = nn.Conv2d(channels, num_states, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.semantic = nn.Conv2d(channels, num_classes, 1)

    def state_logits(self, feat: torch.Tensor) -> torch.Tensor:
        if not self.ordered:
            return self.state_cls(feat)
        score = self.state_score(feat)
        ordered = torch.sort(self.thresholds)[0]
        cumulative = torch.sigmoid(score - ordered.view(1, -1, 1, 1))
        levels = [1.0 - cumulative[:, :1]]
        for index in range(cumulative.shape[1] - 1):
            levels.append(cumulative[:, index : index + 1] - cumulative[:, index + 1 : index + 2])
        levels.append(cumulative[:, -1:])
        return torch.cat(levels, dim=1).clamp_min(1e-6).log()

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        obj_feat = self.object_feat(feat)
        state_feat = self.state_feat(feat)
        fused = self.fuse(torch.cat([feat, obj_feat, state_feat], dim=1))
        return {
            "semantic": self.semantic(fused),
            "object": self.object_cls(obj_feat),
            "state": self.state_logits(state_feat),
        }


def hierarchical_kl(
    semantic: torch.Tensor,
    obj_logits: torch.Tensor,
    obj_matrix: torch.Tensor,
) -> torch.Tensor:
    """KL between the object head and the semantic distribution projected by M_o."""
    pred = obj_logits.softmax(dim=1)
    projected = torch.einsum("oc,nchw->nohw", obj_matrix.float(), semantic.softmax(dim=1))
    projected = projected / projected.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return F.kl_div(pred.clamp_min(1e-6).log(), projected, reduction="batchmean")
