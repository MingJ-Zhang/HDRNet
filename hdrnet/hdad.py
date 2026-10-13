"""Hierarchical Damage-Aware Decoding.

Composite labels are factorized into an object category and, when the object
carries one, a disaster state. RescueNet buildings use an ordered head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StateHead(nn.Module):
    """Categorical states or monotone cumulative ordinal thresholds."""

    def __init__(self, channels: int, states: int, ordered: bool):
        super().__init__()
        self.ordered = ordered
        self.states = states
        self.score = nn.Conv2d(channels, 1 if ordered else states, 1)
        if ordered:
            self.first_threshold = nn.Parameter(torch.tensor(-1.0))
            self.threshold_gaps = nn.Parameter(torch.zeros(states - 2))

    def forward(self, feat):
        score = self.score(feat)
        if not self.ordered:
            return score, None
        thresholds = torch.cat([self.first_threshold.reshape(1),
                                self.first_threshold + F.softplus(self.threshold_gaps).cumsum(0)])
        cumulative_logits = score - thresholds.view(1, -1, 1, 1)
        q = cumulative_logits.sigmoid()
        probabilities = torch.cat([1 - q[:, :1], q[:, :-1] - q[:, 1:], q[:, -1:]], dim=1)
        return probabilities.clamp_min(1e-7).log(), cumulative_logits


class HDAD(nn.Module):
    def __init__(
        self,
        channels: int,
        num_classes: int,
        num_objects: int,
        num_states: int = 2,
        ordered: bool = False,
        state_groups: dict | None = None,
    ) -> None:
        super().__init__()
        self.ordered = ordered
        self.num_states = num_states
        self.object_feat = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True))
        self.state_feat = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True))
        self.object_cls = nn.Conv2d(channels, num_objects, 1)
        groups = state_groups or {"0": {"num_states": num_states, "ordered": ordered}}
        self.state_heads = nn.ModuleDict({name: StateHead(channels, group["num_states"], group["ordered"])
                                          for name, group in groups.items()})
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.semantic = nn.Conv2d(channels, num_classes, 1)

    def forward(self, feat: torch.Tensor) -> dict:
        obj_feat = self.object_feat(feat)
        state_feat = self.state_feat(feat)
        fused = self.fuse(torch.cat([feat, obj_feat, state_feat], dim=1))
        states, ordinal = {}, {}
        for name, head in self.state_heads.items():
            states[name], cumulative = head(state_feat)
            if cumulative is not None:
                ordinal[name] = cumulative
        return {
            "semantic": self.semantic(fused),
            "object": self.object_cls(obj_feat),
            "state": states,
            "ordinal": ordinal,
        }


def hierarchical_kl(
    semantic: torch.Tensor,
    obj_logits: torch.Tensor,
    obj_matrix: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """KL between the object head and the semantic distribution projected by M_o."""
    pred = obj_logits.softmax(dim=1)
    projected = torch.einsum("oc,nchw->nohw", obj_matrix.float(), semantic.softmax(dim=1))
    projected = projected / projected.sum(dim=1, keepdim=True).clamp_min(1e-6)
    divergence = (pred * (pred.clamp_min(1e-6).log() - projected.clamp_min(1e-6).log())).sum(1)
    if valid is None:
        return divergence.mean()
    return (divergence * valid).sum() / valid.sum().clamp_min(1)
