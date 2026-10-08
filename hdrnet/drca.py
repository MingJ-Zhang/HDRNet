"""Disaster Relation-Aware Context Aggregation.

Five context branches are gated by an image-level semantic composition,
then biased by a frozen category co-occurrence prior.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseBranch(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        padding = dilation
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=padding, dilation=dilation, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DRCA(nn.Module):
    def __init__(self, channels: int = 512, num_classes: int = 10, token_dim: int = 256, beta: float = 1.0) -> None:
        super().__init__()
        self.beta = beta
        self.num_classes = num_classes
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(channels, channels, 1, bias=False),
                DepthwiseBranch(channels, dilation=1),
                DepthwiseBranch(channels, dilation=6),
                DepthwiseBranch(channels, dilation=12),
                nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1), nn.GELU()),
            ]
        )
        self.coarse = nn.Linear(channels, num_classes)
        self.scale = nn.Linear(num_classes, len(self.branches))
        self.embed = nn.Embedding(num_classes, token_dim)
        self.qkv = nn.Linear(token_dim, token_dim * 3)
        self.token_out = nn.Linear(token_dim, token_dim)
        self.pix_q = nn.Linear(channels, token_dim)
        self.pix_kv = nn.Linear(token_dim, token_dim * 2)
        self.pix_out = nn.Conv2d(token_dim, channels, 1, bias=False)
        self.out_norm = nn.BatchNorm2d(channels)
        self.register_buffer("prior", torch.zeros(num_classes, num_classes), persistent=True)

    def set_prior(self, relation: torch.Tensor) -> None:
        """Install R_ij = N(c_i, c_j) / N(c_i), with a zero diagonal."""
        prior = relation.detach().float().clone()
        prior.fill_diagonal_(0)
        self.prior.copy_(prior)

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = feat.mean(dim=(2, 3))
        pi = self.coarse(pooled).softmax(dim=-1)
        alpha = self.scale(pi).softmax(dim=-1)
        context = 0.0
        for weight, branch in zip(alpha.unbind(dim=-1), self.branches):
            branch_feat = branch(feat)
            if branch_feat.shape[-2:] != feat.shape[-2:]:
                branch_feat = branch_feat.expand_as(feat)
            context = context + branch_feat * weight[:, None, None, None]

        tokens = pi.unsqueeze(-1) * self.embed.weight.unsqueeze(0)
        q, k, v = self.qkv(tokens).chunk(3, dim=-1)
        scale = q.shape[-1] ** -0.5
        logits = (q @ k.transpose(-2, -1)) * scale + self.beta * self.prior
        tokens = self.token_out(logits.softmax(dim=-1) @ v)

        n, _, h, w = feat.shape
        pix = self.pix_q(feat.flatten(2).transpose(1, 2))
        k_r, v_r = self.pix_kv(tokens).chunk(2, dim=-1)
        attn = (pix @ k_r.transpose(-2, -1) * (pix.shape[-1] ** -0.5)).softmax(dim=-1)
        mixed = (attn @ v_r).transpose(1, 2).reshape(n, -1, h, w)
        refined = self.out_norm(context + self.pix_out(mixed))
        return refined, pi
