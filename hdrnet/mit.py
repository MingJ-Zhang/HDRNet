"""Mix Transformer encoder used by SegFormer-B2 and HDRNet.

The stage layout matches MiT-B2: embed dims 64-128-320-512, depths 3-4-6-3.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int, patch: int, stride: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=stride, padding=patch // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        h, w = x.shape[-2:]
        tokens = x.flatten(2).transpose(1, 2)
        return self.norm(tokens), h, w


class MixFFN(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = self.fc1(x)
        n, _, c = x.shape
        spatial = x.transpose(1, 2).reshape(n, c, h, w)
        spatial = self.dw(spatial)
        x = spatial.flatten(2).transpose(1, 2)
        return self.fc2(F.gelu(x))


class EfficientAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, sr_ratio: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        n, length, dim = x.shape
        q = self.q(x).reshape(n, length, self.num_heads, dim // self.num_heads).permute(0, 2, 1, 3)
        if self.sr is not None and self.norm is not None:
            reduced = self.sr(x.transpose(1, 2).reshape(n, dim, h, w))
            reduced = reduced.flatten(2).transpose(1, 2)
            reduced = self.norm(reduced)
        else:
            reduced = x
        kv = self.kv(reduced).reshape(n, -1, 2, self.num_heads, dim // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(n, length, dim)
        return self.proj(out)


class MixBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, sr_ratio: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, num_heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MixFFN(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), h, w)
        x = x + self.ffn(self.norm2(x), h, w)
        return x


class MixVisionTransformer(nn.Module):
    """MiT-B2 by default. Returns four stride-4/8/16/32 feature maps."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: tuple[int, ...] = (64, 128, 320, 512),
        depths: tuple[int, ...] = (3, 4, 6, 3),
        num_heads: tuple[int, ...] = (1, 2, 5, 8),
        sr_ratios: tuple[int, ...] = (8, 4, 2, 1),
    ) -> None:
        super().__init__()
        patches = ((7, 4), (3, 2), (3, 2), (3, 2))
        in_dims = (in_channels, *embed_dims[:-1])
        self.stages = nn.ModuleList()
        self.embeds = nn.ModuleList()
        for i, (dim, depth, heads, sr) in enumerate(zip(embed_dims, depths, num_heads, sr_ratios)):
            patch, stride = patches[i]
            self.embeds.append(OverlapPatchEmbed(in_dims[i], dim, patch, stride))
            self.stages.append(nn.ModuleList([MixBlock(dim, heads, sr) for _ in range(depth)]))
        self.out_channels = embed_dims

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for embed, blocks in zip(self.embeds, self.stages):
            tokens, h, w = embed(x)
            for block in blocks:
                tokens = block(tokens, h, w)
            x = tokens.transpose(1, 2).reshape(tokens.shape[0], -1, h, w)
            features.append(x)
        return features
