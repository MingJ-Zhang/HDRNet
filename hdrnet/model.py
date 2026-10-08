"""HDRNet: MiT-B2 encoder, relation context, structural boundary, hierarchical head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hdrnet.drca import DRCA
from hdrnet.dsbr import DSBR
from hdrnet.hdad import HDAD
from hdrnet.mit import MixVisionTransformer


class MLPDecoder(nn.Module):
    def __init__(self, in_channels: tuple[int, ...], embed_dim: int = 256) -> None:
        super().__init__()
        self.projections = nn.ModuleList(nn.Conv2d(channels, embed_dim, 1) for channels in in_channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels), embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        size = features[0].shape[-2:]
        aligned = []
        for projection, feature in zip(self.projections, features):
            mapped = projection(feature)
            mapped = F.interpolate(mapped, size=size, mode="bilinear", align_corners=False)
            aligned.append(mapped)
        return self.fuse(torch.cat(aligned, dim=1))


class HDRNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        num_objects: int = 8,
        num_states: int = 2,
        ordered: bool = False,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = MixVisionTransformer()
        channels = self.encoder.out_channels
        self.drca = DRCA(channels[-1], num_classes=num_classes)
        self.decoder = MLPDecoder(channels, embed_dim)
        self.dsbr = DSBR(channels[0], channels[1], embed_dim)
        self.hdad = HDAD(embed_dim, num_classes, num_objects, num_states, ordered)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(image)
        refined, coarse = self.drca(features[-1])
        features[-1] = refined
        decoded = self.decoder(features)
        decoded, boundary = self.dsbr(features[0], features[1], decoded)
        outputs = self.hdad(decoded)
        outputs["coarse"] = coarse
        outputs["boundary"] = boundary
        return outputs
