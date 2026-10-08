"""Small building blocks shared by the reference baselines."""

from __future__ import annotations

import torch.nn as nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, dilation: int = 1) -> None:
        padding = ((kernel - 1) // 2) * dilation
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class Stem(nn.Module):
    """Four-stride convolutional trunk used by the CNN reference models."""

    def __init__(self, widths: tuple[int, ...] = (64, 128, 256, 512)) -> None:
        super().__init__()
        channels = (3, *widths[:-1])
        self.stages = nn.ModuleList(
            nn.Sequential(ConvBNReLU(channels[i], widths[i], stride=2), ConvBNReLU(widths[i], widths[i]))
            for i in range(len(widths))
        )
        self.out_channels = widths

    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features
