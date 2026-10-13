"""Shared blocks for the lightweight comparison adapters."""

from torch import nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
