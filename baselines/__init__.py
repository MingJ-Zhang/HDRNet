"""Comparison models used in the paper tables.

Each module is a short reference implementation of the cited architecture.
HDRNet itself lives in :mod:`hdrnet`.
"""

from baselines.da_segformer import DASegFormer
from baselines.deeplabv3plus import DeepLabV3Plus
from baselines.fastfcn import FastFCN
from baselines.hl_sam_seg import HLSAMSeg
from baselines.knet import KNet
from baselines.mask2former import Mask2Former
from baselines.ocrnet import OCRNet
from baselines.overlock import OverLoCK
from baselines.pidnet import PIDNetS
from baselines.pspnet import PSPNet
from baselines.rs3mamba import RS3Mamba
from baselines.segformer import SegFormer
from baselines.segman import SegMANT
from baselines.spatial_mamba import SpatialMamba

__all__ = [
    "DASegFormer",
    "DeepLabV3Plus",
    "FastFCN",
    "HLSAMSeg",
    "KNet",
    "Mask2Former",
    "OCRNet",
    "OverLoCK",
    "PIDNetS",
    "PSPNet",
    "RS3Mamba",
    "SegFormer",
    "SegMANT",
    "SpatialMamba",
]
