"""Model factory. Names match the comparison table."""

from __future__ import annotations

import torch.nn as nn

from baselines import (
    DASegFormer,
    DeepLabV3Plus,
    FastFCN,
    HLSAMSeg,
    KNet,
    Mask2Former,
    OCRNet,
    OverLoCK,
    PIDNetS,
    PSPNet,
    RS3Mamba,
    SegFormer,
    SegMANT,
    SpatialMamba,
)
from configs.datasets import DATASETS, factor_matrix
from configs.protocol import PROTOCOL
from hdrnet.engine.losses import OhemDiceLoss, SegLoss
from hdrnet.losses import HDRNetLoss
from hdrnet.model import HDRNet

MODELS = {
    "hdrnet": HDRNet,
    "pspnet": PSPNet,
    "deeplabv3plus": DeepLabV3Plus,
    "fastfcn": FastFCN,
    "knet": KNet,
    "mask2former": Mask2Former,
    "ocrnet": OCRNet,
    "pidnet": PIDNetS,
    "segformer": SegFormer,
    "segman": SegMANT,
    "overlock": OverLoCK,
    "spatial-mamba": SpatialMamba,
    "rs3mamba": RS3Mamba,
    "hl-sam-seg": HLSAMSeg,
    "da-segformer": DASegFormer,
}


def build_model(name: str, dataset: str) -> nn.Module:
    spec = DATASETS[dataset]
    num_classes = len(spec["classes"])
    if name == "hdrnet":
        return HDRNet(
            num_classes=num_classes,
            num_objects=spec["num_objects"],
            num_states=spec["num_states"],
            ordered=spec["ordered_damage"],
        )
    if name not in MODELS:
        known = ", ".join(sorted(MODELS))
        raise KeyError(f"unknown model {name}. choices: {known}")
    return MODELS[name](num_classes=num_classes)


def build_criterion(name: str, dataset: str) -> nn.Module:
    spec = DATASETS[dataset]
    if name == "hdrnet":
        return HDRNetLoss(
            num_classes=len(spec["classes"]),
            boundary_classes=spec["boundary_classes"],
            obj_matrix=factor_matrix(spec["object_index"], spec["num_objects"]),
            state_matrix=factor_matrix(spec["state_index"], spec["num_states"]),
            weights=PROTOCOL["loss_weights"],
        )
    if name == "da-segformer":
        return OhemDiceLoss()
    return SegLoss(dice_weight=3.0)
