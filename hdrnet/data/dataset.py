"""Image and mask loading for FloodNet, RescueNet, and FWISD."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from hdrnet.data.transforms import Normalize, TrainTransform

IMAGENET_MEAN = (123.675, 116.28, 103.53)
IMAGENET_STD = (58.395, 57.12, 57.375)


def _pairs(root: Path, split: dict) -> list[tuple[Path, Path]]:
    image_dir = root / split["img"]
    ann_dir = root / split["ann"]
    if not image_dir.is_dir():
        raise FileNotFoundError(f"missing image directory: {image_dir}")
    pairs = []
    suffix = split["img_suffix"]
    ann_suffix = split["ann_suffix"]
    for image_path in sorted(image_dir.rglob(f"*{suffix}")):
        stem = image_path.name[: -len(suffix)]
        mask_path = ann_dir / f"{stem}{ann_suffix}"
        if mask_path.is_file():
            pairs.append((image_path, mask_path))
    if not pairs:
        raise FileNotFoundError(f"no image/mask pairs under {image_dir} and {ann_dir}")
    return pairs


class DisasterDataset(Dataset):
    """Pairs an RGB image with a single-channel class-index mask.

    Training applies the paper pipeline: random scale in ``[0.5, 2.0]`` around
    a 1024 base, a 512 crop that rejects crops dominated by one class, a
    horizontal flip, and photometric distortion. ``class_aware=True`` is the
    DA-SegFormer crop, which centers on a rare class with probability 0.5.
    """

    def __init__(
        self,
        root: str | Path,
        split: dict,
        train: bool = False,
        crop_size: int = 512,
        class_aware: bool = False,
        rare_classes: tuple[int, ...] = (),
        rare_prob: float = 0.5,
    ) -> None:
        self.pairs = _pairs(Path(root), split)
        self.train = train
        self.normalize = Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self.augment = TrainTransform(crop_size, class_aware, rare_classes, rare_prob) if train else None

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image_path, mask_path = self.pairs[index]
        image = np.asarray(Image.open(image_path).convert("RGB"))
        mask = np.asarray(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = mask.astype(np.int64)
        if self.augment is not None:
            image, mask = self.augment(image, mask)
        return {
            "image": self.normalize(image),
            "mask": torch.from_numpy(np.ascontiguousarray(mask)),
            "name": image_path.stem,
        }
