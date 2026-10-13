"""Training augmentations used by the comparison protocol."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance


class Normalize:
    def __init__(self, mean: tuple[float, ...], std: tuple[float, ...]) -> None:
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()
        return (tensor - self.mean) / self.std


def _resize(image: np.ndarray, mask: np.ndarray, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    img = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().unsqueeze(0)
    lab = torch.from_numpy(mask.astype(np.float32)).view(1, 1, *mask.shape)
    img = F.interpolate(img, size=(height, width), mode="bilinear", align_corners=False)
    lab = F.interpolate(lab, size=(height, width), mode="nearest")
    return img.squeeze(0).permute(1, 2, 0).byte().numpy(), lab[0, 0].long().numpy()


class TrainTransform:
    def __init__(
        self,
        crop_size: int = 512,
        class_aware: bool = False,
        rare_classes: tuple[int, ...] = (),
        rare_prob: float = 0.5,
        cat_max_ratio: float = 0.75,
        scale: int = 1024,
        ratio_range: tuple[float, float] = (0.5, 2.0),
        horizontal_prob: float = 0.5,
        vertical_prob: float = 0.5,
        rotation_prob: float = 0.5,
        rotation_degrees: float = 180.0,
    ) -> None:
        self.crop_size = crop_size
        self.class_aware = class_aware
        self.rare_classes = rare_classes
        self.rare_prob = rare_prob
        self.cat_max_ratio = cat_max_ratio
        self.scale = scale
        self.ratio_range = ratio_range
        if crop_size <= 0 or scale <= 0 or not 0 < ratio_range[0] <= ratio_range[1]:
            raise ValueError("crop, scale and resize ratios must be positive")
        if any(not 0 <= p <= 1 for p in (rare_prob, horizontal_prob, vertical_prob, rotation_prob)):
            raise ValueError("augmentation probabilities must lie in [0, 1]")
        if not 0 < cat_max_ratio <= 1 or rotation_degrees < 0:
            raise ValueError("invalid crop ratio or rotation angle")
        self.horizontal_prob = horizontal_prob
        self.vertical_prob = vertical_prob
        self.rotation_prob = rotation_prob
        self.rotation_degrees = rotation_degrees

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        image, mask = self._random_resize(image, mask)
        image, mask = self._crop(image, mask)
        if np.random.rand() < self.horizontal_prob:
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
        if np.random.rand() < self.vertical_prob:
            image = np.ascontiguousarray(image[::-1])
            mask = np.ascontiguousarray(mask[::-1])
        if np.random.rand() < self.rotation_prob:
            angle = float(np.random.uniform(-self.rotation_degrees, self.rotation_degrees))
            image = np.asarray(Image.fromarray(image).rotate(angle, resample=Image.Resampling.BILINEAR))
            mask = np.asarray(Image.fromarray(mask.astype(np.uint8)).rotate(
                angle, resample=Image.Resampling.NEAREST, fillcolor=255)).astype(np.int64)
        image = self._photometric(image)
        return image, mask

    def _random_resize(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ratio = float(np.random.uniform(*self.ratio_range))
        target = max(1, int(round(self.scale * ratio)))
        height, width = image.shape[:2]
        scale = target / max(height, width)
        return _resize(image, mask, max(1, int(round(height * scale))), max(1, int(round(width * scale))))

    def _crop(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        crop = self.crop_size
        height, width = mask.shape
        pad_h = max(0, crop - height)
        pad_w = max(0, crop - width)
        if pad_h or pad_w:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)))
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), constant_values=255)
            height, width = mask.shape
        rare = self._rare_pixels(mask) if self.class_aware and np.random.rand() < self.rare_prob else None
        for _ in range(10):
            top, left = self._origin(height, width, crop, rare)
            crop_mask = mask[top : top + crop, left : left + crop]
            valid = crop_mask != 255
            if valid.sum() == 0:
                continue
            counts = np.bincount(crop_mask[valid].ravel(), minlength=1)
            if counts.max() / valid.sum() < self.cat_max_ratio:
                return image[top : top + crop, left : left + crop], crop_mask
        top, left = self._origin(height, width, crop, rare)
        return image[top : top + crop, left : left + crop], mask[top : top + crop, left : left + crop]

    def _rare_pixels(self, mask: np.ndarray) -> np.ndarray | None:
        hits = np.zeros(mask.shape, dtype=bool)
        for class_id in self.rare_classes:
            hits |= mask == class_id
        points = np.argwhere(hits)
        if len(points) == 0:
            return None
        return points

    def _origin(self, height: int, width: int, crop: int, rare: np.ndarray | None) -> tuple[int, int]:
        if rare is not None:
            y, x = rare[np.random.randint(len(rare))]
            top = int(np.clip(y - crop // 2, 0, height - crop))
            left = int(np.clip(x - crop // 2, 0, width - crop))
            return top, left
        top = int(np.random.randint(0, height - crop + 1))
        left = int(np.random.randint(0, width - crop + 1))
        return top, left

    def _photometric(self, image: np.ndarray) -> np.ndarray:
        out = image.astype(np.float32)
        if np.random.rand() < 0.5:
            out += np.random.uniform(-32, 32)
        order = ["contrast", "saturation", "hue"]
        if np.random.rand() < 0.5:
            order = ["saturation", "hue", "contrast"]
        for name in order:
            if np.random.rand() > 0.5:
                continue
            if name == "contrast":
                out *= np.random.uniform(0.5, 1.5)
            elif name == "saturation":
                rgb = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
                out = np.asarray(ImageEnhance.Color(rgb).enhance(float(np.random.uniform(0.5, 1.5))), dtype=np.float32)
            else:
                hsv = np.array(Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).convert("HSV"))
                delta = int(round(np.random.uniform(-18, 18) * 255 / 360))
                hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + delta) % 256).astype(np.uint8)
                out = np.asarray(Image.fromarray(hsv, mode="HSV").convert("RGB"), dtype=np.float32)
        return np.clip(out, 0, 255).astype(np.uint8)
