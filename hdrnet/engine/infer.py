"""Sliding-window inference at the crop used in training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _pad_to(image: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int, int]:
    height, width = image.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h or pad_w:
        image = F.pad(image, (0, pad_w, 0, pad_h))
    return image, pad_h, pad_w


def logits_of(output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(output, dict):
        return output["semantic"]
    return output


@torch.no_grad()
def slide_inference(
    model: torch.nn.Module,
    image: torch.Tensor,
    num_classes: int,
    crop_size: int = 512,
    stride: int = 384,
) -> torch.Tensor:
    """Average logits over 512 windows. ``image`` is ``1 x 3 x H x W``."""
    model.eval()
    _, _, height, width = image.shape
    if height <= crop_size and width <= crop_size:
        padded, _, _ = _pad_to(image, 32)
        logits = logits_of(model(padded))
        logits = F.interpolate(logits, size=padded.shape[-2:], mode="bilinear", align_corners=False)
        return logits[:, :, :height, :width]

    device = image.device
    accumulator = torch.zeros(1, num_classes, height, width, device=device)
    counts = torch.zeros(1, 1, height, width, device=device)
    ys = list(range(0, max(height - crop_size, 0) + 1, stride))
    xs = list(range(0, max(width - crop_size, 0) + 1, stride))
    if not ys or ys[-1] != height - crop_size:
        ys.append(max(height - crop_size, 0))
    if not xs or xs[-1] != width - crop_size:
        xs.append(max(width - crop_size, 0))
    for top in ys:
        for left in xs:
            patch = image[:, :, top : top + crop_size, left : left + crop_size]
            patch_h, patch_w = patch.shape[-2:]
            padded, _, _ = _pad_to(patch, 32)
            logits = logits_of(model(padded))
            logits = F.interpolate(logits, size=padded.shape[-2:], mode="bilinear", align_corners=False)
            logits = logits[:, :, :patch_h, :patch_w]
            accumulator[:, :, top : top + patch_h, left : left + patch_w] += logits
            counts[:, :, top : top + patch_h, left : left + patch_w] += 1
    return accumulator / counts.clamp_min(1)
