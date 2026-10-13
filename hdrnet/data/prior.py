"""Estimate image-level co-occurrence using training masks only."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def estimate_prior(mask_paths, num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float64)
    joint = torch.zeros(num_classes, num_classes, dtype=torch.float64)
    seen = 0
    for path in mask_paths:
        with Image.open(Path(path)) as image:
            mask = np.array(image)
        if mask.ndim != 2:
            raise ValueError(f"expected an indexed mask: {path}")
        ids = np.unique(mask[mask != 255])
        if (ids < 0).any() or (ids >= num_classes).any():
            raise ValueError(f"mask has unknown class ids: {path}")
        present = torch.zeros(num_classes, dtype=torch.float64)
        present[torch.as_tensor(ids.astype(np.int64))] = 1
        counts += present
        joint += present[:, None] * present[None, :]
        seen += 1
    if not seen:
        raise ValueError("cannot estimate a prior from an empty training split")
    prior = (joint / counts[:, None].clamp_min(1)).float()
    prior.fill_diagonal_(0)
    return prior
