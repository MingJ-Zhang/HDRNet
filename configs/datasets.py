"""Class lists, split layouts, and HDAD factor maps.

Paths match the MMSegmentation dataset configs used for the paper tables.
`data_root` is the dataset directory itself, for example ``.../FloodNet``.
"""

from __future__ import annotations

import torch

DATASETS = {
    "floodnet": dict(
        classes=(
            "Background",
            "Building-flooded",
            "Building-non-flooded",
            "Road-flooded",
            "Road-non-flooded",
            "Water",
            "Tree",
            "Vehicle",
            "Pool",
            "Grass",
        ),
        palette=(
            (0, 0, 0),
            (255, 0, 0),
            (180, 120, 120),
            (255, 165, 0),
            (200, 200, 0),
            (0, 0, 255),
            (0, 128, 0),
            (255, 255, 0),
            (0, 255, 255),
            (128, 255, 0),
        ),
        boundary_classes=(1, 3, 5),
        rare_classes=(1, 3),
        num_objects=8,
        num_states=2,
        ordered_damage=False,
        # object id per class: bg, building, road, water, tree, vehicle, pool, grass
        object_index=(0, 1, 1, 2, 2, 3, 4, 5, 6, 7),
        # flooded=1, dry=0; classes without that state are ignored (-1)
        state_index=(-1, 1, 0, 1, 0, -1, -1, -1, -1, -1),
        splits=dict(
            train=dict(
                img="train_only_crop1024/images",
                ann="train_only_crop1024/labels",
                img_suffix=".jpg",
                ann_suffix=".png",
            ),
            test=dict(
                img="test/test-org-img",
                ann="test/test-label-img",
                img_suffix=".jpg",
                ann_suffix="_lab.png",
            ),
        ),
    ),
    "rescuenet": dict(
        classes=(
            "Background",
            "Water",
            "Building_No_Damage",
            "Building_Minor_Damage",
            "Building_Major_Damage",
            "Building_Total_Destruction",
            "Vehicle",
            "Road-Clear",
            "Road-Blocked",
            "Tree",
            "Pool",
        ),
        palette=(
            (0, 0, 0),
            (0, 0, 255),
            (180, 120, 120),
            (220, 180, 120),
            (255, 100, 0),
            (255, 0, 0),
            (255, 255, 0),
            (200, 200, 200),
            (255, 165, 0),
            (0, 128, 0),
            (0, 255, 255),
        ),
        boundary_classes=(3, 4, 5, 8),
        rare_classes=(3, 4, 5),
        num_objects=7,
        num_states=4,
        ordered_damage=True,
        # bg, water, building, vehicle, road, tree, pool
        object_index=(0, 1, 2, 2, 2, 2, 3, 4, 4, 5, 6),
        # building damage is ordered: none, minor, major, total
        state_index=(-1, -1, 0, 1, 2, 3, -1, -1, -1, -1, -1),
        splits=dict(
            train=dict(
                img="train_only_crop1024/images",
                ann="train_only_crop1024/labels",
                img_suffix=".jpg",
                ann_suffix=".png",
            ),
            test=dict(
                img="test/test-org-img",
                ann="test/test-label-img",
                img_suffix=".jpg",
                ann_suffix="_lab.png",
            ),
        ),
    ),
    "fwisd": dict(
        classes=(
            "Background",
            "Natural Water",
            "Tree",
            "Road-Passable",
            "Road-Flooded",
            "Building-Intact",
            "Building-Damaged",
            "Waterfront-Intact",
            "Waterfront-Damaged",
            "Vehicle-Land",
            "Vehicle-Water",
            "Floodwater",
        ),
        palette=(
            (0, 0, 0),
            (0, 0, 255),
            (0, 128, 0),
            (128, 128, 128),
            (255, 165, 0),
            (180, 120, 120),
            (255, 0, 0),
            (0, 200, 200),
            (200, 0, 100),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ),
        boundary_classes=(4, 6, 7, 8),
        rare_classes=(4, 6, 8),
        num_objects=8,
        num_states=2,
        ordered_damage=False,
        # bg, natural water, tree, road, building, waterfront, vehicle, floodwater
        object_index=(0, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7),
        state_index=(-1, -1, -1, 0, 1, 0, 1, 0, 1, 0, 1, -1),
        splits=dict(
            train=dict(img="img_dir/train", ann="ann_dir/train", img_suffix=".png", ann_suffix=".png"),
            test=dict(img="img_dir/val", ann="ann_dir/val", img_suffix=".png", ann_suffix=".png"),
        ),
    ),
}


def factor_matrix(index: tuple[int, ...], n_rows: int) -> torch.Tensor:
    """One-hot columns. A negative index leaves the column empty (ignored)."""
    matrix = torch.zeros(n_rows, len(index))
    for column, row in enumerate(index):
        if row >= 0:
            matrix[row, column] = 1.0
    return matrix
