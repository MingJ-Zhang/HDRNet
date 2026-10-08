"""Class lists and the disaster-relevant boundary set C_d.

Label ids follow the training order used with the public benchmarks.
Confirm them against the dataset metadata before a new run.
"""

DATASETS = {
    "floodnet": dict(
        classes=(
            "background",
            "building-flooded",
            "building-non-flooded",
            "road-flooded",
            "road-non-flooded",
            "water",
            "tree",
            "vehicle",
            "pool",
            "grass",
        ),
        boundary_classes=(1, 3, 5),
        num_objects=8,
        ordered_damage=False,
    ),
    "rescuenet": dict(
        classes=(
            "background",
            "water",
            "building-no-damage",
            "building-minor",
            "building-major",
            "building-total",
            "vehicle",
            "road-clear",
            "road-blocked",
            "tree",
            "pool",
        ),
        boundary_classes=(3, 4, 5, 8),
        num_objects=6,
        ordered_damage=True,
    ),
    "fwisd": dict(
        classes=(
            "background",
            "natural-water",
            "tree",
            "road-passable",
            "road-flooded",
            "building-intact",
            "building-damaged",
            "waterfront-intact",
            "waterfront-damaged",
            "vehicle-land",
            "vehicle-water",
            "floodwater",
        ),
        boundary_classes=(4, 6, 7, 8),
        num_objects=8,
        ordered_damage=False,
    ),
}
