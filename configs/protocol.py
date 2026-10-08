"""Shared optimization and inference protocol for the three benchmarks."""

PROTOCOL = dict(
    crop_size=512,
    scale_range=(0.5, 2.0),
    cat_max_ratio=0.75,
    batch_size_per_gpu=8,
    num_gpus=2,
    optimizer="AdamW",
    lr=6e-5,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    schedule="poly",
    power=1.0,
    max_iters=80_000,
    slide_crop=512,
    slide_stride=384,
    loss_weights=dict(
        coarse=0.1,
        obj=0.4,
        state=0.6,
        hier=0.3,
        boundary=0.4,
    ),
)
