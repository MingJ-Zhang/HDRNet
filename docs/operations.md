# Training and evaluation operations

## Install

Use Python 3.10 or newer. Install a PyTorch build compatible with your CUDA driver
using the [official selector](https://pytorch.org/get-started/locally/), then run
`python -m pip install -e '.[dev]'`. Core runtime dependencies are PyTorch, NumPy
and Pillow. No dataset or model checkpoint is downloaded automatically.

## Dataset contract

Masks must be 2-D integer class-index images. Background is included as class 0;
255 is ignored. A palette-indexed PNG is supported; an RGB annotation must first
be converted explicitly using its dataset palette. Images and masks must have
identical dimensions. Nested image directories are mirrored under the annotation
directory; missing pairs raise an error rather than silently shrinking a split.

```text
FloodNet/ or RescueNet/
├── train_only_crop1024/images/*.jpg
├── train_only_crop1024/labels/*.png
├── test/test-org-img/*.jpg
└── test/test-label-img/*_lab.png

FWISD/
├── img_dir/train/*.png
├── ann_dir/train/*.png
├── img_dir/val/*.png
└── ann_dir/val/*.png
```

Obtain the datasets from their authors and follow their usage terms. This
repository ships class maps and loaders, not the datasets. The FWISD `val`
directory is the configured evaluation split; changing it changes the protocol.

```bash
python -m tools.check_data --dataset floodnet --data-root /data/FloodNet
python -m tools.prepare_prior --dataset floodnet --data-root /data/FloodNet \
  --output outputs/floodnet_prior.json
```

Data validation reads both configured splits and reports image counts and class
pixel counts. Prior estimation reads **training annotations only**. It counts
image-level class presence, computes conditional co-occurrence and zeros the
diagonal. Classes absent from the source split have zero rows. Training estimates
the prior automatically unless `--prior` supplies a compatible JSON matrix.

## Train

```bash
# Single GPU; effective batch size is 8.
python -m tools.train --model hdrnet --dataset floodnet --data-root /data/FloodNet \
  --work-dir work_dirs/floodnet_seed42 --seed 42

# Two GPUs; per-process batch 8, effective batch 16.
torchrun --standalone --nproc_per_node=2 -m tools.train \
  --model hdrnet --dataset floodnet --data-root /data/FloodNet \
  --work-dir work_dirs/floodnet_2gpu --batch-size 8 --seed 42
```

Default optimization is FP32 AdamW, learning rate `6e-5`, weight decay `0.01`,
750 warm-up iterations and polynomial decay to 80,000 iterations. The loader
uses random scaling `[0.5, 2.0]` around a 1024 base, a 512 crop, horizontal and
vertical flips, random rotation and photometric augmentation. Crop retry is
bounded to ten attempts before falling back to a valid-size crop.

The public rotation defaults are probability `0.5` and angles uniformly sampled
from `[-180, 180]` degrees. RGB uses bilinear interpolation and labels nearest
neighbor with ignored borders. These are explicit reference-implementation
defaults; the manuscript does not supply the original rotation distribution.
Inspect `TrainTransform` before claiming an exact match to an archived experiment.

`--amp` enables optional CUDA FP16. `--sync-bn` enables synchronized BatchNorm
for multi-GPU runs. Both change the numerical training setting and are opt-in.
`--deterministic` seeds Python/NumPy/PyTorch and enables deterministic algorithms;
unsupported nondeterministic kernels fail visibly. CPU runs use `--device cpu`.

The native encoder starts from random initialization unless
`--encoder-checkpoint` supplies a compatible tensor state dict. No original
ImageNet or manuscript weights are bundled. MMSegmentation/NVlabs encoder
weights require a validated key conversion; loading an unrelated state dict
strictly fails instead of partially initializing without notice.

## Run artifacts and recovery

Each work directory contains `run.json`, `train.log`, `metrics.jsonl`, `prior.json`
for HDRNet, and atomic `iter_N.pth` checkpoints. Metadata records the configuration,
effective batch size, class names, software environment, Git revision and dirty
state. Checkpoints contain model, optimizer, AMP scaler, per-rank RNG states,
sampler epoch and consumed-batch offset. Reusing an occupied work directory
requires `--resume`; accidental overwrites raise an error.

```bash
python -m tools.train --model hdrnet --dataset floodnet --data-root /data/FloodNet \
  --work-dir work_dirs/floodnet_seed42 --seed 42 \
  --resume work_dirs/floodnet_seed42/iter_4000.pth
```

Resume requires the same total schedule, batch size, worker count, seed,
precision, world size and crop. Use the original `--iters` value, not an additional
iteration count. Keep input files unchanged. Hardware and PyTorch changes can
affect numerical behavior; this interface does not promise cross-platform bitwise
replay. `--workers 0 --deterministic` is the clearest setting for recovery tests.
Architecture changes in v0.2 require retraining; v0.1 head checkpoints are not
silently converted.

## Evaluate and inspect predictions

```bash
python -m tools.test --model hdrnet --dataset floodnet --data-root /data/FloodNet \
  --checkpoint work_dirs/floodnet_seed42/iter_80000.pth \
  --output outputs/floodnet_metrics.json --save-predictions outputs/floodnet_masks

python -m tools.predict --dataset floodnet --image /data/example.jpg \
  --checkpoint work_dirs/floodnet_seed42/iter_80000.pth --output-dir outputs/example
```

Evaluation uses 512 windows with stride 384, averages logits over overlaps and
includes background in the confusion matrix. Classes with zero ground-truth and
zero predicted support are absent from the macro average; predicted-only classes
are included with zero IoU/F1. `paper_mF1` is a compatibility alias for per-class
IoU-to-F1 conversion averaged afterward, not a conversion of mIoU.

JSON evaluation reports include the checkpoint SHA-256, environment, full
confusion matrix, per-class support/IoU/F1 and aggregate scores. Single-image
inference exports `mask.png` (index IDs), `color.png`, `overlay.png` and metadata.

## Measure performance

```bash
python -m tools.benchmark --model hdrnet --dataset floodnet --size 512 \
  --warmup 20 --repeats 100 --output outputs/benchmark.json
```

Timing synchronizes CUDA before and after every measured forward pass. Reports
contain batch latency, throughput, dispersion, actual parameter count, and peak
allocated CUDA memory. CPU memory is reported as null. Measurements exclude
I/O, preprocessing and sliding-window aggregation and do not claim onboard UAV
deployment performance. Report hardware and dtype with every measurement.
