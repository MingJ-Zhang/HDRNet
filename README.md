# HDRNet: Hierarchical Damage-aware Relation Refinement Network for Post-Disaster UAV Image Segmentation

This repository provides the research code for **HDRNet**, a hierarchical
damage-aware relation refinement network for semantic segmentation of
post-disaster UAV imagery.

HDRNet is built on a SegFormer-B2/MiT-B2 encoder and targets three central
challenges in disaster-scene parsing:

- strong contextual dependencies among flooded areas, buildings, roads,
  vegetation, vehicles, and infrastructure;
- fine-grained and hierarchically organized disaster labels, such as
  flooded/non-flooded objects and multi-level building destruction;
- fragmented or incomplete object boundaries caused by inundation, occlusion,
  shadows, and damaged structures.

The model integrates three task-specific modules:

- **DRCA: Disaster Relation-aware Context Aggregation** for relation-aware
  multi-scale context modeling.
- **HDAD: Hierarchical Damage-aware Decoding** for object-level recognition and
  disaster-state estimation.
- **DSBR: Damage-sensitive Structural Boundary Refinement** for fragmented
  disaster-boundary refinement.

Experiments in the paper are conducted on **FloodNet**, **RescueNet**, and
**FWISD**. HDRNet consistently improves over representative CNN-, Transformer-,
state-space-, and large-kernel segmentation baselines, with the clearest gains
on disaster-critical categories such as flooded roads, damaged buildings,
blocked roads, and damaged waterfront infrastructure.

## Paper Abstract

Semantic segmentation of post-disaster UAV imagery is a key enabling technology
for rapid emergency assessment, supporting flood monitoring, road accessibility
analysis, building damage evaluation, and waterfront infrastructure inspection.
Compared with ordinary scenes, post-disaster UAV imagery presents strong
contextual dependencies among disaster-related objects, fine-grained
hierarchical damage categories, and fragmented object boundaries. HDRNet
addresses these issues with a SegFormer-based framework that combines
relation-aware context aggregation, hierarchical damage-aware decoding, and
damage-sensitive structural boundary refinement.

## Main Contributions

1. We propose HDRNet, a hierarchical damage-aware relation refinement network
   for UAV-based post-disaster semantic segmentation that jointly addresses
   contextual ambiguity, fine-grained damage confusion, and boundary
   fragmentation.
2. We design DRCA to adaptively aggregate multi-scale context conditioned on
   coarse disaster semantics and model category co-occurrence relationships
   among disaster-related objects.
3. We introduce HDAD to decompose disaster-scene segmentation into object-level
   recognition and disaster-state estimation, supporting flooded/non-flooded,
   damaged/intact, and multi-level destruction categories across datasets.
4. We develop DSBR to refine the boundaries of submerged roads, damaged
   buildings, blocked roads, and waterfront infrastructure using automatically
   generated boundary supervision and learnable structural responses.

## Repository Structure

```text
.
├── configs/
│   ├── hdrnet_segformer_b2_common.py      # shared model and training settings
│   ├── hdrnet_segformer_b2_floodnet.py    # FloodNet experiment
│   ├── hdrnet_segformer_b2_rescuenet.py   # RescueNet experiment
│   └── hdrnet_segformer_b2_fwisd.py       # FWISD experiment
├── docs/
│   └── MODULE_DESIGN.md                   # module design notes
├── mmseg_custom/
│   ├── datasets/
│   │   └── disaster_datasets.py           # dataset definitions and palettes
│   └── models/
│       ├── hdrnet_head.py                 # DRCA, HDAD, DSBR, HDRNetHead
│       └── logging_hooks.py               # detailed training log hook
├── scripts/
│   ├── train_2gpu.sh                      # convenience training launcher
│   ├── test_slide.sh                      # sliding-window evaluation launcher
│   └── single_infer.sh                    # single-image inference launcher
├── tools/
│   ├── train.py                           # MMEngine training entry point
│   ├── test.py                            # MMEngine testing entry point
│   ├── image_demo.py                      # MMSegInferencer image demo
│   ├── print_flops.py                     # FLOPs and parameter counter
│   ├── dist_train.sh                      # distributed training launcher
│   └── dist_test.sh                       # distributed testing launcher
├── requirements.txt
└── paper_draft.tex
```

## Environment

The code follows the OpenMMLab 2.x stack and MMSegmentation 1.x.

Recommended environment:

- Python >= 3.8
- PyTorch >= 1.10 with a CUDA version matching your GPU server
- MMEngine >= 0.7
- MMCV >= 2.0
- MMSegmentation >= 1.0

Install dependencies:

```bash
conda create -n hdrnet python=3.10 -y
conda activate hdrnet

# Install PyTorch according to your CUDA version:
# https://pytorch.org/get-started/locally/

pip install -U openmim
mim install "mmengine>=0.7.0"
mim install "mmcv>=2.0.0"
pip install "mmsegmentation>=1.0.0"
pip install -r requirements.txt
```

The helper scripts set `PYTHONPATH` automatically. If you run tools manually,
set it from the repository root:

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"
```

## Dataset Preparation

The paper evaluates HDRNet on three public post-disaster UAV benchmarks:

- **FloodNet**: post-Hurricane Harvey flood-scene segmentation with flooded and
  non-flooded buildings/roads, water, trees, vehicles, pools, grass, and
  background.
- **RescueNet**: natural-disaster damage-assessment segmentation with
  fine-grained building damage levels and road conditions.
- **FWISD**: flood and waterfront infrastructure damage segmentation with intact
  and damaged infrastructure-related classes.

The configs expect datasets under `data/` by default:

```text
data/
├── FloodNet/
│   ├── train_only_crop1024/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── test-org-img/
│       └── test-label-img/
├── RescueNet/
│   ├── images/
│   │   ├── train/
│   │   └── test/
│   └── annotations/
│       ├── train/
│       └── test/
└── FWISD/
    ├── images/
    │   ├── train/
    │   └── test/
    └── annotations/
        ├── train/
        └── test/
```

Annotation masks should be single-channel indexed images. If your downloaded
dataset uses different folder names or suffixes, update `data_root`,
`data_prefix`, `img_suffix`, and `seg_map_suffix` in the corresponding config.

Before training, verify that the class-id order in your masks matches:

- `mmseg_custom/datasets/disaster_datasets.py`
- `configs/hdrnet_segformer_b2_floodnet.py`
- `configs/hdrnet_segformer_b2_rescuenet.py`
- `configs/hdrnet_segformer_b2_fwisd.py`

This is especially important for `object_class_mapping`,
`damage_class_mapping`, and `coarse_class_mapping`, because HDAD derives its
object-level and disaster-state targets from these lists. The FWISD mapping is
provided as a placeholder and should be checked against the label-index file of
your dataset copy.

## Method Overview

HDRNet uses SegFormer-B2 as the encoder and extracts four multi-scale features.
The highest-level feature is enhanced by DRCA, the multi-scale features are
fused by a lightweight decoder, DSBR refines local structures using low-level
features, and HDAD predicts the final semantic mask together with object and
disaster-state auxiliary predictions.

### DRCA

DRCA models disaster-scene relations through:

- parallel multi-scale context branches with pointwise, dilated depthwise, and
  global-context paths;
- coarse disaster-semantic gating that adaptively weights context branches for
  different scene compositions;
- relation modeling that injects disaster-category dependencies into the
  high-level context representation.

### HDAD

HDAD treats disaster labels as composite object-state categories rather than a
flat class list. It predicts:

- object categories, such as building, road, water, vegetation, vehicle, and
  infrastructure;
- disaster states, such as flooded/non-flooded, damaged/intact, or ordered
  damage levels;
- the final semantic segmentation map.

For RescueNet-style ordered damage levels, the implementation supports ordinal
damage prediction.

### DSBR

DSBR targets blurred and fragmented structures in post-disaster scenes. It:

- derives boundary supervision automatically from segmentation masks using a
  morphological-gradient-style target;
- uses learnable directional structural responses initialized with line-like
  kernels;
- injects boundary and structure attention into the decoded feature.

No extra boundary annotation is required.

## Training Settings

The default paper setting uses:

- Backbone: SegFormer-B2 / MiT-B2.
- Crop size: `512 x 512`.
- Training augmentation: random resize, random crop, horizontal/vertical flip,
  random rotation, and photometric distortion.
- Optimizer: AdamW.
- Initial learning rate: `6e-5`.
- Weight decay: `0.01`.
- Scheduler: linear warmup followed by polynomial decay.
- Training length: `80,000` iterations.
- GPUs: 2 GPUs.
- Batch size: 8 per GPU, 16 total.
- Evaluation interval: every `10,000` iterations.
- Inference: sliding-window prediction with crop `512 x 512` and stride
  `384 x 384`.

## Training

Train on two GPUs:

```bash
bash scripts/train_2gpu.sh configs/hdrnet_segformer_b2_floodnet.py
bash scripts/train_2gpu.sh configs/hdrnet_segformer_b2_rescuenet.py
bash scripts/train_2gpu.sh configs/hdrnet_segformer_b2_fwisd.py
```

Select GPUs explicitly:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_2gpu.sh configs/hdrnet_segformer_b2_floodnet.py
```

Use a different number of GPUs:

```bash
GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_2gpu.sh configs/hdrnet_segformer_b2_floodnet.py
```

Override config options from the command line:

```bash
bash tools/dist_train.sh configs/hdrnet_segformer_b2_floodnet.py 2 \
  --cfg-options train_cfg.max_iters=12000 train_cfg.val_interval=2000
```

Resume training:

```bash
bash tools/dist_train.sh configs/hdrnet_segformer_b2_floodnet.py 2 --resume
```

## Evaluation

Run sliding-window evaluation:

```bash
bash scripts/test_slide.sh \
  configs/hdrnet_segformer_b2_floodnet.py \
  path/to/checkpoint.pth \
  2
```

The evaluator reports `mIoU`, `mDice`, and `mFscore` through MMSegmentation's
`IoUMetric`.

## Single-Image Inference

Run inference on one image:

```bash
bash scripts/single_infer.sh \
  path/to/image.jpg \
  configs/hdrnet_segformer_b2_floodnet.py \
  path/to/checkpoint.pth \
  outputs
```

## FLOPs and Parameters

Measure complexity for the configured crop size:

```bash
python tools/print_flops.py configs/hdrnet_segformer_b2_floodnet.py
```

Use a custom input shape:

```bash
python tools/print_flops.py configs/hdrnet_segformer_b2_floodnet.py --shape 512 512
```

Print the per-module table:

```bash
python tools/print_flops.py configs/hdrnet_segformer_b2_floodnet.py --show-table
```

## Paper Results

Main mIoU results reported in the paper:

| Dataset | SegFormer-B2 | HDRNet | Gain |
| --- | ---: | ---: | ---: |
| FloodNet | 71.10 | 73.04 | +1.94 |
| RescueNet | 63.70 | 65.59 | +1.89 |
| FWISD | 60.96 | 62.94 | +1.98 |

Representative disaster-critical category improvements:

- FloodNet road-flooded IoU: `53.28 -> 57.28`.
- FloodNet building-flooded IoU: `72.58 -> 75.58`.
- RescueNet road-blocked IoU: `30.11 -> 35.11`.
- RescueNet building-minor-damage IoU: `51.72 -> 55.22`.
- FWISD road-flooded IoU: `28.82 -> 33.82`.
- FWISD damaged-waterfront IoU: `19.36 -> 25.36`.

Complexity reported in the paper:

| Model | Params (M) | FLOPs (G) | FloodNet mIoU |
| --- | ---: | ---: | ---: |
| SegFormer-B2 | 24.73 | 148 | 71.10 |
| HDRNet | 27.89 | 328 | 73.04 |

## Ablation Studies

The paper starts from a SegFormer-B2 baseline and progressively introduces
DRCA, HDAD, and DSBR. The full HDRNet achieves the best mIoU on all three
datasets:

| DRCA | HDAD | DSBR | FloodNet | RescueNet | FWISD |
| --- | --- | --- | ---: | ---: | ---: |
| no | no | no | 71.10 | 63.70 | 60.96 |
| yes | no | no | 71.83 | 64.36 | 61.74 |
| no | yes | no | 72.05 | 64.71 | 61.55 |
| no | no | yes | 71.68 | 64.22 | 61.92 |
| yes | yes | no | 72.51 | 65.05 | 62.21 |
| yes | no | yes | 72.39 | 64.83 | 62.55 |
| no | yes | yes | 72.64 | 65.18 | 62.43 |
| yes | yes | yes | 73.04 | 65.59 | 62.94 |

For implementation-side ablations, create separate configs or head variants.
The auxiliary loss weights can be adjusted from the command line:

```bash
bash tools/dist_train.sh configs/hdrnet_segformer_b2_floodnet.py 2 \
  --cfg-options model.decode_head.loss_weights.obj=0.0 \
                model.decode_head.loss_weights.damage=0.0 \
                model.decode_head.loss_weights.coarse=0.0 \
                model.decode_head.loss_weights.boundary=0.0
```

## Outputs and Checkpoints

The following generated files are intentionally ignored by Git:

- `work_dirs/`
- `logs/`
- `outputs/`
- `.dist_test/`
- `*.pth`, `*.pt`, `*.ckpt`
- TensorBoard event files
- Python cache files

If you want to release trained weights, upload them as external artifacts
through GitHub Releases, Zenodo, Google Drive, or another storage service, and
link them here instead of committing large checkpoint files.

## Citation

If this work is useful for your research, please cite the corresponding paper.
Replace the publication fields after the paper is accepted.

```bibtex
@article{zhang2026hdrnet,
  title   = {HDRNet: Hierarchical Damage-aware Relation Refinement Network for Post-Disaster UAV Image Segmentation},
  author  = {Zhang, Mingjie and Chen, Kai and Zhou, Yongxu and Shan, Weifeng},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  year    = {2026},
  note    = {Under review}
}
```

## License

Please add a license file before public release if the repository will be shared
outside your group.
