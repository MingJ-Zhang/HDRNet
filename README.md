# HDRNet + SegFormer-B2 for UAV Disaster Semantic Segmentation

This package implements a MMSegmentation-style project for FloodNet, RescueNet, and FWISD.

Model components:

1. **DRCA**: Disaster Relation-aware Context Aggregation.
2. **HDAD**: Hierarchical Damage-aware Decoding Head.
3. **DSBR**: Damage-sensitive Structural Boundary Refinement.

The backbone is **SegFormer-B2 / MiT-B2** initialized with OpenMMLab pretrained weights.

## Expected directory structure

Place data as follows, or edit `data_root` and `data_prefix` in the config files.

```text
data/
  FloodNet/
    images/train/
    images/test/
    annotations/train/
    annotations/test/
  RescueNet/
    images/train/
    images/test/
    annotations/train/
    annotations/test/
  FWISD/
    images/train/
    images/test/
    annotations/train/
    annotations/test/
```

Mask files should be single-channel indexed PNGs. If your labels use JPG/TIF or different suffixes, change `img_suffix` / `seg_map_suffix` in the dataset classes or config.

## Important label-index check

Before training, verify the exact class id order in your annotation masks.

Files to edit if needed:

```text
mmseg_custom/datasets/disaster_datasets.py
configs/hdrnet_segformer_b2_floodnet.py
configs/hdrnet_segformer_b2_rescuenet.py
configs/hdrnet_segformer_b2_fwisd.py
```

The auxiliary object/damage mappings must match your dataset id order. FWISD mappings are placeholders because different released copies may use different index files.

## Training settings already configured

- Backbone: SegFormer-B2 / MiT-B2 pretrained.
- GPUs: 2 GPUs through `dist_train.sh`.
- Batch size: 8 per GPU, global batch size 16.
- Training length: 80k iterations.
- Validation interval: every 10k iterations.
- Training crop size: 512 x 512.
- Sliding-window inference: crop 512 x 512, stride 384 x 384.
- AMP enabled.
- Optimizer: AdamW.
- Scheduler: linear warmup + poly decay.
- Logs: default MMEngine logs + custom grad norm / LR / CUDA memory logs.

## Install

Inside a working MMSegmentation environment:

```bash
pip install -U openmim
mim install "mmengine>=0.7.0"
mim install "mmcv>=2.0.0"
pip install "mmsegmentation>=1.0.0"
```

Then copy this project into your MMSegmentation root directory, or keep it standalone and make sure MMSegmentation's `tools/` and `demo/` are available in the current directory.

## Train

```bash
bash scripts/train_2gpu.sh configs/hdrnet_segformer_b2_floodnet.py
bash scripts/train_2gpu.sh configs/hdrnet_segformer_b2_rescuenet.py
bash scripts/train_2gpu.sh configs/hdrnet_segformer_b2_fwisd.py
```

Use specific GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_2gpu.sh configs/hdrnet_segformer_b2_floodnet.py
```

## Test with sliding-window inference

```bash
bash scripts/test_slide.sh configs/hdrnet_segformer_b2_floodnet.py work_dirs/hdrnet_segformer_b2_floodnet/best_mIoU_iter_*.pth 2
```

The slide inference setting is inside the model config:

```python
test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(384, 384))
```

## Ablation switches

To ablate modules quickly:

- DRCA: replace `self.drca` in `HDRNetHead` with identity-style projection or disable relation in DRCA.
- DSBR: bypass `self.dsbr` and use `dec` directly.
- HDAD: set loss weights `obj=0`, `damage=0`, `coarse=0`, `boundary=0` and compare with a plain SegFormer head baseline.

For paper-quality ablations, create separate head variants rather than editing the same file.

## v2 paper-aligned implementation

This package includes a paper-aligned HDRNet implementation. Compared with the first draft, v2 adds the mechanisms required by the Methodology document:

1. DRCA now uses an explicit `class_cooccurrence_matrix` and relation-biased category attention, not just channel reweighting.
2. HDAD now uses independent object/damage feature heads and implements `loss_hier` as hierarchical KL consistency loss.
3. DSBR now uses both F1 and F2 low-level features and refines decoder features as `F_r = F_d + F_d * M_b`.

See `docs/CODE_PAPER_ALIGNMENT.md` for the detailed code-paper correspondence.
