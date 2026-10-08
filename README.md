# HDRNet

**Hierarchical Damage-Aware Relation Refinement for post-disaster UAV semantic segmentation.**

HDRNet keeps a SegFormer MiT-B2 encoder and adds three modules that the scene actually needs after a disaster: category co-occurrence at the coarsest scale, damage boundaries at the finest scale, and a hierarchical reading of object identity versus disaster state.

<p align="center">

| FloodNet | RescueNet | FWISD | Params | FLOPs | FPS |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **73.04** | **65.59** | **62.94** | 25.83M | 32.36G | 67.2 |

</p>

Numbers are mean IoU. Against SegFormer-B2, trained under the same schedule, the gains are **+1.94 / +1.89 / +1.98**. The extra capacity is 1.10M parameters and 7.07G FLOPs.

This repository is the method release. The table above comes from the paper experiments. It is not produced by a single command in this directory.

## Architecture

```text
image
  └─ MiT-B2  →  F1 F2 F3 F4
                    │
                    ├─ DRCA   relation      on F4, gated by image-level composition
                    ├─ MLP    decode        four scales → 256-d
                    ├─ DSBR   structure     directional kernels on F1/F2, gated into the decoder
                    └─ HDAD   hierarchy     object, disaster state, fused semantic head
```

**DRCA.** Five branches read F4: a 1×1 projection, depthwise 3×3 convolutions at dilations 1, 6 and 12, and a global average-pool branch. An image-level composition π gates the branches. Category tokens then attend to each other with a frozen co-occurrence prior, `R_ij = N(c_i, c_j) / N(c_i)` and `R_ii = 0`, and every pixel attends to those tokens.

**DSBR.** The boundary target is the 3×3 morphological gradient of the disaster set `C_d`, merged by logical OR. Four groups of 3×3 convolutions are initialized as horizontal, vertical, and two diagonal finite differences, then trained. Their response gates the decoder feature. The boundary loss is BCE plus Dice.

**HDAD.** One head predicts the object category. A second head predicts the disaster state of that object. RescueNet building damage is the only ordered state (cumulative thresholds). Flooded versus non-flooded, damaged versus intact, and clear versus blocked stay categorical. A fused head predicts the original composite label. Hierarchical consistency is a KL divergence between each factor head and the corresponding projection of the semantic distribution.

## Results

All comparison models use a 512 crop, 80k iterations, and sliding-window inference at 512 with stride 384. SegFormer, DA-SegFormer, and HDRNet share MiT-B2.

| Method | FloodNet | RescueNet | FWISD |
| --- | ---: | ---: | ---: |
| SegFormer-B2 | 71.10 | 63.70 | 60.96 |
| **HDRNet** | **73.04** | **65.59** | **62.94** |

Component ablation on the SegFormer-B2 host:

| Variant | FloodNet | RescueNet | FWISD |
| --- | ---: | ---: | ---: |
| SegFormer-B2 | 71.10 | 63.70 | 60.96 |
| + DRCA | 71.83 | 64.36 | 61.74 |
| + DSBR | 71.68 | 64.22 | 61.92 |
| + HDAD | 72.05 | 64.71 | 61.55 |
| HDRNet | 73.04 | 65.59 | 62.94 |

Efficiency at 512×512, batch 1, FP32:

| Model | Params | FLOPs | Latency | FPS |
| --- | ---: | ---: | ---: | ---: |
| SegFormer-B2 | 24.73M | 25.29G | 13.59 ms | 73.6 |
| HDRNet | 25.83M | 32.36G | 14.88 ms | 67.2 |

## Layout

```text
hdrnet/
  mit.py drca.py dsbr.py hdad.py model.py losses.py
  data/             FloodNet, RescueNet, FWISD and the train pipeline
  engine/           poly schedule, CE+Dice / OHEM, slide inference, mIoU
  builder.py        model and loss factory
baselines/          one trainable module per comparison row
configs/            class lists, C_d, factor maps, 80k protocol
tools/train.py
tools/test.py
docs/baselines.md   paper, official repository, and the config that was trained
```

## Where the baselines come from

The numbers in the paper were trained in MMSegmentation, not by pasting a few custom layers into an unrelated framework. Each file under `baselines/` names three things: the paper, the official repository, and the config that produced the table row. The full list is in [`docs/baselines.md`](docs/baselines.md).

DA-SegFormer is the SegFormer-B2 graph. Its config only changes the crop and the loss: class-aware sampling (`rare_prob=0.5`) and OHEM + Dice. `tools/train.py --model da-segformer` applies both.

SegMAN, OverLoCK, Spatial-Mamba, RS3Mamba and HL-SAM-Seg depend on official CUDA kernels or a SAM checkpoint. Those files keep the architecture that was compared and point at the upstream repository. They are not drop-in copies of those kernels.

## Data

Put each dataset in its own directory. Masks are single-channel class indices. Background is class 0, not an ignore label. Ignore index is 255.

```text
FloodNet/
  train_only_crop1024/images/*.jpg
  train_only_crop1024/labels/*.png
  test/test-org-img/*.jpg
  test/test-label-img/*_lab.png
RescueNet/
  train_only_crop1024/images/*.jpg
  train_only_crop1024/labels/*.png
  test/test-org-img/*.jpg
  test/test-label-img/*_lab.png
FWISD/
  img_dir/{train,val}/*.png
  ann_dir/{train,val}/*.png
```

Training resizes around a 1024 base with a scale in `[0.5, 2.0]`, crops 512, rejects a crop when one class covers 75% or more, then flips and applies photometric distortion. Normalization is ImageNet mean and standard deviation.

## Train and test

```bash
pip install -r requirements.txt

python tools/train.py --model hdrnet --dataset floodnet --data-root D:/data/FloodNet
python tools/train.py --model segformer --dataset rescuenet --data-root D:/data/RescueNet
python tools/train.py --model da-segformer --dataset fwisd --data-root D:/data/FWISD

python tools/test.py --model hdrnet --dataset floodnet --data-root D:/data/FloodNet ^
    --checkpoint work_dirs/hdrnet_floodnet/iter_80000.pth
```

The schedule is AdamW at `6e-5`, weight decay `0.01`, 750-step warmup, then polynomial decay to 80k. Checkpoints are written every 4000 iterations. Testing is a sliding window of 512 with stride 384. HDRNet loss weights are coarse `0.1`, object `0.4`, state `0.6`, hierarchy `0.3`, boundary `0.4`. Other models use cross-entropy plus Dice with weight 3, except DA-SegFormer, which uses OHEM plus Dice.

The table at the top of this file comes from the paper experiments. A fresh run of this tree is a reimplementation of that protocol, not a bitwise reproduction of the MMSegmentation checkpoints.

Code is released under Apache-2.0. See `LICENSE`.
