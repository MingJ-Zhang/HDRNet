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
  mit.py        MiT-B2 encoder
  drca.py       relation context
  dsbr.py       structural boundary
  hdad.py       hierarchical heads
  losses.py     segmentation, factor, consistency, boundary
  model.py      HDRNet
baselines/      one module per comparison model
configs/        class lists, C_d, and the 80k protocol
```

`baselines/` contains PSPNet, DeepLabV3+, FastFCN, K-Net, Mask2Former, OCRNet, PIDNet-S, SegFormer, SegMAN-T, OverLoCK, Spatial-Mamba, RS3Mamba, HL-SAM-Seg, and DA-SegFormer. Each file names the source paper. DA-SegFormer is the SegFormer graph; its difference is class-aware sampling and an OHEM + Dice loss.

## Use

```bash
pip install torch
```

```python
import torch
from hdrnet import HDRNet

net = HDRNet(num_classes=10, num_objects=8, num_states=2)
out = net(torch.randn(1, 3, 512, 512))
logits = out["semantic"]   # composite label
```

Dataset keys in `configs.datasets` are `floodnet`, `rescuenet`, and `fwisd`. Each entry stores the class names and the boundary set `C_d`. The shared schedule is in `configs.protocol`: AdamW at `6e-5`, weight decay `0.01`, polynomial decay, 80k iterations, crop 512, scale range 0.5–2.0.

```python
from configs.datasets import DATASETS
from hdrnet import HDRNet

spec = DATASETS["rescuenet"]
net = HDRNet(
    num_classes=len(spec["classes"]),
    num_objects=spec["num_objects"],
    ordered=spec["ordered_damage"],
)
```

Loss weights follow the paper: coarse `0.1`, object `0.4`, state `0.6`, hierarchy `0.3`, boundary `0.4`.

## Scope

The modules are a readable realization of the paper. They are not a training framework, and they do not ship pretrained weights. Reported metrics should be cited from the manuscript, not from a rerun of this tree.

Code is released under Apache-2.0. See `LICENSE`.
