# Comparison models

## Integration and release plan

We are organizing comparison-model implementations, experiment configurations and
trained weights under a shared training and evaluation interface. Model-specific
resources will be released progressively after integration and validation.

Full architecture integration for SegMAN, OverLoCK, Spatial-Mamba, RS3Mamba and
HL-SAM-Seg is in progress. The table below provides upstream references and
configuration identifiers from the existing experiment notes; corresponding
release packages and checkpoint links will be added as they become available.

The shared training recipe uses 80,000 iterations, 512-pixel crops, AdamW with
learning rate `6e-5`, polynomial decay and an effective batch size of 16.
See [Operations](operations.md) for the executable workflow and the
[release roadmap](reproducibility.md) for planned artifacts.

| Model | Backbone in the table | Paper | Official code | Config trained for the table |
| --- | --- | --- | --- | --- |
| PSPNet | ResNet-50, stride 8 | [CVPR 2017](https://arxiv.org/abs/1612.01105) | [hszhao/PSPNet](https://github.com/hszhao/PSPNet) | `pspnet_r50-d8_2xb8-80k_*-crop1024.py` |
| DeepLabV3+ | ResNet-50, stride 8 | [ECCV 2018](https://arxiv.org/abs/1802.02611) | [tensorflow/models deeplab](https://github.com/tensorflow/models/tree/master/research/deeplab) | `deeplabv3plus_r50-d8_4xb4-80k_*-crop1024_v3.py` |
| FastFCN | ResNet-50, stride 32 + JPU | [arXiv 2019](https://arxiv.org/abs/1903.11816) | [wuhuikai/FastFCN](https://github.com/wuhuikai/FastFCN) | `fastfcn_r50-d32_jpu_psp_2xb8-80k_*-crop1024.py` |
| K-Net | ResNet-50, stride 8, 3 updates | [NeurIPS 2021](https://arxiv.org/abs/2106.14855) | [ZwwWayne/K-Net](https://github.com/ZwwWayne/K-Net) | `knet-s3_r50-d8_pspnet_2xb8-adamw-80k_*-crop1024.py` |
| Mask2Former | ResNet-50 | [CVPR 2022](https://arxiv.org/abs/2112.01527) | [facebookresearch/Mask2Former](https://github.com/facebookresearch/Mask2Former) | `mask2former_r50_2xb8-80k_*-crop1024.py` |
| OCRNet | HRNet-W48 | [ECCV 2020](https://arxiv.org/abs/1909.11065) | [HRNet/HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation) | `ocrnet_hr48_2xb8-80k_*-crop1024.py` |
| PIDNet-S | PIDNet-S | [CVPR 2023](https://arxiv.org/abs/2206.02066) | [XuJiacong/PIDNet](https://github.com/XuJiacong/PIDNet) | `pidnet-s_2xb8-80k_*-crop1024.py` |
| SegFormer | MiT-B2 | [NeurIPS 2021](https://arxiv.org/abs/2105.15203) | [NVlabs/SegFormer](https://github.com/NVlabs/SegFormer) | `segformer_mit-b2_2xb8-80k_*-crop1024.py` |
| DA-SegFormer | MiT-B2, class-aware crop, OHEM+Dice | same graph as SegFormer | [NVlabs/SegFormer](https://github.com/NVlabs/SegFormer) | `da-segformer_mit-b2_2xb8-80k_*.py` |
| SegMAN-T | SegMAN-Tiny | [CVPR 2025](https://arxiv.org/abs/2412.11890) | [yunxiangfu2001/SegMAN](https://github.com/yunxiangfu2001/SegMAN) | same 512 / 80k schedule |
| OverLoCK-T | OverLoCK-T + UPerNet | [CVPR 2025](https://arxiv.org/abs/2502.20087) | [LMMMEng/OverLoCK](https://github.com/LMMMEng/OverLoCK) | `upernet_overlock-t_2xb8-80k_*-crop1024.py` |
| Spatial-Mamba | Spatial-Mamba | [ICLR 2025](https://arxiv.org/abs/2410.15091) | [EdwardChasel/Spatial-Mamba](https://github.com/EdwardChasel/Spatial-Mamba) | same 512 / 80k schedule |
| RS3Mamba | CNN + Mamba | [GRSL 2024](https://arxiv.org/abs/2404.02457) | [sstary/SSRS](https://github.com/sstary/SSRS) | same 512 / 80k schedule |
| HL-SAM-Seg | SAM image encoder + hierarchical head | Segment Anything | [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) | same 512 / 80k schedule |

`*` is `floodnet`, `rescuenet`, or `fwisd`. FloodNet and RescueNet train on `train_only_crop1024` and test on the original tiles. FWISD trains on `img_dir/train` and evaluates on `img_dir/val`.

### Configuration alignment

The configuration archive is being consolidated with the public training tools.
The existing notes record two settings to align during checkpoint packaging:

- The manuscript states a sliding-window stride of 384. `tools/test.py` defaults to 384. The MMSegmentation configs used stride 341.
- SegFormer's MMSegmentation config sets the decode-head learning rate to 10 times the backbone. `tools/train.py` uses one AdamW learning rate of `6e-5` for every parameter, which is the rate written in the paper.
