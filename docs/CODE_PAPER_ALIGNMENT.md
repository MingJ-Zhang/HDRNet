# HDRNet v2 Code-Paper Alignment Notes

This version re-implements the code according to the Methodology description in `HDRNet_UAV_Disaster_Segmentation_Method_Summary.md`.

## 1. DRCA: Disaster Relation-aware Context Aggregation

The implementation now contains the mechanisms described in the paper:

- Multi-scale branches: point branch, dilated depthwise separable convolution branches, and global context branch.
- Coarse disaster semantic distribution `p_c` predicted by `coarse_cls`.
- Dynamic branch fusion using `Softmax(W_g p_c)`.
- Explicit `class_cooccurrence_matrix` passed from each dataset config.
- Relation-biased category attention:

```text
A = Softmax(QK^T / sqrt(d) + beta R)
```

- Spatial cross-attention injects relation-enhanced category tokens back into the context feature.

Main code locations:

- `CategoryRelationAttention`
- `DRCA`
- dataset configs: `class_cooccurrence_matrix`

## 2. HDAD: Hierarchical Damage-aware Decoding Head

The implementation now matches the paper-level description:

- Independent object feature head: `object_feat_head`.
- Independent damage/status feature head: `damage_feat_head`.
- Object logits from object features.
- Damage logits from damage features.
- Final semantic prediction from `[refined_feature, object_feature, damage_feature]` rather than directly from branch logits.
- Hierarchical KL consistency loss is implemented as `loss_hier`.

The consistency loss projects final semantic probabilities into object and damage/status spaces using:

- `object_class_mapping`
- `damage_class_mapping`

Then it computes:

```text
D_KL(P_object || Project(P_semantic)) + D_KL(P_damage || Project(P_semantic))
```

Main code locations:

- `HDRNetHead.loss()`
- `_project_semantic_prob`
- `_masked_kl`
- `_ordinal_probs_from_logits`

## 3. DSBR: Damage-sensitive Structural Boundary Refinement

The implementation now uses both low-level features:

- `F1`: first encoder stage feature.
- `F2`: second encoder stage feature.

The module now follows the paper formula:

```text
F_r = F_d + F_d * M_b
```

where `M_b` is generated from decoder features, structural response features, and boundary probability. The previous extra additive term `F_struct * gate` has been removed to keep the implementation consistent with the written method.

Main code locations:

- `DSBR.forward(dec_feat, low1_feat, low2_feat)`
- `_boundary_from_seg`

## 4. Config changes

Each dataset config now explicitly enables:

- `coarse_class_mapping`
- `object_class_mapping`
- `damage_class_mapping`
- `class_cooccurrence_matrix`
- `loss_weights['hier'] = 0.3`

This ensures the paper-described mechanisms are actually used during training.

## 5. Training settings retained

The requested training/inference settings are preserved:

- Backbone: SegFormer-B2 / MiT-B2 pretrained weight.
- GPUs: 2 GPUs via `scripts/train_2gpu.sh`.
- Per-GPU batch size: 8.
- Total iterations: 80k.
- Validation interval: 10k.
- Training crop: 512 x 512.
- Sliding-window inference: crop 512 x 512, stride 384 x 384.
- Detailed logging hook: LR, gradient norm, CUDA memory.
