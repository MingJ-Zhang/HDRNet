# HDRNet Module Design Notes

## 1. DRCA: Disaster Relation-aware Context Aggregation

Position: highest-level feature after SegFormer-B2 Stage 4 projection.

Purpose: replace a plain ASPP/PPM-style context block with a disaster-aware context aggregator.

Main operations:

- point branch: 1x1 conv;
- local/mid/large branches: dilated depth-wise separable convolution;
- global branch: GAP + MLP;
- coarse disaster semantic gate: image-level coarse class distribution generates branch weights;
- relation projection: coarse semantic distribution modulates channels as a lightweight relation prior.

Auxiliary loss: image-level coarse CE generated from the dominant object-level label.

## 2. HDAD: Hierarchical Damage-aware Decoding Head

Position: final decode head after feature fusion/refinement.

Purpose: avoid flat softmax-only prediction by adding object-level and damage/status-level supervision.

Branches:

- final semantic branch: normal segmentation logits;
- object branch: background/building/road/water/vegetation/vehicle/etc.;
- damage branch: flooded/non-flooded, damaged/intact, or ordinal damage levels.

For RescueNet-style multi-level damage, ordinal regression is used:

```text
P(y >= k | f) = sigmoid(w_k^T f + b_k), k=1,...,K-1
```

Auxiliary losses:

- object CE;
- damage CE or ordinal BCE;
- final segmentation CE.

## 3. DSBR: Damage-sensitive Structural Boundary Refinement

Position: decoder feature refinement stage, using low-level SegFormer features.

Purpose: refine boundaries of submerged roads, damaged buildings, blocked roads, and waterfront infrastructure.

Main operations:

- low-level feature projection;
- learnable directional structural convolution initialized by line kernels;
- auxiliary boundary head;
- boundary/structure attention injected into decoder features.

Boundary labels are generated automatically from segmentation masks by morphological gradient. No extra annotation is required.
