# Release roadmap and version notes

## Open-source progress

We are refining and organizing the HDRNet codebase, experiment configurations and
trained model weights for progressive public release. The current release includes
modular model components, training and evaluation tools, prediction utilities,
structured experiment records and automated tests.

Planned updates include:

- Dataset-specific HDRNet weights for FloodNet, RescueNet and FWISD.
- Comparison-model integrations with their configurations and checkpoints.
- Dataset split manifests, initialization details and evaluation recipes.
- Runtime measurement scripts and environment records accompanying each release.

Checkpoint download links will be published once packaging and validation are
complete. We will continue maintaining these resources to support research and
applications in disaster scene understanding.

## Version-specific measurements

| Model version | Parameter count | Measurement context |
|:--|--:|:--|
| Manuscript model | 25.83M | Paper-reported experiment |
| v0.2 FloodNet implementation | 30,870,552 (30.87M) | Python 3.12 / PyTorch 2.14.1 CPU |

These measurements refer to different model graphs. Alignment with the manuscript
configuration is part of the checkpoint release workflow. Each released checkpoint
will be accompanied by its model configuration and evaluation record.

The existing configuration notes in [Comparison models](baselines.md) track
sliding-window and optimizer settings during this consolidation.

## v0.2 compatibility

- State heads now belong to individual object categories. RescueNet building
  damage uses monotone cumulative thresholds; clear/blocked roads use a separate
  categorical head. FWISD's floodwater/natural-water factor shares the water
  object category.
- State probabilities are conditioned on image features.
- Boundary supervision consumes raw logits, and ignored pixels do not contribute
  to BCE/Dice. Coarse composition uses soft-label cross-entropy.
- Hierarchical consistency includes object and state factors, with valid-pixel
  normalization. Empty state regions contribute finite zero losses.
- The prior is estimated from training masks, installed before training and saved
  in model checkpoints.
- Metrics retain the original confusion matrix and reject invalid labels. A
  per-class F1 must agree with `200 * IoU / (100 + IoU)` when IoU is in percent;
  the arithmetic mean of F1 is not derived from mean IoU.

Version 0.2 updates the model graph and loss interfaces. Use matching code versions
and checkpoints; the loader validates state-dict compatibility. Earlier versions
remain available through the Git history.

## Sharing results

Attach evaluation JSON and the corresponding `run.json` when reporting results.
Record the model version, configuration and checkpoint hash so others can follow
the same evaluation workflow.
