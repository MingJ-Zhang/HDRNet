# Security policy

Report a suspected security issue privately to xy_zhang@bjut.edu.cn with a minimal
reproduction and affected commit. Avoid disclosing secrets in public issues.

Use checkpoints from trusted sources. Loading uses PyTorch's `weights_only=True`
and strict model keys, but tensor deserialization still requires resource limits
when the source is untrusted. This project is a research library, not a network
service with an isolation boundary or service-level guarantee.

Datasets and images should come from trusted sources. Keep PyTorch, Pillow and
other dependencies updated and avoid exposing input/output directories publicly.
