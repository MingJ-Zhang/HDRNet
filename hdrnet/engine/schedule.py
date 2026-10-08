"""Iteration learning-rate schedule: 750-step warmup, then linear poly decay."""

from __future__ import annotations


def learning_rate(step: int, base_lr: float, warmup: int = 750, max_iters: int = 80_000, power: float = 1.0) -> float:
    if step < warmup:
        start = base_lr * 1e-6
        return start + (base_lr - start) * step / max(warmup, 1)
    progress = (step - warmup) / max(max_iters - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    return base_lr * (1.0 - progress) ** power


def set_learning_rate(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr
