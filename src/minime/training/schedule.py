"""Learning-rate schedule: linear warmup into a cosine decay.

Written as a pure function of ``step`` rather than a stateful scheduler
object.  Resuming then needs no scheduler state at all -- the LR is recomputed
from the step number, so a resumed run cannot silently continue on a different
part of the curve.
"""

from __future__ import annotations

import math
from typing import Sequence


def lr_at_step(
    step: int,
    base_lr: float,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    min_lr = base_lr * min_lr_ratio

    if warmup_steps > 0 and step < warmup_steps:
        # step+1 so the very first step is not a no-op at lr=0.
        return base_lr * (step + 1) / warmup_steps

    decay_steps = max(1, max_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / decay_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (base_lr - min_lr)


def apply_schedule(
    optimizer,
    peak_lrs: Sequence[float],
    step: int,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> list[float]:
    """Put every param group on the same curve, each scaled to its own peak.

    The Muon arm has two peaks (Muon's for the hidden weights, AdamW's for
    the rest); sharing the curve's shape keeps the schedule a controlled
    variable in E4.  ``peak_lrs`` is one entry per group, in group order.
    """
    groups = optimizer.param_groups
    if len(peak_lrs) != len(groups):
        raise ValueError(f"{len(peak_lrs)} peak LRs for {len(groups)} param groups")
    lrs = [lr_at_step(step, peak, warmup_steps, max_steps, min_lr_ratio) for peak in peak_lrs]
    for group, lr in zip(groups, lrs):
        group["lr"] = lr
    return lrs
