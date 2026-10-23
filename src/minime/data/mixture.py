"""Batch sampling over a weighted mixture of token shards.

Experiment E2 varies nothing but the weights in this sampler, so two
properties matter more than speed:

* **Reproducibility.**  The sampler owns an explicit RNG whose state is
  checkpointed, so a resumed run draws the same sequence of windows it would
  have drawn had it never stopped.
* **Honest proportions.**  ``realised_mixture`` reports what the model actually
  saw, which is what goes in the report -- not the requested weights.

Validation deliberately does *not* use this sampler.  Random validation
batches add sampling noise to exactly the number the experiments compare, so
``make_val_batches`` walks a fixed, evenly spaced set of windows instead.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch

from .binidx import TokenShard


@dataclass
class MixtureSampler:
    shards: dict[str, TokenShard]
    weights: dict[str, float]
    context_length: int
    batch_size: int
    seed: int = 1234

    def __post_init__(self) -> None:
        if not self.shards:
            raise ValueError("mixture needs at least one shard")
        missing = set(self.weights) - set(self.shards)
        if missing:
            raise KeyError(f"mixture references unknown shard(s): {sorted(missing)}")

        self._names = sorted(self.weights)
        probs = np.array([self.weights[n] for n in self._names], dtype=np.float64)
        if probs.min() < 0:
            raise ValueError("mixture weights must be non-negative")
        if probs.sum() <= 0:
            raise ValueError("mixture weights must sum to a positive number")
        self._probs = probs / probs.sum()

        need = self.context_length + 1
        for name in self._names:
            shard = self.shards[name]
            if shard.n_tokens < need:
                raise ValueError(
                    f"shard '{name}' has {shard.n_tokens} tokens, which is fewer than the "
                    f"{need} needed for one context window"
                )

        self._rng = np.random.default_rng(self.seed)
        self._drawn = Counter({name: 0 for name in self._names})
        self._steps = 0

    # ----------------------------------------------------------- sampling

    def batch(self, device: torch.device | str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """One training batch as ``(inputs, targets)``, targets shifted by one."""
        picks = self._rng.choice(len(self._names), size=self.batch_size, p=self._probs)
        xs = np.empty((self.batch_size, self.context_length), dtype=np.int64)
        ys = np.empty((self.batch_size, self.context_length), dtype=np.int64)

        for row, pick in enumerate(picks):
            name = self._names[pick]
            shard = self.shards[name]
            high = shard.n_tokens - self.context_length - 1
            start = int(self._rng.integers(0, high + 1))
            xs[row] = shard.window(start, self.context_length)
            ys[row] = shard.window(start + 1, self.context_length)
            self._drawn[name] += self.context_length

        self._steps += 1
        x = torch.from_numpy(xs)
        y = torch.from_numpy(ys)
        if device is not None:
            # Non-blocking on CUDA alone: a copy from pageable memory is staged
            # before the call returns.  MPS may read xs and ys after this
            # function has freed them and the next batch has reused their
            # memory, so a model would train on other windows' inputs and
            # targets (HANDOFF §3 M10g).
            non_blocking = torch.device(device).type == "cuda"
            x, y = x.to(device, non_blocking=non_blocking), y.to(device, non_blocking=non_blocking)
        return x, y

    # -------------------------------------------------------- bookkeeping

    @property
    def tokens_drawn(self) -> int:
        return sum(self._drawn.values())

    @property
    def realised_mixture(self) -> dict[str, float]:
        total = self.tokens_drawn
        if total == 0:
            return {name: 0.0 for name in self._names}
        return {name: self._drawn[name] / total for name in self._names}

    @property
    def requested_mixture(self) -> dict[str, float]:
        return {name: float(p) for name, p in zip(self._names, self._probs)}

    def state_dict(self) -> dict:
        return {
            "rng": self._rng.bit_generator.state,
            "drawn": dict(self._drawn),
            "steps": self._steps,
            "seed": self.seed,
        }

    def load_state_dict(self, state: dict) -> None:
        self._rng.bit_generator.state = state["rng"]
        self._drawn = Counter(state["drawn"])
        self._steps = int(state["steps"])


def make_val_batches(
    shard: TokenShard,
    context_length: int,
    batch_size: int,
    n_batches: int,
    device: torch.device | str | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Fixed, evenly spaced validation windows.

    The same windows every time, for every run, so a 0.01 difference in
    validation loss between two arms is a difference between the models and not
    between two draws of the validation set.
    """
    n_windows = batch_size * n_batches
    usable = shard.n_tokens - context_length - 1
    if usable < 1:
        raise ValueError("validation shard is smaller than one context window")

    stride = max(1, usable // max(1, n_windows))
    starts = [(i * stride) % (usable + 1) for i in range(n_windows)]

    batches = []
    for b in range(n_batches):
        chunk = starts[b * batch_size : (b + 1) * batch_size]
        xs = np.stack([shard.window(s, context_length) for s in chunk])
        ys = np.stack([shard.window(s + 1, context_length) for s in chunk])
        x, y = torch.from_numpy(xs), torch.from_numpy(ys)
        if device is not None:
            x, y = x.to(device), y.to(device)
        batches.append((x, y))
    return batches
