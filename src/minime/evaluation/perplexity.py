"""Cross-entropy on held-out token shards: both axes of the frontier.

Personal CE is the primary voice metric; general CE is how forgetting shows up
during adaptation.  Three choices make the numbers comparable across runs:

* **Every scored token is scored once.**  Windows are non-overlapping and
  aligned to multiples of the context length, so no token is counted twice
  and no target is silently dropped.  A small shard (the personal ``val`` is
  ~16k tokens) is covered completely, including its final partial window.
* **Capped shards are sampled evenly, not from the front.**  A general ``val``
  of millions of tokens is scored on ``max_tokens`` worth of windows spread
  across the whole shard; its front is one stretch of the source's stream
  order.
* **The eval set is hashed.**  ``digest`` covers the window plan and every
  token scored, so two results with the same digest were measured on the same
  text, and a changed shard cannot pass for the old one.

``bits_per_byte`` divides the same total by the UTF-8 length of the targets.
Loss per token is comparable across runs (one frozen tokenizer); bits per byte
is also comparable across *sources*, whose chars-per-token differ (personal
chat 3.3, web prose 4.2).
"""

from __future__ import annotations

import hashlib
import math
from contextlib import nullcontext
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..data.binidx import TOKEN_DTYPE, TokenShard

Window = tuple[int, int]  # (start, length): inputs tokens[s:s+length], targets shifted by one


def plan_windows(n_tokens: int, context_length: int, max_tokens: int | None = None) -> list[Window]:
    """Non-overlapping windows covering the shard, or an even sample of them.

    Without a cap (or when the shard fits under it) every target token is in
    exactly one window and the last window may be shorter.  With a cap, only
    full windows are used, ``max_tokens // context_length`` of them (at least
    one), spread evenly from the start of the shard to its end.
    """
    if context_length < 1:
        raise ValueError("context_length must be positive")
    scorable = n_tokens - 1  # the first token is never a target
    if scorable < 1:
        raise ValueError(f"a shard of {n_tokens} token(s) has nothing to score")

    n_full, rest = divmod(scorable, context_length)
    if max_tokens is None or scorable <= max_tokens or n_full == 0:
        windows = [(i * context_length, context_length) for i in range(n_full)]
        if rest:
            windows.append((n_full * context_length, rest))
        return windows

    n = max(1, max_tokens // context_length)  # n <= n_full because scorable > max_tokens
    return [((i * n_full) // n * context_length, context_length) for i in range(n)]


def windows_digest(shard: TokenShard, windows: Sequence[Window]) -> str:
    """Hash of the window plan and every token it reads."""
    h = hashlib.sha256()
    for start, length in windows:
        h.update(f"{start}:{length};".encode())
        h.update(np.asarray(shard.tokens[start : start + length + 1], dtype=TOKEN_DTYPE).tobytes())
    return h.hexdigest()[:16]


@torch.no_grad()
def cross_entropy(
    model: torch.nn.Module,
    shard: TokenShard,
    context_length: int,
    batch_size: int = 16,
    device: torch.device | str = "cpu",
    max_tokens: int | None = None,
    byte_lengths: Sequence[int] | None = None,
    autocast=None,
) -> dict:
    """Mean next-token loss over ``shard`` (see ``plan_windows`` for which tokens).

    ``autocast`` is a context-manager factory (``lambda: autocast_ctx(spec)``).
    Logits are cast to float32 before the loss; each batch's sum is taken on
    the device and the batches are accumulated in float64 on the host.
    """
    windows = plan_windows(shard.n_tokens, context_length, max_tokens)
    lengths = np.asarray(byte_lengths, dtype=np.int64) if byte_lengths is not None else None
    autocast = autocast or nullcontext

    was_training = model.training
    model.eval()
    nats = 0.0
    n_targets = n_bytes = 0
    try:
        # Only the last window can be shorter, so batching by length is cheap.
        by_length: dict[int, list[int]] = {}
        for start, length in windows:
            by_length.setdefault(length, []).append(start)
        for length, starts in by_length.items():
            for b in range(0, len(starts), batch_size):
                chunk = starts[b : b + batch_size]
                xs = np.stack([shard.window(s, length) for s in chunk])
                ys = np.stack([shard.window(s + 1, length) for s in chunk])
                x = torch.from_numpy(xs).to(device)
                y = torch.from_numpy(ys).to(device)
                with autocast():
                    logits, _ = model(x)
                loss = F.cross_entropy(
                    logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
                )
                nats += loss.item()  # Python float: float64 across batches (MPS has none)
                n_targets += ys.size
                if lengths is not None:
                    n_bytes += int(lengths[ys].sum())
    finally:
        if was_training:
            model.train()

    mean = nats / n_targets
    return {
        "loss": mean,
        "ppl": math.exp(min(mean, 50.0)),
        "bits_per_byte": nats / math.log(2) / n_bytes if n_bytes else None,
        "tokens": n_targets,
        "windows": len(windows),
        "coverage": n_targets / (shard.n_tokens - 1),
        "digest": windows_digest(shard, windows),
    }


def weighted_mean(results: dict[str, dict], weights: dict[str, float], key: str = "loss") -> float:
    """Mixture-weighted average of per-slice results (weights are normalised)."""
    missing = set(weights) - set(results)
    if missing:
        raise KeyError(f"no result for weighted slice(s) {sorted(missing)}")
    total = sum(weights.values())
    return sum(results[name][key] * w / total for name, w in weights.items())
