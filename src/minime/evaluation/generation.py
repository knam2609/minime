"""Batched, seeded sampling for the generation-based metrics.

``MiniMeLM.generate`` is fine for a single prompt; the harness needs hundreds
of continuations that are reproducible and comparable across runs:

* **Each row has its own RNG**, seeded from ``(seed, prompt index)``.  A
  continuation depends on its prompt and the model, not on the batch size or
  on which other prompts shared its batch -- so a Kaggle run with batch 64 and
  a laptop run with batch 8 draw the same random numbers for the same prompt.
* **Rows stop independently** at ``<|endoftext|>``; the EOT is not part of
  the continuation, and whether a row reached it is recorded.
* **Entropy is of the model's own distribution** (before temperature and
  top-k), averaged over every generated position: the next-token entropy the
  degeneration metrics report.

Prompts in one call must share a length.  Every prompt set in the harness is
built that way (a fixed number of tokens taken from a shard), which avoids
padding and attention masks entirely.
"""

from __future__ import annotations

import hashlib
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class Generation:
    tokens: list[list[int]]  # one continuation per prompt, EOT excluded
    ended: list[bool]  # True if the row stopped at EOT rather than at max_new_tokens
    entropy: float  # mean nats over generated positions, raw distribution
    positions: int  # generated positions the entropy averages over

    @property
    def eot_rate(self) -> float:
        return sum(self.ended) / len(self.ended) if self.ended else 0.0

    @property
    def mean_length(self) -> float:
        return sum(map(len, self.tokens)) / len(self.tokens) if self.tokens else 0.0


def row_seed(seed: int, index: int) -> int:
    digest = hashlib.sha256(f"{seed}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


@torch.no_grad()
def sample(
    model: torch.nn.Module,
    prompts: np.ndarray,
    max_new_tokens: int,
    eot_id: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int = 0,
    batch_size: int = 32,
    device: torch.device | str = "cpu",
    autocast=None,
) -> Generation:
    """Continue each row of ``prompts`` (shape ``[n, prompt_len]``).

    ``temperature <= 0`` is greedy decoding.  The context is cropped to the
    model's window, as in ``MiniMeLM.generate``.
    """
    prompts = np.asarray(prompts, dtype=np.int64)
    if prompts.ndim != 2 or prompts.shape[1] < 1:
        raise ValueError(f"prompts must be a non-empty [n, length] array, got shape {prompts.shape}")
    context = model.cfg.context_length
    autocast = autocast or nullcontext

    was_training = model.training
    model.eval()
    out_tokens: list[list[int]] = []
    out_ended: list[bool] = []
    entropy_sum = 0.0
    positions = 0
    try:
        for b in range(0, len(prompts), batch_size):
            rows = range(b, min(b + batch_size, len(prompts)))
            rngs = [np.random.default_rng(row_seed(seed, i)) for i in rows]
            idx = torch.from_numpy(prompts[b : b + batch_size]).to(device)
            active = torch.ones(len(rows), dtype=torch.bool)
            generated: list[list[int]] = [[] for _ in rows]
            ended = [False] * len(rows)

            for _ in range(max_new_tokens):
                with autocast():
                    logits, _ = model(idx[:, -context:])
                # Keep a compact copy of the last position, so the full
                # [batch, length, vocab] block is freed now.  Each step's block
                # is a new size, and on MPS the cached ones piled up to an
                # out-of-memory error (8 GiB at 32 x 384 x 16k) on an 8 GB
                # laptop.  The numbers are unchanged: the same forward pass.
                logits = logits[:, -1, :].float().clone()
                _release_cached_memory(idx.device)

                logp = F.log_softmax(logits, dim=-1)
                ent = -(logp.exp() * logp).sum(-1).cpu()
                entropy_sum += float(ent[active].double().sum())
                positions += int(active.sum())

                next_ids = _draw(logits, temperature, top_k, [r.random() for r in rngs])
                for row, token in enumerate(next_ids.tolist()):
                    if not active[row]:
                        continue
                    if token == eot_id:
                        active[row] = False
                        ended[row] = True
                    else:
                        generated[row].append(token)
                if not bool(active.any()):
                    break
                idx = torch.cat((idx, next_ids.to(idx.device).unsqueeze(1)), dim=1)

            out_tokens.extend(generated)
            out_ended.extend(ended)
    finally:
        if was_training:
            model.train()

    return Generation(
        tokens=out_tokens,
        ended=out_ended,
        entropy=entropy_sum / positions if positions else 0.0,
        positions=positions,
    )


def _release_cached_memory(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()


def _draw(logits: torch.Tensor, temperature: float, top_k: int | None, uniforms: list[float]) -> torch.Tensor:
    """One token per row by inverse-CDF sampling with the caller's uniforms."""
    if temperature <= 0:
        return logits.argmax(dim=-1).cpu()
    logits = logits / temperature
    if top_k is not None and top_k < logits.size(-1):
        threshold = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    # float64 on the CPU: MPS has no float64, and a float32 CDF over 16k
    # entries can end visibly short of 1.
    cdf = F.softmax(logits, dim=-1).cpu().double().cumsum(dim=-1)
    u = torch.tensor(uniforms, dtype=torch.float64).unsqueeze(1) * cdf[:, -1:]
    picks = torch.searchsorted(cdf, u, right=True).squeeze(1)
    return picks.clamp_(max=logits.size(-1) - 1)
