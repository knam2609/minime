"""The gate that must pass before any real training run.

A model that cannot memorise a single batch has a bug -- in the mask, the
loss, the optimiser wiring or the target shift -- and no amount of GPU time
will fix it.  Running this first turns a class of silent correctness failures
into a five-second local check.

The test deliberately uses *real* token sequences rather than random ids:
random targets are unlearnable beyond the unigram prior, so a broken model and
a working one look identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..config import ModelConfig
from ..model import build_model
from ..utils.device import resolve
from ..utils.seed import seed_everything


@dataclass
class OverfitResult:
    losses: list[float] = field(default_factory=list)
    steps_to_threshold: int | None = None
    threshold: float = 0.1
    initial_loss: float = 0.0

    @property
    def final_loss(self) -> float:
        return self.losses[-1] if self.losses else float("nan")

    @property
    def passed(self) -> bool:
        return self.final_loss < self.threshold

    def summary(self) -> str:
        reached = (
            f"reached <{self.threshold} at step {self.steps_to_threshold}"
            if self.steps_to_threshold is not None
            else f"never reached <{self.threshold}"
        )
        return (
            f"initial {self.initial_loss:.3f} -> final {self.final_loss:.4f} "
            f"({reached}, {len(self.losses)} steps)"
        )


def run_overfit(
    cfg: ModelConfig | None = None,
    tokens: torch.Tensor | None = None,
    steps: int = 300,
    lr: float = 3e-3,
    batch_size: int = 4,
    threshold: float = 0.1,
    seed: int = 0,
    device: str = "auto",
    log_every: int = 0,
) -> OverfitResult:
    """Drive one fixed batch to near-zero loss.

    Args:
        tokens: ``(batch, context+1)`` of real token ids.  Generated as a
            repeating structured pattern when omitted, which is memorisable but
            not trivially copyable from the input.
    """
    seed_everything(seed)
    cfg = cfg or ModelConfig(
        vocab_size=64, d_model=64, n_heads=4, n_layers=2, context_length=64, dropout=0.0
    )
    spec = resolve(device, "fp32")

    if tokens is None:
        tokens = _structured_tokens(batch_size, cfg.context_length + 1, cfg.vocab_size, seed)
    tokens = tokens.to(spec.device)
    x, y = tokens[:, :-1], tokens[:, 1:]

    model = build_model(cfg).to(spec.device)
    model.train()
    optimiser = torch.optim.AdamW(model.param_groups(weight_decay=0.0), lr=lr, betas=(0.9, 0.95))

    result = OverfitResult(threshold=threshold)
    for step in range(steps):
        _, loss = model(x, targets=y)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()

        value = float(loss.detach())
        result.losses.append(value)
        if step == 0:
            result.initial_loss = value
        if result.steps_to_threshold is None and value < threshold:
            result.steps_to_threshold = step
        if log_every and step % log_every == 0:
            print(f"  step {step:4d}  loss {value:.4f}")

    return result


def _structured_tokens(batch: int, length: int, vocab: int, seed: int) -> torch.Tensor:
    """A learnable but non-trivial sequence.

    Each row is a short random motif repeated with a per-position offset, so
    predicting the next token requires actually using context -- a model that
    only learned the unigram distribution cannot drive this to zero.
    """
    generator = torch.Generator().manual_seed(seed)
    motif_len = max(4, length // 8)
    motif = torch.randint(0, vocab, (batch, motif_len), generator=generator)
    repeats = (length // motif_len) + 1
    seq = motif.repeat(1, repeats)[:, :length]
    offsets = torch.arange(length) // motif_len
    return (seq + offsets.unsqueeze(0)) % vocab
