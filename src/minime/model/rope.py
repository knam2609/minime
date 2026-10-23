"""Rotary position embeddings.

RoPE injects position by rotating query/key pairs by an angle proportional to
their absolute index, which makes the attention logit depend only on the
*relative* offset.  That invariance is the property the unit test asserts --
it is the whole reason RoPE works, and it is easy to break silently by getting
the dimension pairing wrong.
"""

from __future__ import annotations

import torch


class RotaryCache(torch.nn.Module):
    """Precomputed cos/sin tables, stored as buffers so ``.to(device)`` works."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE needs an even head dimension")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        cos, sin = self._build(max_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _build(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        pos = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)          # (T, head_dim/2)
        return freqs.cos(), freqs.sin()

    def get(self, seq_len: int, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        end = seq_len + offset
        if end > self.cos.shape[0]:
            raise ValueError(
                f"sequence length {end} exceeds the RoPE cache ({self.cos.shape[0]})"
            )
        # (1, 1, T, head_dim/2) so it broadcasts over batch and heads.
        cos = self.cos[offset:end].view(1, 1, seq_len, -1)
        sin = self.sin[offset:end].view(1, 1, seq_len, -1)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` of shape (B, H, T, head_dim) in the (i, i+d/2) planes."""
    cos = torch.cat((cos, cos), dim=-1).to(x.dtype)
    sin = torch.cat((sin, sin), dim=-1).to(x.dtype)
    return x * cos + rotate_half(x) * sin
