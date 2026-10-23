"""A pre-norm decoder block.

Pre-norm (normalise going *into* each sublayer, add the raw residual) rather
than post-norm: it is what makes deep stacks trainable without a long warmup
and keeps the residual stream an unnormalised identity path.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig
from .attention import CausalSelfAttention
from .ffn import make_ffn
from .norm import make_norm
from .rope import RotaryCache


class DecoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, rotary: RotaryCache | None = None) -> None:
        super().__init__()
        self.norm_attn = make_norm(cfg)
        self.attn = CausalSelfAttention(cfg, rotary=rotary)
        self.norm_ffn = make_norm(cfg)
        self.ffn = make_ffn(cfg)

    def forward(self, x: torch.Tensor, use_sdpa: bool = True) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x), use_sdpa=use_sdpa)
        x = x + self.ffn(self.norm_ffn(x))
        return x
