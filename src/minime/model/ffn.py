"""Feed-forward blocks: SwiGLU vs GELU (an ablation axis).

The widths are set in ``ModelConfig.ffn_hidden`` so that the SwiGLU variant's
third matrix does not make it a larger model -- otherwise "SwiGLU is better"
would just mean "more parameters are better".
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig


class GELUFeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = cfg.ffn_hidden
        self.up_proj = nn.Linear(cfg.d_model, hidden, bias=cfg.bias)
        self.down_proj = nn.Linear(hidden, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.gelu(self.up_proj(x))))


class SwiGLUFeedForward(nn.Module):
    """Gated variant: one branch produces values, the other gates them."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = cfg.ffn_hidden
        self.gate_proj = nn.Linear(cfg.d_model, hidden, bias=cfg.bias)
        self.up_proj = nn.Linear(cfg.d_model, hidden, bias=cfg.bias)
        self.down_proj = nn.Linear(hidden, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


def make_ffn(cfg: ModelConfig) -> nn.Module:
    return SwiGLUFeedForward(cfg) if cfg.ffn == "swiglu" else GELUFeedForward(cfg)
