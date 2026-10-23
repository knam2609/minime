"""Normalisation layers.

RMSNorm vs LayerNorm is a candidate ablation, so both live behind one factory
and the block code never knows which it got.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig


class RMSNorm(nn.Module):
    """Root-mean-square norm: LayerNorm without the mean subtraction or bias.

    Cheaper (one reduction instead of two) and empirically no worse, which is
    why every recent open model uses it.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reduce in fp32 even under autocast: the mean of squares is exactly
        # the quantity that overflows in fp16.
        dtype = x.dtype
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed.to(dtype)) * self.weight

    def extra_repr(self) -> str:
        return f"dim={tuple(self.weight.shape)}, eps={self.eps}"


def make_norm(cfg: ModelConfig, dim: int | None = None) -> nn.Module:
    dim = dim or cfg.d_model
    if cfg.norm == "rmsnorm":
        return RMSNorm(dim)
    return nn.LayerNorm(dim, bias=cfg.bias)
