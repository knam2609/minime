"""Causal multi-head self-attention.

Two code paths compute the same thing:

* ``_sdpa`` -- ``F.scaled_dot_product_attention``, which dispatches to a fused
  kernel and is what every real run uses.
* ``_manual`` -- the textbook implementation, kept deliberately so the test
  suite can assert the fused path agrees with it.  A masking bug in the fused
  path is invisible in a loss curve (the model just quietly cheats), so the
  equivalence test is the only thing standing between us and a corrupt result.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .rope import RotaryCache, apply_rotary


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, rotary: RotaryCache | None = None) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.dropout = cfg.dropout
        self.rotary = rotary

        # One fused projection for q, k and v: a third of the kernel launches.
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, h, t, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * d)

    def _manual(self, q, k, v, dropout_p: float) -> torch.Tensor:
        t = q.shape[-2]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
        weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
        weights = F.dropout(weights, p=dropout_p, training=self.training)
        return weights @ v

    def _sdpa(self, q, k, v, dropout_p: float) -> torch.Tensor:
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)

    def forward(self, x: torch.Tensor, use_sdpa: bool = True) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv_proj(x).split(c, dim=2)
        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)

        if self.rotary is not None:
            cos, sin = self.rotary.get(t)
            q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)

        dropout_p = self.dropout if self.training else 0.0
        attn = self._sdpa(q, k, v, dropout_p) if use_sdpa else self._manual(q, k, v, dropout_p)
        return self.resid_dropout(self.out_proj(self._merge_heads(attn)))
