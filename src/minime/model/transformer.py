"""The MiniMe decoder-only language model.

A single implementation covers every architecture arm.  ``ModelConfig``
decides RMSNorm vs LayerNorm, RoPE vs learned positions, SwiGLU vs GELU, so an
ablation is a one-line config diff and never a second copy of the model.
"""

from __future__ import annotations

import math
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .block import DecoderBlock
from .norm import make_norm
from .rope import RotaryCache


class MiniMeLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = (
            nn.Embedding(cfg.context_length, cfg.d_model) if cfg.pos == "learned" else None
        )
        self.rotary = (
            RotaryCache(cfg.head_dim, cfg.context_length, cfg.rope_theta)
            if cfg.pos == "rope"
            else None
        )

        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(DecoderBlock(cfg, self.rotary) for _ in range(cfg.n_layers))
        self.norm_out = make_norm(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            # Tying saves vocab_size*d_model params -- 37% of the probe model --
            # and is a well-established small-model win.
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        self._scale_residual_projections()

    # ------------------------------------------------------------------ init

    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def _scale_residual_projections(self) -> None:
        """Shrink the projections that write into the residual stream.

        Each layer adds to the residual, so without this the stream's variance
        grows with depth and deep models start out badly conditioned.  Scaling
        by 1/sqrt(2 * n_layers) (two residual writes per block) keeps the
        activation scale roughly depth-independent at init.
        """
        std = self.cfg.init_std / math.sqrt(2 * self.cfg.n_layers)
        for name, param in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=std)

    # --------------------------------------------------------------- forward

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        use_sdpa: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the model, optionally computing the loss.

        ``targets`` must already be shifted by the caller: ``targets[t]`` is the
        token that follows ``idx[t]``.  The loss is *not* shifted internally.
        Passing an unshifted ``targets=idx`` asks the model to predict the token
        it was just given, which weight tying makes trivially easy -- it looks
        like a suspiciously good loss rather than an error.
        """
        _, t = idx.shape
        if t > self.cfg.context_length:
            raise ValueError(f"sequence length {t} exceeds context {self.cfg.context_length}")

        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            pos = torch.arange(t, device=idx.device)
            x = x + self.pos_emb(pos)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x, use_sdpa=use_sdpa)
        x = self.norm_out(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        return logits, loss

    # ----------------------------------------------------------- bookkeeping

    def num_parameters(self, non_embedding: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.tok_emb.weight.numel()
            if self.pos_emb is not None:
                total -= self.pos_emb.weight.numel()
            if not self.cfg.tie_embeddings:
                total -= self.lm_head.weight.numel()
        return total

    def flops_per_token(self) -> float:
        """Approximate training FLOPs per token, for MFU reporting.

        6*N covers forward+backward through the weights; the second term is the
        attention score/value matmuls, which do not live in any parameter and
        are not negligible at these context lengths.
        """
        cfg = self.cfg
        n = self.num_parameters(non_embedding=True)
        attention = 12 * cfg.n_layers * cfg.d_model * cfg.context_length
        return 6 * n + attention

    def param_groups(self, weight_decay: float) -> list[dict]:
        """Decay every parameter with two or more dimensions; nothing else.

        Norm gains and biases are exempt.  The embeddings -- the token table,
        which is also the tied output head, and the learned position table when
        ``pos=learned`` -- *are* decayed, as in nanoGPT and llm.c, the reference
        recipe the E4 learning-rate grid is centred on.
        The owner fixed this as the baseline before the first sweep; exempting
        the embedding (OLMo 2's choice) would be an ablation, and one that
        invalidates every sweep run before it.
        """
        decay, no_decay = [], []
        seen: set[int] = set()
        for param in self.parameters():
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            (decay if param.dim() >= 2 else no_decay).append(param)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    # ------------------------------------------------------------ generation

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Plain autoregressive sampling (no KV cache -- fine at this scale)."""
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            window = idx[:, -self.cfg.context_length :]
            logits, _ = self(window)
            logits = logits[:, -1, :]
            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    threshold = torch.topk(logits, k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
            if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                break
        if was_training:
            self.train()
        return idx


def build_model(cfg: ModelConfig) -> MiniMeLM:
    return MiniMeLM(cfg)


def iter_named_2d(model: nn.Module) -> Iterator[tuple[str, nn.Parameter]]:
    """Hidden 2D weights -- the parameters Muon orthogonalises."""
    for name, param in model.named_parameters():
        if param.dim() == 2 and "emb" not in name and "lm_head" not in name:
            yield name, param
