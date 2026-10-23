"""Optimiser construction.

Experiment E4 swaps the optimiser and nothing else, so both optimisers are
built from the same parameter grouping and the choice lives entirely in the
config.  In the Muon arm the hidden 2D weights move to Muon; every other
parameter keeps exactly the AdamW treatment it gets in the AdamW arm.
"""

from __future__ import annotations

import torch

from ...config import TrainingConfig
from ...model import iter_named_2d
from .muon import MuonWithAdamW, orthogonalise

# Weights that fuse several matrices into one kernel, and how many.  Muon
# orthogonalises each block on its own: q, k and v are three maps, and
# orthogonalising their stack would couple their scales.
FUSED_BLOCKS = {"qkv_proj.weight": 3}


def build_optimizer(model, cfg: TrainingConfig, device_type: str = "cpu"):
    groups = model.param_groups(weight_decay=cfg.weight_decay)

    if cfg.optimizer == "adamw":
        # The fused kernel is a meaningful speedup at these model sizes, where
        # the optimiser step is not negligible next to the forward/backward.
        kwargs = {"lr": cfg.learning_rate, "betas": tuple(cfg.betas), "eps": 1e-8}
        if device_type == "cuda":
            kwargs["fused"] = True
        return torch.optim.AdamW(groups, **kwargs)

    if cfg.optimizer == "muon":
        return MuonWithAdamW(muon_param_groups(model, cfg, groups))

    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


def muon_param_groups(model, cfg: TrainingConfig, adamw_groups: list[dict]) -> list[dict]:
    """Split ``adamw_groups`` into Muon groups (hidden 2D) and AdamW groups (the rest)."""
    by_blocks: dict[int, list] = {}
    hidden: set[int] = set()
    for name, param in iter_named_2d(model):
        if not param.requires_grad:
            continue
        blocks = next((n for suffix, n in FUSED_BLOCKS.items() if name.endswith(suffix)), 1)
        by_blocks.setdefault(blocks, []).append(param)
        hidden.add(id(param))

    muon = [
        {
            "params": params,
            "use_muon": True,
            "blocks": blocks,
            "lr": cfg.muon_lr,
            "momentum": cfg.muon_momentum,
            "nesterov": True,
            "weight_decay": cfg.muon_weight_decay,
        }
        for blocks, params in sorted(by_blocks.items())
    ]
    adamw = [
        {
            **group,
            "params": [p for p in group["params"] if id(p) not in hidden],
            "use_muon": False,
            "lr": cfg.learning_rate,
            "betas": tuple(cfg.betas),
            "eps": 1e-8,
        }
        for group in adamw_groups
    ]
    return [g for g in muon + adamw if g["params"]]


__all__ = ["FUSED_BLOCKS", "MuonWithAdamW", "build_optimizer", "muon_param_groups", "orthogonalise"]
