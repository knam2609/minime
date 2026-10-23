"""General-corpus sources: streaming loaders that yield ``Document`` objects.

Each slice of the ~900M-token general corpus is one pinned Hugging Face
dataset.  The loaders only convert rows to documents; tokenising, splitting
and writing shards is ``minime.data.ingest``'s job, so every source goes
through exactly the same path.

The targets are the plan's 70 / 20 / 10 split of ~900M tokens.  Why each slice
exists, and what its chosen dataset cannot do, is in the slice's module
docstring and the README's limitations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterator

from ..document import Document
from . import codeparrot, discord, fineweb
from .hf import HFSource, SchemaDriftError, stream_rows


@dataclass(frozen=True)
class GeneralSlice:
    name: str
    source: HFSource
    iter_documents: Callable[..., Iterator[Document]]
    target_tokens: int
    purpose: str


GENERAL_SLICES: dict[str, GeneralSlice] = {
    "web": GeneralSlice(
        "web", fineweb.SOURCE, fineweb.iter_documents, 630_000_000,
        "clean, well-formed prose",
    ),
    "dialogue": GeneralSlice(
        "dialogue", discord.SOURCE, discord.iter_documents, 180_000_000,
        "conversational turn-taking, so adapting onto chat is less of a register cliff",
    ),
    "code": GeneralSlice(
        "code", codeparrot.SOURCE, codeparrot.iter_documents, 90_000_000,
        "a code axis for the data-mixture experiment",
    ),
}

DEFAULT_TARGETS = ",".join(f"{s.name}={s.target_tokens // 1_000_000}M" for s in GENERAL_SLICES.values())

_SCALE = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}


def parse_targets(text: str) -> dict[str, int]:
    """``"web=630M,dialogue=180M"`` -> ``{"web": 630_000_000, ...}``, in the order given."""
    targets: dict[str, int] = {}
    for part in filter(None, (p.strip() for p in text.split(","))):
        name, _, amount = part.partition("=")
        name = name.strip()
        if name not in GENERAL_SLICES:
            raise KeyError(f"unknown general slice {name!r}; expected one of {sorted(GENERAL_SLICES)}")
        match = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([KMB]?)\s*", amount.upper())
        if not match:
            raise ValueError(f"bad token target {amount!r} for {name!r}; use e.g. 630M or 2.5B")
        targets[name] = int(float(match.group(1)) * _SCALE[match.group(2)])
        if targets[name] <= 0:
            raise ValueError(f"token target for {name!r} must be positive")
    if not targets:
        raise ValueError("no general slices requested")
    return targets


__all__ = [
    "DEFAULT_TARGETS",
    "GENERAL_SLICES",
    "GeneralSlice",
    "HFSource",
    "SchemaDriftError",
    "parse_targets",
    "stream_rows",
]
