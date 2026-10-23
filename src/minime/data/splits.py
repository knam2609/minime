"""Train/validation/test splitting, at document granularity.

Splitting a personal corpus by *line* rather than by document is the single
easiest way to get an impressive and completely meaningless result: the same
conversation lands on both sides, the model has effectively seen the test set,
and every style metric improves.  So the split is by whole document, always.

A document can belong to a larger unit that must not straddle the boundary
either: chat documents are bursts of utterances, dozens to a conversation, so
the Meta parser gives each a ``split_group`` (one month of one thread) and
the whole group is assigned together.  ``split_key`` is the one place that
decides what a split is assigned by.

Assignment is a hash of that key rather than a shuffle, which buys two
things:

* stability -- adding new documents later does not reshuffle the existing
  ones, so the held-out test set stays held out across corpus versions;
* reproducibility -- no RNG state to carry around, and anyone can recompute
  which split a given document belongs to.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Iterable, Sequence

from .document import Document

DEFAULT_RATIOS = {"train": 0.90, "val": 0.05, "test": 0.05}


def _fraction(doc_id: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{doc_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def split_key(doc: Document) -> str:
    """What decides a document's split: its group if it has one, else its own id."""
    return doc.meta.get("split_group") or doc.id


def assign_split(doc_id: str, ratios: dict[str, float] | None = None, salt: str = "minime") -> str:
    ratios = ratios or DEFAULT_RATIOS
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("split ratios must sum to a positive number")

    point = _fraction(doc_id, salt) * total
    cumulative = 0.0
    for name in sorted(ratios):            # sorted: order must not depend on dict insertion
        cumulative += ratios[name]
        if point < cumulative:
            return name
    return sorted(ratios)[-1]


def split_documents(
    docs: Iterable[Document],
    ratios: dict[str, float] | None = None,
    salt: str = "minime",
) -> dict[str, list[Document]]:
    ratios = ratios or DEFAULT_RATIOS
    out: dict[str, list[Document]] = {name: [] for name in ratios}
    for doc in docs:
        out[assign_split(split_key(doc), ratios, salt)].append(doc)
    return out


def check_no_leakage(splits: dict[str, Sequence[Document]]) -> None:
    """Raise if any document id, or any split group, appears in more than one split."""
    seen: dict[str, str] = {}
    groups: dict[str, str] = {}
    for name, docs in splits.items():
        for doc in docs:
            if doc.id in seen and seen[doc.id] != name:
                raise ValueError(
                    f"document {doc.id} appears in both '{seen[doc.id]}' and '{name}'"
                )
            seen[doc.id] = name
            key = split_key(doc)
            if key in groups and groups[key] != name:
                raise ValueError(
                    f"split group {key} appears in both '{groups[key]}' and '{name}'"
                )
            groups[key] = name


def describe_splits(splits: dict[str, Sequence[Document]]) -> dict[str, dict]:
    summary = {}
    for name, docs in splits.items():
        summary[name] = {
            "documents": len(docs),
            "characters": sum(d.n_chars for d in docs),
            "words": sum(d.n_words for d in docs),
            "by_source": dict(Counter(d.source for d in docs)),
        }
    return summary
