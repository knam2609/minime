"""Tokenizers.

``CharTokenizer`` has no dependencies and keeps the unit suite and the
overfitting gate runnable anywhere.  ``BPETokenizer`` (``minime.tokenizer.bpe``)
is the one real experiments use; it is imported lazily because it needs the
optional ``tokenizers`` package.
"""

from __future__ import annotations

import json
from pathlib import Path

from .char import EOT, CharTokenizer

__all__ = ["CharTokenizer", "EOT", "is_dev_tokenizer", "load_tokenizer"]


def load_tokenizer(path: str | Path):
    """Load either tokenizer kind from its saved JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") == "minime-bpe":
        from .bpe import BPETokenizer

        return BPETokenizer.load(path)
    if "itos" in payload:
        return CharTokenizer.load(path)
    raise ValueError(f"{path} is not a tokenizer file this project wrote")


def is_dev_tokenizer(tokenizer_id: str) -> bool:
    """True for a ``ShardMeta.tokenizer`` id that must never back a real experiment.

    Relies on the naming rule ``train_tokenizer.py`` enforces: a dev BPE
    tokenizer's name ends in ``-dev``; the char tokenizer is dev by definition.
    """
    name = tokenizer_id.split("@", 1)[0]
    return name == "char" or name.endswith("-dev")
