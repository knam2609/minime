"""A character-level tokenizer for development.

Not used for any real experiment -- the BPE tokenizer is -- but it keeps the
tiny model, the overfitting test and the unit suite runnable with no
dependency on the ``tokenizers`` package and no vocabulary to train.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

EOT = "<|endoftext|>"


class CharTokenizer:
    name = "char"
    dev = True  # never used for a real experiment

    def __init__(self, chars: Sequence[str], specials: Sequence[str] = (EOT,)) -> None:
        self.specials = list(specials)
        self.itos = self.specials + [c for c in chars if c not in self.specials]
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    @classmethod
    def fit(cls, texts: Iterable[str], specials: Sequence[str] = (EOT,)) -> "CharTokenizer":
        chars: set[str] = set()
        for text in texts:
            chars.update(text)
        return cls(sorted(chars), specials)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def eot_id(self) -> int:
        return self.stoi[EOT]

    def encode(self, text: str) -> list[int]:
        # Unknown characters are dropped rather than mapped to a shared UNK:
        # at char level an UNK token teaches the model nothing useful.
        return [self.stoi[c] for c in text if c in self.stoi]

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.itos[i] for i in ids if 0 <= i < len(self.itos) and self.itos[i] not in self.specials)

    def byte_lengths(self) -> list[int]:
        """UTF-8 bytes per id; a special counts as one byte (see ``BPETokenizer``)."""
        return [1 if tok in self.specials else len(tok.encode("utf-8")) for tok in self.itos]

    def fingerprint(self) -> str:
        blob = json.dumps({"itos": self.itos, "specials": self.specials}, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    @property
    def id(self) -> str:
        return f"{self.name}@{self.fingerprint()}"

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"itos": self.itos, "specials": self.specials}, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        payload = json.loads(Path(path).read_text())
        tok = cls.__new__(cls)
        tok.specials = payload["specials"]
        tok.itos = payload["itos"]
        tok.stoi = {t: i for i, t in enumerate(tok.itos)}
        return tok
