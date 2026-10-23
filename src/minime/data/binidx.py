"""On-disk token storage: a flat ``.bin`` of uint16 ids plus a ``.idx`` of
document offsets.

Why not just pickle a list of lists: training reads billions of tokens in
random order, so the corpus has to be memory-mapped rather than loaded.  uint16
halves the file size versus int32 and is the reason ``vocab_size`` is capped at
65535 in the config.

The ``.idx`` offsets are not used by the training sampler (which samples flat
windows across document boundaries, as every LM trainer does) but they are
essential everywhere else: per-document train/val/test splits, stylometry over
whole documents, and corpus statistics.  Losing document boundaries at
tokenisation time is a one-way door.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np

TOKEN_DTYPE = np.uint16
MAX_TOKEN_ID = np.iinfo(TOKEN_DTYPE).max

_SUFFIXES = (".bin", ".idx.npy", ".idx", ".meta.json")


def shard_stem(path: str | Path) -> Path:
    """Strip any shard suffix, leaving the stem the three files share.

    Deliberately not ``Path.with_suffix("")``: a shard called ``personal.v2``
    would silently become ``personal``, and two corpus versions would then
    overwrite each other.
    """
    text = str(path)
    for suffix in _SUFFIXES:
        if text.endswith(suffix):
            return Path(text[: -len(suffix)])
    return Path(text)


@dataclass(frozen=True)
class ShardMeta:
    name: str
    n_tokens: int
    n_docs: int
    vocab_size: int
    dtype: str = "uint16"
    tokenizer: str = ""
    created: str = ""
    extra: dict | None = None

    def to_dict(self) -> dict:
        payload = {
            "name": self.name,
            "n_tokens": self.n_tokens,
            "n_docs": self.n_docs,
            "vocab_size": self.vocab_size,
            "dtype": self.dtype,
            "tokenizer": self.tokenizer,
            "created": self.created,
        }
        if self.extra:
            payload["extra"] = self.extra
        return payload


class ShardWriter:
    """Streams documents to disk so a corpus never has to fit in memory.

    Usage::

        with ShardWriter(path, vocab_size=16384) as w:
            for doc in docs:
                w.add_document(token_ids)
    """

    def __init__(
        self,
        path: str | Path,
        vocab_size: int,
        tokenizer: str = "",
        eot_id: int | None = None,
        extra: dict | None = None,
    ) -> None:
        self.path = shard_stem(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.eot_id = eot_id
        self.extra = extra
        self._offsets: list[int] = [0]
        self._cursor = 0
        self._handle = open(self.bin_path, "wb")

    @property
    def bin_path(self) -> Path:
        return Path(f"{self.path}.bin")

    @property
    def idx_path(self) -> Path:
        # np.save appends .npy, so the name is spelled out here rather than
        # letting writer and reader disagree about it.
        return Path(f"{self.path}.idx.npy")

    @property
    def meta_path(self) -> Path:
        return Path(f"{self.path}.meta.json")

    @property
    def n_tokens(self) -> int:
        """Tokens written so far, EOT included."""
        return self._cursor

    def add_document(self, tokens: Sequence[int] | np.ndarray) -> None:
        arr = np.asarray(tokens, dtype=np.int64)
        if arr.size == 0:
            return
        if arr.min() < 0 or arr.max() >= self.vocab_size:
            raise ValueError(
                f"token id out of range for vocab_size={self.vocab_size}: "
                f"[{int(arr.min())}, {int(arr.max())}]"
            )
        if self.eot_id is not None:
            arr = np.append(arr, self.eot_id)
        arr.astype(TOKEN_DTYPE).tofile(self._handle)
        self._cursor += arr.size
        self._offsets.append(self._cursor)

    def add_documents(self, docs: Iterable[Sequence[int]]) -> None:
        for doc in docs:
            self.add_document(doc)

    def close(self) -> ShardMeta:
        self._handle.close()
        offsets = np.asarray(self._offsets, dtype=np.int64)
        np.save(self.idx_path.with_suffix(""), offsets, allow_pickle=False)
        meta = ShardMeta(
            name=self.path.name,
            n_tokens=self._cursor,
            n_docs=len(self._offsets) - 1,
            vocab_size=self.vocab_size,
            tokenizer=self.tokenizer,
            created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            extra=self.extra,
        )
        self.meta_path.write_text(json.dumps(meta.to_dict(), indent=2))
        return meta

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *exc) -> None:
        if not self._handle.closed:
            self.close()


class TokenShard:
    """Read-only, memory-mapped view over one ``.bin``/``.idx`` pair."""

    def __init__(self, path: str | Path) -> None:
        self.path = shard_stem(path)
        if not self.bin_path.exists():
            raise FileNotFoundError(f"no token shard at {self.bin_path}")
        if self.bin_path.stat().st_size == 0:
            # mmap cannot map an empty file.  An empty shard is legitimate (a
            # small split can receive no documents); the sampler and the
            # validation windows then fail with their own, clearer errors.
            self.tokens = np.zeros(0, dtype=TOKEN_DTYPE)
        else:
            self.tokens = np.memmap(self.bin_path, dtype=TOKEN_DTYPE, mode="r")
        self.offsets = np.load(self.idx_path) if self.idx_path.exists() else None
        self.meta = (
            json.loads(self.meta_path.read_text()) if self.meta_path.exists() else {}
        )

    # paths -------------------------------------------------------------
    @property
    def bin_path(self) -> Path:
        return Path(f"{self.path}.bin")

    @property
    def idx_path(self) -> Path:
        return Path(f"{self.path}.idx.npy")

    @property
    def meta_path(self) -> Path:
        return Path(f"{self.path}.meta.json")

    # access ------------------------------------------------------------
    @property
    def n_tokens(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def n_docs(self) -> int:
        return 0 if self.offsets is None else int(self.offsets.shape[0] - 1)

    def document(self, i: int) -> np.ndarray:
        if self.offsets is None:
            raise RuntimeError("shard has no document index")
        start, end = int(self.offsets[i]), int(self.offsets[i + 1])
        return np.asarray(self.tokens[start:end])

    def documents(self) -> Iterator[np.ndarray]:
        for i in range(self.n_docs):
            yield self.document(i)

    def window(self, start: int, length: int) -> np.ndarray:
        return np.asarray(self.tokens[start : start + length], dtype=np.int64)

    def __len__(self) -> int:
        return self.n_tokens

    def __repr__(self) -> str:
        return f"TokenShard({self.path.name}, {self.n_tokens:,} tokens, {self.n_docs:,} docs)"


def write_shard(
    path: str | Path,
    docs: Iterable[Sequence[int]],
    vocab_size: int,
    tokenizer: str = "",
    eot_id: int | None = None,
    extra: dict | None = None,
) -> ShardMeta:
    with ShardWriter(path, vocab_size, tokenizer, eot_id, extra) as writer:
        writer.add_documents(docs)
        return writer.close()


def common_tokenizer(shards: Mapping[str, TokenShard]) -> str:
    """The single tokenizer id every shard was made with; raise if they differ.

    Mixing tokenizers is the quiet way to break the one-frozen-tokenizer rule:
    the ids are all in range, training runs, and the losses are meaningless.
    """
    by_id: dict[str, list[str]] = {}
    for name, shard in shards.items():
        by_id.setdefault(shard.meta.get("tokenizer", ""), []).append(name)
    if len(by_id) > 1:
        detail = "; ".join(f"{tid or '<unrecorded>'}: {', '.join(sorted(names))}"
                           for tid, names in sorted(by_id.items()))
        raise ValueError(f"shards were made by different tokenizers -- {detail}")
    return next(iter(by_id), "")
