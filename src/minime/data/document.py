"""The one record format every data source converts into.

Meta exports, long-form writing, GitHub repos and the synthetic stand-in
corpus all produce ``Document`` objects, so tokenisation, splitting and
evaluation each have exactly one input shape to handle.

``author`` exists because the evaluation harness needs negatives: "does this
text read like me?" is only answerable against somebody else's writing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

SOURCES = ("chat", "comment", "longform", "code", "commit", "general")


@dataclass
class Document:
    id: str
    source: str
    text: str
    author: str = "me"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"unknown source {self.source!r}; expected one of {SOURCES}")

    @property
    def n_chars(self) -> int:
        return len(self.text)

    @property
    def n_words(self) -> int:
        return len(self.text.split())

    def to_dict(self) -> dict:
        return asdict(self)


def make_id(*parts: object) -> str:
    """Stable id from the source identity, so re-running a parser is idempotent."""
    blob = "|".join(str(p) for p in parts)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def write_jsonl(path: str | Path, docs: Iterable[Document]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> Iterator[Document]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield Document(**json.loads(line))
