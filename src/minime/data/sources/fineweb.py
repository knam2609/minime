"""Web prose: FineWeb-Edu, the ``sample-10BT`` subset.

Educational web pages filtered by a quality classifier: clean, well-formed
prose.  It is the bulk of the general corpus and the reference for "general
language ability" on the Pareto frontier.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Iterator

from ..document import Document, make_id
from .hf import HFSource, field, stream_rows

SOURCE = HFSource(
    dataset="HuggingFaceFW/fineweb-edu",
    config="sample-10BT",
    splits=("train",),
    revision="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
    license="odc-by",
)


def row_to_document(row: dict) -> Document | None:
    text = field(row, "text", SOURCE)
    if not text or not text.strip():
        return None
    return Document(
        id=make_id("fineweb-edu", field(row, "id", SOURCE)),
        source="general",
        text=text,
        author="fineweb-edu",
        meta={"slice": "web"},
    )


def iter_documents(rows: Iterable[dict] | None = None, stats: Counter | None = None) -> Iterator[Document]:
    stats = stats if stats is not None else Counter()
    for row in stream_rows(SOURCE) if rows is None else rows:
        doc = row_to_document(row)
        if doc is None:
            stats["skipped_empty"] += 1
            continue
        stats["kept"] += 1
        yield doc
