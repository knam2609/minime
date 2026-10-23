"""Dialogue: Discord-Dialogues (mookiezi/Discord-Dialogues), real two-person chat.

7.3M anonymized exchanges between two people on public Discord servers, from
spring to autumn 2025: ~16.9M turns, ~139M words, Apache-2.0 as released.  It
replaces SODA, which filled the slice but was written by an LLM -- turn-taking
without the orthography of real chat.  Discord supplies the real thing:
lowercase messages, fragments, slang, one-line replies.

What it still is not, recorded as limitations in the README:

* **Cleaned.**  The publisher's pipeline removed links, code blocks and PII,
  normalised some slang, and stripped nearly all emoji (0.03-0.1% of
  exchanges keep one in the row groups sampled, against 5.7% of the owner's
  messages).
* **One register.**  Gaming/hobby Discord in 2025, not DMs between friends.
* **Barely enough.**  Turn text is 49-55% of the raw ChatML characters, so
  ~610-690M characters: roughly 175-200M tokens with document boundaries,
  against a 180M target.  If the stream runs dry, ingestion records
  ``exhausted: true`` and the realised share rather than failing.

Rows are ChatML strings: ``<|im_start|>user\\n...<|im_end|>`` per turn, then
``<|end_of_text|>``.  The role names only alternate between the two speakers,
so they are dropped, and turns are joined by newlines with no labels -- the
format of the personal ``chat`` documents.  Nothing ChatML survives into the
text; it would otherwise be a format the personal corpus never uses.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Iterator

from ..document import Document, make_id
from .hf import HFSource, SchemaDriftError, field, stream_rows

SOURCE = HFSource(
    dataset="mookiezi/Discord-Dialogues",
    config=None,
    splits=("train",),
    revision="a8b2294bd5b4acfe4ce537b688e7eee111c50fe2",
    license="apache-2.0",
)

MIN_TURNS = 2
TURN = re.compile(r"<\|im_start\|>\w+\n(.*?)<\|im_end\|>", re.DOTALL)
MARKER = "<|im_"


def parse_turns(text: str) -> list[str]:
    """ChatML -> stripped, non-empty turn texts.  Raises if there is no ChatML at all."""
    if MARKER not in text:
        raise SchemaDriftError(
            f"{SOURCE.dataset}@{SOURCE.revision[:10]}: 'text' has no ChatML turns; the "
            "dataset format differs from the one this converter was written against"
        )
    return [turn.strip() for turn in TURN.findall(text) if turn.strip()]


def row_to_document(row: dict, stats: Counter | None = None) -> Document | None:
    stats = stats if stats is not None else Counter()
    turns = parse_turns(field(row, "text", SOURCE))
    if any(MARKER in turn for turn in turns):
        stats["skipped_malformed"] += 1
        return None
    if len(turns) < MIN_TURNS:
        stats["skipped_short"] += 1
        return None
    stats["kept"] += 1
    text = "\n".join(turns)
    return Document(
        # Content-addressed: the dataset has no per-exchange id of its own.
        id=make_id("discord-dialogues", text),
        source="general",
        text=text,
        author="discord-dialogues",
        meta={"slice": "dialogue", "turns": len(turns)},
    )


def iter_documents(rows: Iterable[dict] | None = None, stats: Counter | None = None) -> Iterator[Document]:
    stats = stats if stats is not None else Counter()
    for row in stream_rows(SOURCE) if rows is None else rows:
        doc = row_to_document(row, stats)
        if doc is not None:
            yield doc
