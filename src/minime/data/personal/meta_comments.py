"""Parser for comments in a Meta "Download Your Information" export.

Comments are a separate export file from messages, with a different and less
stable schema, so they get their own parser rather than being bolted onto
``meta_export``.

Why comments are worth the extra parser: they are short *public* writing.  DMs
and comments are different registers from the same person -- the corpus is
better for holding both, and keeping them as separate sources lets the mixture
experiment weight them independently.

**The schemas below are written from knowledge of past export versions and have
not yet been validated against a real export.**  Meta changes these files
without notice, so:

* several shapes are accepted rather than one;
* a file that parses to zero comments but clearly held records is reported
  loudly instead of returning an empty list, because a silently empty corpus
  is the failure mode that wastes a week;
* ``scripts/inspect_export.py`` prints the *structure* of a real export (keys
  and types, never values) so the parser can be checked against it without any
  message text being read.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from ..document import Document, make_id
from .language import admit
from .meta_export import _norm_name, fix_mojibake, is_appledouble

# Top-level keys that have wrapped the comment array in some export version.
RECORD_KEYS = (
    "comments_v2",            # Facebook, current
    "comments",               # Facebook, older
    "comments_media_comments",  # Instagram posts
    "comments_reels_comments",  # Instagram reels
    "post_comments",
    "reels_comments",
)

# Keys whose string values are never comment text.  Used to tell a reaction
# record (metadata only) apart from a comment record whose schema has moved --
# without that distinction, every "comments and reactions" file would report a
# false schema drift, which is worse than no diagnostic at all.
METADATA_KEYS = {
    "title", "author", "name", "uri", "url", "type", "sender_name",
    "media owner", "media_owner", "external_context", "share",
}

# Filename fragments that indicate a comments file.
FILENAME_HINTS = ("comment",)
# ...and ones that look like comments but are not mine to train on.
FILENAME_EXCLUDE = ("reaction", "like", "received", "on_your")


@dataclass
class CommentsConfig:
    me: list[str] = field(default_factory=list)
    min_chars: int = 2
    platform: str = "meta"
    # Instagram's export contains only the owner's activity and often omits an
    # author field entirely. Facebook's includes one. When absent, the comment
    # is assumed to be mine; when present, it must match.
    assume_mine_when_author_missing: bool = True
    # Same rule as messages: drop mainly-Vietnamese comments; None keeps all.
    max_vietnamese_share: float | None = 0.5
    language_counts: Counter = field(default_factory=Counter, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.me, str):
            self.me = [self.me]
        self._me_norm = {_norm_name(n) for n in self.me}

    def is_me(self, author: str | None) -> bool:
        if not author:
            return self.assume_mine_when_author_missing
        if not self._me_norm:
            return True
        return _norm_name(author) in self._me_norm


def find_comment_files(root: str | Path) -> list[Path]:
    """Every JSON file in the export that looks like my own comments."""
    root = Path(root)
    found: list[Path] = []
    for path in root.rglob("*.json"):
        if is_appledouble(path):
            continue
        name = path.name.lower()
        if not any(hint in name for hint in FILENAME_HINTS):
            continue
        if any(bad in name for bad in FILENAME_EXCLUDE):
            continue
        found.append(path)
    return sorted(set(found))


def _records(payload: Any) -> list[dict]:
    """Find the array of comment records, whatever wraps it this year."""
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        for key in RECORD_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                return [r for r in value if isinstance(r, dict)]
        # Fall back to a lone list value, which covers a renamed key.
        lists = [v for v in payload.values() if isinstance(v, list)]
        if len(lists) == 1:
            return [r for r in lists[0] if isinstance(r, dict)]
    return []


def _extract(record: dict) -> tuple[str, int | None, str | None] | None:
    """Pull ``(text, timestamp, author)`` out of one record.

    Returns None for records with no comment text -- which is how reaction
    entries in Facebook's combined "comments and reactions" file get dropped:
    a reaction has a title but no comment body.
    """
    # Facebook: {"timestamp":…, "data":[{"comment":{"comment":…, "author":…}}], "title":…}
    data = record.get("data")
    if isinstance(data, list):
        for entry in data:
            if not isinstance(entry, dict):
                continue
            comment = entry.get("comment")
            if isinstance(comment, dict) and isinstance(comment.get("comment"), str):
                return (
                    comment["comment"],
                    record.get("timestamp") or comment.get("timestamp"),
                    comment.get("author"),
                )

    # Instagram: {"string_map_data": {"Comment": {"value": …}, "Time": {"timestamp": …}}}
    string_map = record.get("string_map_data")
    if isinstance(string_map, dict):
        text = (string_map.get("Comment") or {}).get("value")
        if isinstance(text, str):
            timestamp = (string_map.get("Time") or {}).get("timestamp")
            # No author field: Instagram exports only the owner's own activity.
            # "Media Owner" is whose post was commented on, never the author,
            # so it must not be used for the authorship check.
            return text, timestamp, None

    # Flat shape: {"comment": "…", "author": "…", "timestamp": …}
    if isinstance(record.get("comment"), str):
        return record["comment"], record.get("timestamp"), record.get("author")

    return None


def _candidate_strings(value: Any, depth: int = 0) -> Iterator[str]:
    """Strings in a record that could plausibly be comment text."""
    if depth > 4:
        return
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).strip().lower() in METADATA_KEYS:
                continue
            yield from _candidate_strings(nested, depth + 1)
    elif isinstance(value, list):
        for item in value:
            yield from _candidate_strings(item, depth + 1)
    elif isinstance(value, str) and value.strip():
        yield value


def looks_like_comment_record(record: dict) -> bool:
    """True if the record carries text that a comment parser ought to have found.

    A reaction entry has a title and a timestamp and nothing else, so it is
    correctly skipped and must not be counted as a parsing failure.
    """
    return any(len(text.strip()) >= 2 for text in _candidate_strings(record))


def parse_comment_file(path: str | Path, cfg: CommentsConfig | None = None) -> list[Document]:
    cfg = cfg or CommentsConfig()
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return []

    records = _records(payload)
    docs: list[Document] = []
    for index, record in enumerate(records):
        extracted = _extract(record)
        if extracted is None:
            continue
        raw_text, timestamp, author = extracted
        if not cfg.is_me(author):
            continue
        text = fix_mojibake(raw_text).strip()
        if len(text) < cfg.min_chars:
            continue
        if not admit(text, cfg.max_vietnamese_share, cfg.language_counts):
            continue
        docs.append(Document(
            id=make_id(cfg.platform, path.name, timestamp, index),
            source="comment",
            text=text,
            author="me",
            meta={"platform": cfg.platform, "timestamp": timestamp},
        ))
    return docs


def iter_comments(root: str | Path, cfg: CommentsConfig | None = None) -> Iterator[Document]:
    cfg = cfg or CommentsConfig()
    for path in find_comment_files(root):
        docs = parse_comment_file(path, cfg)
        unrecognised = _unrecognised_count(path)
        if unrecognised:
            # Records that carry text but matched no known shape: real drift.
            print(
                f"  WARNING: {path.name} has {unrecognised} record(s) containing text "
                f"that no known comment schema matched. The export format has probably "
                f"changed -- run `python scripts/inspect_export.py {path.parent}` "
                f"(it prints structure only, never text) and update "
                f"meta_comments._extract.",
                flush=True,
            )
        yield from docs


def _unrecognised_count(path: Path) -> int:
    """Records that look like comments but that ``_extract`` could not read."""
    try:
        records = _records(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return 0
    return sum(
        1 for record in records
        if _extract(record) is None and looks_like_comment_record(record)
    )


def parse_comments(root: str | Path, cfg: CommentsConfig | None = None) -> list[Document]:
    return list(iter_comments(root, cfg))
