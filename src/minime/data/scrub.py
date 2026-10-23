"""PII redaction for the personal corpus.

This is a safety net, not a guarantee.  The primary protections are structural
-- only messages I wrote are kept, and the corpus is never committed -- but a
model trained on unredacted text can memorise and later emit a phone number,
so the text is scrubbed as well.

Every pattern here is deliberately conservative.  An over-eager phone-number
regex eats dates, version numbers and loss values, which would quietly corrupt
the corpus in a way no test of the *privacy* behaviour would catch.  The tests
therefore assert both directions: that PII is removed, and that ordinary text
survives untouched.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Collection, Iterable

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]*\w", re.UNICODE)
URL = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
# Handles only after whitespace or start, so an email's @ is never a handle.
HANDLE = re.compile(r"(?<![\w@.])@[A-Za-z0-9_.]{2,30}")

PHONE_PATTERNS = [
    re.compile(r"\+\d[\d\s().-]{7,}\d"),                       # +61 412 345 678
    re.compile(r"\b0\d{1,3}[\s.-]?\d{3,4}[\s.-]?\d{3,4}\b"),   # 0412 345 678
    re.compile(r"\b\d{3}[-.]\d{3}[-.]\d{4}\b"),                # 415-555-0134
]
LONG_NUMBER = re.compile(r"\b\d{9,}\b")

PLACEHOLDERS = {
    "email": "<EMAIL>",
    "url": "<URL>",
    "handle": "<HANDLE>",
    "phone": "<PHONE>",
    "number": "<NUMBER>",
    "name": "<NAME>",
}


@dataclass
class ScrubConfig:
    emails: bool = True
    phones: bool = True
    handles: bool = True
    urls: bool = True
    long_numbers: bool = False   # off by default: eats legitimate numbers in code
    names: list[str] = field(default_factory=list)

    @classmethod
    def from_names_file(cls, path, **kwargs) -> "ScrubConfig":
        """Load other people's names from a gitignored file, one per line."""
        from pathlib import Path

        names = []
        file = Path(path)
        if file.exists():
            names = [
                line.strip() for line in file.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")
            ]
        return cls(names=names, **kwargs)


def _name_pattern(names: list[str]) -> re.Pattern | None:
    cleaned = sorted({n.strip() for n in names if len(n.strip()) >= 3}, key=len, reverse=True)
    if not cleaned:
        return None
    # Longest first so "Anna Marie" wins over "Anna".
    return re.compile(r"\b(?:" + "|".join(re.escape(n) for n in cleaned) + r")\b", re.IGNORECASE)


def scrub(text: str, config: ScrubConfig | None = None) -> tuple[str, Counter]:
    """Return ``(redacted_text, counts_by_category)``."""
    config = config or ScrubConfig()
    counts: Counter = Counter()

    def replace(pattern: re.Pattern, kind: str, value: str) -> str:
        placeholder = PLACEHOLDERS[kind]
        value, n = pattern.subn(placeholder, value)
        if n:
            counts[kind] += n
        return value

    # Emails first: their @ would otherwise be consumed as a handle.
    if config.emails:
        text = replace(EMAIL, "email", text)
    if config.urls:
        text = replace(URL, "url", text)
    if config.handles:
        text = replace(HANDLE, "handle", text)
    if config.phones:
        for pattern in PHONE_PATTERNS:
            text = replace(pattern, "phone", text)
    if config.long_numbers:
        text = replace(LONG_NUMBER, "number", text)
    if pattern := _name_pattern(config.names):
        text = replace(pattern, "name", text)

    return text, counts


def scrub_documents(docs, config: ScrubConfig | None = None):
    """Scrub in place across an iterable of Documents, returning totals."""
    totals: Counter = Counter()
    out = []
    for doc in docs:
        doc.text, counts = scrub(doc.text, config)
        totals.update(counts)
        out.append(doc)
    return out, totals


NAMES_FILE_HEADER = [
    "# Other people's names, redacted from the personal corpus as <NAME>. One per line.",
    "# Generated from the export's participants -- review it: delete anything that is",
    "# not a name. Gitignored; never commit it.",
    "#",
    "# Names match case-insensitively, so a part that is also an ordinary word is written",
    "# commented out ('# common word: Grace' would otherwise redact every 'grace').",
    "# Uncomment one only if redacting that word everywhere is acceptable.",
]


def names_file_lines(names: Iterable[str], common_words: Collection[str] = ()) -> list[str]:
    """A names file for the scrubber: every full name, then every part of 3+ characters.

    ``common_words`` are lowercase ordinary words; a one-word entry found there
    is written commented out rather than dropped, so the reviewer sees it.
    """
    entries: dict[str, str] = {}
    for name in names:
        name = " ".join(name.split())
        candidates = [name] + re.findall(r"[\w'\u2019-]+", name)
        for entry in candidates:
            if len(entry) >= 3 and any(c.isalpha() for c in entry):
                entries.setdefault(entry.casefold(), entry)
    lines = list(NAMES_FILE_HEADER) + [""]
    for key, entry in sorted(entries.items()):
        common = " " not in entry and key in common_words
        lines.append(f"# common word: {entry}" if common else entry)
    return lines


def contains_obvious_pii(text: str) -> bool:
    """Cheap check used by tests and by the inventory's sample printer."""
    if EMAIL.search(text):
        return True
    return any(pattern.search(text) for pattern in PHONE_PATTERNS)
