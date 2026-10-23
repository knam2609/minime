"""Corpus census.

Step one of the personal-data pipeline, and the number every later decision
depends on: whether "me" can be mixed into pretraining or only used for
adaptation is a question about how many tokens exist, and nobody knows that
until it is counted.

Prints aggregates only.  The corpus itself never reaches a terminal, a log or
this repository.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from .document import Document
from .scrub import contains_obvious_pii
from .splits import DEFAULT_RATIOS, assign_split, split_key

# English byte-level BPE averages a little under 4 characters per token.
CHARS_PER_TOKEN = 3.9


@dataclass
class SourceReport:
    source: str
    documents: int
    characters: int
    words: int

    @property
    def est_tokens(self) -> int:
        return int(self.characters / CHARS_PER_TOKEN)

    @property
    def mean_chars(self) -> float:
        return self.characters / self.documents if self.documents else 0.0


def inventory(docs: Iterable[Document]) -> dict[str, SourceReport]:
    buckets: dict[str, list[int]] = {}
    counts: Counter = Counter()
    for doc in docs:
        entry = buckets.setdefault(doc.source, [0, 0])
        entry[0] += doc.n_chars
        entry[1] += doc.n_words
        counts[doc.source] += 1
    return {
        source: SourceReport(source, counts[source], chars, words)
        for source, (chars, words) in sorted(buckets.items())
    }


def split_preview(
    docs: Sequence[Document], ratios: dict[str, float] | None = None, groups: bool = False,
) -> dict[str, int]:
    """Document counts per split -- or, with ``groups``, split-group counts -- including empty ones.

    Reporting only non-empty buckets hides the case that matters: a corpus so
    small that the validation split came out empty, which silently breaks
    every later comparison.  Groups matter because they are the unit actually
    assigned: a thousand documents in val can still be three conversations.
    """
    ratios = ratios or DEFAULT_RATIOS
    counts: Counter = Counter({name: 0 for name in ratios})
    seen: set[str] = set()
    for doc in docs:
        key = split_key(doc)
        if groups and key in seen:
            continue
        seen.add(key)
        counts[assign_split(key, ratios)] += 1
    return dict(counts)


def pii_audit(docs: Sequence[Document]) -> dict[str, int]:
    """How many documents still trip the obvious-PII detector after scrubbing."""
    flagged = sum(1 for doc in docs if contains_obvious_pii(doc.text))
    return {"documents_checked": len(docs), "documents_flagged": flagged}


def format_table(reports: dict[str, SourceReport]) -> str:
    if not reports:
        return "no documents found"

    total_chars = sum(r.characters for r in reports.values())
    header = f"{'source':<12}{'docs':>9}{'chars':>14}{'words':>12}{'~tokens':>12}{'mean':>9}{'share':>8}"
    lines = [header, "-" * len(header)]
    for report in reports.values():
        share = report.characters / total_chars * 100 if total_chars else 0
        lines.append(
            f"{report.source:<12}{report.documents:>9,}{report.characters:>14,}"
            f"{report.words:>12,}{report.est_tokens:>12,}{report.mean_chars:>9.0f}{share:>7.1f}%"
        )

    total_docs = sum(r.documents for r in reports.values())
    total_words = sum(r.words for r in reports.values())
    total_tokens = sum(r.est_tokens for r in reports.values())
    lines.append("-" * len(header))
    lines.append(
        f"{'TOTAL':<12}{total_docs:>9,}{total_chars:>14,}{total_words:>12,}{total_tokens:>12,}"
    )
    return "\n".join(lines)


def verdict(total_tokens: int) -> str:
    """Turn the token count into the decision it actually drives."""
    if total_tokens < 1_000_000:
        return (
            "ADAPTATION ONLY. Under ~1M tokens the personal corpus cannot support "
            "mixing into pretraining; expect overfitting within a few hundred steps, "
            "and treat that as the finding rather than a problem to hide."
        )
    if total_tokens < 7_000_000:
        return (
            "ADAPTATION + REPLAY. Enough to adapt on and to mix a replayed fraction "
            "into late pretraining. The full mixture experiment is viable."
        )
    return (
        "FULL MIXTURE. Enough for a proper personal-data mixture sweep and a "
        "meaningful held-out personal test set."
    )
