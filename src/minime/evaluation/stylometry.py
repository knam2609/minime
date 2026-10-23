"""Stylometry: does generated text have the owner's measurable habits?

Perplexity says how well a model predicts the owner's text; stylometry asks
whether what it *writes* has the same surface statistics.  Two instruments:

* **Profiles and distances.**  A ``StyleProfile`` summarises a set of texts
  (length per document and per line, casing, emoji, punctuation, vocabulary
  richness, function-word use); ``compare`` turns two profiles into distances.
  Every distance is reported beside the same distance for a *real* held-out
  text and for a control author, because a number like "JSD 0.04" means
  nothing until you know what a genuine sample and an impostor score.
* **Burrows' Delta.**  The classic authorship-attribution measure: z-scored
  frequencies of the corpus's most frequent words, and the mean absolute
  z-difference between a questioned text and each candidate author.  Texts
  are pooled into fixed-size word chunks, because Delta on a ten-word chat
  message is noise.

The vocabulary is fixed by the *reference* texts, never by the text being
judged, so a model cannot shift the yardstick by what it generates.  Nothing
here returns words or text -- only numbers -- so results are safe to commit.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np

# Letter runs with internal apostrophes ("don't", "it's"); digits are not words.
WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")
PUNCTUATION = tuple(".,!?'\"’:;-()…~*/")
# The pictographic blocks (emoticons, symbols and pictographs, transport,
# supplemental, dingbats, misc symbols, regional-indicator flags).  An
# approximation of Unicode's Extended_Pictographic property without the
# third-party `regex` module; applied identically to every text compared.
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿⌀-⏿]")

FUNCTION_WORDS = tuple(sorted(set("""
a about above after again against all am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further
had has have having he her here hers herself him himself his how i if in into is it its itself
just me more most my myself no nor not now of off on once only or other our ours ourselves out
over own same she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when where which
while who whom why will with would you your yours yourself yourselves
""".split())))

MATTR_WINDOW = 50


def words(text: str) -> list[str]:
    return [w.lower().replace("’", "'") for w in WORD.findall(text)]


def mattr(tokens: Sequence[str], window: int = MATTR_WINDOW) -> float | None:
    """Moving-average type-token ratio: TTR averaged over every ``window``-word span.

    Plain TTR falls as text gets longer, so it would reward whichever sample
    happened to be shorter.  MATTR does not depend on length.
    """
    n = len(tokens)
    if n == 0:
        return None
    if n <= window:
        return len(set(tokens)) / n
    counts = Counter(tokens[:window])
    ratios = [len(counts) / window]
    for i in range(window, n):
        counts[tokens[i]] += 1
        old = tokens[i - window]
        counts[old] -= 1
        if counts[old] == 0:
            del counts[old]
        ratios.append(len(counts) / window)
    return float(np.mean(ratios))


@dataclass
class StyleProfile:
    docs: int
    words: int
    chars: int
    words_per_doc: float
    chars_per_doc: float
    words_per_line: float
    empty_doc_share: float
    mattr: float | None
    uppercase_share: float | None  # uppercase letters / cased letters
    emoji_per_100_chars: float
    docs_with_emoji: float
    punctuation_per_100_chars: float

    def scalars(self) -> dict:
        return asdict(self)


@dataclass
class _Counts:
    profile: StyleProfile
    doc_words: list[int]
    punctuation: Counter
    function_words: Counter
    words: list[str]


def _count(texts: Sequence[str]) -> _Counts:
    doc_words, all_words = [], []
    punct: Counter = Counter()
    lines = line_words = chars = upper = cased = emoji = docs_with_emoji = empty = 0
    for text in texts:
        ws = words(text)
        doc_words.append(len(ws))
        all_words.extend(ws)
        chars += len(text)
        empty += not text.strip()
        for line in text.splitlines() or [text]:
            if line.strip():
                lines += 1
                line_words += len(words(line))
        for c in text:
            if c.isupper():
                upper += 1
                cased += 1
            elif c.islower():
                cased += 1
            if c in PUNCTUATION:
                punct[c] += 1
        found = len(EMOJI.findall(text))
        emoji += found
        docs_with_emoji += found > 0

    n = len(texts)
    per100 = (lambda x: 100 * x / chars) if chars else (lambda x: 0.0)
    profile = StyleProfile(
        docs=n,
        words=len(all_words),
        chars=chars,
        words_per_doc=len(all_words) / n if n else 0.0,
        chars_per_doc=chars / n if n else 0.0,
        words_per_line=line_words / lines if lines else 0.0,
        empty_doc_share=empty / n if n else 0.0,
        mattr=mattr(all_words),
        uppercase_share=upper / cased if cased else None,
        emoji_per_100_chars=per100(emoji),
        docs_with_emoji=docs_with_emoji / n if n else 0.0,
        punctuation_per_100_chars=per100(sum(punct.values())),
    )
    fw = Counter(w for w in all_words if w in FUNCTION_WORDS)
    return _Counts(profile, doc_words, punct, fw, all_words)


def profile(texts: Sequence[str]) -> StyleProfile:
    return _count(texts).profile


def jensen_shannon(p: Counter, q: Counter, support: Iterable[str]) -> float | None:
    """JSD in bits between two count distributions over ``support`` (0 = identical, 1 = disjoint)."""
    support = list(support)
    a = np.array([p.get(k, 0) for k in support], dtype=np.float64)
    b = np.array([q.get(k, 0) for k in support], dtype=np.float64)
    if a.sum() == 0 or b.sum() == 0:
        return None
    a, b = a / a.sum(), b / b.sum()
    m = (a + b) / 2

    def kl(x: np.ndarray) -> float:
        mask = x > 0
        return float((x[mask] * np.log2(x[mask] / m[mask])).sum())

    return (kl(a) + kl(b)) / 2


def ks_statistic(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Two-sample Kolmogorov-Smirnov distance between two length distributions."""
    if not len(a) or not len(b):
        return None
    a, b = np.sort(np.asarray(a, dtype=np.float64)), np.sort(np.asarray(b, dtype=np.float64))
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.abs(cdf_a - cdf_b).max())


def compare(candidate: Sequence[str], reference: Sequence[str]) -> dict:
    """Distances from ``candidate`` texts to ``reference`` texts (smaller = closer)."""
    c, r = _count(candidate), _count(reference)
    cp, rp = c.profile, r.profile

    def gap(x, y):
        return None if x is None or y is None else abs(x - y)

    return {
        "function_word_jsd": jensen_shannon(c.function_words, r.function_words, FUNCTION_WORDS),
        "punctuation_jsd": jensen_shannon(c.punctuation, r.punctuation, PUNCTUATION),
        "words_per_doc_ks": ks_statistic(c.doc_words, r.doc_words),
        "words_per_line_gap": gap(cp.words_per_line, rp.words_per_line),
        "mattr_gap": gap(cp.mattr, rp.mattr),
        "uppercase_share_gap": gap(cp.uppercase_share, rp.uppercase_share),
        "emoji_per_100_chars_gap": gap(cp.emoji_per_100_chars, rp.emoji_per_100_chars),
        "punctuation_per_100_chars_gap": gap(cp.punctuation_per_100_chars, rp.punctuation_per_100_chars),
    }


# ------------------------------------------------------------ Burrows' Delta


def chunk_words(texts: Sequence[str], size: int) -> list[list[str]]:
    """Pool the texts' words in order and cut them into ``size``-word chunks.

    The trailing partial chunk is dropped: a short chunk has noisier
    frequencies and would be judged on a different footing from the rest.
    """
    pooled = [w for text in texts for w in words(text)]
    return [pooled[i : i + size] for i in range(0, len(pooled) - size + 1, size)]


def _frequencies(chunks: list[list[str]], vocab: list[str]) -> np.ndarray:
    index = {w: i for i, w in enumerate(vocab)}
    out = np.zeros((len(chunks), len(vocab)), dtype=np.float64)
    for row, chunk in enumerate(chunks):
        for w in chunk:
            j = index.get(w)
            if j is not None:
                out[row, j] += 1
        out[row] /= len(chunk)
    return out


def burrows_delta(
    authors: dict[str, Sequence[str]],
    questioned: dict[str, Sequence[str]],
    target: str,
    chunk_size: int = 300,
    mfw: int = 100,
) -> dict:
    """Delta from each questioned text set to each reference author.

    ``authors`` maps an author to reference texts; their chunks define the
    most frequent words and the z-score statistics, and each author's profile
    is the mean z-vector of its chunks.  Pass the same number of words per
    author, or the majority author dominates the z-scores.  For every
    questioned set, reports the mean Delta to each author and the share of its
    chunks whose nearest author is ``target``.
    """
    if target not in authors:
        raise KeyError(f"target author {target!r} is not among the reference authors")
    ref_chunks = {a: chunk_words(texts, chunk_size) for a, texts in authors.items()}
    for author, chunks in ref_chunks.items():
        if len(chunks) < 2:
            raise ValueError(f"author {author!r} has {len(chunks)} chunk(s) of {chunk_size} words; "
                             "Delta needs at least two per author")

    counts = Counter(w for chunks in ref_chunks.values() for chunk in chunks for w in chunk)
    # Ties broken alphabetically, so the vocabulary is a function of the texts alone.
    vocab = [w for w, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:mfw]]

    names = sorted(ref_chunks)
    ref = np.concatenate([_frequencies(ref_chunks[a], vocab) for a in names])
    mu, sd = ref.mean(axis=0), ref.std(axis=0, ddof=1)
    keep = sd > 0
    z_ref = (ref[:, keep] - mu[keep]) / sd[keep]
    bounds = np.cumsum([0] + [len(ref_chunks[a]) for a in names])
    centroids = {a: z_ref[bounds[i] : bounds[i + 1]].mean(axis=0) for i, a in enumerate(names)}

    results = {}
    for label, texts in questioned.items():
        chunks = chunk_words(texts, chunk_size)
        if not chunks:
            results[label] = {"chunks": 0, "delta": None, "nearest_is_target": None}
            continue
        z = (_frequencies(chunks, vocab)[:, keep] - mu[keep]) / sd[keep]
        deltas = {a: np.abs(z - centroids[a]).mean(axis=1) for a in names}
        stacked = np.stack([deltas[a] for a in names])
        nearest = np.array(names)[stacked.argmin(axis=0)]
        results[label] = {
            "chunks": len(chunks),
            "delta": {a: float(deltas[a].mean()) for a in names},
            "nearest_is_target": float((nearest == target).mean()),
        }
    return {
        "chunk_words": chunk_size,
        "mfw": int(keep.sum()),
        "reference_chunks": {a: len(ref_chunks[a]) for a in names},
        "questioned": results,
    }


def take_words(texts: Iterable[str], budget: int) -> list[str]:
    """Texts in order until ``budget`` words have been taken (for balanced references)."""
    out, taken = [], 0
    for text in texts:
        if taken >= budget:
            break
        out.append(text)
        taken += len(words(text))
    return out
