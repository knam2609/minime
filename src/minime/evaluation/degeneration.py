"""Degeneration: repetition and collapse in generated text.

Small models fail by looping ("i think i think i think") or by collapsing to a
few safe outputs.  Perplexity can look fine while either happens, so these are
measured directly on sampled continuations, always beside the same numbers for
the *real* continuations of the same prompts:

* ``rep_n``      -- within one continuation, the share of its n-grams that
  repeat an earlier n-gram of that continuation (Welleck et al., 2020).
  Higher = loopier.
* ``distinct_n`` -- across all continuations, unique n-grams over total
  n-grams (Li et al., 2016).  Lower = less diverse.
* ``self_bleu``  -- mean BLEU-4 of each continuation against all the others
  (Zhu et al., 2018).  Higher = the samples resemble each other.

Next-token entropy comes from ``generation.sample``.  N-grams are over words
and punctuation marks, so the numbers do not depend on the tokenizer.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

TOKEN = re.compile(r"\w+|[^\w\s]")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text)


def ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def rep_n(texts: Sequence[str], n: int) -> float | None:
    """Mean over texts with at least one n-gram of ``1 - unique/total``."""
    values = []
    for text in texts:
        grams = ngrams(tokenize(text), n)
        if grams:
            values.append(1 - len(set(grams)) / len(grams))
    return sum(values) / len(values) if values else None


def distinct_n(texts: Sequence[str], n: int) -> float | None:
    grams = [g for text in texts for g in ngrams(tokenize(text), n)]
    return len(set(grams)) / len(grams) if grams else None


def self_bleu(texts: Sequence[str], max_n: int = 4, epsilon: float = 0.1) -> float | None:
    """Mean BLEU of each text against every other text as references.

    Corpus-free BLEU with Chen & Cherry's smoothing 1 (an ``epsilon`` count for
    an empty n-gram order), uniform weights and the closest-length brevity
    penalty.  Texts shorter than ``max_n`` tokens are skipped as hypotheses
    but still serve as references.  Linear in the total number of n-grams: for
    each n-gram only the two largest per-text counts are kept, which is all
    "max count over the *other* texts" needs.
    """
    tokens = [tokenize(t) for t in texts]
    if len(tokens) < 2:
        return None
    counts = [[Counter(ngrams(toks, n)) for n in range(1, max_n + 1)] for toks in tokens]

    # For every n-gram: (largest count, text holding it, second-largest count).
    top: list[dict[tuple[str, ...], tuple[int, int, int]]] = [{} for _ in range(max_n)]
    for i, per_order in enumerate(counts):
        for order, counter in enumerate(per_order):
            table = top[order]
            for gram, c in counter.items():
                best, holder, second = table.get(gram, (0, -1, 0))
                if c > best:
                    table[gram] = (c, i, best)
                elif c > second:
                    table[gram] = (best, holder, c)

    lengths = sorted((len(t), i) for i, t in enumerate(tokens))
    scores = []
    for i, toks in enumerate(tokens):
        if len(toks) < max_n:
            continue
        log_precision = 0.0
        for order in range(max_n):
            total = sum(counts[i][order].values())
            clipped = 0
            for gram, c in counts[i][order].items():
                best, holder, second = top[order][gram]
                clipped += min(c, second if holder == i else best)
            log_precision += math.log((clipped or epsilon) / total)
        ref_len = min((abs(n - len(toks)), n) for n, j in lengths if j != i)[1]
        bp = 1.0 if len(toks) > ref_len else math.exp(1 - ref_len / len(toks))
        scores.append(bp * math.exp(log_precision / max_n))
    return sum(scores) / len(scores) if scores else None


def summarise(texts: Sequence[str], orders: Sequence[int] = (2, 3, 4)) -> dict:
    return {
        "texts": len(texts),
        "mean_words": sum(len(tokenize(t)) for t in texts) / len(texts) if texts else 0.0,
        **{f"rep_{n}": rep_n(texts, n) for n in orders},
        **{f"distinct_{n}": distinct_n(texts, n) for n in (1, 2, 3)},
        "self_bleu": self_bleu(texts),
    }
