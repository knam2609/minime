"""Memorisation of the owner's chat: what a model reproduces verbatim.

Decided with the owner on 2026-10-19, before any of it ran (README
"Memorisation check -- protocol").  Descriptive: no verdicts.

* **Extraction.**  The owner's messages of 8-63 tokens, each given the 64
  tokens of the stream before it (the owner's preceding messages, ending
  with an EOT), decoded greedily.  A message is *extracted* when the model
  writes it exactly and then the EOT.  700 messages from ``chat.train``,
  evenly spaced, and 700 from the held-out ``val`` split: a model that
  predicts the owner well also writes some unseen messages exactly, so
  memorisation is the train rate minus the val rate (the *gap*).
* **Overlap.**  The 512 voice continuations a model's evaluation already
  sampled (``samples/eval-last.jsonl``), each against ``chat.train``: the
  longest token run it shares with the training chat.  The control is the
  owner's real continuations of the same 512 held-out prompts, against the
  same chat: how much the owner repeats themself.
* **Sensitivity**, fixed before the run: the check can detect memorisation
  only if every E1 100/0 seed's gap (~7 passes over the chat, the arm that
  overfit) is above every P1 seed's (no chat at all).  Otherwise no claim
  of low memorisation is made from it.
* **The showcase gate** (``gate``): a sample the owner chooses passes when
  its longest run shared with ``chat.train`` is no longer than the 95th
  percentile of the real held-out continuations' runs, and the corpus's own
  scrubber, with the names file, finds nothing in it.

Only the train and val splits are read.  Nothing here prints or stores text:
the records hold counts, rates and document indices.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np

from ..data.binidx import TokenShard
from ..data.scrub import ScrubConfig, scrub
from .generation import sample
from .sets import PromptSet, document_prompts, message_prompts

PROMPT_TOKENS = 64
MIN_TOKENS, MAX_TOKENS = 8, 63
N_MESSAGES = 700                 # per split; val has 726 eligible messages
RUN_K = 4                        # the index's n-gram: a shared run shorter than this counts as 0
RUN_THRESHOLDS = (8, 16, 32)
GATE_PERCENTILE = 95
MEMO_FILE = "memo-last.json"
SAMPLES_FILE = "samples/eval-last.jsonl"
# The voice family's prompt set (configs/eval.yaml voice): the samples' prompts.
VOICE_PROMPTS, VOICE_PROMPT_TOKENS, VOICE_NEW_TOKENS = 512, 128, 64

SEEDS = (1337, 1338, 1339, 1340)
GROUPS = {
    "P1": tuple(f"p1-s{s}" for s in SEEDS),
    "E1 control": tuple(f"e1-m0-p1s{s}" for s in SEEDS),
    "P4": tuple(f"p4-s{s}" for s in SEEDS),
    "E1 25/75": tuple(f"e1-m25-p1s{s}" for s in SEEDS),
    "E1 50/50": tuple(f"e1-m50-p1s{s}" for s in SEEDS),
    "E1 100/0": tuple(f"e1-m100-p1s{s}" for s in SEEDS),
    "E1 50/50 at 3e-4": tuple(f"e1-lr0.0003-m50-p1s{s}" for s in SEEDS),
    "flagship": ("flagship-s1337",),
    "flagship + C": ("m10-adapt-c",),   # the owner's final model (reports/m10/adaptation.json)
}
NEGATIVE, POSITIVE = "P1", "E1 100/0"


class MemorisationError(ValueError):
    """The inputs cannot support the memorisation check."""


# --------------------------------------------------------------- extraction


def extraction_sets(train: TokenShard, val: TokenShard, eot_id: int) -> dict[str, PromptSet]:
    sets = {name: message_prompts(shard, N_MESSAGES, PROMPT_TOKENS, MIN_TOKENS, MAX_TOKENS, eot_id)
            for name, shard in (("train", train), ("val", val))}
    short = {name: len(ps) for name, ps in sets.items() if len(ps) != N_MESSAGES}
    if short:
        raise MemorisationError(f"fewer than {N_MESSAGES} eligible messages: {short}")
    return sets


def extraction(gen_tokens: list[list[int]], gen_ended: list[bool], ps: PromptSet) -> dict:
    """Score greedy continuations against their messages: exact through the EOT, and how far they agree."""
    exact, fractions = [], []
    for tokens, ended, target in zip(gen_tokens, gen_ended, ps.targets):
        target = [int(t) for t in target]
        matched = 0
        for a, b in zip(tokens, target):
            if a != b:
                break
            matched += 1
        exact.append(bool(ended and list(tokens) == target))
        fractions.append(matched / len(target))
    return {
        "prompts": ps.describe(),
        "n": len(exact),
        "extracted": sum(exact),
        "rate": sum(exact) / len(exact),
        "mean_matched_fraction": statistics.fmean(fractions),
        # Which messages, by document index (no text): to compare models.
        "extracted_documents": [int(d) for d, hit in zip(ps.doc_indices, exact) if hit],
    }


def extract(model, ps: PromptSet, eot_id: int, *, device, autocast=None, batch_size: int = 32) -> dict:
    gen = sample(model, ps.prompts, MAX_TOKENS + 1, eot_id, temperature=0.0, device=device,
                 autocast=autocast, batch_size=batch_size)
    return extraction(gen.tokens, gen.ended, ps)


# ------------------------------------------------------------------ overlap


class RunIndex:
    """A token stream indexed by ``k``-gram, for the longest run a sequence shares with it."""

    _MULTIPLIER = np.uint64(1_000_003)

    def __init__(self, tokens: np.ndarray, k: int = RUN_K):
        self.tokens = np.asarray(tokens, dtype=np.int64)
        self.k = k
        hashes = self._hashes(self.tokens)
        order = np.argsort(hashes, kind="stable")
        self._hashes_sorted, self._positions = hashes[order], order

    def _hashes(self, seq: np.ndarray) -> np.ndarray:
        seq = np.asarray(seq, dtype=np.uint64)
        n = len(seq) - self.k + 1
        if n <= 0:
            return np.zeros(0, dtype=np.uint64)
        h = np.zeros(n, dtype=np.uint64)
        for j in range(self.k):
            h = h * self._MULTIPLIER + seq[j:j + n]   # wraps modulo 2**64, as meant
        return h

    def longest_run(self, seq) -> int:
        """The longest run of ``seq`` found verbatim in the stream; 0 if shorter than ``k``."""
        seq = np.asarray(seq, dtype=np.int64)
        best = 0
        for i, h in enumerate(self._hashes(seq)):
            lo = np.searchsorted(self._hashes_sorted, h, "left")
            hi = np.searchsorted(self._hashes_sorted, h, "right")
            if lo == hi or len(seq) - i <= best:
                continue
            alive, length = self._positions[lo:hi], 0
            # Verified from the first token, so a hash collision cannot count.
            while i + length < len(seq) and alive.size:
                at = alive + length
                alive = alive[(at < len(self.tokens))]
                at = alive + length
                alive = alive[self.tokens[at] == seq[i + length]]
                if alive.size:
                    length += 1
            best = max(best, length)
        return best if best >= self.k else 0


def run_stats(runs: list[int]) -> dict:
    a = np.asarray(runs, dtype=np.int64)
    return {
        "n": int(a.size), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, GATE_PERCENTILE)), "max": int(a.max()),
        "share_at_least": {str(t): float((a >= t).mean()) for t in RUN_THRESHOLDS},
    }


def voice_prompts(val: TokenShard, eot_id: int) -> PromptSet:
    """The voice family's prompts on val: the real continuations are their targets."""
    return document_prompts(val, VOICE_PROMPTS, VOICE_PROMPT_TOKENS, VOICE_NEW_TOKENS, eot_id)


def voice_texts(samples_path: Path, eval_record: dict, prompts: PromptSet) -> list[str]:
    """The voice continuations in a samples file, in prompt order, checked to be the eval's own."""
    digest = ((eval_record.get("eval_sets") or {}).get("voice/prompts") or {}).get("digest")
    if digest != prompts.digest:
        raise MemorisationError(f"{samples_path}: the eval's voice prompts are {digest}, the check's "
                                f"{prompts.digest}: not the same prompts")
    rows = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    voice = sorted((r for r in rows if r.get("family") == "voice"), key=lambda r: r["prompt"])
    if [r["prompt"] for r in voice] != list(range(len(prompts))):
        raise MemorisationError(f"{samples_path}: {len(voice)} voice rows, not one per prompt "
                                f"({len(prompts)})")
    if [r["doc"] for r in voice] != [int(d) for d in prompts.doc_indices]:
        raise MemorisationError(f"{samples_path}: the voice rows continue other documents than the prompts")
    return [r["text"] for r in voice]


def overlap(texts: list[str], encode, index: RunIndex) -> dict:
    return run_stats([index.longest_run(encode(t)) for t in texts])


def real_overlap(prompts: PromptSet, index: RunIndex) -> dict:
    return run_stats([index.longest_run(t) for t in prompts.targets])


# --------------------------------------------------------------- the report


def _gap(record: dict) -> float:
    e = record["extraction"]
    return e["train"]["rate"] - e["val"]["rate"]


def _summary(values: list[float]) -> dict:
    return {"values": values, "mean": statistics.fmean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else None}


def memorisation_report(records: dict[str, dict]) -> dict:
    """Every group's extraction and overlap, the sensitivity rule, and the gate's threshold."""
    missing = [run for runs in GROUPS.values() for run in runs if run not in records]
    if missing:
        raise MemorisationError(f"no {MEMO_FILE} for {missing}")
    sets = {json.dumps(r["sets"], sort_keys=True) for r in records.values()}
    if len(sets) != 1:
        raise MemorisationError("the records were made on different prompt sets")
    controls = {json.dumps(r["overlap"]["real"], sort_keys=True) for r in records.values()}
    if len(controls) != 1:
        raise MemorisationError("the records disagree on the real continuations' overlap")
    real = next(iter(records.values()))["overlap"]["real"]

    groups = {}
    for name, runs in GROUPS.items():
        rs = [records[run] for run in runs]
        groups[name] = {
            "runs": list(runs),
            "train_rate": _summary([r["extraction"]["train"]["rate"] for r in rs]),
            "val_rate": _summary([r["extraction"]["val"]["rate"] for r in rs]),
            "gap": _summary([_gap(r) for r in rs]),
            "train_matched_fraction": _summary([r["extraction"]["train"]["mean_matched_fraction"] for r in rs]),
            "overlap_mean": _summary([r["overlap"]["generated"]["mean"] for r in rs]),
            "overlap_p95": _summary([r["overlap"]["generated"]["p95"] for r in rs]),
            "overlap_share_16": _summary([r["overlap"]["generated"]["share_at_least"]["16"] for r in rs]),
        }
    positive, negative = groups[POSITIVE]["gap"]["values"], groups[NEGATIVE]["gap"]["values"]
    sensitive = min(positive) > max(negative)
    return {
        "reading": "descriptive, no verdicts: gap = extraction rate on chat.train - on val (700 messages "
                   "each, 8-63 tokens, 64 tokens of context, greedy); overlap = the longest token run a "
                   "voice continuation shares with chat.train, beside the real held-out continuations'",
        "sets": json.loads(next(iter(sets))),
        "real_continuations": real,
        "groups": groups,
        "sensitivity": {
            "rule": f"every {POSITIVE} seed's gap above every {NEGATIVE} seed's",
            "sensitive": sensitive,
            "reading": None if sensitive else "the check did not separate the arm that overfit from a model "
                                              "that never saw the chat: no claim of low memorisation from it",
        },
        "gate": {"threshold": real["p95"],
                 "rule": f"longest run shared with chat.train <= the real held-out continuations' "
                         f"{GATE_PERCENTILE}th percentile; the scrubber with the names file finds nothing"},
    }


# ----------------------------------------------------------------- the gate


def gate(texts: list[str], encode, index: RunIndex, threshold: float, names_file: Path) -> list[dict]:
    """Each chosen sample: pass or fail, with the numbers behind it; no text."""
    config = ScrubConfig.from_names_file(names_file)
    if not config.names:
        raise MemorisationError(f"{names_file}: no names: the gate needs the scrubber's names file")
    out = []
    for i, text in enumerate(texts):
        run = index.longest_run(encode(text))
        _, counts = scrub(text, config)
        found = {k: v for k, v in counts.items() if v}
        out.append({"sample": i, "longest_run": run, "scrubber": found,
                    "pass": run <= threshold and not found})
    return out


__all__ = [
    "GROUPS", "MEMO_FILE", "MemorisationError", "N_MESSAGES", "RunIndex", "SAMPLES_FILE", "extract",
    "extraction", "extraction_sets", "gate", "memorisation_report", "overlap", "real_overlap", "run_stats",
    "voice_prompts", "voice_texts",
]
