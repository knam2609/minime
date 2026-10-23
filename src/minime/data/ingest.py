"""Stream documents straight into token shards: no raw text is ever written.

``ingest_slice`` consumes a document stream, assigns each document to
train/val/test by hashing its id (the same rule as the personal corpus, so the
held-out sets stay held out), tokenises in batches and appends to three
shards, stopping once the *train* shard reaches its token target.  Documents
are never cut to hit the target exactly.  Batches shrink as the target nears,
so the overshoot is about one document, not one batch -- a 512-file batch of
code is ~1.5M tokens, which overshot a 1M smoke target by 61%.

Two properties the rest of the project depends on:

* **Determinism.**  Pinned dataset revision + stream order + hash split +
  frozen tokenizer means a re-run produces byte-identical shards, and in
  particular the same validation shard -- which is what makes the fixed
  validation windows fixed across runs.
* **No half-written shards.**  A slice is written under ``_incomplete/`` and
  moved into place only when all three of its shards are closed.  A crashed
  ingestion leaves nothing a training run could mistake for data.  Completed
  slices are skipped on a re-run, so a network failure costs one slice, not
  the whole corpus.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

from .binidx import ShardMeta, ShardWriter, TokenShard
from .document import Document
from .splits import _fraction, assign_split, split_key

# The general corpus is large, so 0.5% each is ~4.5M tokens of val and of
# test at full scale -- far more than the fixed validation windows read.
GENERAL_RATIOS = {"train": 0.99, "val": 0.005, "test": 0.005}
SPLITS = ("train", "val", "test")
_SHARD_FILES = (".bin", ".idx.npy", ".meta.json")  # meta last: its presence means complete


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- disk space


def estimate_bytes(targets: dict[str, int], ratios: dict[str, float] = GENERAL_RATIOS) -> int:
    """Upper-ish estimate of what an ingestion will write, in bytes."""
    held_out = (ratios["val"] + ratios["test"]) / ratios["train"]
    tokens = sum(targets.values()) * (1 + 2 * held_out)  # x2: per-slice + combined val/test
    return int(tokens * 2 * 1.05) + 64 * 2**20  # uint16, 5% overshoot, index + slack


def check_free_space(out_dir: str | Path, need_bytes: int) -> None:
    out_dir = Path(out_dir)
    probe = out_dir
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if free < need_bytes:
        raise RuntimeError(
            f"{out_dir} has {free / 2**30:.1f} GiB free but this ingestion needs about "
            f"{need_bytes / 2**30:.1f} GiB. Point MINIME_DATA_ROOT (or --out) at the "
            "external drive."
        )


# ------------------------------------------------------------ one slice


def slice_paths(out_dir: str | Path, name: str) -> dict[str, Path]:
    return {split: Path(out_dir) / f"{name}.{split}" for split in SPLITS}


def slice_settings(
    source: dict,
    tokenizer_id: str,
    target_tokens: int,
    ratios: dict[str, float] = GENERAL_RATIOS,
    salt: str = "minime",
) -> dict:
    """Everything that determines a slice's shards.  Equal settings, equal bytes."""
    return {
        "source": source,
        "tokenizer": tokenizer_id,
        "target_tokens": int(target_tokens),
        "ratios": dict(ratios),
        "salt": salt,
    }


def completed_slice(out_dir: str | Path, name: str, settings: dict) -> dict | None:
    """The slice's recorded summary if it was finished with identical settings."""
    metas = {}
    for split, stem in slice_paths(out_dir, name).items():
        meta_path = Path(f"{stem}.meta.json")
        if not meta_path.exists() or not Path(f"{stem}.bin").exists():
            return None
        metas[split] = json.loads(meta_path.read_text())
    extra = metas["train"].get("extra") or {}
    if not extra.get("complete") or extra.get("settings") != settings:
        return None
    return extra.get("summary")


def ingest_slice(
    docs: Iterable[Document],
    tokenizer,
    out_dir: str | Path,
    name: str,
    target_tokens: int,
    source: dict | None = None,
    stats: Counter | None = None,
    ratios: dict[str, float] = GENERAL_RATIOS,
    salt: str = "minime",
    batch_docs: int = 512,
    log: Callable[[str], None] = print,
    log_every: float = 30.0,
) -> dict:
    """Tokenise ``docs`` into ``<name>.{train,val,test}`` until train reaches ``target_tokens``.

    ``source`` describes the dataset (id, revision, ...); together with the
    tokenizer, target and split rule it forms the slice's settings, and a later
    run whose settings are equal can skip the slice (``completed_slice``).
    ``stats`` is the source's own counter of kept/skipped rows, recorded so a
    filter that silently discarded most of a source would show up.
    """
    out_dir = Path(out_dir)
    stats = stats if stats is not None else Counter()
    settings = slice_settings(source or {}, tokenizer.id, target_tokens, ratios, salt)

    staging = out_dir / "_incomplete" / name
    if staging.exists():
        shutil.rmtree(staging)  # a previous run died here; nothing in it is trustworthy
    staging.mkdir(parents=True)

    writers = {
        split: ShardWriter(
            staging / f"{name}.{split}",
            vocab_size=tokenizer.vocab_size,
            tokenizer=tokenizer.id,
            eot_id=tokenizer.eot_id,
        )
        for split in SPLITS
    }
    counts = {split: Counter() for split in SPLITS}
    started = last_log = time.monotonic()

    def flush(batch: list[tuple[str, Document]]) -> None:
        for (split, doc), ids in zip(batch, tokenizer.encode_batch([d.text for _, d in batch])):
            writers[split].add_document(ids)
            counts[split]["docs"] += 1
            counts[split]["chars"] += len(doc.text)
        batch.clear()

    def chars_per_token() -> float:
        # Measured on this slice so far.  Before any data, 1.0 is conservative
        # for byte-level BPE on most text (CJK can go lower; the cost is only a
        # few extra small batches).
        tokens = sum(w.n_tokens for w in writers.values())
        return sum(c["chars"] for c in counts.values()) / tokens if tokens else 1.0

    batch: list[tuple[str, Document]] = []
    batch_chars = 0
    exhausted = True
    for doc in docs:
        batch.append((assign_split(split_key(doc), ratios, salt), doc))
        batch_chars += len(doc.text)
        # Flush early once this batch would fill half of what the target still
        # needs, so batches shrink geometrically as the target nears.
        remaining = target_tokens - writers["train"].n_tokens
        if len(batch) < batch_docs and batch_chars < 0.5 * remaining * chars_per_token():
            continue
        flush(batch)
        batch_chars = 0
        now = time.monotonic()
        done = writers["train"].n_tokens
        if now - last_log >= log_every:
            last_log = now
            rate = done / (now - started)
            eta = (target_tokens - done) / rate if rate else float("inf")
            log(f"  {name:9s} {done / 1e6:9.1f}M / {target_tokens / 1e6:.1f}M train tokens "
                f"({100 * done / target_tokens:5.1f}%)  {rate / 1e3:6.0f}k tok/s  eta {eta / 60:5.1f} min")
        if done >= target_tokens:
            exhausted = False
            break
    if batch:
        flush(batch)
        exhausted = exhausted and writers["train"].n_tokens < target_tokens

    metas: dict[str, ShardMeta] = {split: writers[split].close() for split in SPLITS}
    summary = {
        "target_tokens": int(target_tokens),
        "exhausted": exhausted,
        "seconds": round(time.monotonic() - started, 1),
        "splits": {
            split: {"tokens": metas[split].n_tokens, **counts[split]} for split in SPLITS
        },
        "source_stats": dict(stats),
    }
    if exhausted:
        log(f"  WARNING: {name} ran out of documents at {metas['train'].n_tokens:,} train tokens, "
            f"short of the {target_tokens:,} target. The shortfall is recorded in the manifest.")

    # Rewrite each meta with the split, settings and (on train) the summary,
    # then move the finished slice into place, meta files last.
    for split in SPLITS:
        meta = metas[split]
        extra = {"split": split, "slice": name}
        if split == "train":
            extra.update(complete=True, settings=settings, summary=summary)
        payload = {**meta.to_dict(), "extra": extra}
        Path(f"{staging / f'{name}.{split}'}.meta.json").write_text(json.dumps(payload, indent=2))

    finals = slice_paths(out_dir, name)
    for split in SPLITS:
        Path(f"{finals[split]}.meta.json").unlink(missing_ok=True)
    for suffix in _SHARD_FILES:
        for split in SPLITS:
            os.replace(f"{staging / f'{name}.{split}'}{suffix}", f"{finals[split]}{suffix}")
    shutil.rmtree(staging)
    try:
        (out_dir / "_incomplete").rmdir()
    except OSError:
        pass  # another slice is still staged there
    return summary


# ------------------------------------------------------------ combining


def combine_shards(
    parts: Sequence[str | Path],
    out_path: str | Path,
    tokenizer_id: str,
    extra: dict | None = None,
) -> ShardMeta:
    """Concatenate shards document by document into one (e.g. the general ``val``).

    The parts already end every document with EOT, so nothing is appended.
    """
    shards = [TokenShard(p) for p in parts]
    ids = {s.meta.get("tokenizer", "") for s in shards}
    if len(ids) != 1:
        raise ValueError(f"cannot combine shards made by different tokenizers: {sorted(ids)}")
    vocab = {s.meta.get("vocab_size") for s in shards}
    if len(vocab) != 1:
        raise ValueError(f"cannot combine shards with different vocab sizes: {sorted(vocab)}")
    with ShardWriter(out_path, vocab_size=vocab.pop(), tokenizer=tokenizer_id, extra=extra) as writer:
        for shard in shards:
            writer.add_documents(shard.documents())
        return writer.close()


# --------------------------------------------------- tokenizer training sample


def sample_documents(
    docs: Iterable[Document],
    char_budget: int,
    seed: int,
    keep_fraction: float = 0.5,
    ratios: dict[str, float] = GENERAL_RATIOS,
    salt: str = "minime",
) -> Iterator[Document]:
    """A deterministic tokenizer-training sample: ``f(stream, seed)``.

    Only *train*-split documents are eligible, so the tokenizer's merges are
    never fitted to held-out text.  Within those, the seed picks a
    ``keep_fraction`` subset by hashing, in stream order, until ``char_budget``
    characters have been taken.  No RNG state, so the sample is reproducible
    from the arguments alone.
    """
    if not 0 < keep_fraction <= 1:
        raise ValueError("keep_fraction must be in (0, 1]")
    taken = 0
    for doc in docs:
        if taken >= char_budget:
            return
        if assign_split(split_key(doc), ratios, salt) != "train":
            continue
        if keep_fraction < 1 and _fraction(doc.id, f"tokenizer-sample:{seed}") >= keep_fraction:
            continue
        taken += len(doc.text)
        yield doc
