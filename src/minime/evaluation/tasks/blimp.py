"""BLiMP: grammatical knowledge by minimal pairs, no generation needed.

Each of BLiMP's 67 paradigms holds 1,000 sentence pairs that differ in one
grammatical point ("The cats *are*/*is* sleeping").  The model is right when
it gives the grammatical sentence the higher total log-probability.  It needs
no sampling and no instruction following, which is why it is meaningful at
17M parameters when HellaSwag and LAMBADA are not.

The data is fetched once by ``scripts/prepare_eval.py`` from a pinned
revision into ``data_root()/eval/blimp.jsonl``, and ``load_pairs`` refuses a
file whose hash differs from the pinned one: an edited or partial file cannot
silently change the benchmark.  Accuracy is reported per paradigm, averaged
into per-field and overall (macro) scores, so a paradigm with more pairs
counts no more than one with fewer.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from ...data.sources.hf import HFSource

SOURCE = HFSource(
    dataset="nyu-mll/blimp",
    config=None,
    splits=("train",),
    revision="877fba0801ffb7cbd8c39c1ff314a46f053f6036",
    license="cc-by-4.0",
)
FIELDS = ("sentence_good", "sentence_bad", "field", "linguistics_term", "UID", "pair_id")
# sha256 of the jsonl ``download`` writes from SOURCE.revision (verified
# byte-identical across two downloads, 2026-09-28); ``load_pairs`` refuses
# anything else.
PINNED_SHA256: str | None = "690c1685acc338d86fb0a93d2285769fe24e583da520ad3727eeccd37290a4ce"


def default_path() -> Path:
    from ...config import data_root

    return data_root() / "eval" / "blimp.jsonl"


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(out_path: str | Path, log=print) -> dict:
    """Fetch every paradigm at the pinned revision into one sorted jsonl."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    from ...config import data_root

    cache = data_root() / "cache" / "huggingface"
    files = sorted(
        f for f in HfApi().list_repo_files(SOURCE.dataset, repo_type="dataset", revision=SOURCE.revision)
        if f.endswith(".parquet")
    )
    if not files:
        raise RuntimeError(f"{SOURCE.dataset}@{SOURCE.revision[:10]} lists no parquet files")
    rows = []
    for name in files:
        local = hf_hub_download(SOURCE.dataset, name, repo_type="dataset",
                                revision=SOURCE.revision, cache_dir=str(cache))
        table = pq.read_table(local)
        missing = set(FIELDS) - set(table.column_names)
        if missing:
            from ...data.sources.hf import SchemaDriftError

            raise SchemaDriftError(f"{name}: missing field(s) {sorted(missing)}")
        for row in table.select(list(FIELDS)).to_pylist():
            rows.append({
                # One paradigm spells its field "syntax/semantics", the rest
                # "syntax_semantics"; one spelling keeps the per-field means whole.
                "paradigm": row["UID"], "field": row["field"].replace("/", "_"),
                "phenomenon": row["linguistics_term"],
                "pair_id": int(row["pair_id"]), "good": row["sentence_good"], "bad": row["sentence_bad"],
            })
    rows.sort(key=lambda r: (r["paradigm"], r["pair_id"]))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(out_path)
    summary = {"path": str(out_path), "pairs": len(rows), "paradigms": len(files),
               "sha256": file_sha256(out_path), **SOURCE.describe()}
    log(f"BLiMP: {summary['pairs']:,} pairs in {summary['paradigms']} paradigms -> {out_path}")
    return summary


def load_pairs(path: str | Path | None = None, per_paradigm: int | None = None,
               expected_sha256: str | None = "pinned") -> tuple[list[dict], str]:
    """The first ``per_paradigm`` pairs (by pair id) of every paradigm, and the file hash.

    ``expected_sha256`` defaults to ``PINNED_SHA256`` (read at call time);
    ``None`` skips the check, for hand-made test files.
    """
    path = Path(path) if path else default_path()
    if not path.exists():
        raise FileNotFoundError(f"no BLiMP data at {path}; fetch it with `python scripts/prepare_eval.py`")
    digest = file_sha256(path)
    if expected_sha256 == "pinned":
        expected_sha256 = PINNED_SHA256
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"{path} has sha256 {digest[:16]}..., not the pinned {expected_sha256[:16]}...; "
                         "re-fetch it with scripts/prepare_eval.py")
    by_paradigm: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            by_paradigm[row["paradigm"]].append(row)
    pairs = []
    for paradigm in sorted(by_paradigm):
        rows = sorted(by_paradigm[paradigm], key=lambda r: r["pair_id"])
        pairs.extend(rows[:per_paradigm] if per_paradigm else rows)
    return pairs, digest


@torch.no_grad()
def sentence_logprobs(
    model: torch.nn.Module,
    token_lists: Sequence[Sequence[int]],
    bos_id: int,
    batch_size: int = 64,
    device: torch.device | str = "cpu",
    autocast=None,
) -> list[float]:
    """Total log-probability of each token list, conditioned on a leading ``bos_id``.

    Rows are right-padded; a causal model never attends to the padding of a
    later position, and padded targets are masked out of the sum.
    """
    autocast = autocast or nullcontext
    context = model.cfg.context_length
    was_training = model.training
    model.eval()
    out: list[float] = []
    try:
        for b in range(0, len(token_lists), batch_size):
            batch = [[bos_id, *ids][: context + 1] for ids in token_lists[b : b + batch_size]]
            width = max(len(ids) for ids in batch)
            x = torch.full((len(batch), width - 1), bos_id, dtype=torch.long)
            y = torch.full((len(batch), width - 1), -100, dtype=torch.long)
            for row, ids in enumerate(batch):
                x[row, : len(ids) - 1] = torch.tensor(ids[:-1])
                y[row, : len(ids) - 1] = torch.tensor(ids[1:])
            x, y = x.to(device), y.to(device)
            with autocast():
                logits, _ = model(x)
            nll = F.cross_entropy(logits.float().transpose(1, 2), y, ignore_index=-100, reduction="none")
            out.extend((-nll.sum(dim=1)).cpu().double().tolist())
    finally:
        if was_training:
            model.train()
    return out


def evaluate(
    model: torch.nn.Module,
    tokenizer,
    pairs: Sequence[dict],
    batch_size: int = 64,
    device: torch.device | str = "cpu",
    autocast=None,
) -> dict:
    """Macro accuracy over paradigms, per field, and the per-paradigm table."""
    sentences = [s for pair in pairs for s in (pair["good"], pair["bad"])]
    scores = sentence_logprobs(model, tokenizer.encode_batch(sentences), tokenizer.eot_id,
                               batch_size, device, autocast)
    correct: dict[str, list[bool]] = defaultdict(list)
    field_of: dict[str, str] = {}
    for i, pair in enumerate(pairs):
        correct[pair["paradigm"]].append(scores[2 * i] > scores[2 * i + 1])
        field_of[pair["paradigm"]] = pair["field"]
    paradigms = {p: sum(v) / len(v) for p, v in sorted(correct.items())}
    fields: dict[str, list[float]] = defaultdict(list)
    for p, acc in paradigms.items():
        fields[field_of[p]].append(acc)
    return {
        "accuracy": sum(paradigms.values()) / len(paradigms) if paradigms else None,
        "pairs": len(pairs),
        "paradigms": len(paradigms),
        "by_field": {f: sum(v) / len(v) for f, v in sorted(fields.items())},
        "by_paradigm": paradigms,
    }
