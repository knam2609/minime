#!/usr/bin/env python3
"""The memorisation check (README "Memorisation check -- protocol").

    python scripts/memorisation.py run       # every checkpoint of GROUPS, one at a time
    python scripts/memorisation.py report    # -> reports/memorisation/memorisation.json

``run``: for each checkpoint, greedy extraction of 700 ``chat.train`` and 700
``val`` messages (MPS, one job at a time: HANDOFF §6), and the overlap of the
voice continuations its evaluation sampled (CPU).  Each writes
``memo-last.json`` beside its checkpoint (``$MINIME_DATA_ROOT``), copied into
``experiments/<run>/``; a run already there is skipped, so the command picks
up where an interruption stopped it.  Numbers and document indices only: no
text is printed or stored.  ``report`` sets the groups side by side, applies
the sensitivity rule fixed before the run, and records the showcase gate's
threshold.  Only the train and val splits are read.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from minime.config import data_root  # noqa: E402
from minime.data.binidx import TokenShard  # noqa: E402
from minime.evaluation.harness import load_model  # noqa: E402
from minime.evaluation.memorisation import (  # noqa: E402
    GROUPS,
    MEMO_FILE,
    SAMPLES_FILE,
    MemorisationError,
    RunIndex,
    extract,
    extraction_sets,
    memorisation_report,
    overlap,
    real_overlap,
    voice_prompts,
    voice_texts,
)
from minime.tokenizer import load_tokenizer  # noqa: E402
from minime.training.m10 import checkpoint_path  # noqa: E402
from minime.training.noise_floor import EVAL_FILE  # noqa: E402
from minime.utils.device import autocast_ctx, describe, resolve  # noqa: E402
from minime.utils.throughput import format_duration  # noqa: E402

TOKENIZER = REPO / "tokenizer" / "minime-16k.json"
REPORT = REPO / "reports" / "memorisation" / "memorisation.json"


def run_all(root: Path, runs: list[str], device: str, dry_run: bool) -> int:
    missing = [str(checkpoint_path(root, r)) for r in runs if not checkpoint_path(root, r).exists()]
    missing += [str(checkpoint_path(root, r).parent / SAMPLES_FILE) for r in runs
                if not (checkpoint_path(root, r).parent / SAMPLES_FILE).exists()]
    if missing:
        print("error: not starting, missing:\n  " + "\n  ".join(missing), file=sys.stderr)
        return 1
    todo = [r for r in runs if not (REPO / "experiments" / r / MEMO_FILE).exists()]
    for r in sorted(set(runs) - set(todo)):
        print(f"{r}: {MEMO_FILE} there already, skipped")
    if dry_run or not todo:
        print("would run: " + ", ".join(todo) if todo else "nothing to run")
        return 0

    tokenizer = load_tokenizer(TOKENIZER)
    personal = root / "shards" / "personal"
    train, val = TokenShard(personal / "chat.train"), TokenShard(personal / "val")
    sets = extraction_sets(train, val, tokenizer.eot_id)
    prompts = voice_prompts(val, tokenizer.eot_id)
    index = RunIndex(train.tokens)
    real = real_overlap(prompts, index)
    spec = resolve(device)
    for run in todo:
        checkpoint = checkpoint_path(root, run)
        eval_record = json.loads((checkpoint.parent / EVAL_FILE).read_text())
        started = time.monotonic()
        print(f"{run}: ...", flush=True)
        model, info = load_model(checkpoint, spec.device)
        evaluated = eval_record.get("checkpoint") or {}
        if (evaluated.get("config_fingerprint"), evaluated.get("step")) != (info["config_fingerprint"], info["step"]):
            raise MemorisationError(f"{checkpoint.parent / EVAL_FILE} evaluates another checkpoint: its "
                                    "samples are not this model's")
        texts = voice_texts(checkpoint.parent / SAMPLES_FILE, eval_record, prompts)
        record = {
            "checkpoint": {k: info[k] for k in ("step", "config_name", "config_fingerprint")},
            "tokenizer": tokenizer.id,
            "device": describe(spec),
            "sets": {name: ps.describe() for name, ps in sets.items()} | {"voice": prompts.describe()},
            "extraction": {name: extract(model, ps, tokenizer.eot_id, device=spec.device,
                                         autocast=lambda: autocast_ctx(spec))
                           for name, ps in sets.items()},
            "overlap": {"generated": overlap(texts, tokenizer.encode, index), "real": real},
            "seconds": round(time.monotonic() - started, 1),
        }
        del model
        out = checkpoint.parent / MEMO_FILE
        out.write_text(json.dumps(record, indent=2) + "\n")
        here = REPO / "experiments" / run
        here.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out, here / MEMO_FILE)
        e, o = record["extraction"], record["overlap"]["generated"]
        print(f"{run}: extracted train {e['train']['extracted']}/{e['train']['n']}, val "
              f"{e['val']['extracted']}/{e['val']['n']}; overlap mean {o['mean']:.2f}, p95 {o['p95']:.0f} "
              f"(real {real['mean']:.2f}, {real['p95']:.0f}) in {format_duration(time.monotonic() - started)}",
              flush=True)
    return 0


def report(out: Path) -> int:
    records = {}
    for runs in GROUPS.values():
        for run in runs:
            path = REPO / "experiments" / run / MEMO_FILE
            if path.exists():
                records[run] = json.loads(path.read_text())
    try:
        result = memorisation_report(records)
    except MemorisationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    real = result["real_continuations"]
    print(f"real held-out continuations: longest shared run mean {real['mean']:.2f}, p95 {real['p95']:.0f}")
    for name, g in result["groups"].items():
        sd = g["gap"]["sd"]
        print(f"  {name:18s} gap {g['gap']['mean']:+.4f}{f' (s {sd:.4f})' if sd is not None else ''}  "
              f"train {g['train_rate']['mean']:.4f}  val {g['val_rate']['mean']:.4f}  "
              f"overlap mean {g['overlap_mean']['mean']:.2f}  p95 {g['overlap_p95']['mean']:.1f}")
    print(f"sensitive: {result['sensitivity']['sensitive']} ({result['sensitivity']['rule']})")
    print(f"gate threshold: {result['gate']['threshold']:.0f} tokens")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"report -> {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=["run", "report"])
    parser.add_argument("--runs", nargs="*", default=None, help="a subset of the groups' runs")
    parser.add_argument("--root", type=Path, default=None, help="data root (default: $MINIME_DATA_ROOT)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=REPORT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.phase == "report":
        return report(args.out)
    every = [run for runs in GROUPS.values() for run in runs]
    runs = args.runs if args.runs is not None else every
    unknown = sorted(set(runs) - set(every))
    if unknown:
        parser.error(f"not in the check's groups: {unknown}")
    try:
        return run_all(args.root or data_root(), runs, args.device, args.dry_run)
    except MemorisationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
