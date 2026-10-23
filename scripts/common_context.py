#!/usr/bin/env python3
"""The M10 report's common-context CE: every row scored on 512-token windows (HANDOFF §4a).

    python scripts/common_context.py eval      # evaluate.py with configs/eval_common512.yaml, row by row
    python scripts/common_context.py report    # -> reports/m10/common512.json

Rows: P1, P4, E1 25/75 and E1 50/50 (four runs each), the unadapted flagship,
and the adaptation ``reports/m10/adaptation.json`` accepted, if any.  CE only,
on the val split.  Each row's ``eval-last-ctx512.json`` is written beside its
checkpoint (``$MINIME_DATA_ROOT/checkpoints/<run>/`` for the Kaggle runs,
``runs/<run>/`` for the laptop's) and copied into ``experiments/<run>/``; a
row already there is skipped, so the command can be re-run after an
interruption.  One MPS job at a time (HANDOFF §6).  ``report`` refuses rows
evaluated unlike each other, and a probe whose 512-token CE differs from its
native evaluation's (the same windows) by more than 1e-4.  Descriptive: the
table carries no verdicts.  ``--dry-run`` prints the commands.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from minime.config import data_root  # noqa: E402
from minime.training.m10 import (  # noqa: E402
    COMMON_EVAL_FILE,
    AdaptationError,
    checkpoint_path,
    common_context_table,
    common_eval_command,
    common_rows,
)
from minime.training.noise_floor import PRIMARY_EVAL_METRICS  # noqa: E402
from minime.utils.throughput import format_duration  # noqa: E402

ADAPTATION_REPORT = REPO / "reports" / "m10" / "adaptation.json"


def adaptation() -> dict | None:
    return json.loads(ADAPTATION_REPORT.read_text()) if ADAPTATION_REPORT.exists() else None


def eval_all(root: Path, dry_run: bool) -> int:
    runs = [run for group in common_rows(adaptation()).values() for run in group]
    missing = [str(checkpoint_path(root, r)) for r in runs if not checkpoint_path(root, r).exists()]
    if missing:
        print("error: not starting, no checkpoint at:\n  " + "\n  ".join(missing), file=sys.stderr)
        return 1
    for run in runs:
        here = REPO / "experiments" / run
        if (here / COMMON_EVAL_FILE).exists():
            print(f"{run}: evaluated already, skipped")
            continue
        checkpoint = checkpoint_path(root, run)
        cmd = common_eval_command(checkpoint, repo=REPO, root=root, python=sys.executable)
        if dry_run:
            print(" ".join(cmd))
            continue
        started = time.monotonic()
        print(f"{run}: evaluating ...", flush=True)
        log = root / "runs" / "logs" / f"{run}.ctx512.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as handle:
            handle.write("$ " + " ".join(cmd) + "\n")
            handle.flush()
            code = subprocess.run(cmd, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT).returncode
        written = checkpoint.parent / COMMON_EVAL_FILE
        if code != 0 or not written.exists():
            print(f"error: evaluating {run} failed (exit {code}); see {log}", file=sys.stderr)
            return 1
        shutil.copy2(written, here / COMMON_EVAL_FILE)
        print(f"{run}: evaluated in {format_duration(time.monotonic() - started)}")
    return 0


def report(out: Path) -> int:
    try:
        table = common_context_table(REPO / "experiments", adaptation())
    except AdaptationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    personal, general = PRIMARY_EVAL_METRICS
    print(f"common-context CE: {table['context_length']}-token val windows, eval {table['eval']['eval_config_fingerprint']}")
    for group, g in table["groups"].items():
        sd = g["sd"] or {}
        spread = lambda m: f" (s {sd[m]:.4f})" if m in sd else ""  # noqa: E731
        print(f"  {group:10s} n={len(g['runs'])}  personal {g['mean'][personal]:.4f}{spread(personal)}  "
              f"general-weighted {g['mean'][general]:.4f}{spread(general)}")
    print(f"  probes against their native evaluations: max |difference| "
          f"{table['native_agreement']['max_abs_difference']:.2e}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=2) + "\n")
    print(f"report -> {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=["eval", "report"])
    parser.add_argument("--root", type=Path, default=None, help="data root (default: $MINIME_DATA_ROOT)")
    parser.add_argument("--out", type=Path, default=REPO / "reports" / "m10" / "common512.json")
    parser.add_argument("--dry-run", action="store_true", help="eval: print the commands, run nothing")
    args = parser.parse_args()
    if args.phase == "eval":
        return eval_all(args.root or data_root(), args.dry_run)
    return report(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
