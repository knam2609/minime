#!/usr/bin/env python3
"""The one test-split phase of the M10 report (HANDOFF §4a *Test split*).

    python scripts/final_eval.py eval      # every evaluation on test, once
    python scripts/final_eval.py report    # -> reports/m10/test.json

Waits for the owner's frozen final choice (``reports/m10/adaptation.json``,
``final_choice_frozen``).  Then, with the eval settings of the val phase
unchanged: the full eval (``configs/eval_flagship.yaml``) of the unadapted
flagship and the chosen adaptation -> ``eval-last-test.json``; the common
512-token CE (``configs/eval_common512.yaml``) of every row of the report --
the probes' only test number, the same windows as their native CE ->
``eval-last-ctx512-test.json``.  Each file is written beside its checkpoint
(``$MINIME_DATA_ROOT``); one MPS job at a time (HANDOFF §6).  A file already
there is never redone: the test split is read once, so a re-run after an
interruption picks up where it stopped.  ``eval`` refuses a working tree with
uncommitted changes, so every test evaluation records a clean revision.

``report`` copies the files into ``experiments/<run>/`` (refusing one that
differs from a copy already there) and sets test beside val.  Test results
never trigger retuning or another choice: the report applies no rule.
``--dry-run`` prints the commands.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from minime.config import data_root  # noqa: E402
from minime.training.logging import git_revision  # noqa: E402
from minime.training.m10 import (  # noqa: E402
    TEST_EVAL_FILE,
    AdaptationError,
    adapt_eval_command,
    checkpoint_path,
    common_eval_command,
    final_report,
    final_report_runs,
)
from minime.training.noise_floor import PRIMARY_EVAL_METRICS  # noqa: E402
from minime.utils.throughput import format_duration  # noqa: E402

def read_adaptation(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def eval_all(root: Path, adaptation: dict | None, dry_run: bool) -> int:
    try:
        runs = final_report_runs(adaptation)
    except AdaptationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    revision = git_revision(REPO)
    if not dry_run and (revision.endswith("-dirty") or revision == "unknown"):
        print(f"error: not starting at revision {revision}: commit first, so every test evaluation "
              "records a clean one", file=sys.stderr)
        return 1
    missing = sorted({str(checkpoint_path(root, r)) for r, _ in runs if not checkpoint_path(root, r).exists()})
    if missing:
        print("error: not starting, no checkpoint at:\n  " + "\n  ".join(missing), file=sys.stderr)
        return 1
    for run, eval_file in runs:
        checkpoint = checkpoint_path(root, run)
        written = checkpoint.parent / eval_file
        if written.exists():
            print(f"{run}: {eval_file} read already, never redone")
            continue
        command = adapt_eval_command if eval_file == TEST_EVAL_FILE else common_eval_command
        cmd = command(checkpoint, repo=REPO, root=root, python=sys.executable, split="test")
        if dry_run:
            print(" ".join(cmd))
            continue
        started = time.monotonic()
        print(f"{run}: {eval_file} ...", flush=True)
        log = root / "runs" / "logs" / f"{run}.{eval_file.removeprefix('eval-last-').removesuffix('.json')}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as handle:
            handle.write("$ " + " ".join(cmd) + "\n")
            handle.flush()
            code = subprocess.run(cmd, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT).returncode
        if code != 0 or not written.exists():
            print(f"error: evaluating {run} failed (exit {code}); see {log}", file=sys.stderr)
            return 1
        print(f"{run}: {eval_file} in {format_duration(time.monotonic() - started)}")
    return 0


def collect(root: Path, experiments: Path, runs: list[tuple[str, str]]) -> None:
    """Copy each test file into ``experiments/<run>/``; a copy already there must be the same file."""
    for run, eval_file in runs:
        source = checkpoint_path(root, run).parent / eval_file
        here = experiments / run / eval_file
        if not source.exists():
            continue   # the SSD is not attached, or never evaluated: final_report says which
        if here.exists():
            if not filecmp.cmp(source, here, shallow=False):
                raise AdaptationError(f"{here} differs from {source}: the test split is read once")
            continue
        shutil.copy2(source, here)


def report(root: Path, experiments: Path, adaptation: dict | None, out: Path) -> int:
    try:
        runs = final_report_runs(adaptation)
        collect(root, experiments, runs)
        result = final_report(experiments, adaptation)
    except AdaptationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    personal, general = PRIMARY_EVAL_METRICS
    print(f"the test split (final choice {result['final_run']}); val beside it, no rule applied")
    for run, splits in result["full"]["runs"].items():
        print(f"  {run:16s} " + "  ".join(
            f"{split} personal {v[personal]:.4f} general-weighted {v[general]:.4f}" for split, v in splits.items()))
    table = result["common_context"]
    print(f"common-context CE on test: {table['context_length']}-token windows, "
          f"eval {table['eval']['eval_config_fingerprint']}")
    for group, g in table["groups"].items():
        sd = g["sd"] or {}
        spread = lambda m: f" (s {sd[m]:.4f})" if m in sd else ""  # noqa: E731
        print(f"  {group:10s} n={len(g['runs'])}  personal {g['mean'][personal]:.4f}{spread(personal)}  "
              f"general-weighted {g['mean'][general]:.4f}{spread(general)}")
    print("P4's contrasts, described (not verdicts; the verdicts are on val, reports/p4/verdicts.json):")
    for split in ("val", "test"):
        for name, metrics in result["p4_described"][split].items():
            print(f"  {split:4s} P4 - {name:6s} " + "  ".join(
                f"{m.split('.')[1]} {e['difference']:+.4f} ± {e['half_width']:.4f}" for m, e in metrics.items()))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"report -> {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=["eval", "report"])
    parser.add_argument("--root", type=Path, default=None, help="data root (default: $MINIME_DATA_ROOT)")
    parser.add_argument("--adaptation", type=Path, default=REPO / "reports" / "m10" / "adaptation.json",
                        help="the gate's report, with the owner's frozen choice")
    parser.add_argument("--experiments", type=Path, default=REPO / "experiments")
    parser.add_argument("--out", type=Path, default=REPO / "reports" / "m10" / "test.json")
    parser.add_argument("--dry-run", action="store_true", help="eval: print the commands, run nothing")
    args = parser.parse_args()
    root = args.root or data_root()
    adaptation = read_adaptation(args.adaptation)
    if args.phase == "eval":
        return eval_all(root, adaptation, args.dry_run)
    return report(root, args.experiments, adaptation, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
