#!/usr/bin/env python3
"""Run E1 on the laptop: the LR pilot, the main runs, then their evaluations.

    python scripts/run_e1.py pilot                   # e1-pilot-{0.0001,0.0003,0.001}
    python scripts/pick_e1_lr.py experiments/e1-pilot-*
    python scripts/run_e1.py pilot --extend 0.003    # only if the pick says so (exit 2)
    # set training.learning_rate in configs/experiments/e1_base.yaml to the pick
    python scripts/run_e1.py main                    # 16 runs: 4 arms x 4 P1 bases
    python scripts/run_e1.py eval                    # evaluate the 16 final checkpoints
    python scripts/e1_paired.py experiments/e1-m*-p1s*
    python scripts/run_e1.py followup                # the 3e-4 follow-up: 8 runs, control + 50/50
    python scripts/run_e1.py followup-eval
    python scripts/e1_followup.py experiments/e1-lr0.0003-m*-p1s*

One job at a time (one MPS job at a time: HANDOFF §6).  Each run writes to
``$MINIME_DATA_ROOT/runs/<run>`` (checkpoints never touch the internal disk),
its output to ``runs/logs/<run>.log``; a finished or diverged run's
``manifest.json`` and ``metrics.jsonl`` are copied into ``experiments/<run>/``,
and ``eval`` copies each ``eval-last.json`` there.  A run already in
``experiments/`` is skipped, so the command can be re-run after an
interruption; a run directory left half-written is refused -- resume it by
hand (``train.py --resume``, same arguments) or delete it.  ``main`` refuses
until the pilot has picked an LR and ``e1_base.yaml`` uses it; ``followup``,
unless its configs run at ``FOLLOWUP_LR``.  ``--dry-run``
prints the commands.
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

from minime.config import data_root, load_config, parse_cli_overrides  # noqa: E402
from minime.training.adaptation import (  # noqa: E402
    FOLLOWUP_LR,
    PILOT_EXTENSIONS,
    E1Error,
    E1Job,
    eval_command,
    followup_jobs,
    main_jobs,
    pilot_jobs,
    train_command,
)
from minime.training.lr_sweep import SweepError, read_run  # noqa: E402
from minime.training.noise_floor import EVAL_FILE  # noqa: E402
from minime.utils.throughput import format_duration  # noqa: E402

PILOT_REPORT = REPO / "reports" / "e1" / "pilot.json"
COPIED = ("manifest.json", "metrics.jsonl")


def status(run_dir: Path) -> str:
    """``absent``, ``finished``, ``diverged`` or ``incomplete``."""
    if not (run_dir / "manifest.json").exists():
        return "absent" if not run_dir.exists() else "incomplete"
    try:
        return read_run(run_dir)[1]
    except SweepError:
        return "incomplete"


def preflight(jobs: list[E1Job], root: Path, phase: str) -> None:
    """Everything that can be checked before the laptop spends an hour."""
    problems = []
    for path in (root / "shards" / "general", root / "shards" / "personal"):
        if not path.is_dir():
            problems.append(f"no shard directory {path}")
    for base in sorted({job.base for job in jobs}):
        if not (root / "checkpoints" / base / "last.pt").exists():
            problems.append(f"no base checkpoint {root / 'checkpoints' / base / 'last.pt'}")
    lrs = set()
    for job in jobs:
        try:
            cfg = load_config(REPO / job.config, overrides=parse_cli_overrides(list(job.overrides)))
        except (ValueError, FileNotFoundError) as exc:
            problems.append(f"{job.config}: {exc}")
            continue
        if cfg.role != job.role:
            problems.append(f"{job.config}: role {cfg.role!r}, expected {job.role!r}")
        lrs.add(cfg.training.learning_rate)
    if phase == "main" and not problems:
        if not PILOT_REPORT.exists():
            problems.append(f"no {PILOT_REPORT.relative_to(REPO)}: run the pilot and scripts/pick_e1_lr.py first")
        else:
            pilot = json.loads(PILOT_REPORT.read_text())
            if pilot.get("status") != "picked":
                problems.append(f"the pilot has not picked an LR yet (status {pilot.get('status')!r}, "
                                f"next {pilot.get('next')})")
            elif lrs != {pilot["pick"]}:
                problems.append(f"the main configs use LR {sorted(lrs)}, the pilot picked {pilot['pick']}: "
                                "set it in configs/experiments/e1_base.yaml")
    if phase == "followup" and not problems and lrs != {FOLLOWUP_LR}:
        problems.append(f"the follow-up configs use LR {sorted(lrs)}, the follow-up's is {FOLLOWUP_LR}")
    if problems:
        raise SystemExit("error: not starting:\n  " + "\n  ".join(problems))


def copy_results(run_dir: Path, names: tuple[str, ...]) -> Path:
    out = REPO / "experiments" / run_dir.name
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copy2(run_dir / name, out / name)
    return out


def run(cmd: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n")
        handle.flush()
        return subprocess.run(cmd, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT).returncode


def train_all(jobs: list[E1Job], root: Path, dry_run: bool) -> int:
    runs = root / "runs"
    for job in jobs:
        run_dir = runs / job.run_name
        cmd = train_command(job, repo=REPO, root=root, run_dir=run_dir, python=sys.executable)
        done_here = status(REPO / "experiments" / job.run_name)
        if done_here in ("finished", "diverged"):
            print(f"{job.run_name}: {done_here} already, skipped")
            continue
        if done_here == "incomplete":
            print(f"error: experiments/{job.run_name} holds an unfinished run: check it by hand", file=sys.stderr)
            return 1
        on_disk = status(run_dir)
        if on_disk in ("finished", "diverged"):
            copy_results(run_dir, COPIED)
            print(f"{job.run_name}: {on_disk} on disk, copied into experiments/")
            continue
        if on_disk == "incomplete":
            print(f"error: {run_dir} holds an interrupted run: resume it with the same command plus "
                  f"--resume {run_dir}/step_<N>.pt, or delete the directory", file=sys.stderr)
            return 1
        if dry_run:
            print(" ".join(cmd))
            continue
        started = time.monotonic()
        print(f"{job.run_name}: training ...", flush=True)
        code = run(cmd, runs / "logs" / f"{job.run_name}.log")
        result = status(run_dir)
        if result in ("finished", "diverged"):
            copy_results(run_dir, COPIED)
            print(f"{job.run_name}: {result} in {format_duration(time.monotonic() - started)}")
            continue
        print(f"error: {job.run_name} failed (exit {code}); see {runs / 'logs' / (job.run_name + '.log')}",
              file=sys.stderr)
        return 1
    return 0


def eval_all(jobs: list[E1Job], root: Path, dry_run: bool) -> int:
    for job in jobs:
        run_dir = root / "runs" / job.run_name
        here = REPO / "experiments" / job.run_name
        if (here / EVAL_FILE).exists():
            print(f"{job.run_name}: evaluated already, skipped")
            continue
        result = status(here)
        if result == "diverged":
            print(f"{job.run_name}: diverged, nothing to evaluate")
            continue
        if result != "finished":
            print(f"error: {job.run_name} has not finished ({result}): train it first", file=sys.stderr)
            return 1
        cmd = eval_command(run_dir, repo=REPO, root=root, python=sys.executable)
        if dry_run:
            print(" ".join(cmd))
            continue
        started = time.monotonic()
        print(f"{job.run_name}: evaluating ...", flush=True)
        code = run(cmd, root / "runs" / "logs" / f"{job.run_name}.eval.log")
        if code != 0 or not (run_dir / EVAL_FILE).exists():
            print(f"error: evaluating {job.run_name} failed (exit {code})", file=sys.stderr)
            return 1
        copy_results(run_dir, (EVAL_FILE,))
        print(f"{job.run_name}: evaluated in {format_duration(time.monotonic() - started)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=["pilot", "main", "eval", "followup", "followup-eval"])
    parser.add_argument("--extend", type=float, default=None, metavar="LR",
                        help=f"pilot: add the extension point the pick asked for ({sorted(PILOT_EXTENSIONS.values())})")
    parser.add_argument("--root", type=Path, default=None, help="data root (default: $MINIME_DATA_ROOT)")
    parser.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    args = parser.parse_args()
    if args.extend is not None and args.phase != "pilot":
        parser.error("--extend is for the pilot")

    root = args.root or data_root()
    try:
        if args.phase == "pilot":
            jobs = pilot_jobs(args.extend)
        else:
            jobs = followup_jobs() if args.phase.startswith("followup") else main_jobs()
    except E1Error as exc:
        parser.error(str(exc))
    if args.phase.endswith("eval"):
        return eval_all(jobs, root, args.dry_run)
    preflight(jobs, root, args.phase)
    return train_all(jobs, root, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
