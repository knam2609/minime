#!/usr/bin/env python3
"""Run M10's adaptation candidates on the laptop, then evaluate them (HANDOFF §4a).

    python scripts/run_m10_adapt.py train      # m10-adapt-{a,b,c}, from the flagship's last.pt
    python scripts/run_m10_adapt.py eval       # the flagship (if not yet), then the three
    python scripts/pick_m10_adaptation.py experiments/flagship-s1337 experiments/m10-adapt-*

    # the same three, configs unchanged, retrained after the MPS batch fix
    # (HANDOFF §3 M10g): m10-rerun-{a,b,c}, gated as a description
    python scripts/run_m10_adapt.py train --rerun
    python scripts/run_m10_adapt.py eval --rerun
    python scripts/pick_m10_adaptation.py experiments/flagship-s1337 experiments/m10-rerun-* --rerun

One job at a time (one MPS job at a time: HANDOFF §6).  Each run writes to
``$MINIME_DATA_ROOT/runs/<run>`` (checkpoints never touch the internal disk),
its output to ``runs/logs/<run>.log``; a finished or diverged run's
``manifest.json`` and ``metrics.jsonl`` are copied into ``experiments/<run>/``,
and ``eval`` copies each ``eval-last.json`` there -- the flagship's into
``experiments/flagship-s1337/``.  Evaluations use
``configs/eval_flagship.yaml``: the native context, identical for the four.
A run already in ``experiments/`` is skipped, so the command can be re-run
after an interruption; a run directory left half-written is refused --
resume it by hand (``train.py --resume``, same arguments) or delete it.
``train`` refuses before any step unless the flagship's ``last.pt`` is the
one its session checksummed and every config is its candidate exactly.
``--dry-run`` prints the commands.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from minime.config import data_root, load_config  # noqa: E402
from minime.training.lr_sweep import SweepError, read_run  # noqa: E402
from minime.training.m10 import (  # noqa: E402
    FLAGSHIP_RUN,
    AdaptationError,
    adapt_eval_command,
    adapt_train_command,
    adaptation_jobs,
    candidate_of,
    final_lr,
)
from minime.training.noise_floor import EVAL_FILE  # noqa: E402
from minime.utils.throughput import format_duration  # noqa: E402

FLAGSHIP_HERE = REPO / "experiments" / FLAGSHIP_RUN
COPIED = ("manifest.json", "metrics.jsonl")


def status(run_dir: Path) -> str:
    """``absent``, ``finished``, ``diverged`` or ``incomplete``."""
    if not (run_dir / "manifest.json").exists():
        return "absent" if not run_dir.exists() else "incomplete"
    try:
        return read_run(run_dir)[1]
    except SweepError:
        return "incomplete"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def preflight(root: Path) -> None:
    """Everything that can be checked before the laptop spends an hour."""
    problems = []
    for path in (root / "shards" / "general", root / "shards" / "personal"):
        if not path.is_dir():
            problems.append(f"no shard directory {path}")
    checkpoint = root / "checkpoints" / FLAGSHIP_RUN / "last.pt"
    sums = FLAGSHIP_HERE / "SHA256SUMS"
    if not checkpoint.exists():
        problems.append(f"no flagship checkpoint {checkpoint}")
    elif not sums.exists():
        problems.append(f"no {sums.relative_to(REPO)}: the flagship's session checksums")
    else:
        wanted = next((line.split()[0] for line in sums.read_text().splitlines() if line.split()[1:] == ["last.pt"]),
                      None)
        if sha256(checkpoint) != wanted:
            problems.append(f"{checkpoint} is not the last.pt the flagship's session wrote (SHA256SUMS)")
    try:
        f = final_lr(FLAGSHIP_HERE)
        for job in adaptation_jobs():
            name = candidate_of(load_config(REPO / job.config).to_dict(), f, job.config)
            if name != job.candidate:
                problems.append(f"{job.config} is candidate {name.upper()}, not {job.candidate.upper()}")
    except (AdaptationError, SweepError, ValueError, FileNotFoundError) as exc:
        problems.append(str(exc))
    if problems:
        raise SystemExit("error: not starting:\n  " + "\n  ".join(problems))


def copy_results(run_dir: Path, names: tuple[str, ...], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copy2(run_dir / name, out / name)


def run(cmd: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n")
        handle.flush()
        return subprocess.run(cmd, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT).returncode


def train_all(root: Path, dry_run: bool, rerun: bool = False) -> int:
    runs = root / "runs"
    for job in adaptation_jobs(rerun):
        run_dir = runs / job.run_name
        here = REPO / "experiments" / job.run_name
        cmd = adapt_train_command(job, repo=REPO, root=root, run_dir=run_dir, python=sys.executable)
        done_here = status(here)
        if done_here in ("finished", "diverged"):
            print(f"{job.run_name}: {done_here} already, skipped")
            continue
        if done_here == "incomplete":
            print(f"error: experiments/{job.run_name} holds an unfinished run: check it by hand", file=sys.stderr)
            return 1
        on_disk = status(run_dir)
        if on_disk in ("finished", "diverged"):
            copy_results(run_dir, COPIED, here)
            print(f"{job.run_name}: {on_disk} on disk, copied into experiments/")
            continue
        if on_disk == "incomplete":
            print(f"error: {run_dir} holds an interrupted run: resume it with the same command plus "
                  f"--resume {run_dir}/last.pt, or delete the directory", file=sys.stderr)
            return 1
        if dry_run:
            print(" ".join(cmd))
            continue
        started = time.monotonic()
        print(f"{job.run_name}: training ...", flush=True)
        code = run(cmd, runs / "logs" / f"{job.run_name}.log")
        result = status(run_dir)
        if result in ("finished", "diverged"):
            copy_results(run_dir, COPIED, here)
            print(f"{job.run_name}: {result} in {format_duration(time.monotonic() - started)}")
            continue
        print(f"error: {job.run_name} failed (exit {code}); see {runs / 'logs' / (job.run_name + '.log')}",
              file=sys.stderr)
        return 1
    return 0


def eval_all(root: Path, dry_run: bool, rerun: bool = False) -> int:
    # The flagship first: it is every candidate's baseline.
    targets = [(FLAGSHIP_RUN, root / "checkpoints" / FLAGSHIP_RUN, FLAGSHIP_HERE)]
    targets += [(job.run_name, root / "runs" / job.run_name, REPO / "experiments" / job.run_name)
                for job in adaptation_jobs(rerun)]
    for name, run_dir, here in targets:
        if (here / EVAL_FILE).exists():
            print(f"{name}: evaluated already, skipped")
            continue
        result = status(here)
        if result == "diverged":
            print(f"{name}: diverged, nothing to evaluate")
            continue
        if result != "finished":
            print(f"error: {name} has not finished ({result}): train it first", file=sys.stderr)
            return 1
        cmd = adapt_eval_command(run_dir / "last.pt", repo=REPO, root=root, python=sys.executable)
        if dry_run:
            print(" ".join(cmd))
            continue
        started = time.monotonic()
        print(f"{name}: evaluating ...", flush=True)
        code = run(cmd, root / "runs" / "logs" / f"{name}.eval.log")
        if code != 0 or not (run_dir / EVAL_FILE).exists():
            print(f"error: evaluating {name} failed (exit {code})", file=sys.stderr)
            return 1
        copy_results(run_dir, (EVAL_FILE,), here)
        print(f"{name}: evaluated in {format_duration(time.monotonic() - started)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=["train", "eval"])
    parser.add_argument("--root", type=Path, default=None, help="data root (default: $MINIME_DATA_ROOT)")
    parser.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    parser.add_argument("--rerun", action="store_true",
                        help="the candidates retrained after the MPS batch fix, as m10-rerun-{a,b,c}")
    args = parser.parse_args()

    root = args.root or data_root()
    if args.phase == "eval":
        return eval_all(root, args.dry_run, args.rerun)
    preflight(root)
    return train_all(root, args.dry_run, args.rerun)


if __name__ == "__main__":
    raise SystemExit(main())
