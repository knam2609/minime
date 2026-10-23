#!/usr/bin/env python3
"""Pick the flagship's peak LR from the M10 sweep (HANDOFF §4a).

    python scripts/pick_m10_lr.py experiments/m10-lr-*

Writes ``reports/m10/sweep.json`` (numbers only).  The rule, fixed before
any sweep run: the lowest final general validation loss at 431 steps, read
unrounded from each manifest; an exact tie goes to the lower LR; a diverged
point is reported and excluded.  Exit 0: picked -- set
``training.learning_rate`` in ``configs/flagship.yaml`` to it.  Exit 2: the
grid's winner is at 8e-4 or 6.4e-3; run the one extension it names, then pick
again (the notebook queues it itself when ``SWEEP_EXTENSION`` is on).  Exit 3:
revisit -- no point finished, or the extension won; that is the owner's call,
not a rule's.  Refuses (exit 1) runs that are not the sweep or are
incomplete: complete or rerun them, never drop them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from minime.training.lr_sweep import SweepError  # noqa: E402
from minime.training.m10 import pick_sweep_lr, sweep_config  # noqa: E402

EXIT = {"picked": 0, "extend": 2, "revisit": 3}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="the sweep's run directories")
    parser.add_argument("--out", type=Path, default=REPO / "reports" / "m10" / "sweep.json")
    args = parser.parse_args()

    try:
        report = pick_sweep_lr(args.runs)
    except SweepError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"M10 LR sweep: {report['steps']} steps, {report['token_budget'] / 1e6:.1f}M tokens a run, "
          f"seeds {report['seed']}/{report['data_seed']}, {report['precision']}")
    for point in report["points"]:
        loss = point["final_val_loss"]
        shown = f"{loss:.6f}" if loss is not None else "diverged"
        mark = "  <- pick" if point["run"] == report["pick_run"] and report["status"] == "picked" else ""
        print(f"  {point['lr']:<10g} {shown:>10}   {point['run']}{mark}")
    if report["margin"] is not None:
        print(f"  margin to the runner-up: {report['margin']:.6f} (one run per LR: no noise floor at this scale)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report -> {args.out}")

    if report["status"] == "extend":
        print(f"EXTEND: {report['reason']}; run {sweep_config(report['next'])}, then pick again", file=sys.stderr)
    elif report["status"] == "revisit":
        print(f"REVISIT: {report['reason']}; the grid goes back to the owner", file=sys.stderr)
    else:
        print(f"pick: {report['pick']:g} -> training.learning_rate in configs/flagship.yaml")
    return EXIT[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
