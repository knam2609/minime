#!/usr/bin/env python3
"""Pick an optimiser's learning rate from its E4 sweep (the LR protocol).

    # 1. AdamW first
    python scripts/pick_lr.py adamw experiments/e4-lr-adamw-*

    # 2. set learning_rate in configs/experiments/e4_lr_muon_base.yaml to the
    #    AdamW winner, run the Muon sweep, then
    python scripts/pick_lr.py muon experiments/e4-lr-muon-*

Writes ``reports/lr_sweep/<sweep>.json`` (numbers only) and prints the grid.
The Muon pick reads the AdamW report and refuses runs made at any other AdamW
LR.  Exits 2 when the winner is at the edge of the grid -- the report is
still written, but the sweep needs another point beyond that edge before the
winner means anything.  The protocol is in ``minime.training.lr_sweep`` and
the README.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minime.training.lr_sweep import SWEEPS, SweepError, pick_lr  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep", choices=sorted(SWEEPS))
    parser.add_argument("runs", nargs="+", type=Path, help="the sweep's run directories")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/lr_sweep"))
    args = parser.parse_args()

    anchor_report = None
    if (anchor := SWEEPS[args.sweep].anchor) is not None:
        anchor_path = args.out_dir / f"{anchor[0]}.json"
        if anchor_path.exists():
            anchor_report = json.loads(anchor_path.read_text())

    try:
        report = pick_lr(args.sweep, args.runs, anchor_report=anchor_report)
    except SweepError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"{report['sweep']} sweep over {report['key']} "
          f"({report['token_budget'] / 1e6:.1f}M tokens per run, seed {report['seed']})")
    if report["anchor"]:
        print(f"  anchored at {report['anchor']['key']} = {report['anchor']['value']:g}")
    for point in report["points"]:
        loss = point["final_val_loss"]
        shown = f"{loss:.4f}" if loss is not None else "diverged"
        mark = "  <- winner" if point["run"] == report["winner_run"] else ""
        print(f"  {point['value']:<10g} {shown:>10}   {point['run']}{mark}")
    if report["margin"] is not None:
        print(f"  margin to the runner-up: {report['margin']:.4f} (compare with the seed-noise floor)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.sweep}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report -> {out}")

    if report["at_edge"]:
        print(
            f"EDGE: the winner {report['winner']:g} is the grid's "
            f"{'lowest' if report['winner'] == report['points'][0]['value'] else 'highest'} value. "
            "Add a point beyond it (a factor of 2) and pick again.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
