#!/usr/bin/env python3
"""Pick E1's peak LR from the pilot runs.

    python scripts/pick_e1_lr.py experiments/e1-pilot-*

Writes ``reports/e1/pilot.json`` (numbers only).  The rule, decided before
any pilot ran (README "E1 adaptation protocol"): the lowest final personal
validation loss, an exact tie to the lower LR; a diverged pilot is reported
and drops out.  When the grid's winner is its lowest or highest point the
report asks for one point beyond it and the script exits 2: run
``scripts/run_e1.py pilot --extend <next>``, then pick again.  The search
stops after that point; if it wins, it is the best *tested* LR, flagged
``at_edge``.  Refuses (exit 1) runs that are not the protocol's pilot: another
role, mixture, data seed or starting checkpoint (``--reference``: the P1 run
the pilots must start from), anything but the LR differing, LRs off the grid.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from minime.training.adaptation import PILOT_BASE, E1Error, pick_pilot_lr  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="the pilot run directories")
    parser.add_argument("--reference", type=Path, default=REPO / "experiments" / PILOT_BASE,
                        help=f"the P1 run the pilots start from (default experiments/{PILOT_BASE})")
    parser.add_argument("--out", type=Path, default=REPO / "reports" / "e1" / "pilot.json")
    args = parser.parse_args()

    try:
        report = pick_pilot_lr(args.runs, references={args.reference.name: args.reference})
    except E1Error as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"E1 LR pilot ({report['token_budget']:,} tokens per run, 50/50 from {PILOT_BASE}); "
          f"ranked on {report['metric']}")
    for point in report["points"]:
        if point["status"] != "finished":
            print(f"  {point['lr']:<8g} {point['status']:>8}   {point['run']}")
            continue
        print(f"  {point['lr']:<8g} personal {point['personal']:.4f}  general {point['general']:.4f}   {point['run']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] == "extend":
        print(f"the grid's winner is at its edge: run scripts/run_e1.py pilot --extend {report['next']}, "
              "then pick again")
        print(f"report -> {args.out}")
        return 2
    edge = "  (still at the edge of everything tested: the best tested LR)" if report["at_edge"] else ""
    print(f"pick: {report['pick']} ({report['pick_run']}){edge}")
    print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
