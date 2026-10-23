#!/usr/bin/env python3
"""E1's verdicts: each arm against its base's control, paired over the P1 bases.

    python scripts/e1_paired.py experiments/e1-m*-p1s*

Writes ``reports/e1/paired.json`` (numbers only).  For each base and arm,
d = arm - control on each eval metric of the runs' final checkpoints.  The two
primary metrics get the six verdicts (three arms x two metrics), each by a
Bonferroni interval, mean d ± t(1 - 0.05/12, n-1) · s_d / √n -- ±3.12 s_d at
four bases.  An interval containing 0 is "unresolved".  The other eval
metrics are exploratory: differences only.  Beside them, each arm's change
from the unchanged P1 checkpoint (``--reference``), descriptive only.

Refuses (exit 1): a run not in the ``main`` role, runs that differ in
anything but the mixture and data seed, an LR other than the pilot's pick
(``--pilot``) or a run on the pilot's data seed, a base missing an arm, arms
of one base on different data seeds or two bases on one, and eval files that
do not evaluate their own run's final checkpoint the same way.  A diverged
run is its strategy failing on that base: its contrast gets no verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from minime.training.adaptation import BASES, E1Error, paired_analysis  # noqa: E402
from minime.training.noise_floor import EVAL_FILE  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="the main runs' directories")
    parser.add_argument("--pilot", type=Path, default=REPO / "reports" / "e1" / "pilot.json")
    parser.add_argument("--reference", nargs="*", type=Path,
                        default=[REPO / "experiments" / base for base in BASES],
                        help="the unchanged P1 runs (default experiments/p1-s1337..p1-s1340)")
    parser.add_argument("--eval-file", default=EVAL_FILE)
    parser.add_argument("--out", type=Path, default=REPO / "reports" / "e1" / "paired.json")
    args = parser.parse_args()

    if not args.pilot.exists():
        print(f"error: no pilot report {args.pilot}: the main runs' LR comes from it", file=sys.stderr)
        return 1
    try:
        report = paired_analysis(
            args.runs, pilot=json.loads(args.pilot.read_text()),
            references={ref.name: ref for ref in args.reference}, eval_file=args.eval_file,
        )
    except E1Error as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"E1 paired analysis: {report['n_bases']} bases (planned {report['planned_bases']}), "
          f"LR {report['learning_rate']}, {report['token_budget']:,} tokens per run, {report['precision']}")
    print(f"  each interval: mean d ± {report['multiplier']:.2f} s_d (Bonferroni over 6 verdicts)")
    for arm, contrast in report["contrasts"].items():
        for metric, entry in contrast["metrics"].items():
            if entry["role"] != "primary":
                continue
            if "half_width" not in entry:
                print(f"  {arm:>3}% vs control  {metric}: {entry['verdict']}")
                continue
            diffs = " ".join(f"{d:+.4f}" for d in entry["differences"])
            print(f"  {arm:>3}% vs control  {metric}: {entry['mean']:+.4f} ± {entry['half_width']:.4f} "
                  f"[{diffs}] -> {entry['verdict']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
