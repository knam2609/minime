#!/usr/bin/env python3
"""P4's verdicts: four P4 runs against P1's four and E1's 50/50 arm, unpaired.

    python scripts/p4_compare.py        # experiments/p4-s1337..1340 by default

Writes ``reports/p4/verdicts.json`` (numbers only).  Two contrasts, P4 - P1
and P4 - E1 50/50, on the two primary eval metrics of the runs' final
checkpoints: four verdicts, each mean(P4) - mean(other) ± t(1 - 0.05/8, 3) ·
√(s²_P4/4 + s²_other/4) -- Welch at the conservative degrees of freedom,
Bonferroni over the four (README, "P4 protocol").  An interval containing 0
is "unresolved".  Descriptive beside them: the exploratory metrics, P4
against E1's control, 25/75 and 100/0, and P4's validation loss beside P1's
floor.

Refuses (exit 1): other than four runs a side; P4 runs that differ from P1
in anything but the mixture and the seeds, share a seed, or mix in other
than the protocol's 1,048,576 chat tokens; E1 runs outside the ``main``
role or an arm without four bases; and eval files that do not evaluate their
own run's final checkpoint the same way.  A diverged P4 run is the recipe
failing on that seed: the report says so and gives no verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from minime.training.adaptation import BASES  # noqa: E402
from minime.training.noise_floor import EVAL_FILE  # noqa: E402
from minime.training.p4 import P4Error, p4_analysis  # noqa: E402

SEEDS = [base.removeprefix("p1-") for base in BASES]   # s1337 .. s1340


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--p4", nargs="+", type=Path, default=[REPO / "experiments" / f"p4-{s}" for s in SEEDS])
    parser.add_argument("--p1", nargs="+", type=Path, default=[REPO / "experiments" / f"p1-{s}" for s in SEEDS])
    parser.add_argument("--e1", nargs="+", type=Path, default=sorted((REPO / "experiments").glob("e1-m*-p1s*")),
                        help="E1's main runs (default experiments/e1-m*-p1s*)")
    parser.add_argument("--eval-file", default=EVAL_FILE)
    parser.add_argument("--out", type=Path, default=REPO / "reports" / "p4" / "verdicts.json")
    args = parser.parse_args()

    try:
        report = p4_analysis(args.p4, args.p1, args.e1, eval_file=args.eval_file)
    except P4Error as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"P4: {report['n_runs']} runs a side, {report['token_budget']:,} tokens per run, "
          f"{report['precision']}; each interval ± {report['t']:.2f} · se (Bonferroni over "
          f"{report['n_verdicts']} verdicts)")
    if report["diverged"]:
        print(f"  diverged: {report['diverged']}")
    for name, metrics in report["contrasts"].items():
        for metric, entry in metrics.items():
            if entry["role"] != "primary":
                continue
            if "half_width" not in entry:
                print(f"  P4 - {name:6s} {metric}: {entry['verdict']}")
                continue
            print(f"  P4 - {name:6s} {metric}: {entry['difference']:+.4f} ± {entry['half_width']:.4f} "
                  f"-> {entry['verdict']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
