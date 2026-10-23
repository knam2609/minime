#!/usr/bin/env python3
"""E1's 3e-4 follow-up: its two verdicts, and what it says beside 1e-3 and P4.

    python scripts/e1_followup.py experiments/e1-lr0.0003-m*-p1s*

Writes ``reports/e1/followup.json`` (numbers only).  The follow-up is the
control and the 50/50 arm from each P1 base at 3e-4 (README, "E1 follow-up
at 3e-4"), a second recipe reported beside the main runs' 1e-3, never
instead of it.  Two verdicts, 50/50 - control on the two primary metrics,
each mean d ± t(1 - 0.05/4, n-1) · s_d / √n -- ±2.09 s_d at four bases.
Descriptive beside them: each arm's change from the unchanged P1 checkpoint
(the control's general change is the restart's own cost at 3e-4); each run
against the main run of its arm on its base (``--main``, the LR alone
changed); and P4 - 50/50 at 3e-4 by P4's verdict arithmetic, P4's numbers
from its report (``--p4``), described, not judged.

Refuses (exit 1): what ``e1_paired.py`` refuses, with the follow-up's role
and LR; a run on the pilot's data seed; a follow-up run without its main
counterpart; and a P4 report evaluated unlike the follow-up.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from minime.training.adaptation import BASES, E1Error, against_main, paired_analysis  # noqa: E402
from minime.training.noise_floor import EVAL_FILE  # noqa: E402
from minime.training.p4 import CONTRASTS, P4Error, p4_against_followup  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="the follow-up runs' directories")
    parser.add_argument("--main", nargs="+", type=Path,
                        default=sorted((REPO / "experiments").glob("e1-m*-p1s*")),
                        help="E1's main runs (default experiments/e1-m*-p1s*)")
    parser.add_argument("--pilot", type=Path, default=REPO / "reports" / "e1" / "pilot.json")
    parser.add_argument("--p4", type=Path, default=REPO / "reports" / "p4" / "verdicts.json")
    parser.add_argument("--reference", nargs="*", type=Path,
                        default=[REPO / "experiments" / base for base in BASES],
                        help="the unchanged P1 runs (default experiments/p1-s1337..p1-s1340)")
    parser.add_argument("--eval-file", default=EVAL_FILE)
    parser.add_argument("--out", type=Path, default=REPO / "reports" / "e1" / "followup.json")
    args = parser.parse_args()

    for path in (args.pilot, args.p4):
        if not path.exists():
            print(f"error: no {path}", file=sys.stderr)
            return 1
    try:
        report = paired_analysis(
            args.runs, pilot=json.loads(args.pilot.read_text()),
            references={ref.name: ref for ref in args.reference}, eval_file=args.eval_file, followup=True,
        )
        lr = against_main(args.runs, args.main, eval_file=args.eval_file)
        e1_50 = {m: entry["followup"] for m, entry in lr["arms"]["50"]["metrics"].items()}
        report["against_main"] = lr["arms"]
        report["p4_described"] = {
            "reading": "description, not a verdict: P4's verdict arithmetic (Welch at df 3, Bonferroni over "
                       "four) against the follow-up's 50/50 arm; P4's numbers are reports/p4/verdicts.json's",
            "metrics": p4_against_followup(json.loads(args.p4.read_text()), e1_50, report["eval"]),
        }
    except (E1Error, P4Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"E1 follow-up: {report['n_bases']} bases (planned {report['planned_bases']}), "
          f"LR {report['learning_rate']}, {report['token_budget']:,} tokens per run, {report['precision']}")
    print(f"  each interval: mean d ± {report['multiplier']:.2f} s_d (Bonferroni over 2 verdicts)")
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
    print("descriptive:")
    for arm, metrics in report["from_base"].items():
        print(f"  {arm:>3}% from its P1 base   " + "  ".join(
            f"{m.split('.')[1]} {e['mean']:+.4f}" for m, e in metrics.items() if e["mean"] is not None))
    for arm, entry in report["against_main"].items():
        print(f"  {arm:>3}% against 1e-3      " + "  ".join(
            f"{m.split('.')[1]} {e['mean']:+.4f}" for m, e in entry["metrics"].items() if e["mean"] is not None))
    for metric, entry in report["p4_described"]["metrics"].items():
        if "difference" in entry:
            print(f"  P4 - {CONTRASTS[1]} at 3e-4  {metric}: {entry['difference']:+.4f} ± {entry['half_width']:.4f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
