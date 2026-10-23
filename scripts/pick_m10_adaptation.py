#!/usr/bin/env python3
"""Apply M10's adaptation gate: the flagship against candidates A, B and C (HANDOFF §4a).

    python scripts/pick_m10_adaptation.py experiments/flagship-s1337 experiments/m10-adapt-*

    # the owner's final choice, after reading the voice samples
    ... --owner-choice c --chosen-on 2026-10-19      # or --owner-choice flagship

Writes ``reports/m10/adaptation.json`` (numbers only).  The rule, fixed
before any attempt: a candidate is accepted only if, against the unadapted
flagship on the same native-context val evaluation, personal CE falls by 0.10
or more and general-weighted CE rises by 0.02 or less; among those accepted,
the largest personal fall wins, an exact tie to the lower general CE.  None
accepted: the unadapted flagship stays.  Either way the choice is not final
until the owner has read the voice samples; ``--owner-choice`` then freezes
it, the gate's pick or the flagship, nothing else.  ``--rerun`` gates the candidates retrained after the
MPS batch fix (``m10-rerun-*``, HANDOFF §3 M10g) as a description: it writes
``reports/m10/rerun.json`` and freezes nothing, since the final choice stays
``adaptation.json``'s.  Exit 0 with a result;
exit 1 when the runs or eval files are not the protocol's (a candidate
missing or twice, another starting checkpoint, evaluations unlike the
flagship's) or the choice is not one the owner can make.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from minime.training.m10 import RERUN_PREFIX, AdaptationError, freeze_choice, gate_adaptation  # noqa: E402
from minime.training.noise_floor import PRIMARY_EVAL_METRICS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("flagship", type=Path, help="the flagship's run directory (with SHA256SUMS and eval-last.json)")
    parser.add_argument("candidates", nargs="+", type=Path, help="the three candidates' run directories")
    parser.add_argument("--out", type=Path, default=None,
                        help="default reports/m10/adaptation.json, or rerun.json with --rerun")
    parser.add_argument("--owner-choice", default=None,
                        help="the owner's final choice after reading the samples: the pick's letter or 'flagship'")
    parser.add_argument("--chosen-on", type=datetime.date.fromisoformat, default=None,
                        help="the day the owner chose (YYYY-MM-DD); required with --owner-choice")
    parser.add_argument("--rerun", action="store_true",
                        help="the candidates retrained after the MPS batch fix: a description, no choice")
    args = parser.parse_args()
    if (args.owner_choice is None) != (args.chosen_on is None):
        parser.error("--owner-choice and --chosen-on go together")
    if args.rerun and args.owner_choice is not None:
        parser.error("--rerun describes the retrained candidates; the final choice stays adaptation.json's")
    if args.out is None:
        args.out = REPO / "reports" / "m10" / ("rerun.json" if args.rerun else "adaptation.json")
    # Neither set may stand in for the other: the reruns never write the frozen report.
    wrong = [d.name for d in args.candidates if d.name.startswith(RERUN_PREFIX) != args.rerun]
    if wrong:
        print(f"error: {', '.join(wrong)}: " + ("not a rerun (m10-rerun-*)" if args.rerun
                                               else "a rerun: pass --rerun"), file=sys.stderr)
        return 1

    try:
        report = gate_adaptation(args.flagship, args.candidates)
        if args.owner_choice is not None:
            report = freeze_choice(report, args.owner_choice, args.chosen_on.isoformat())
    except AdaptationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.rerun:
        report["reading"] = ("description: the three candidates retrained, configs unchanged, after the MPS batch "
                             "fix (HANDOFF §3 M10g), gated as the originals were; the final choice stays "
                             "reports/m10/adaptation.json's, and none of these is evaluated on test")

    personal, general = PRIMARY_EVAL_METRICS
    base = report["flagship"]["values"]
    print(f"M10 adaptation gate: F {report['flagship']['final_lr']:.6g}, L {report['l']:.6g}, "
          f"{report['token_budget']:,} tokens a candidate, data.seed {report['data_seed']}, {report['precision']}")
    print(f"  flagship        personal {base[personal]:.4f}  general-weighted {base[general]:.4f}")
    for name, c in report["candidates"].items():
        if c["status"] != "finished":
            print(f"  {name.upper()} {c['run']:<12} diverged: not accepted")
            continue
        change = c["change"]
        verdict = "accepted" if c["accepted"] else f"not accepted ({c['reason']})"
        mark = ("  <- the gate's pick" if args.rerun else "  <- pick") if name == report["pick"] else ""
        print(f"  {name.upper()} {c['run']:<12} personal {change[personal]:+.4f}  "
              f"general-weighted {change[general]:+.4f}  {verdict}{mark}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report -> {args.out}")
    if args.rerun:
        print("a description: the final choice stays reports/m10/adaptation.json's")
    elif report["final_choice_frozen"]:
        print(f"final choice, frozen: {report['final_run']} ({report['final_choice_made']})")
    elif report["pick"]:
        print(f"pick: candidate {report['pick'].upper()} ({report['pick_run']}), pending the owner's sample reading")
    else:
        print("none accepted: the unadapted flagship stays, pending the owner's sample reading")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
