#!/usr/bin/env python3
"""Measure the seed-noise floor from the baseline's repeated runs.

    python scripts/noise_floor.py experiments/p1-s*
    python scripts/noise_floor.py experiments/p1-s* --compare experiments/p2-s1337
    python scripts/noise_floor.py experiments/p1-s* --eval --compare experiments/p2-s1337
    python scripts/noise_floor.py experiments/p1-s* --eval blimp.accuracy code.valid

Writes ``reports/noise_floor/<config>.json`` (numbers only) and prints the
interval an arm must leave to count as different: the 95% prediction interval
of the runs' final validation losses.  Refuses (exit 1) fewer than three runs,
an unfinished or diverged run, and runs that differ in anything but their two
seeds -- each run must have its own ``training.seed`` and ``data.seed``.
``--compare`` judges finished arm runs against the interval, and refuses
(exit 1) an arm validated on other windows, at another token budget or in
another precision.

``--eval`` measures the same interval on eval metrics instead -- dotted paths
into each run's ``eval-last.json`` (``--eval-file``), by default the fixed
list ``EVAL_METRICS`` -- and writes ``reports/noise_floor/<config>-eval.json``.
Only ``PRIMARY_EVAL_METRICS`` (personal and general-weighted CE) get verdicts
that count as results; the others are marked exploratory.
Each run's file must evaluate its own final checkpoint on the val split, and
every file, an arm's included, the same eval sets with the same eval config on
the same device.  The protocol is in ``minime.training.noise_floor`` and the
README.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minime.training.lr_sweep import SweepError, read_run  # noqa: E402
from minime.training.noise_floor import (  # noqa: E402
    EVAL_FILE,
    EVAL_METRICS,
    VAL_METRIC,
    FloorError,
    check_comparable,
    check_eval_comparable,
    compare,
    eval_metric,
    measure_floor,
    metric_role,
    read_eval,
)


def _print_floor(floor: dict) -> None:
    metric = floor["metric"]
    print(f"noise floor of '{floor['config']}' on {metric} over {floor['n']} seed pairs "
          f"({floor['token_budget'] / 1e6:.1f}M tokens per run, {floor['precision']})")
    for run in floor["runs"]:
        print(f"  seeds {run['training_seed']}/{run['data_seed']}  {run[metric]:.4f}   {run['run']}")
    print(f"  mean {floor['mean']:.4f}  std {floor['std']:.4f}  range {floor['range']:.4f}")
    print(f"  an arm differs only outside {floor['mean']:.4f} ± {floor['half_width']:.4f}")


def _judge(floors: dict[str, dict], arms: list[Path], eval_file: str | None,
           baseline_dir: Path) -> tuple[dict, int]:
    """Each arm against each floor; ``(comparisons, exit code)``."""
    baseline_manifest = read_run(baseline_dir)[0]
    baseline_eval = read_eval(baseline_dir, baseline_manifest, eval_file) if eval_file else None
    comparisons: dict[str, dict] = {}
    for arm in arms:
        try:
            manifest, status, loss = read_run(arm)
            check_comparable(baseline_manifest, manifest, validation=eval_file is None)
            record = None
            if eval_file and status == "finished":
                record = read_eval(arm, manifest, eval_file)
                check_eval_comparable(baseline_eval, record)
        except (SweepError, FloorError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return comparisons, 1
        if status != "finished":
            print(f"  {arm.name}: {status}")
            comparisons[arm.name] = {"status": status}
            continue
        results = {}
        for metric, floor in floors.items():
            try:
                value = loss if record is None else eval_metric(record, metric, arm / eval_file)
            except FloorError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return comparisons, 1
            results[metric] = {"value": value, **compare(floor, value), "role": metric_role(metric)}
            label = "" if results[metric]["role"] == "primary" else "  [exploratory]"
            print(f"  {arm.name}: {metric} {value:.4f} ({results[metric]['difference']:+.4f}) "
                  f"-> {results[metric]['verdict']}{label}")
        comparisons[arm.name] = results
    return comparisons, 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="the baseline's run directories")
    parser.add_argument("--compare", nargs="*", type=Path, default=[], metavar="RUN",
                        help="arm run directories to judge against the floor")
    parser.add_argument("--eval", nargs="*", default=None, metavar="METRIC",
                        help="floor eval metrics (dotted eval.json paths) instead of validation loss; "
                             "no METRIC means the fixed list EVAL_METRICS")
    parser.add_argument("--eval-file", default=EVAL_FILE, help=f"each run's eval file (default {EVAL_FILE})")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/noise_floor"))
    args = parser.parse_args()

    metrics = [VAL_METRIC] if args.eval is None else (args.eval or list(EVAL_METRICS))
    eval_file = None if args.eval is None else args.eval_file
    floors: dict[str, dict] = {}
    try:
        for metric in metrics:
            floors[metric] = measure_floor(args.runs, metric, args.eval_file)
    except FloorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for floor in floors.values():
        _print_floor(floor)

    comparisons, code = _judge(floors, args.compare, eval_file, args.runs[0])

    first = next(iter(floors.values()))
    if eval_file is None:
        report, out = dict(first), args.out_dir / f"{first['config']}.json"
    else:
        report = {"config": first["config"], "eval": first["eval"],
                  "metrics": {m: {k: v for k, v in f.items() if k != "eval"} for m, f in floors.items()}}
        out = args.out_dir / f"{first['config']}-eval.json"
    if comparisons:
        report["comparisons"] = comparisons
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report -> {out}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
