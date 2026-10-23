"""P4: P1 with the owner's chat from step 0, judged against P1 and E1's 50/50 arm.

The protocol was decided with the owner on 2026-10-13, before any P4 run
(README, "P4 protocol"; HANDOFF §4):

* P4 is P1 with one change, its mixture: chat at 20 parts in 5,750, so
  1,048,576 of P1's 301,465,600 tokens are expected from it -- the E1 50/50
  arm's personal tokens.  It runs at the floor's four seed pairs.
* P1's runs drew their initial weights before the seed fix, and E1's runs
  started from those, so no P4 run shares a starting point with either:
  each contrast is four runs against four, unpaired.
* Two contrasts on the two primary eval metrics of the final checkpoints,
  P4 - P1 and P4 - E1 50/50: four verdicts, each by Welch's interval at the
  conservative degrees of freedom (the smaller group's n - 1) with Bonferroni
  over the four, mean(P4) - mean(other) ± t(1 - 0.05/8, 3) ·
  √(s²_P4/4 + s²_other/4).  An interval containing zero is "unresolved".  A
  diverged P4 run is the recipe failing on that seed: no verdict.
* Everything else is descriptive: P4 against E1's other arms, the eight
  exploratory metrics, and P4's validation loss beside P1's floor.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from pathlib import Path

from .adaptation import ARMS, CONTROL, E1Error, GENERAL_RATIO, read_e1_run, share_text
from .lr_sweep import SweepError, check_controlled, controlled_view, read_run
from .noise_floor import (
    EVAL_FILE,
    EVAL_METRICS,
    PRIMARY_EVAL_METRICS,
    SEED_KEYS,
    FloorError,
    check_eval_comparable,
    compare,
    eval_metric,
    eval_summary,
    measure_floor,
    read_eval,
)


class P4Error(ValueError):
    """The runs cannot support P4's verdicts."""


P4_CONFIG = "p4"
P1_CONFIG = "p1"
PERSONAL_SOURCE = "chat"
# The E1 50/50 arm's expected personal tokens: 256 steps x 8,192 x 0.5.
CHAT_TOKENS = 1_048_576
N_RUNS = 4                       # per group: P4's seeds, P1's floor, E1's bases
VERDICT_ARM = 0.5                # E1's arm in the second contrast
CONTRASTS = ("p1", f"e1-{share_text(VERDICT_ARM)}")
N_VERDICTS = len(CONTRASTS) * len(PRIMARY_EVAL_METRICS)
FAMILY_ALPHA = 0.05
# t(1 - 0.05/8, 3): Bonferroni over the four, Welch's conservative df at four
# runs a side.  Checked against scipy by a test; scipy is not a dependency.
T_P4 = 5.3919


# ------------------------------------------------------------------ reading


def _read(run_dir: Path) -> tuple[dict, str, float | None]:
    try:
        return read_run(run_dir)
    except SweepError as exc:
        raise P4Error(str(exc)) from None


def _budget(view: dict) -> int:
    return (view["training.max_steps"] * view["training.batch_size"]
            * view["training.gradient_accumulation"] * view["model.context_length"])


def _check_mixture(view: dict, where: Path) -> None:
    """P4's mixture, refused unless it is the protocol's."""
    mixture = {k.removeprefix("data.mixture."): v for k, v in view.items() if k.startswith("data.mixture.")}
    expected = {*GENERAL_RATIO, PERSONAL_SOURCE}
    if set(mixture) != expected:
        raise P4Error(f"{where}: mixture {mixture} must name exactly {sorted(expected)}")
    total = sum(mixture.values())
    chat = mixture[PERSONAL_SOURCE] / total * _budget(view)
    if not math.isclose(chat, CHAT_TOKENS, abs_tol=1.0):
        raise P4Error(f"{where}: {chat:,.0f} chat tokens expected, the protocol's are {CHAT_TOKENS:,}")
    general = total - mixture[PERSONAL_SOURCE]
    ratio_total = sum(GENERAL_RATIO.values())
    for name, part in GENERAL_RATIO.items():
        if not math.isclose(mixture[name] / general, part / ratio_total, rel_tol=1e-9):
            raise P4Error(f"{where}: the general part of {mixture} is not {GENERAL_RATIO}")


def _check_pretrains(p4: dict[Path, tuple], p1: dict[Path, tuple]) -> dict:
    """P4's runs are P1's config with its mixture changed, each at its own seed pair."""
    for group, name in ((p4, P4_CONFIG), (p1, P1_CONFIG)):
        if len(group) != N_RUNS:
            raise P4Error(f"{len(group)} {name} run(s); the protocol has {N_RUNS}")
        for run_dir, (manifest, _, _) in group.items():
            if manifest["config"]["name"] != name:
                raise P4Error(f"{run_dir}: config {manifest['config']['name']!r}, not {name!r}")
    views = {d: controlled_view(m) for d, (m, _, _) in {**p1, **p4}.items()}
    for group in (p4, p1):
        for key in SEED_KEYS:
            values = [views[d][key] for d in group]
            if len(set(values)) != len(values):
                raise P4Error(f"runs share a {key} ({sorted(values)}): each has its own seed pair")
    mixture = {f"data.mixture.{n}" for n in (*GENERAL_RATIO, PERSONAL_SOURCE)}
    try:
        check_controlled(views, {*SEED_KEYS, *mixture, f"manifest.data.shards.{PERSONAL_SOURCE}"})
    except SweepError as exc:
        raise P4Error(f"P4 is P1 with its mixture changed, and nothing else: {exc}") from None
    for run_dir in p4:
        _check_mixture(views[run_dir], run_dir)
    return views[next(iter(p4))]


def _e1_arms(run_dirs: list[Path]) -> dict[float, list]:
    """E1's main runs by arm, four bases each."""
    arms: dict[float, list] = defaultdict(list)
    try:
        runs = [read_e1_run(d) for d in run_dirs]
    except E1Error as exc:
        raise P4Error(str(exc)) from None
    for run in runs:
        if run.role != "main":
            raise P4Error(f"{run.run_dir}: role {run.role!r}, not 'main' -- pilots stay out of the analysis")
        arms[run.share].append(run)
    for share in ARMS:
        runs = arms.get(share, [])
        if len(runs) != N_RUNS or len({r.init["sha256"] for r in runs}) != N_RUNS:
            raise P4Error(f"E1's {share_text(share)}% arm: {len(runs)} run(s) from "
                          f"{len({r.init['sha256'] for r in runs})} base(s); the protocol has {N_RUNS}")
    return arms


# ------------------------------------------------------------------ numbers


def describe_difference(p4: list[float], other: list[float]) -> dict:
    """The verdicts' arithmetic, mean(P4) - mean(other) ± T_P4 · se, with no verdict.

    For numbers the protocol gives no verdict on: the test split's CE, which
    the test phase reads once and only describes.
    """
    var_p4, var_other = statistics.variance(p4), statistics.variance(other)
    se = math.sqrt(var_p4 / len(p4) + var_other / len(other))
    difference = statistics.fmean(p4) - statistics.fmean(other)
    half = T_P4 * se
    return {"difference": difference, "sd_p4": math.sqrt(var_p4), "sd_other": math.sqrt(var_other), "se": se,
            "t": T_P4, "half_width": half, "low": difference - half, "high": difference + half}


def _welch(p4: list[float], other: list[float], other_name: str) -> dict:
    described = describe_difference(p4, other)
    difference, half = described.pop("difference"), described["half_width"]
    if half == 0:
        verdict = "not judged: neither side varies"
    elif difference - half <= 0 <= difference + half:
        verdict = "unresolved"
    else:
        verdict = f"{'lower' if difference < 0 else 'higher'} than {other_name}"
    return {**described, "verdict": verdict}


def p4_analysis(
    p4_dirs: list[str | Path],
    p1_dirs: list[str | Path],
    e1_dirs: list[str | Path],
    *,
    eval_file: str = EVAL_FILE,
) -> dict:
    """P4's four verdicts, and the descriptive comparisons beside them."""
    p4 = {Path(d): _read(Path(d)) for d in p4_dirs}
    p1 = {Path(d): _read(Path(d)) for d in p1_dirs}
    shared = _check_pretrains(p4, p1)
    unfinished = [d.name for d, (_, status, _) in p1.items() if status != "finished"]
    if unfinished:
        raise P4Error(f"P1 runs {unfinished} did not finish: the baseline must")
    arms = _e1_arms([Path(d) for d in e1_dirs])
    diverged = sorted(d.name for d, (_, status, _) in p4.items() if status != "finished")
    if len(diverged) == len(p4):
        raise P4Error("every P4 run diverged: the recipe failed on every seed -- report that; "
                      "there is nothing to compare")

    # Every eval file evaluates its own run's final checkpoint, all alike.
    records: dict[Path, dict] = {}
    try:
        for run_dir, (manifest, status, _) in {**p4, **p1}.items():
            if status == "finished":
                records[run_dir] = read_eval(run_dir, manifest, eval_file)
        for runs in arms.values():
            for run in runs:
                if run.status == "finished":
                    records[run.run_dir] = read_eval(run.run_dir, run.manifest, eval_file)
        first = records[next(iter(p1))]
        for record in records.values():
            check_eval_comparable(first, record)
    except FloorError as exc:
        raise P4Error(str(exc)) from None

    def values(run_dirs, metric: str) -> list[float]:
        return [eval_metric(records[d], metric, d / eval_file) for d in run_dirs if d in records]

    p4_finished = [d for d in p4 if d in records]
    groups = {
        CONTRASTS[0]: list(p1),
        CONTRASTS[1]: [r.run_dir for r in arms[VERDICT_ARM]],
    }
    failed_e1 = [r.run_dir.name for r in arms[VERDICT_ARM] if r.status != "finished"]

    contrasts: dict[str, dict] = {}
    for name, other in groups.items():
        per_metric = {}
        for metric in EVAL_METRICS:
            role = "primary" if metric in PRIMARY_EVAL_METRICS else "exploratory"
            try:
                a, b = values(p4_finished, metric), values(other, metric)
            except FloorError as exc:
                if role == "primary":
                    raise P4Error(str(exc)) from None
                # An exploratory number a model cannot produce is reported
                # missing, not allowed to block the verdicts.
                per_metric[metric] = {"role": role, "unavailable": str(exc)}
                continue
            entry = {"role": role, "p4": a, name: b, "difference": statistics.fmean(a) - statistics.fmean(b)}
            if role == "primary":
                if diverged:
                    entry["verdict"] = f"no verdict: P4 diverged on {diverged} -- the recipe failed there"
                elif failed_e1 and name == CONTRASTS[1]:
                    entry["verdict"] = f"no verdict: E1 runs {failed_e1} diverged"
                else:
                    entry.update(_welch(a, b, name))
            per_metric[metric] = entry
        contrasts[name] = per_metric

    # P4 beside E1's other arms: means only, no verdict.
    descriptive = {}
    for share in sorted(ARMS):
        if share == VERDICT_ARM:
            continue
        run_dirs = [r.run_dir for r in arms[share]]
        descriptive[f"e1-{share_text(share)}"] = {
            metric: {"difference": statistics.fmean(values(p4_finished, metric))
                     - statistics.fmean(values(run_dirs, metric)),
                     "role": "control" if share == CONTROL else "arm"}
            for metric in PRIMARY_EVAL_METRICS
        }

    # P4's validation loss beside P1's floor: the same windows, descriptive.
    try:
        floor = measure_floor(list(p1))
    except FloorError as exc:
        raise P4Error(str(exc)) from None
    val_loss = {
        "role": "descriptive",
        "floor": {k: floor[k] for k in ("n", "mean", "std", "half_width", "rule")},
        "p4": [{"run": d.name, "final_val_loss": loss, **compare(floor, loss)}
               for d, (_, status, loss) in p4.items() if status == "finished"],
    }

    return {
        "rule": f"mean(P4) - mean(other) ± t(1 - {FAMILY_ALPHA}/{2 * N_VERDICTS}, n-1) · "
                "√(s²_P4/n + s²_other/n): Welch at the smaller group's n - 1, Bonferroni over "
                f"{N_VERDICTS} verdicts; an interval containing 0 is 'unresolved'",
        "n_runs": N_RUNS,
        "n_verdicts": N_VERDICTS,
        "t": T_P4,
        "p4_runs": [{"run": d.name, "training_seed": m["config"]["training"]["seed"],
                     "data_seed": m["config"]["data"]["seed"], "status": status}
                    for d, (m, status, _) in p4.items()],
        "diverged": diverged,
        "chat_tokens_expected": CHAT_TOKENS,
        "token_budget": _budget(shared),
        "precision": shared["manifest.device.precision"],
        "eval": eval_summary(first, eval_file),
        "contrasts": contrasts,
        # P4 against E1's other arms and its validation loss beside P1's floor: no verdicts.
        "descriptive": descriptive,
        "val_loss": val_loss,
    }


def p4_against_followup(p4_report: dict, followup_values: dict[str, list[float | None]],
                        followup_eval: dict) -> dict:
    """P4 - E1 50/50 at the 3e-4 follow-up, by the verdicts' arithmetic, described: no verdict.

    P4's numbers are its report's (``reports/p4/verdicts.json``), which must
    have evaluated like the follow-up (``eval_summary`` alike, file aside).
    A diverged run on either side: nothing to describe.
    """
    def alike(summary: dict) -> dict:
        return {k: v for k, v in summary.items() if k != "file"}

    if alike(p4_report["eval"]) != alike(followup_eval):
        raise P4Error(f"P4's report was evaluated with {alike(p4_report['eval'])}, the follow-up with "
                      f"{alike(followup_eval)}: not alike")
    other = CONTRASTS[1]
    described = {}
    for metric in PRIMARY_EVAL_METRICS:
        p4_values, values = p4_report["contrasts"][other][metric]["p4"], followup_values[metric]
        if p4_report["diverged"] or any(v is None for v in values) or len(values) != N_RUNS:
            described[metric] = {"described": "nothing: a run diverged or is missing"}
            continue
        described[metric] = {"p4": p4_values, f"{other}-followup": values, **describe_difference(p4_values, values)}
    return described


__all__ = ["CHAT_TOKENS", "CONTRASTS", "N_RUNS", "N_VERDICTS", "P4Error", "T_P4", "describe_difference",
           "p4_against_followup", "p4_analysis"]
