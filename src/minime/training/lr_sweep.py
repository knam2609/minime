"""The E4 learning-rate protocol: each optimiser runs at its own best LR.

Comparing two optimisers at one shared learning rate measures which of them
that rate happens to suit.  So each optimiser gets a short sweep -- the probe
model at a quarter of its token budget, one run per LR, everything else
fixed (``configs/experiments/e4_lr_*``) -- and its E4 arm runs at the LR with
the lowest final validation loss.

The sweeps are sequential.  AdamW goes first; the Muon arm keeps AdamW for
the embedding and the norms, at the AdamW sweep's winning LR, so that the two
arms differ in the hidden matrices and nowhere else.  The Muon sweep then
varies Muon's own LR.

This module reads finished sweep runs and makes the pick.  It refuses, loudly,
when the runs are not a controlled sweep (anything but the swept LR differs),
when a run did not finish, when the Muon sweep was run at a different AdamW
LR than the AdamW sweep chose, and when the winner is at the edge of the grid,
where the optimum may lie beyond it.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .logging import read_metrics


class SweepError(ValueError):
    """The runs cannot support a learning-rate choice."""


@dataclass(frozen=True)
class Sweep:
    optimizer: str
    key: str  # dotted config path of the swept LR
    # (sweep, key): this sweep's runs must use that sweep's winner for that key.
    anchor: tuple[str, str] | None = None


SWEEPS = {
    "adamw": Sweep(optimizer="adamw", key="training.learning_rate"),
    "muon": Sweep(
        optimizer="muon", key="training.muon_lr", anchor=("adamw", "training.learning_rate")
    ),
}

MIN_POINTS = 3
# Differ between sweep points by construction.  ``role`` is a label too
# (pilot / main), and runs from before it existed have none.
_UNCONTROLLED = {"name", "notes", "role"}
# Where the data lives is not what the data is: a Kaggle path and a local path
# to the same shards are the same experiment.
_DATA_UNCONTROLLED = {"data_dir"}


@dataclass
class SweepPoint:
    run: str
    value: float
    status: str  # "finished" | "diverged"
    final_val_loss: float | None


def flatten(tree: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in tree.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, f"{dotted}."))
        else:
            out[dotted] = value
    return out


def read_run(run_dir: str | Path) -> tuple[dict, str, float | None]:
    """``(manifest, status, final_val_loss)`` for one sweep run."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise SweepError(f"{run_dir}: no manifest.json -- is this a training run directory?")
    manifest = json.loads(manifest_path.read_text())

    summary = manifest.get("summary")
    if summary:
        max_steps = manifest["config"]["training"]["max_steps"]
        if summary.get("steps") != max_steps:
            raise SweepError(f"{run_dir}: ran {summary.get('steps')} of {max_steps} steps")
        loss = summary.get("final_val_loss")
        if loss is None or not math.isfinite(loss):
            raise SweepError(f"{run_dir}: finished without a finite final validation loss")
        return manifest, "finished", float(loss)

    if paused := manifest.get("paused"):
        raise SweepError(f"{run_dir}: paused at step {paused.get('step')} of "
                         f"{manifest['config']['training']['max_steps']}: resume it to finish, then pick")
    # A diverged run is a result -- too high an LR -- not a missing one.
    metrics = run_dir / "metrics.jsonl"
    if metrics.exists() and any(r.get("phase") == "diverged" for r in read_metrics(metrics)):
        return manifest, "diverged", None
    raise SweepError(f"{run_dir}: neither finished nor diverged (killed mid-run?)")


def controlled_view(manifest: dict) -> dict[str, Any]:
    """Everything about a run that must match its comparison runs, flattened.

    The config, plus the manifest's data block (tokenizer id, shard sizes) --
    the config alone does not say what the model was trained on -- and the
    realised precision.
    """
    data = {
        f"manifest.data.{k}": v
        for k, v in flatten(manifest.get("data", {})).items()
        if k not in _DATA_UNCONTROLLED
    }
    # So is the precision: the config usually says "auto", and what that
    # resolved to depends on the GPU the run landed on.
    data["manifest.device.precision"] = manifest.get("device", {}).get("precision")
    return {**flatten(manifest["config"]), **data}


def pick_lr(
    sweep_name: str,
    run_dirs: list[str | Path],
    anchor_report: dict | None = None,
) -> dict:
    """Check that ``run_dirs`` form a controlled sweep and pick the winning LR.

    ``anchor_report`` is the pick of the sweep this one is anchored to (the
    AdamW report, for the Muon sweep).  Returns a JSON-able report; raises
    ``SweepError`` for anything that makes the choice untrustworthy, except a
    winner at the grid's edge, which the report flags as ``at_edge``.
    """
    if sweep_name not in SWEEPS:
        raise SweepError(f"unknown sweep {sweep_name!r} (known: {sorted(SWEEPS)})")
    sweep = SWEEPS[sweep_name]
    if len(run_dirs) < MIN_POINTS:
        raise SweepError(f"a sweep needs at least {MIN_POINTS} runs, got {len(run_dirs)}")

    runs = [(Path(d), *read_run(d)) for d in run_dirs]
    configs = {}
    points: list[SweepPoint] = []
    for run_dir, manifest, status, loss in runs:
        config = controlled_view(manifest)
        if config["training.optimizer"] != sweep.optimizer:
            raise SweepError(
                f"{run_dir}: optimizer {config['training.optimizer']!r} "
                f"in the {sweep_name} sweep"
            )
        configs[run_dir] = config
        points.append(SweepPoint(run_dir.name, float(config[sweep.key]), status, loss))

    check_controlled(configs, {sweep.key})

    values = [p.value for p in points]
    if len(set(values)) != len(values):
        raise SweepError(f"two runs share a {sweep.key} value: {sorted(values)}")

    anchor = None
    if sweep.anchor is not None:
        anchor = _check_anchor(sweep_name, sweep.anchor, anchor_report, configs)

    finished = [p for p in points if p.status == "finished"]
    if not finished:
        raise SweepError("every run diverged: lower the grid")
    points.sort(key=lambda p: p.value)
    ranked = sorted(finished, key=lambda p: p.final_val_loss)
    winner = ranked[0]
    at_edge = winner.value in (points[0].value, points[-1].value)

    shared = next(iter(configs.values()))
    return {
        "sweep": sweep_name,
        "key": sweep.key,
        "optimizer": sweep.optimizer,
        "winner": winner.value,
        "winner_run": winner.run,
        "winner_final_val_loss": winner.final_val_loss,
        # Gap to the runner-up.  A gap under the seed-noise floor means the
        # two LRs are tied and the choice between them is arbitrary.
        "margin": (ranked[1].final_val_loss - winner.final_val_loss) if len(ranked) > 1 else None,
        "at_edge": at_edge,
        "anchor": anchor,
        "token_budget": shared["training.max_steps"]
        * shared["training.batch_size"]
        * shared["training.gradient_accumulation"]
        * shared["model.context_length"],
        "seed": shared["training.seed"],
        "points": [asdict(p) for p in points],
    }


def check_controlled(configs: dict[Path, dict], varied: set[str]) -> None:
    """Refuse runs (``controlled_view``s) that differ in anything but ``varied``."""
    all_keys = set().union(*configs.values())
    differing = sorted(
        k
        for k in all_keys - _UNCONTROLLED - varied
        if len({json.dumps(c.get(k), sort_keys=True) for c in configs.values()}) > 1
    )
    if differing:
        detail = "; ".join(
            f"{k}: " + ", ".join(f"{d.name}={c.get(k)!r}" for d, c in configs.items())
            for k in differing[:5]
        )
        raise SweepError(
            f"not a controlled comparison: besides {sorted(varied)}, the runs differ in "
            f"{differing}. {detail}"
        )


def _check_anchor(
    sweep_name: str,
    anchor: tuple[str, str],
    report: dict | None,
    configs: dict[Path, dict],
) -> dict:
    anchor_sweep, anchor_key = anchor
    if report is None:
        raise SweepError(
            f"the {sweep_name} sweep is anchored to the {anchor_sweep} sweep's pick: "
            f"pick {anchor_sweep} first"
        )
    if report.get("sweep") != anchor_sweep or report.get("key") != anchor_key:
        raise SweepError(f"anchor report is for {report.get('sweep')}/{report.get('key')}")
    if report.get("at_edge"):
        raise SweepError(
            f"the {anchor_sweep} winner is at the edge of its grid; extend that sweep first"
        )
    wanted = report["winner"]
    wrong = {d.name: c[anchor_key] for d, c in configs.items() if c[anchor_key] != wanted}
    if wrong:
        raise SweepError(
            f"the {sweep_name} runs must use {anchor_key}={wanted} (the {anchor_sweep} "
            f"winner); these did not: {wrong}"
        )
    return {"sweep": anchor_sweep, "key": anchor_key, "value": wanted}


__all__ = [
    "SWEEPS", "Sweep", "SweepError", "SweepPoint", "check_controlled", "controlled_view",
    "flatten", "pick_lr", "read_run",
]
