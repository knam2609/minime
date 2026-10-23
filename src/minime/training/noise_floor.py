"""The seed-noise floor: how far a run moves when only its seeds change.

Before any two pretrains are compared, the baseline (P1) runs at four seed
pairs.  Both seeds vary together (decided with the owner, 2026-10-03):
``training.seed`` sets the initial weights, ``data.seed`` which windows the
sampler draws.  A floor measured over one data order would be too small for
an arm whose data differs from P1's -- P4 by design -- and the stricter
floor costs a same-data arm (P2, P3) at most a "no detectable difference".
(Until 2026-10-05 ``scripts/train.py`` built the model before seeding, so
the floor's runs drew their initial weights at random rather than from
``training.seed``.  They still varied, which is what the floor needs; see
the README's "seed bug".)

The rule was fixed before any floor run finished: an arm's single run differs
from P1 only if its final validation loss lies outside the 95% prediction
interval of the P1 runs,

    mean ± t(0.975, n-1) · s · sqrt(1 + 1/n),

which is ±3.56 s at four seeds (±4.97 s at three, ±3.04 s at five).  It
accounts for ``s`` being estimated from a few runs: a plain "2 s" would call
pure luck a difference ~18% of the time at n = 3.  The fourth seed was added
before any floor run, once fp16 made a run ~75 min (decided with the owner,
2026-10-03): it fills a GPU slot that would otherwise sit idle, and fixing n
up front avoids adding seeds only when a result looks borderline.

This module reads the finished runs, refuses anything but a controlled
repetition, and reports the interval; ``compare`` applies it to an arm, once
``check_comparable`` has found the arm's loss measured on the same validation
windows, budget and precision.

The same interval is measured on any number in a run's ``eval.json``
(``metric="perplexity.personal.loss"``, say), once each run's last checkpoint
has been evaluated and the file copied into its run directory.  The extra
checks: every file is an evaluation of *that* run's final checkpoint (its
config fingerprint, seeds included, and its last step), on the ``val`` split,
and all of them -- the arm's too -- share the eval config, the hashed eval
sets and the device the samples were drawn on.  The training-time validation
windows do not matter there, so an arm with another val set can still be
judged on eval metrics though not on its validation loss.  Only the
``PRIMARY_EVAL_METRICS`` get verdicts; the rest are exploratory.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

from .lr_sweep import SweepError, check_controlled, controlled_view, read_run


class FloorError(ValueError):
    """The runs cannot support a noise floor."""


SEED_KEYS = ("training.seed", "data.seed")
MIN_RUNS = 3
# What an arm shares with the baseline whatever its variable: the same
# validation windows, token budget and precision (``check_comparable``).
COMPARABLE_KEYS = (
    "model.context_length",
    "training.batch_size",
    "training.gradient_accumulation",
    "training.max_steps",
    "training.validation_batches",
    "manifest.data.tokenizer",
    "manifest.data.tokenizer_dev",
    "manifest.device.precision",
)
COMPARABLE_PREFIXES = ("manifest.data.val_shard.",)
# The keys of COMPARABLE_KEYS that only define the validation windows.
VALIDATION_KEYS = ("training.validation_batches",)

VAL_METRIC = "final_val_loss"
EVAL_FILE = "eval-last.json"  # what scripts/evaluate.py writes for last.pt
# The eval metrics a floor is measured on unless others are named.  Decided
# with the owner (2026-10-05), before any floor checkpoint's evaluation was
# read: only the two axes of the frontier -- voice fit and general ability --
# get verdicts, each by the 95% interval.  Ten metrics judged alike would call
# some arm "different" on at least one by luck up to ~40% of the time; two
# keep that under ~10%.  The rest have their floors measured and reported
# beside them as exploratory, never as a result.
PRIMARY_EVAL_METRICS = (
    "perplexity.personal.loss",
    "perplexity.general_weighted.loss",
)
EVAL_METRICS = (
    *PRIMARY_EVAL_METRICS,
    "voice.authorship.discriminator.auc",
    "voice.authorship.attribution.generated.attributed_to_target",
    "voice.stylometry.delta.questioned.generated.nearest_is_target",
    "voice.stylometry.distance_to_real.generated.function_word_jsd",
    "degeneration.generated.rep_4",
    "degeneration.generated.distinct_2",
    "code.valid",
    "blimp.accuracy",
)
# What every eval file of a floor, and an arm's, must share.  The samples are
# drawn per prompt from seeded RNGs, whose streams depend on the device.
EVAL_COMPARABLE_KEYS = (
    "eval_version",
    "eval_config_fingerprint",
    "split",
    "tokenizer.id",
    "device.device",
    "device.precision",
    "device.torch",
)

# Two-sided 95% quantiles of Student's t, t(0.975, df).  Checked against scipy
# by a test; scipy is not a dependency.
T_975 = {
    2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706, 6: 2.4469, 7: 2.3646, 8: 2.3060,
    9: 2.2622, 10: 2.2281, 11: 2.2010, 12: 2.1788, 13: 2.1604, 14: 2.1448,
    15: 2.1314, 16: 2.1199, 17: 2.1098, 18: 2.1009, 19: 2.0930, 20: 2.0860,
}


def measure_floor(run_dirs: list[str | Path], metric: str = VAL_METRIC,
                  eval_file: str = EVAL_FILE) -> dict:
    """The prediction interval of ``run_dirs``' values of ``metric``.

    ``metric`` is the final validation loss, or a dotted path into each run's
    ``eval_file`` (``perplexity.personal.loss``).  Raises ``FloorError``
    unless the runs are one config repeated with every run at its own
    ``training.seed`` *and* its own ``data.seed`` -- and, for an eval metric,
    unless every run's eval file is a controlled evaluation of its own final
    checkpoint (``read_eval``, ``check_eval_comparable``).
    """
    if len(run_dirs) < MIN_RUNS:
        raise FloorError(f"a noise floor needs at least {MIN_RUNS} runs, got {len(run_dirs)}")
    if len(run_dirs) - 1 not in T_975:
        raise FloorError(f"no t quantile for {len(run_dirs)} runs (table: {min(T_975) + 1}-{max(T_975) + 1})")

    manifests: dict[Path, dict] = {}
    views: dict[Path, dict] = {}
    losses: dict[Path, float] = {}
    for run_dir in map(Path, run_dirs):
        try:
            manifest, status, loss = read_run(run_dir)
        except SweepError as exc:
            raise FloorError(str(exc)) from None
        if status != "finished":
            raise FloorError(f"{run_dir}: the baseline {status} -- fix that before measuring a floor")
        manifests[run_dir] = manifest
        views[run_dir] = controlled_view(manifest)
        losses[run_dir] = loss

    try:
        check_controlled(views, set(SEED_KEYS))
    except SweepError as exc:
        raise FloorError(str(exc)) from None
    for key in SEED_KEYS:
        values = [v[key] for v in views.values()]
        if len(set(values)) != len(values):
            raise FloorError(
                f"runs share a {key} ({sorted(values)}): the floor varies both seeds, "
                "so each run needs its own training.seed and data.seed"
            )

    evaluation = None
    if metric == VAL_METRIC:
        values = losses
    else:
        records = {d: read_eval(d, manifests[d], eval_file) for d in manifests}
        first = next(iter(records.values()))
        for record in records.values():
            check_eval_comparable(first, record)
        values = {d: eval_metric(r, metric, d / eval_file) for d, r in records.items()}
        evaluation = eval_summary(first, eval_file)

    shared = next(iter(views.values()))
    floor = {
        "config": shared["name"],
        "metric": metric,
        **_interval(list(values.values())),
        "token_budget": shared["training.max_steps"]
        * shared["training.batch_size"]
        * shared["training.gradient_accumulation"]
        * shared["model.context_length"],
        "precision": shared["manifest.device.precision"],
        "runs": [
            {
                "run": run_dir.name,
                "training_seed": views[run_dir]["training.seed"],
                "data_seed": views[run_dir]["data.seed"],
                metric: values[run_dir],
            }
            for run_dir in views
        ],
    }
    if evaluation is not None:
        floor["eval"] = evaluation
        floor["role"] = metric_role(metric)
    return floor


def metric_role(metric: str) -> str:
    """``primary`` (its verdict is a result) or ``exploratory`` (shown beside its floor)."""
    return "primary" if metric in (VAL_METRIC, *PRIMARY_EVAL_METRICS) else "exploratory"


def _interval(values: list[float]) -> dict:
    n = len(values)
    std = statistics.stdev(values)
    t = T_975[n - 1]
    return {
        "n": n,
        "mean": statistics.fmean(values),
        "std": std,
        "range": max(values) - min(values),
        "rule": "95% prediction interval: mean ± t(0.975, n-1) · std · sqrt(1 + 1/n)",
        "t": t,
        "half_width": t * std * math.sqrt(1 + 1 / n),
    }


# ------------------------------------------------------------ eval metrics


def config_fingerprint(config: dict) -> str:
    """What ``evaluation.harness.load_model`` records as a checkpoint's
    ``config_fingerprint``, computed from a run manifest's config."""
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:12]


def read_eval(run_dir: str | Path, manifest: dict, eval_file: str = EVAL_FILE, split: str = "val") -> dict:
    """``run_dir``'s eval file, refused unless it evaluates that run's final checkpoint on ``split``.

    The checkpoint's config fingerprint covers both seeds, so an eval file of
    another seed's checkpoint -- same config name, same step -- is caught.
    Only the final report reads ``split="test"``.
    """
    run_dir = Path(run_dir)
    path = run_dir / eval_file
    if not path.exists():
        raise FloorError(f"{run_dir}: no {eval_file} -- evaluate the run's last checkpoint "
                         "(scripts/evaluate.py) and copy the file into its run directory")
    record = json.loads(path.read_text())
    checkpoint = record.get("checkpoint") or {}
    if record.get("split") != split:
        raise FloorError(f"{path}: evaluated on the {record.get('split')!r} split; "
                         + ("a floor reads 'val' (the test split is for the final report)" if split == "val"
                            else f"this reads {split!r}"))
    if checkpoint.get("config_fingerprint") != config_fingerprint(manifest["config"]):
        raise FloorError(f"{path}: evaluates a checkpoint of another config or seed "
                         f"({checkpoint.get('config_name')!r}, {checkpoint.get('path')}), not {run_dir.name}'s")
    steps = manifest["config"]["training"]["max_steps"]
    if checkpoint.get("step") != steps:
        raise FloorError(f"{path}: evaluates step {checkpoint.get('step')}, not the final step {steps}")
    if record.get("tokenizer", {}).get("id") != manifest.get("data", {}).get("tokenizer"):
        raise FloorError(f"{path}: evaluated with tokenizer {record.get('tokenizer', {}).get('id')!r}, "
                         f"but the run trained on {manifest.get('data', {}).get('tokenizer')!r}")
    return record


def _eval_view(record: dict) -> dict:
    view = {}
    for key in EVAL_COMPARABLE_KEYS:
        node = record
        for part in key.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        view[key] = node
    view["eval_sets"] = json.dumps(record.get("eval_sets"), sort_keys=True)
    return view


def check_eval_comparable(baseline_eval: dict, arm_eval: dict) -> None:
    """Raise ``FloorError`` unless two eval files measured the same thing the same way."""
    base, arm = _eval_view(baseline_eval), _eval_view(arm_eval)
    differing = sorted(k for k in base if base[k] != arm[k])
    if differing:
        shown = ", ".join(k if k == "eval_sets" else f"{k} ({base[k]!r} vs {arm[k]!r})" for k in differing)
        name = (arm_eval.get("checkpoint") or {}).get("path")
        raise FloorError(f"{name}: not evaluated like the baseline: {shown}")


def eval_metric(record: dict, metric: str, where: str | Path = "eval.json") -> float:
    """The number at dotted path ``metric`` in an eval record."""
    node = record
    for part in metric.split("."):
        if not isinstance(node, dict) or part not in node:
            raise FloorError(f"{where}: no {metric!r} (no {part!r} under it)")
        node = node[part]
    if isinstance(node, bool) or not isinstance(node, (int, float)) or not math.isfinite(node):
        raise FloorError(f"{where}: {metric!r} is {node!r}, not a finite number")
    return float(node)


def eval_summary(record: dict, eval_file: str) -> dict:
    """What a floor report records about the evaluation its values came from."""
    view = _eval_view(record)
    return {
        "file": eval_file,
        **{k: view[k] for k in EVAL_COMPARABLE_KEYS},
        "eval_sets_digest": hashlib.sha256(view["eval_sets"].encode()).hexdigest()[:16],
    }


def check_comparable(baseline_manifest: dict, arm_manifest: dict, validation: bool = True) -> None:
    """Raise ``FloorError`` unless an arm's validation loss can be set beside the baseline's.

    An arm changes one variable, but not these: the validation windows (they
    depend on the tokenizer, the val shard, the context length, the batch
    size and the number of batches), the token budget, and the precision.  A
    loss measured on other windows or after another budget is not a
    difference the floor can judge, however far outside it lands.

    ``validation=False`` is for eval metrics, whose sets are the eval files'
    (``check_eval_comparable``): the training-time validation windows are
    then not checked, the budget and precision still are.
    """
    base, arm = controlled_view(baseline_manifest), controlled_view(arm_manifest)
    if validation:
        keys = set(COMPARABLE_KEYS) | {
            k for k in base.keys() | arm.keys() if k.startswith(COMPARABLE_PREFIXES)
        }
    else:
        keys = set(COMPARABLE_KEYS) - set(VALIDATION_KEYS)
    differing = sorted(k for k in keys if base.get(k) != arm.get(k))
    if differing:
        shown = ", ".join(f"{k} ({base.get(k)!r} vs {arm.get(k)!r})" for k in differing)
        raise FloorError(f"{arm['name']}: not comparable with the baseline: {shown}")


def compare(floor: dict, value: float) -> dict:
    """Where one arm's run falls against the floor's interval.

    A floor whose runs all scored the same (a coarse metric -- 123 code
    prompts -- can) has measured no noise, not zero noise: it judges nothing.
    """
    difference = value - floor["mean"]
    if floor["half_width"] == 0:
        verdict = "not judged: the floor's runs are identical on this metric"
    elif abs(difference) <= floor["half_width"]:
        verdict = "within the floor"
    else:
        verdict = "lower than the baseline" if difference < 0 else "higher than the baseline"
    return {"difference": difference, "half_width": floor["half_width"], "verdict": verdict}


__all__ = [
    "COMPARABLE_KEYS", "EVAL_COMPARABLE_KEYS", "EVAL_FILE", "EVAL_METRICS", "FloorError", "MIN_RUNS",
    "PRIMARY_EVAL_METRICS",
    "SEED_KEYS", "T_975", "VAL_METRIC", "check_comparable", "check_eval_comparable",
    "compare", "config_fingerprint", "eval_metric", "eval_summary", "measure_floor", "metric_role",
    "read_eval",
]
