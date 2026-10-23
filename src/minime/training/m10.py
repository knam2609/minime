"""M10, the flagship: its recipe and the LR sweep's pick (HANDOFF §4a).

Decided with the owner on 2026-10-14, before any M10 code:

* **The recipe** (D1 A): the owner's chat mixed in from step 0 at P4's
  expected exposure -- ``FLAGSHIP_MIXTURE``, 20 parts in 17,250, so
  1,048,576 of the flagship's 904,396,800 tokens, ~3.6 passes over the chat.
* **The LR** (D3 B): AdamW, a sweep at 1/8 of the budget on the flagship's
  own mixture, at four peak LRs fixed upfront (``SWEEP_GRID``).
* **The pick, mechanical**: among completed, finite points, the lowest
  *final* general validation loss at the fixed 431-step budget, read
  unrounded from the manifest (``summary.final_val_loss``; ``metrics.jsonl``
  rounds to 5 decimals); an exact tie goes to the lower LR.  Personal CE,
  samples, throughput, intermediate or ``best.pt`` losses do not rank.  An
  incomplete run is completed or rerun, never dropped; a diverged one is
  reported and excluded.  No finite point -> revisit the grid.
* **The edge**: the end points of all tested LRs, diverged ones included.
  Only a winner at 8e-4 or 6.4e-3 earns one extension, a factor of two
  beyond it (``SWEEP_EXTENSIONS``); ranked again by the same rule, an
  extension that wins means revisiting before the flagship, never a second
  automatic extension.  The winner's margin is reported, with no claim of a
  flagship noise floor: there is none.
* **The adaptation, tested on the flagship itself** (fixed in form the same
  day): three candidates (``CANDIDATES``), each from the unadapted
  flagship's weights with a fresh AdamW, 1,048,576 tokens = 128 steps of
  8,192 at context 1024, 8 warmup steps, cosine to 10%; with F the
  flagship's final optimiser LR and L = min(3e-5, F/4): A 10/90 at L, B
  25/75 at L, C 25/75 at 2L.  Identical seeds and batch layout.  **A
  candidate is accepted only if**, against the unadapted flagship on the
  same native-context val evaluation, personal CE falls by 0.10 or more and
  general-weighted CE rises by 0.02 or less -- practical thresholds, not
  significance claims.  Among those accepted, the largest personal gain; an
  exact tie goes to the lower general CE.  None accepted -> the unadapted
  flagship stays.  Every attempt is reported.  The owner reads the voice
  samples before the final model choice is frozen (``freeze_choice``: the
  gate's pick or the unadapted flagship, nothing else).
* **The test split, once** (``final_report``), after the choice and
  every eval setting are frozen: the common 512-token CE of every row of
  the report, and the full eval of the flagship and the chosen adaptation.
  Reported beside val, never acted on.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from .adaptation import GENERAL_RATIO
from .logging import read_metrics
from .lr_sweep import SweepError, check_controlled, controlled_view, flatten, read_run
from .noise_floor import (
    EVAL_FILE,
    EVAL_METRICS,
    PRIMARY_EVAL_METRICS,
    FloorError,
    check_eval_comparable,
    config_fingerprint,
    eval_metric,
    eval_summary,
    read_eval,
)
from .p4 import describe_difference

FLAGSHIP_CONFIG = "configs/flagship.yaml"
FLAGSHIP_MIXTURE = {"web": 12061, "dialogue": 3446, "code": 1723, "chat": 20}
FLAGSHIP_STEPS = 3450
FLAGSHIP_TOKENS_PER_STEP = 262_144
CHAT_TOKENS = 1_048_576                # P4's expected exposure, and E1 50/50's

SWEEP_GRID = (8e-4, 1.6e-3, 3.2e-3, 6.4e-3)
SWEEP_EXTENSIONS = {8e-4: 4e-4, 6.4e-3: 1.28e-2}
SWEEP_STEPS = 431                      # 1/8 of 3,450 (431.25)
SWEEP_TOKENS = SWEEP_STEPS * FLAGSHIP_TOKENS_PER_STEP   # 112,984,064
SWEEP_KEY = "training.learning_rate"
SWEEP_RULE = ("the lowest final general validation loss at 431 steps (manifest summary.final_val_loss, "
              "unrounded) among completed finite points; an exact tie goes to the lower LR; a winner at "
              "8e-4 or 6.4e-3 earns one point a factor of two beyond it; if that point wins, revisit")


def sweep_config(lr: float) -> str:
    """The config of the sweep point at ``lr``, relative to the repository root."""
    return f"configs/experiments/m10_lr_{lr:.4f}.yaml"


@dataclass
class SweepPoint:
    run: str
    lr: float
    status: str                     # "finished" | "diverged"
    final_val_loss: float | None    # general validation, unrounded


def _is(value: float, candidates) -> bool:
    return any(math.isclose(value, c) for c in candidates)


def pick_sweep_lr(run_dirs: list[str | Path]) -> dict:
    """Check the runs are M10's sweep and apply the rule.

    The report's ``status`` is ``"picked"`` (``pick`` is the flagship's LR),
    ``"extend"`` (run ``next``, then pick again) or ``"revisit"`` (no finite
    point, or the extension won: the grid needs the owner, not a rule).
    Raises ``SweepError`` for runs that are not the protocol's sweep, and for
    an incomplete run.
    """
    if not run_dirs:
        raise SweepError("no sweep runs")
    runs = [(Path(d), *read_run(d)) for d in run_dirs]
    views = {run_dir: controlled_view(manifest) for run_dir, manifest, _, _ in runs}
    for run_dir, view in views.items():
        if view["training.optimizer"] != "adamw":
            raise SweepError(f"{run_dir}: optimizer {view['training.optimizer']!r}; the M10 sweep is AdamW")
        if view["training.max_steps"] != SWEEP_STEPS:
            raise SweepError(f"{run_dir}: {view['training.max_steps']} steps; the sweep runs {SWEEP_STEPS}")
        mixture = {k[len("data.mixture."):]: v for k, v in view.items() if k.startswith("data.mixture.")}
        if mixture != FLAGSHIP_MIXTURE:
            raise SweepError(f"{run_dir}: mixture {mixture}; the sweep runs the flagship's {FLAGSHIP_MIXTURE}")
    check_controlled(views, {SWEEP_KEY})
    shared = next(iter(views.values()))
    tokens = (shared["training.max_steps"] * shared["training.batch_size"]
              * shared["training.gradient_accumulation"] * shared["model.context_length"])
    if tokens != SWEEP_TOKENS:
        raise SweepError(f"{tokens:,} tokens a run; the sweep's budget is {SWEEP_TOKENS:,}")

    points = sorted((SweepPoint(run_dir.name, float(views[run_dir][SWEEP_KEY]), status, loss)
                     for run_dir, _, status, loss in runs), key=lambda p: p.lr)
    lrs = [p.lr for p in points]
    if len(set(lrs)) != len(lrs):
        raise SweepError(f"two runs share an LR: {lrs}")
    missing = [lr for lr in SWEEP_GRID if not _is(lr, lrs)]
    if missing:
        raise SweepError(f"the grid's {missing} have no run")
    extras = [lr for lr in lrs if not _is(lr, SWEEP_GRID)]
    if len(extras) > 1 or any(not _is(x, SWEEP_EXTENSIONS.values()) for x in extras):
        raise SweepError(f"LRs {extras} are not the grid {list(SWEEP_GRID)} plus at most one of "
                         f"{list(SWEEP_EXTENSIONS.values())}")

    def ranked(candidates):
        return sorted((p for p in candidates if p.status == "finished"), key=lambda p: (p.final_val_loss, p.lr))

    report = {
        "metric": "summary.final_val_loss (general val, unrounded)",
        "rule": SWEEP_RULE,
        "grid": list(SWEEP_GRID),
        "extension": extras[0] if extras else None,
        "steps": SWEEP_STEPS,
        "token_budget": tokens,
        "seed": shared["training.seed"],
        "data_seed": shared["data.seed"],
        "precision": shared["manifest.device.precision"],
        "points": [asdict(p) for p in points],
        "diverged": [p.run for p in points if p.status == "diverged"],
    }
    on_grid = ranked(p for p in points if _is(p.lr, SWEEP_GRID))
    if not on_grid:
        return {**report, "status": "revisit", "reason": "no point on the grid finished",
                "pick": None, "pick_run": None, "next": None, "margin": None}
    wanted = next((e for edge, e in SWEEP_EXTENSIONS.items() if math.isclose(on_grid[0].lr, edge)), None)
    extension = report["extension"]
    if extension is not None and (wanted is None or not math.isclose(extension, wanted)):
        raise SweepError(f"the extension {extension} was not called for: the grid's winner is {on_grid[0].lr}")
    if wanted is not None and extension is None:
        return {**report, "status": "extend", "reason": f"the grid's winner {on_grid[0].lr:g} is at its edge",
                "pick": None, "pick_run": None, "next": wanted, "margin": None}

    order = ranked(points)
    winner = order[0]
    margin = order[1].final_val_loss - winner.final_val_loss if len(order) > 1 else None
    if extension is not None and math.isclose(winner.lr, extension):
        return {**report, "status": "revisit", "reason": f"the extension {extension:g} won too",
                "pick": None, "pick_run": winner.run, "next": None, "margin": margin}
    return {**report, "status": "picked", "reason": None, "pick": winner.lr, "pick_run": winner.run,
            "next": None, "margin": margin}


def next_sweep_job(run_dirs: list[str | Path]) -> tuple[str | None, str]:
    """For the notebook, after the four points: the extension's config, or None, and why.

    Never raises: an incomplete or refused sweep queues nothing, and the
    session still packs its results.
    """
    try:
        report = pick_sweep_lr(run_dirs)
    except (SweepError, OSError, KeyError) as exc:
        return None, f"no extension: the sweep cannot be ranked ({exc})"
    if report["status"] == "extend":
        return sweep_config(report["next"]), f"extension {report['next']:g}: {report['reason']}"
    if report["status"] == "picked":
        return None, f"no extension: {report['pick']:g} wins inside the grid"
    return None, f"no extension: {report['reason']} -- revisit with the owner"


# ------------------------------------------------------- the adaptation


class AdaptationError(ValueError):
    """The runs or eval files cannot support the adaptation gate."""


FLAGSHIP_RUN = "flagship-s1337"
FLAGSHIP_EVAL_CONFIG = "configs/eval_flagship.yaml"

ADAPT_STEPS = 128
ADAPT_TOKENS = 1_048_576               # 128 x 8,192
ADAPT_CONTEXT = 1024
ADAPT_WARMUP = 8
ADAPT_MIN_LR_RATIO = 0.1
ADAPT_LR_CAP = 3e-5
# The configs write L to four significant digits: 2.0e-5 for F/4 = 2.0000045e-5.
LR_REL_TOL = 1e-4
# Candidate -> (chat share, peak LR in units of L).
CANDIDATES = {"a": (0.10, 1), "b": (0.25, 1), "c": (0.25, 2)}
PERSONAL_GAIN = 0.10                   # personal CE must fall by at least this
GENERAL_CAP = 0.02                     # general-weighted CE may rise by at most this
GATE_RULE = (f"accepted iff, against the unadapted flagship on the same native-context val evaluation, "
             f"personal CE falls by >= {PERSONAL_GAIN} and general-weighted CE rises by <= {GENERAL_CAP}; "
             "among those accepted, the largest personal fall, an exact tie to the lower general CE; "
             "none accepted -> the unadapted flagship stays")
ADAPT_SOURCES = ("web", "dialogue", "code", "chat")
# The candidates retrained, configs unchanged, after the MPS batch fix
# (HANDOFF §3 M10g): reported beside the originals, never in their place.
RERUN_PREFIX = "m10-rerun-"


@dataclass(frozen=True)
class AdaptJob:
    candidate: str      # "a" | "b" | "c"
    run_name: str
    config: str         # relative to the repository root
    experiment: str = "M10-adaptation"


def adaptation_jobs(rerun: bool = False) -> list[AdaptJob]:
    if rerun:
        return [AdaptJob(c, f"{RERUN_PREFIX}{c}", f"configs/experiments/m10_adapt_{c}.yaml", "M10-rerun")
                for c in CANDIDATES]
    return [AdaptJob(c, f"m10-adapt-{c}", f"configs/experiments/m10_adapt_{c}.yaml") for c in CANDIDATES]


def adapt_train_command(job: AdaptJob, *, repo: Path, root: Path, run_dir: Path, python: str) -> list[str]:
    """``scripts/train.py`` for one candidate: from the flagship's last.pt, validating on both sets."""
    shards = root / "shards"
    return [
        python, "-u", str(repo / "scripts" / "train.py"),
        "--config", str(repo / job.config),
        "--data", str(shards / "general"), str(shards / "personal"),
        "--val-shard", f"general={shards / 'general' / 'val'}", f"personal={shards / 'personal' / 'val'}",
        "--init-from", str(root / "checkpoints" / FLAGSHIP_RUN / "last.pt"),
        "--run-name", job.run_name,
        "--run-dir", str(run_dir),
        "--experiment", job.experiment,
        "--venue", "local",
    ]


def _split_args(split: str) -> list[str]:
    if split not in ("val", "test"):
        raise ValueError(f"split must be 'val' or 'test', not {split!r}")
    return ["--split", "test", "--final-report"] if split == "test" else []


def adapt_eval_command(checkpoint: Path, *, repo: Path, root: Path, python: str, split: str = "val") -> list[str]:
    """``scripts/evaluate.py`` at the flagship's native context: the gate's evaluation (on test, the report's)."""
    return [
        python, "-u", str(repo / "scripts" / "evaluate.py"),
        "--checkpoint", str(checkpoint),
        "--tokenizer", str(repo / "tokenizer" / "minime-16k.json"),
        "--personal", str(root / "shards" / "personal"),
        "--general", str(root / "shards" / "general"),
        "--eval-config", str(repo / FLAGSHIP_EVAL_CONFIG),
        *_split_args(split),
    ]


def final_lr(run_dir: str | Path) -> float:
    """F: the LR of a finished run's last step, as ``metrics.jsonl`` logs it.

    ``apply_schedule`` writes each step's LR into the optimiser's param
    groups, so this is also what the run's ``last.pt`` holds.
    """
    run_dir = Path(run_dir)
    manifest, status, _ = read_run(run_dir)
    steps = manifest["config"]["training"]["max_steps"]
    last = [r for r in read_metrics(run_dir / "metrics.jsonl") if r.get("phase") == "train" and r.get("step") == steps]
    if status != "finished" or not last:
        raise AdaptationError(f"{run_dir}: no logged LR at its final step {steps}")
    return float(last[-1]["lr"])


def candidate_lr(f: float, multiple: int) -> float:
    return multiple * min(ADAPT_LR_CAP, f / 4)


def _sha256s(run_dir: Path) -> dict[str, str]:
    path = run_dir / "SHA256SUMS"
    if not path.exists():
        raise AdaptationError(f"{run_dir}: no SHA256SUMS -- the flagship's checksums came with its results")
    return {name.strip(): digest for digest, name in (line.split(maxsplit=1) for line in path.read_text().splitlines()
                                                     if line.strip())}


def _candidate(view: dict, f: float, where: Path) -> str:
    """Which candidate a run's config is, refused unless it is one of the three exactly."""
    mixture = {k[len("data.mixture."):]: v for k, v in view.items() if k.startswith("data.mixture.")}
    if set(mixture) != set(ADAPT_SOURCES):
        raise AdaptationError(f"{where}: mixture {mixture} must name exactly {list(ADAPT_SOURCES)}")
    total = sum(mixture.values())
    general = total - mixture["chat"]
    ratio_total = sum(GENERAL_RATIO.values())
    if general <= 0 or any(not math.isclose(mixture[n] / general, part / ratio_total, rel_tol=1e-9)
                           for n, part in GENERAL_RATIO.items()):
        raise AdaptationError(f"{where}: mixture {mixture}: the general part is not {GENERAL_RATIO}")
    share, lr = mixture["chat"] / total, float(view["training.learning_rate"])
    for name, (c_share, multiple) in CANDIDATES.items():
        if math.isclose(share, c_share, rel_tol=1e-9) and math.isclose(lr, candidate_lr(f, multiple),
                                                                       rel_tol=LR_REL_TOL):
            return name
    raise AdaptationError(f"{where}: chat share {share:g} at LR {lr:g} is none of the candidates "
                          f"(L = {candidate_lr(f, 1):.6g}: {CANDIDATES})")


def _check_recipe(view: dict, where: Path) -> None:
    expected = {
        "role": "adaptation",
        "training.optimizer": "adamw",
        "training.max_steps": ADAPT_STEPS,
        "training.warmup_steps": ADAPT_WARMUP,
        "training.min_lr_ratio": ADAPT_MIN_LR_RATIO,
        "model.context_length": ADAPT_CONTEXT,
    }
    wrong = {k: view.get(k) for k, v in expected.items() if view.get(k) != v}
    if wrong:
        raise AdaptationError(f"{where}: not the adaptation recipe: {wrong} (expected {expected})")
    tokens = (view["training.max_steps"] * view["training.batch_size"]
              * view["training.gradient_accumulation"] * view["model.context_length"])
    if tokens != ADAPT_TOKENS:
        raise AdaptationError(f"{where}: {tokens:,} tokens; a candidate runs {ADAPT_TOKENS:,}")


def candidate_of(config: dict, f: float, where: str | Path) -> str:
    """Which candidate ``config`` is (a manifest's, or ``Config.to_dict()``).

    Refused unless it is the adaptation recipe and exactly one of the three
    candidates at the LR that ``f``, the flagship's final LR, gives.
    """
    view = flatten(config)
    _check_recipe(view, where)
    return _candidate(view, f, where)


def _native(record: dict, where: Path) -> None:
    ppl = record.get("perplexity") or {}
    if "context_length" not in ppl or ppl["context_length"] != ppl.get("native_context_length"):
        raise AdaptationError(f"{where}: not a native-context evaluation (CE on {ppl.get('context_length')} "
                              f"of {ppl.get('native_context_length')}); the gate reads native ones")


def _values(record: dict, where: Path) -> dict:
    out = {}
    for metric in EVAL_METRICS:
        try:
            out[metric] = eval_metric(record, metric, where)
        except FloorError as exc:
            if metric in PRIMARY_EVAL_METRICS:
                raise AdaptationError(str(exc)) from None
            out[metric] = None  # exploratory and missing: reported, never blocking
    return out


def gate_adaptation(flagship_dir: str | Path, candidate_dirs: list[str | Path],
                    eval_file: str = EVAL_FILE) -> dict:
    """Check the candidates are the protocol's and apply the acceptance gate.

    ``flagship_dir`` holds the flagship's manifest, metrics, ``SHA256SUMS``
    and its native-context ``eval_file``; each candidate's directory its
    manifest and ``eval_file``.  The report's ``status`` is ``"accepted"``
    (``pick``) or ``"none accepted"`` (the unadapted flagship stays).  A
    diverged candidate is reported and fails the gate.  Raises
    ``AdaptationError`` for anything else that is not the protocol: a
    candidate missing, twice, from another checkpoint, or evaluated unlike
    the flagship.
    """
    flagship_dir = Path(flagship_dir)
    try:
        flagship, f_status, _ = read_run(flagship_dir)
    except SweepError as exc:
        raise AdaptationError(str(exc)) from None
    mixture = flagship["config"]["data"]["mixture"]
    if f_status != "finished" or mixture != FLAGSHIP_MIXTURE:
        raise AdaptationError(f"{flagship_dir}: not the finished flagship ({f_status}, mixture {mixture})")
    f = final_lr(flagship_dir)
    flagship_sha = _sha256s(flagship_dir).get("last.pt")
    flagship_fp = config_fingerprint(flagship["config"])

    runs = {}
    for d in candidate_dirs:
        d = Path(d)
        try:
            manifest, status, _ = read_run(d)
        except SweepError as exc:
            raise AdaptationError(str(exc)) from None
        init = manifest.get("init_from") or {}
        if init.get("sha256") != flagship_sha or init.get("config_fingerprint") != flagship_fp:
            raise AdaptationError(f"{d}: started from {init.get('path') or 'no checkpoint'} "
                                  f"(sha256 {str(init.get('sha256'))[:12]}), not the flagship's last.pt "
                                  f"({str(flagship_sha)[:12]})")
        name = candidate_of(manifest["config"], f, d)
        view = controlled_view(manifest)
        if name in runs:
            raise AdaptationError(f"two runs of candidate {name.upper()}: {runs[name][0].name}, {d.name}")
        runs[name] = (d, manifest, status, view)
    missing = sorted(set(CANDIDATES) - set(runs))
    if missing:
        raise AdaptationError(f"candidate(s) {[m.upper() for m in missing]} have no run: every attempt is reported, "
                              "so the gate waits for all three")
    try:
        check_controlled({d: view for d, _, _, view in runs.values()},
                         {"training.learning_rate", *(f"data.mixture.{s}" for s in ADAPT_SOURCES)})
    except SweepError as exc:
        raise AdaptationError(str(exc)) from None

    try:
        base_eval = read_eval(flagship_dir, flagship, eval_file)
        _native(base_eval, flagship_dir / eval_file)
        evals = {}
        for name, (d, manifest, status, _) in runs.items():
            if status != "finished":
                continue
            evals[name] = read_eval(d, manifest, eval_file)
            _native(evals[name], d / eval_file)
            check_eval_comparable(base_eval, evals[name])
    except FloorError as exc:
        raise AdaptationError(str(exc)) from None

    personal, general = PRIMARY_EVAL_METRICS
    base_values = _values(base_eval, flagship_dir / eval_file)
    candidates = {}
    for name in CANDIDATES:
        d, manifest, status, view = runs[name]
        share, multiple = CANDIDATES[name]
        entry = {"run": d.name, "status": status, "chat_share": share, "lr": float(view["training.learning_rate"]),
                 "lr_in_l": multiple, "final_val_losses": (manifest.get("summary") or {}).get("final_val_losses")}
        if status != "finished":
            candidates[name] = {**entry, "accepted": False, "reason": "diverged"}
            continue
        values = _values(evals[name], d / eval_file)
        change = {m: None if values[m] is None or base_values[m] is None else values[m] - base_values[m]
                  for m in EVAL_METRICS}
        personal_ok = change[personal] <= -PERSONAL_GAIN
        general_ok = change[general] <= GENERAL_CAP
        reason = None if personal_ok and general_ok else "; ".join(
            text for ok, text in (
                (personal_ok, f"personal CE {change[personal]:+.4f}, needs <= {-PERSONAL_GAIN:+.2f}"),
                (general_ok, f"general-weighted CE {change[general]:+.4f}, needs <= {GENERAL_CAP:+.2f}"),
            ) if not ok)
        candidates[name] = {**entry, "values": values, "change": change,
                            "accepted": personal_ok and general_ok, "reason": reason}

    accepted = sorted((n for n, c in candidates.items() if c["accepted"]),
                      key=lambda n: (candidates[n]["change"][personal], candidates[n]["values"][general]))
    pick = accepted[0] if accepted else None
    return {
        "rule": GATE_RULE,
        "status": "accepted" if pick else "none accepted",
        "pick": pick,
        "pick_run": candidates[pick]["run"] if pick else FLAGSHIP_RUN,
        "final_choice_frozen": False,   # until the owner has read the voice samples
        "flagship": {"run": flagship_dir.name, "last_sha256": flagship_sha, "final_lr": f,
                     "values": base_values},
        "l": candidate_lr(f, 1),
        "token_budget": ADAPT_TOKENS,
        "data_seed": next(iter(runs.values()))[3]["data.seed"],
        "precision": next(iter(runs.values()))[3]["manifest.device.precision"],
        "eval": eval_summary(base_eval, eval_file),
        "candidates": candidates,
    }


def freeze_choice(report: dict, choice: str, date: str) -> dict:
    """The gate's report with the owner's final model choice frozen into it.

    The owner chooses after reading the voice samples, between the gate's
    pick and the unadapted flagship: an accepted candidate can be turned
    down, a rejected one never chosen.  ``choice`` is the pick's letter or
    ``"flagship"``; ``date`` the day it was made (ISO).
    """
    allowed = ["flagship", *([report["pick"]] if report["pick"] else [])]
    if choice not in allowed:
        raise AdaptationError(f"final choice {choice!r}: the owner chooses between {allowed}, "
                              "the gate's pick and the unadapted flagship")
    run = report["flagship"]["run"] if choice == "flagship" else report["candidates"][choice]["run"]
    frozen = {}
    for key, value in report.items():
        frozen[key] = value
        if key == "final_choice_frozen":
            frozen.update(final_choice_frozen=True, final_choice=choice, final_run=run,
                          final_choice_made=f"{date}, by the owner, after reading the voice samples")
    return frozen


def chosen_adaptation(adaptation: dict | None) -> str | None:
    """The adaptation run the report carries: the gate's pick, unless the owner kept the flagship."""
    if adaptation is None or adaptation.get("status") != "accepted":
        return None
    if adaptation.get("final_choice_frozen") and adaptation["final_run"] != adaptation["pick_run"]:
        return None
    return adaptation["pick_run"]


# ------------------------------------------------- common-context CE


COMMON_EVAL_CONFIG = "configs/eval_common512.yaml"
COMMON_CONTEXT = 512
COMMON_EVAL_FILE = f"eval-last-ctx{COMMON_CONTEXT}.json"
COMMON_EVAL_FILES = {"val": COMMON_EVAL_FILE, "test": f"eval-last-ctx{COMMON_CONTEXT}-test.json"}
# The report's rows (HANDOFF §4a), each scored on 512-token windows: the
# probes four runs each, the unadapted flagship, and the chosen adaptation.
# Values: where each run's checkpoint is under the data root -- the Kaggle
# runs' downloads in checkpoints/, the laptop's runs in runs/.
COMMON_GROUPS = {
    "P1": tuple(f"p1-s{s}" for s in range(1337, 1341)),
    "P4": tuple(f"p4-s{s}" for s in range(1337, 1341)),
    "E1 25/75": tuple(f"e1-m25-p1s{s}" for s in range(1337, 1341)),
    "E1 50/50": tuple(f"e1-m50-p1s{s}" for s in range(1337, 1341)),
    "flagship": (FLAGSHIP_RUN,),
}
LAPTOP_RUN_PREFIXES = ("e1-", "m10-adapt-", RERUN_PREFIX)
COMMON_METRICS = (*PRIMARY_EVAL_METRICS, "perplexity.general.web.loss",
                  "perplexity.general.dialogue.loss", "perplexity.general.code.loss")
# A probe's native context is 512: its common-context CE scores the same
# windows as its native evaluation, only batched differently.
NATIVE_AGREEMENT = 1e-4


def common_rows(adaptation: dict | None) -> dict[str, tuple[str, ...]]:
    """``COMMON_GROUPS``, plus the adaptation the gate accepted and the owner kept (``chosen_adaptation``)."""
    rows = dict(COMMON_GROUPS)
    run = chosen_adaptation(adaptation)
    if run is not None:
        rows["adaptation"] = (run,)
    return rows


def non_dominated(points: dict[str, tuple[float, float]]) -> list[str]:
    """The names of the points no other point matches or beats on both coordinates (lower is better).

    Sorted by the first coordinate: the frontier's group means, for the plot.
    Two identical points both stay.
    """
    front = [name for name, (x, y) in points.items()
             if not any(ox <= x and oy <= y and (ox, oy) != (x, y)
                        for other, (ox, oy) in points.items() if other != name)]
    return sorted(front, key=lambda name: points[name])


def checkpoint_path(root: Path, run: str) -> Path:
    where = "runs" if run.startswith(LAPTOP_RUN_PREFIXES) else "checkpoints"
    return root / where / run / "last.pt"


def common_eval_command(checkpoint: Path, *, repo: Path, root: Path, python: str, split: str = "val") -> list[str]:
    """``scripts/evaluate.py``, CE only on 512-token windows: writes ``COMMON_EVAL_FILE`` (or the test one)."""
    return [
        python, "-u", str(repo / "scripts" / "evaluate.py"),
        "--checkpoint", str(checkpoint),
        "--tokenizer", str(repo / "tokenizer" / "minime-16k.json"),
        "--personal", str(root / "shards" / "personal"),
        "--general", str(root / "shards" / "general"),
        "--eval-config", str(repo / COMMON_EVAL_CONFIG),
        "--no-samples",
        *_split_args(split),
    ]


def common_context_table(experiments: str | Path, adaptation: dict | None, split: str = "val") -> dict:
    """The common-context CE of every row, checked alike; descriptive, no verdicts.

    Each run's ``COMMON_EVAL_FILES[split]`` must evaluate its own final
    checkpoint on 512-token windows of ``split``, and every file must be
    comparable with every other (``check_eval_comparable``).  On val, for a
    run whose own context is 512, the values must agree with its native
    ``EVAL_FILE`` to ``NATIVE_AGREEMENT``; the test split has no native CE to
    check against for those runs (the test phase scores them once).
    """
    experiments = Path(experiments)
    eval_file = COMMON_EVAL_FILES[split]
    rows = common_rows(adaptation)
    first, groups, agreement = None, {}, 0.0
    try:
        for group, runs in rows.items():
            members = {}
            for run in runs:
                run_dir = experiments / run
                manifest, status, _ = read_run(run_dir)
                if status != "finished":
                    raise AdaptationError(f"{run_dir}: {status}; every row is a finished run")
                record = read_eval(run_dir, manifest, eval_file, split)
                ppl = record.get("perplexity") or {}
                if ppl.get("context_length") != COMMON_CONTEXT:
                    raise AdaptationError(f"{run_dir / eval_file}: CE on {ppl.get('context_length')}-token "
                                          f"windows, not {COMMON_CONTEXT}")
                first = first or record
                check_eval_comparable(first, record)
                values = {m: eval_metric(record, m, run_dir / eval_file) for m in COMMON_METRICS}
                if split == "val" and ppl["native_context_length"] == COMMON_CONTEXT:
                    native = read_eval(run_dir, manifest, EVAL_FILE)
                    for m in PRIMARY_EVAL_METRICS:
                        native_value = eval_metric(native, m, run_dir / EVAL_FILE)
                        gap = abs(values[m] - native_value)
                        if gap > NATIVE_AGREEMENT:
                            raise AdaptationError(f"{run_dir}: {m} on its own 512-token windows is {values[m]:.6f} "
                                                  f"here, {native_value:.6f} natively -- the same windows scored "
                                                  "differently")
                        agreement = max(agreement, gap)
                members[run] = {"native_context_length": ppl["native_context_length"], "values": values}
            groups[group] = {
                "runs": members,
                "mean": {m: statistics.fmean(v["values"][m] for v in members.values()) for m in COMMON_METRICS},
                "sd": ({m: statistics.stdev(v["values"][m] for v in members.values()) for m in COMMON_METRICS}
                       if len(members) > 1 else None),
            }
    except (SweepError, FloorError) as exc:
        raise AdaptationError(str(exc)) from None
    return {
        "split": split,
        "context_length": COMMON_CONTEXT,
        "eval_file": eval_file,
        "eval": eval_summary(first, eval_file),
        "reading": "descriptive: the probes are the controlled study, the flagship and its adaptation single runs "
                   "of another size, budget and context; no difference is attributed to any one of them",
        "adaptation_final": None if adaptation is None else adaptation.get("final_choice_frozen"),
        "native_agreement": ({"max_abs_difference": agreement, "allowed": NATIVE_AGREEMENT}
                             if split == "val" else None),
        "groups": groups,
    }


# ------------------------------------------------------ the test split, once


TEST_EVAL_FILE = "eval-last-test.json"    # the full eval on test, at the native context
TEST_READING = ("the personal test split, read once after the final choice and every eval setting were "
                "frozen; reported beside val, never acted on -- no rule is applied to these numbers and they "
                "choose nothing.  BLiMP's items do not depend on the split")


# P4's two contrasts, described on each split's common-context CE (the owner, 2026-10-19).
P4_DESCRIBED = {"p1": "P1", "e1-50": "E1 50/50"}
P4_DESCRIBED_READING = ("description, not a verdict: P4's verdict arithmetic (Welch at df 3, Bonferroni over four) "
                        "on each split's 512-token CE, the probes' own CE.  The verdicts are reports/p4/verdicts.json, "
                        "on val, and stand; the val rows here reproduce them")


def p4_described(table: dict) -> dict:
    """P4 - P1 and P4 - E1 50/50 on one common-context table, by the verdicts' arithmetic, with no verdict."""
    groups = table["groups"]

    def values(group: str, metric: str) -> list[float]:
        return [run["values"][metric] for run in groups[group]["runs"].values()]

    return {name: {metric: describe_difference(values("P4", metric), values(group, metric))
                   for metric in PRIMARY_EVAL_METRICS}
            for name, group in P4_DESCRIBED.items()}


def full_eval_rows(adaptation: dict | None) -> tuple[str, ...]:
    """The runs given the full eval on test: the unadapted flagship and the chosen adaptation, if any."""
    run = chosen_adaptation(adaptation)
    return (FLAGSHIP_RUN,) if run is None else (FLAGSHIP_RUN, run)


def final_report_runs(adaptation: dict | None) -> list[tuple[str, str]]:
    """Every evaluation of the test phase, as ``(run, eval file)``: the full ones first, then CE."""
    if adaptation is None or not adaptation.get("final_choice_frozen"):
        raise AdaptationError("the final model choice is not frozen (reports/m10/adaptation.json): the test "
                              "split waits until it is")
    common = [(run, COMMON_EVAL_FILES["test"]) for runs in common_rows(adaptation).values() for run in runs]
    return [*((run, TEST_EVAL_FILE) for run in full_eval_rows(adaptation)), *common]


def final_report(experiments: str | Path, adaptation: dict | None) -> dict:
    """The test split's numbers, checked as the val ones were, set beside them.

    The full evals of ``full_eval_rows`` must be native-context evaluations of
    each run's final checkpoint on test, comparable with each other; their
    val counterparts are the gate's ``EVAL_FILE``.  The common-context rows
    are ``common_context_table`` on test.  Differences are described, not
    judged; P4's two contrasts too (``p4_described``), on both splits.
    """
    final_report_runs(adaptation)   # refuses an unfrozen choice
    experiments = Path(experiments)
    files = {"val": EVAL_FILE, "test": TEST_EVAL_FILE}
    full, first = {}, None
    try:
        for run in full_eval_rows(adaptation):
            run_dir = experiments / run
            manifest, status, _ = read_run(run_dir)
            if status != "finished":
                raise AdaptationError(f"{run_dir}: {status}; every row is a finished run")
            evals = {split: read_eval(run_dir, manifest, f, split) for split, f in files.items()}
            for split, record in evals.items():
                _native(record, run_dir / files[split])
            first = first or evals
            for split in files:
                check_eval_comparable(first[split], evals[split])
            full[run] = {split: _values(evals[split], run_dir / files[split]) for split in files}
    except (SweepError, FloorError) as exc:
        raise AdaptationError(str(exc)) from None
    if first["val"]["eval_config_fingerprint"] != first["test"]["eval_config_fingerprint"]:
        raise AdaptationError(f"{FLAGSHIP_RUN}: evaluated on test with eval "
                              f"{first['test']['eval_config_fingerprint']}, on val with "
                              f"{first['val']['eval_config_fingerprint']} -- the test phase changes no setting")
    val_table = common_context_table(experiments, adaptation, "val")
    test_table = common_context_table(experiments, adaptation, "test")
    chosen = chosen_adaptation(adaptation)
    change = None
    if chosen is not None:
        change = {split: {m: None if full[chosen][split][m] is None or full[FLAGSHIP_RUN][split][m] is None
                          else full[chosen][split][m] - full[FLAGSHIP_RUN][split][m] for m in EVAL_METRICS}
                  for split in ("val", "test")}
    return {
        "reading": TEST_READING,
        "final_run": adaptation["final_run"],
        "final_choice_made": adaptation["final_choice_made"],
        "full": {"eval": {"val": eval_summary(first["val"], EVAL_FILE),
                          "test": eval_summary(first["test"], TEST_EVAL_FILE)},
                 "runs": full,
                 "adaptation_minus_flagship": change},
        "common_context": test_table,
        "p4_described": {"reading": P4_DESCRIBED_READING, "val": p4_described(val_table),
                         "test": p4_described(test_table)},
    }


__all__ = [
    "ADAPT_TOKENS", "CANDIDATES", "CHAT_TOKENS", "COMMON_CONTEXT", "COMMON_EVAL_CONFIG", "COMMON_EVAL_FILE",
    "COMMON_EVAL_FILES", "COMMON_GROUPS", "checkpoint_path", "chosen_adaptation", "common_context_table",
    "common_eval_command", "common_rows", "FLAGSHIP_CONFIG", "FLAGSHIP_EVAL_CONFIG", "FLAGSHIP_MIXTURE",
    "FLAGSHIP_RUN", "GENERAL_CAP", "PERSONAL_GAIN", "RERUN_PREFIX", "SWEEP_EXTENSIONS", "SWEEP_GRID", "SWEEP_STEPS",
    "SWEEP_TOKENS", "TEST_EVAL_FILE", "AdaptJob", "AdaptationError", "SweepPoint", "adapt_eval_command",
    "adapt_train_command", "adaptation_jobs", "candidate_lr", "candidate_of", "final_lr", "full_eval_rows",
    "freeze_choice", "gate_adaptation", "next_sweep_job", "non_dominated", "pick_sweep_lr", "sweep_config", "final_report",
    "final_report_runs", "p4_described",
]
