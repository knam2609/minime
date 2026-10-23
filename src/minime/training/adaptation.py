"""E1: adapting a pretrained P1 checkpoint to the owner's messages.

The protocol was decided with the owner on 2026-10-06, before any adaptation
run (README, "E1 adaptation protocol"; HANDOFF §4):

* Every run starts from a P1 final checkpoint's weights (``train.py
  --init-from``) with a fresh AdamW and schedule: 2,097,152 tokens, 256 steps
  of 8,192, on the laptop -- every arm and its control alike.
* Four arms set the personal share of the windows: 0 (the control: general
  only), 25, 50 and 100%.  Personal means ``chat``; ``comment`` (800 tokens)
  stays at 0.  The general part keeps the corpus's 7 : 2 : 1.
* An LR pilot -- 1e-4, 3e-4, 1e-3, each the 50/50 arm from ``p1-s1337`` on a
  ``data.seed`` no main run uses -- picks the peak LR by the lowest final
  personal validation loss (an exact tie to the lower LR).  An edge winner
  earns one point beyond it (3e-5 or 3e-3), and the search stops there: if
  that point wins too, it is the best *tested* LR, still at the edge.  A
  diverged pilot is reported and drops out of the ranking.
* The main runs: every arm from each of the four P1 checkpoints, at the
  picked LR.  The arms from one base share a ``data.seed``; the bases do not.
  Runs are paired by the sha256 of the checkpoint they started from and by
  that seed, never by name.
* Six verdicts -- three arms against the control, on the two primary eval
  metrics -- each on the paired differences d_i = arm - control over the
  bases, by a Bonferroni interval: mean d ± t(1 - 0.05/12, n-1) · s_d / √n,
  ±3.12 s_d at four bases, at least 95% coverage for the six together.  An
  interval containing zero is "unresolved", never "no difference".  A run
  that diverged is its strategy failing on that base: its contrast gets no
  verdict rather than one over fewer bases.

The 3e-4 follow-up (the owner, 2026-10-19; README "E1 follow-up at 3e-4"),
fixed before any of its runs: the control and the 50/50 arm from each P1
base at the pilot's 3e-4, every other setting the main runs', each on its
base's main-run ``data.seed``.  A second recipe, reported beside 1e-3's and
never instead of it: its own family of two verdicts (50/50 - control on the
two primary metrics, ``T_BONFERRONI_FOLLOWUP``), and descriptive beside them,
the change from the unchanged base (the control's is the restart's own cost)
and from the same arm at 1e-3 on the same base (``against_main``).  Val
only: the test split was read once, before it.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from .lr_sweep import SweepError, check_controlled, controlled_view, read_run
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


class E1Error(ValueError):
    """The runs cannot support an E1 pick or verdict."""


PERSONAL_SOURCES = ("chat", "comment")
# The general part of every arm keeps the corpus's own proportions.
GENERAL_RATIO = {"web": 7, "dialogue": 2, "code": 1}
# comment.train is 800 tokens (288 window starts): any weight near chat's
# would replay those few windows hundreds of times.
ZERO_SOURCES = ("comment",)

# Validation sets, as scripts/run_e1.py names them.  The first is val_loss.
GENERAL_VAL = "general"
PERSONAL_VAL = "personal"

# Personal share -> the arm's config.  0 is the control.
ARMS = {0.0: "e1_m0.yaml", 0.25: "e1_m25.yaml", 0.5: "e1_m50.yaml", 1.0: "e1_m100.yaml"}
CONTROL = 0.0
CONFIG_DIR = "configs/experiments"

# The four P1 checkpoints, and the data.seed their arms share.
BASES = {"p1-s1337": 2001, "p1-s1338": 2002, "p1-s1339": 2003, "p1-s1340": 2004}
PILOT_BASE = "p1-s1337"  # chosen by convention, not by its score
PILOT_SHARE = 0.5
PILOT_DATA_SEED = 2000
PILOT_GRID = (1e-4, 3e-4, 1e-3)
# An edge winner -> the one point added beyond it.
PILOT_EXTENSIONS = {1e-4: 3e-5, 1e-3: 3e-3}

# The six verdicts share a 5% family-wise error: each interval is at
# 1 - 0.05/6 two-sided, t(1 - 0.05/12, df).  Checked against scipy by a test;
# scipy is not a dependency.
N_VERDICTS = (len(ARMS) - 1) * len(PRIMARY_EVAL_METRICS)
FAMILY_ALPHA = 0.05
T_BONFERRONI = {
    2: 10.8859, 3: 6.2315, 4: 4.8510, 5: 4.2193, 6: 3.8630, 7: 3.6358, 8: 3.4789,
    9: 3.3642, 10: 3.2768,
}
PLANNED_BASES = len(BASES)

# The 3e-4 follow-up: two arms, so two verdicts at 1 - 0.05/2 each,
# t(1 - 0.05/4, df).  Checked against scipy by a test.
FOLLOWUP_LR = 3e-4
FOLLOWUP_ARMS = {0.0: "e1_lr0.0003_m0.yaml", 0.5: "e1_lr0.0003_m50.yaml"}
T_BONFERRONI_FOLLOWUP = {
    2: 6.2053, 3: 4.1765, 4: 3.4954, 5: 3.1634, 6: 2.9687, 7: 2.8412, 8: 2.7515,
    9: 2.6850, 10: 2.6338,
}


# ------------------------------------------------------------------ the plan


def lr_text(lr: float) -> str:
    """``3e-4`` -> ``"0.0003"``: how configs and run names spell an LR."""
    return f"{lr:.10f}".rstrip("0")


def share_text(share: float) -> str:
    return str(round(share * 100))


@dataclass(frozen=True)
class E1Job:
    run_name: str
    config: str                      # relative to the repository root
    base: str                        # the P1 run whose last.pt the weights come from
    role: str                        # "pilot" | "main"
    overrides: tuple[str, ...] = ()  # ``--set`` arguments


def pilot_jobs(extension: float | None = None) -> list[E1Job]:
    """The pilot's three runs, plus the one extension point when the rule calls for it."""
    lrs = list(PILOT_GRID)
    if extension is not None:
        if not any(math.isclose(extension, e) for e in PILOT_EXTENSIONS.values()):
            raise E1Error(f"{extension} is not an extension point (allowed: {sorted(PILOT_EXTENSIONS.values())})")
        lrs.append(extension)
    return [
        E1Job(f"e1-pilot-{lr_text(lr)}", f"{CONFIG_DIR}/e1_pilot_{lr_text(lr)}.yaml", PILOT_BASE, "pilot")
        for lr in lrs
    ]


def main_jobs() -> list[E1Job]:
    """Every arm from every base, base by base: an interrupted queue leaves whole bases."""
    return [
        E1Job(f"e1-m{share_text(share)}-{base.replace('-', '')}", f"{CONFIG_DIR}/{config}", base, "main",
              (f"data.seed={seed}",))
        for base, seed in BASES.items()
        for share, config in ARMS.items()
    ]


def followup_jobs() -> list[E1Job]:
    """The follow-up's eight runs, base by base, each on its base's main-run data seed."""
    return [
        E1Job(f"e1-lr{lr_text(FOLLOWUP_LR)}-m{share_text(share)}-{base.replace('-', '')}",
              f"{CONFIG_DIR}/{config}", base, "followup", (f"data.seed={seed}",))
        for base, seed in BASES.items()
        for share, config in FOLLOWUP_ARMS.items()
    ]


def train_command(job: E1Job, *, repo: Path, root: Path, run_dir: Path, python: str) -> list[str]:
    """``scripts/train.py`` for one E1 run: general + personal shards, both val sets."""
    shards = root / "shards"
    cmd = [
        python, "-u", str(repo / "scripts" / "train.py"),
        "--config", str(repo / job.config),
        "--data", str(shards / "general"), str(shards / "personal"),
        "--val-shard", f"{GENERAL_VAL}={shards / 'general' / 'val'}",
        f"{PERSONAL_VAL}={shards / 'personal' / 'val'}",
        "--init-from", str(root / "checkpoints" / job.base / "last.pt"),
        "--run-name", job.run_name,
        "--run-dir", str(run_dir),
        "--experiment", f"E1-{job.role}",
        "--venue", "local",
    ]
    if job.overrides:
        cmd += ["--set", *job.overrides]   # last: --set takes every argument after it
    return cmd


def eval_command(run_dir: Path, *, repo: Path, root: Path, python: str) -> list[str]:
    """``scripts/evaluate.py`` on a run's final checkpoint, as the floor's were."""
    return [
        python, "-u", str(repo / "scripts" / "evaluate.py"),
        "--checkpoint", str(run_dir / "last.pt"),
        "--tokenizer", str(repo / "tokenizer" / "minime-16k.json"),
        "--personal", str(root / "shards" / "personal"),
        "--general", str(root / "shards" / "general"),
    ]


# ----------------------------------------------------------------- mixtures


def personal_share(mixture: dict[str, float]) -> float:
    """The personal share of a mixture, refused unless it is shaped like an E1 arm."""
    total = sum(mixture.values())
    if total <= 0:
        raise E1Error(f"mixture {mixture} has no weight")
    expected = set(GENERAL_RATIO) | set(PERSONAL_SOURCES)
    if set(mixture) != expected:
        raise E1Error(f"mixture {mixture} must name exactly {sorted(expected)}")
    for name in ZERO_SOURCES:
        if mixture[name] != 0:
            raise E1Error(f"mixture {mixture}: {name} must be 0")
    general = sum(mixture[n] for n in GENERAL_RATIO)
    if general > 0:
        ratio_total = sum(GENERAL_RATIO.values())
        for name, part in GENERAL_RATIO.items():
            if not math.isclose(mixture[name] / general, part / ratio_total, rel_tol=1e-9):
                raise E1Error(f"mixture {mixture}: the general part is not {GENERAL_RATIO}")
    return sum(mixture[n] for n in PERSONAL_SOURCES) / total


def _arm(share: float) -> float:
    for arm in ARMS:
        if math.isclose(share, arm, abs_tol=1e-12):
            return arm
    raise E1Error(f"personal share {share} is not an E1 arm ({sorted(ARMS)})")


# ------------------------------------------------------------------ reading


@dataclass
class E1Run:
    run_dir: Path
    manifest: dict
    status: str              # "finished" | "diverged"
    role: str
    share: float
    lr: float
    data_seed: int
    init: dict               # manifest["init_from"]

    @property
    def summary(self) -> dict:
        return self.manifest.get("summary") or {}


def read_e1_run(run_dir: str | Path) -> E1Run:
    run_dir = Path(run_dir)
    try:
        manifest, status, _ = read_run(run_dir)
    except SweepError as exc:
        raise E1Error(str(exc)) from None
    config = manifest["config"]
    init = manifest.get("init_from")
    if not init or not init.get("sha256"):
        raise E1Error(f"{run_dir}: no init_from in its manifest -- not started with train.py --init-from")
    return E1Run(
        run_dir=run_dir,
        manifest=manifest,
        status=status,
        role=config.get("role", ""),
        share=_arm(personal_share(config["data"]["mixture"])),
        lr=float(config["training"]["learning_rate"]),
        data_seed=config["data"]["seed"],
        init=init,
    )


def _check_controlled(runs: list[E1Run], varied: set[str]) -> dict:
    views = {r.run_dir: controlled_view(r.manifest) for r in runs}
    try:
        check_controlled(views, varied)
    except SweepError as exc:
        raise E1Error(str(exc)) from None
    return next(iter(views.values()))


def _budget(view: dict) -> int:
    return (view["training.max_steps"] * view["training.batch_size"]
            * view["training.gradient_accumulation"] * view["model.context_length"])


def _reference_manifests(references: dict[str, str | Path]) -> dict[str, dict]:
    try:
        return {name: read_run(ref_dir)[0] for name, ref_dir in references.items()}
    except SweepError as exc:
        raise E1Error(str(exc)) from None


def _base_names(runs: list[E1Run], manifests: dict[str, dict]) -> dict[str, str | None]:
    """init sha256 -> the reference run (a P1 run's name) whose checkpoint it is.

    A checkpoint records its config's fingerprint, which covers both seeds, so
    it matches exactly one P1 run's manifest.
    """
    by_fingerprint = {config_fingerprint(m["config"]): name for name, m in manifests.items()}
    return {r.init["sha256"]: by_fingerprint.get(r.init.get("config_fingerprint")) for r in runs}


def _metric(record: dict, metric: str, where) -> float:
    try:
        return eval_metric(record, metric, where)
    except FloorError as exc:
        raise E1Error(str(exc)) from None


# -------------------------------------------------------------- the pilot


@dataclass
class PilotPoint:
    run: str
    lr: float
    status: str
    personal: float | None
    general: float | None


def pick_pilot_lr(run_dirs: list[str | Path], references: dict[str, str | Path] | None = None) -> dict:
    """Check the pilot is the protocol's and pick the LR, or name the extension point.

    ``references`` maps P1 run names to their run directories; with it, the
    pilots must have started from ``PILOT_BASE``'s checkpoint.  The report's
    ``status`` is ``"picked"`` or ``"extend"`` (run ``next``, then pick again).
    """
    if not run_dirs:
        raise E1Error("no pilot runs")
    runs = [read_e1_run(d) for d in run_dirs]
    for r in runs:
        if r.role != "pilot":
            raise E1Error(f"{r.run_dir}: role {r.role!r}, not 'pilot'")
        if r.share != PILOT_SHARE:
            raise E1Error(f"{r.run_dir}: personal share {r.share}, the pilot runs {PILOT_SHARE}")
        if r.data_seed != PILOT_DATA_SEED:
            raise E1Error(f"{r.run_dir}: data.seed {r.data_seed}, the pilot's is {PILOT_DATA_SEED}")
    shared = _check_controlled(runs, {"training.learning_rate"})
    inits = {r.init["sha256"] for r in runs}
    if len(inits) != 1:
        raise E1Error(f"the pilots started from {len(inits)} different checkpoints; they share one")
    if references is not None:
        base = _base_names(runs, _reference_manifests(references))[next(iter(inits))]
        if base != PILOT_BASE:
            raise E1Error(f"the pilots started from {base or 'an unknown checkpoint'}, not {PILOT_BASE}")

    lrs = [r.lr for r in runs]
    if len(set(lrs)) != len(lrs):
        raise E1Error(f"two pilots share an LR: {sorted(lrs)}")
    missing = [lr for lr in PILOT_GRID if not any(math.isclose(lr, x) for x in lrs)]
    if missing:
        raise E1Error(f"the grid's {missing} have no run")
    extras = [lr for lr in lrs if not any(math.isclose(lr, g) for g in PILOT_GRID)]
    allowed = list(PILOT_EXTENSIONS.values())
    if len(extras) > 1 or any(not any(math.isclose(x, a) for a in allowed) for x in extras):
        raise E1Error(f"LRs {extras} are not the grid {PILOT_GRID} plus at most one of {allowed}")

    points = []
    for r in runs:
        losses = r.summary.get("final_val_losses") or {}
        if r.status == "finished" and PERSONAL_VAL not in losses:
            raise E1Error(f"{r.run_dir}: no final '{PERSONAL_VAL}' validation loss")
        points.append(PilotPoint(r.run_dir.name, r.lr, r.status,
                                 losses.get(PERSONAL_VAL), losses.get(GENERAL_VAL)))
    points.sort(key=lambda p: p.lr)

    def ranked(candidates):
        # Lowest final personal loss; an exact tie goes to the lower LR.
        return sorted((p for p in candidates if p.status == "finished"), key=lambda p: (p.personal, p.lr))

    on_grid = ranked(p for p in points if any(math.isclose(p.lr, g) for g in PILOT_GRID))
    if not on_grid:
        raise E1Error("every pilot on the grid diverged: the grid is too high")
    grid_winner = on_grid[0]
    wanted = next((e for edge, e in PILOT_EXTENSIONS.items() if math.isclose(grid_winner.lr, edge)), None)
    extension = extras[0] if extras else None
    if extension is not None and (wanted is None or not math.isclose(extension, wanted)):
        raise E1Error(f"the extension {extension} was not called for: the grid's winner is {grid_winner.lr}")

    report = {
        "metric": f"final_val_losses.{PERSONAL_VAL}",
        "rule": "lowest final personal validation loss; an exact tie goes to the lower LR; "
                "an edge winner earns one point beyond it, then the search stops",
        "grid": list(PILOT_GRID),
        "extension": extension,
        "init_from": runs[0].init,
        "data_seed": PILOT_DATA_SEED,
        "personal_share": PILOT_SHARE,
        "token_budget": _budget(shared),
        "points": [vars(p) for p in points],
        "diverged": [p.run for p in points if p.status == "diverged"],
    }
    if wanted is not None and extension is None:
        return {**report, "status": "extend", "pick": None, "next": wanted, "at_edge": True}
    winner = ranked(points)[0]
    tested = [p.lr for p in points]
    return {**report, "status": "picked", "pick": winner.lr, "pick_run": winner.run, "next": None,
            # Still at the edge of everything tested: the best tested LR, not a located optimum.
            "at_edge": winner.lr in (min(tested), max(tested))}


# ------------------------------------------------------- the paired analysis


def _interval(diffs: list[float], table: dict[int, float] = T_BONFERRONI) -> dict:
    n = len(diffs)
    if n - 1 not in table:
        raise E1Error(f"no Bonferroni t quantile for {n} bases (table: {min(table) + 1}-{max(table) + 1})")
    mean = statistics.fmean(diffs)
    sd = statistics.stdev(diffs)
    t = table[n - 1]
    half = t * sd / math.sqrt(n)
    if half == 0:
        verdict = "not judged: the differences do not vary"
    elif mean - half <= 0 <= mean + half:
        verdict = "unresolved"
    else:
        verdict = "lower than the control" if mean < 0 else "higher than the control"
    return {"n": n, "mean": mean, "sd": sd, "t": t, "half_width": half,
            "low": mean - half, "high": mean + half, "verdict": verdict}


def paired_analysis(
    run_dirs: list[str | Path],
    *,
    pilot: dict | None = None,
    references: dict[str, Path] | None = None,
    eval_file: str = EVAL_FILE,
    followup: bool = False,
) -> dict:
    """E1's six verdicts: each arm against its base's control, paired over the bases.

    ``pilot`` is the pilot's report (``pick_pilot_lr``): every main run must
    use its pick, and none its data seed.  ``references`` maps P1 run names
    to their run directories, whose eval files give the unchanged bases'
    numbers -- the descriptive "change from the base", never a verdict.
    ``followup`` judges the 3e-4 follow-up instead: its two arms, at
    ``FOLLOWUP_LR``, two verdicts.
    """
    arms, role, table = (FOLLOWUP_ARMS, "followup", T_BONFERRONI_FOLLOWUP) if followup else (ARMS, "main", T_BONFERRONI)
    n_verdicts = (len(arms) - 1) * len(PRIMARY_EVAL_METRICS)
    runs = [read_e1_run(d) for d in run_dirs]
    for r in runs:
        if r.role != role:
            raise E1Error(f"{r.run_dir}: role {r.role!r}, not {role!r} -- pilots stay out of the analysis")
    mixture_keys = {f"data.mixture.{n}" for n in (*GENERAL_RATIO, *PERSONAL_SOURCES)}
    shared = _check_controlled(runs, {"data.seed", *mixture_keys})

    if followup:
        wrong = {r.run_dir.name: r.lr for r in runs if not math.isclose(r.lr, FOLLOWUP_LR)}
        if wrong:
            raise E1Error(f"runs not at the follow-up's {FOLLOWUP_LR}: {wrong}")
    if pilot is not None:
        if pilot.get("status") != "picked":
            raise E1Error(f"the pilot has not picked an LR (status {pilot.get('status')!r})")
        wrong = {r.run_dir.name: r.lr for r in runs if not math.isclose(r.lr, pilot["pick"])}
        if wrong and not followup:
            raise E1Error(f"runs not at the pilot's pick {pilot['pick']}: {wrong} -- "
                          "an LR changed after the pilot is a separate follow-up")
        reused = [r.run_dir.name for r in runs if r.data_seed == pilot["data_seed"]]
        if reused:
            raise E1Error(f"runs on the pilot's data.seed {pilot['data_seed']}: {reused}")

    # Pair by the checkpoint the run started from, never by its name.
    bases: dict[str, dict[float, E1Run]] = {}
    for r in runs:
        arms = bases.setdefault(r.init["sha256"], {})
        if r.share in arms:
            raise E1Error(f"two {r.share} runs from one base: {arms[r.share].run_dir.name}, {r.run_dir.name}")
        arms[r.share] = r
    for sha, present in bases.items():
        if set(present) != set(arms):
            names = sorted(a.run_dir.name for a in present.values())
            raise E1Error(f"base {sha[:12]} ({names}) lacks arm(s) {sorted(set(arms) - set(present))}")
        seeds = {a.data_seed for a in present.values()}
        if len(seeds) != 1:
            raise E1Error(f"base {sha[:12]}: its arms use data.seeds {sorted(seeds)}; they share one")
    base_seeds = [next(iter(present.values())).data_seed for present in bases.values()]
    if len(set(base_seeds)) != len(base_seeds):
        raise E1Error(f"bases share a data.seed ({sorted(base_seeds)}); each base has its own")

    # Every eval file evaluates its own run's final checkpoint, all alike.
    records: dict[Path, dict] = {}
    try:
        for r in runs:
            if r.status == "finished":
                records[r.run_dir] = read_eval(r.run_dir, r.manifest, eval_file)
        first = next(iter(records.values()), None)
        if first is None:
            raise E1Error("every run diverged")
        for record in records.values():
            check_eval_comparable(first, record)
    except FloorError as exc:
        raise E1Error(str(exc)) from None

    reference_values: dict[str, dict] = {}
    names: dict[str, str | None] = {}
    if references:
        ref_manifests = _reference_manifests(references)
        try:
            for name, ref_dir in references.items():
                record = read_eval(ref_dir, ref_manifests[name], eval_file)
                check_eval_comparable(first, record)
                reference_values[name] = record
        except FloorError as exc:
            raise E1Error(str(exc)) from None
        names = _base_names(runs, ref_manifests)
        unknown = [sha[:12] for sha in bases if names.get(sha) is None]
        if unknown:
            raise E1Error(f"no reference run for base(s) {unknown}")

    order = sorted(bases, key=lambda sha: (names.get(sha) or "", sha))
    n = len(order)
    if n - 1 not in table:
        raise E1Error(f"{n} base(s): the paired analysis needs {min(table) + 1}-{max(table) + 1}")

    def value(run: E1Run, metric: str) -> float | None:
        if run.status != "finished":
            return None
        return _metric(records[run.run_dir], metric, run.run_dir / eval_file)

    contrasts: dict[str, dict] = {}
    for share in sorted(arms):
        if share == CONTROL:
            continue
        failed = [names.get(sha) or sha[:12] for sha in order
                  if bases[sha][share].status != "finished" or bases[sha][CONTROL].status != "finished"]
        per_metric = {}
        for metric in EVAL_METRICS:
            role = "primary" if metric in PRIMARY_EVAL_METRICS else "exploratory"
            if failed:
                per_metric[metric] = {
                    "role": role,
                    "verdict": f"no verdict: a run diverged on {failed} -- that strategy failed there",
                }
                continue
            try:
                diffs = [value(bases[sha][share], metric) - value(bases[sha][CONTROL], metric) for sha in order]
            except E1Error as exc:
                if role == "primary":
                    raise
                # An exploratory number an adapted model cannot produce is
                # reported missing, not allowed to block the verdicts.
                per_metric[metric] = {"role": role, "unavailable": str(exc)}
                continue
            entry = {"role": role, "differences": diffs, "mean": statistics.fmean(diffs)}
            if role == "primary":
                entry.update(_interval(diffs, table))
            per_metric[metric] = entry
        contrasts[share_text(share)] = {"failed_bases": failed, "metrics": per_metric}

    from_base = {}
    if references:
        for share in sorted(arms):
            per_metric = {}
            for metric in PRIMARY_EVAL_METRICS:
                changes = []
                for sha in order:
                    v = value(bases[sha][share], metric)
                    base_value = _metric(reference_values[names[sha]], metric, names[sha])
                    changes.append(None if v is None else v - base_value)
                finite = [c for c in changes if c is not None]
                per_metric[metric] = {"changes": changes,
                                      "mean": statistics.fmean(finite) if finite else None}
            from_base[share_text(share)] = per_metric

    return {
        "rule": f"d_i = arm - control per base; mean d ± t(1 - {FAMILY_ALPHA}/{2 * n_verdicts}, n-1) · s_d / √n "
                f"(Bonferroni over {n_verdicts} verdicts); an interval containing 0 is 'unresolved'",
        "n_bases": n,
        "planned_bases": PLANNED_BASES,
        "multiplier": table[n - 1] / math.sqrt(n),
        "bases": [
            {"base": names.get(sha), "init_sha256": sha, "data_seed": next(iter(bases[sha].values())).data_seed,
             "runs": {share_text(s): bases[sha][s].run_dir.name for s in sorted(arms)}}
            for sha in order
        ],
        "learning_rate": runs[0].lr,
        "token_budget": _budget(shared),
        "precision": shared["manifest.device.precision"],
        "eval": eval_summary(first, eval_file),
        "contrasts": contrasts,
        # Change from the unchanged P1 checkpoint: descriptive, no verdict.
        "from_base": from_base,
    }


def against_main(followup_dirs: list[str | Path], main_dirs: list[str | Path], *, eval_file: str = EVAL_FILE) -> dict:
    """Each follow-up run against the main run of its arm on its base: what the LR changed, descriptive.

    Paired by the starting checkpoint's sha256, the data seed and the arm,
    never by name; the two may differ in their LR alone.  A diverged run on
    either side leaves that base's change empty.
    """
    followups = [read_e1_run(d) for d in followup_dirs]
    mains = [read_e1_run(d) for d in main_dirs]
    for runs, role in ((followups, "followup"), (mains, "main")):
        odd = [r.run_dir.name for r in runs if r.role != role]
        if odd:
            raise E1Error(f"{odd}: not in the {role!r} role")
    mixture_keys = {f"data.mixture.{n}" for n in (*GENERAL_RATIO, *PERSONAL_SOURCES)}
    _check_controlled([*followups, *mains], {"data.seed", "training.learning_rate", *mixture_keys})
    by_key = {(r.init["sha256"], r.data_seed, r.share): r for r in mains}
    records: dict[Path, dict] = {}
    try:
        for r in (*followups, *mains):
            if r.status == "finished":
                records[r.run_dir] = read_eval(r.run_dir, r.manifest, eval_file)
        first = next(iter(records.values()), None)
        if first is None:
            raise E1Error("every run diverged")
        for record in records.values():
            check_eval_comparable(first, record)
    except FloorError as exc:
        raise E1Error(str(exc)) from None

    def value(run: E1Run, metric: str) -> float | None:
        return None if run.status != "finished" else _metric(records[run.run_dir], metric, run.run_dir / eval_file)

    result = {}
    for share in sorted(FOLLOWUP_ARMS):
        arm = sorted((r for r in followups if r.share == share), key=lambda r: r.data_seed)
        pairs = []
        for r in arm:
            main = by_key.get((r.init["sha256"], r.data_seed, share))
            if main is None:
                raise E1Error(f"{r.run_dir.name}: no main run of its arm from its base on data.seed {r.data_seed}")
            pairs.append((r, main))
        per_metric = {}
        for metric in PRIMARY_EVAL_METRICS:
            followup_values = [value(r, metric) for r, _ in pairs]
            main_values = [value(m, metric) for _, m in pairs]
            changes = [None if f is None or m is None else f - m for f, m in zip(followup_values, main_values)]
            finite = [c for c in changes if c is not None]
            per_metric[metric] = {"followup": followup_values, "main": main_values, "changes": changes,
                                  "mean": statistics.fmean(finite) if finite else None}
        result[share_text(share)] = {"pairs": [{"followup": r.run_dir.name, "main": m.run_dir.name,
                                                "data_seed": r.data_seed} for r, m in pairs],
                                     "main_lr": sorted({m.lr for _, m in pairs}),
                                     "metrics": per_metric}
    return {"eval": eval_summary(first, eval_file), "arms": result}


__all__ = [
    "ARMS", "BASES", "CONTROL", "E1Error", "E1Job", "FOLLOWUP_ARMS", "FOLLOWUP_LR", "GENERAL_RATIO",
    "GENERAL_VAL", "N_VERDICTS", "PERSONAL_SOURCES", "PERSONAL_VAL", "PILOT_BASE", "PILOT_DATA_SEED",
    "PILOT_EXTENSIONS", "PILOT_GRID", "PILOT_SHARE", "T_BONFERRONI", "T_BONFERRONI_FOLLOWUP", "against_main",
    "eval_command", "followup_jobs", "lr_text", "main_jobs", "paired_analysis", "personal_share",
    "pick_pilot_lr", "pilot_jobs", "read_e1_run", "share_text", "train_command",
]
