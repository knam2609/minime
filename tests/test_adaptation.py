"""E1: the configs, the run plan, the pilot's LR pick and the paired analysis.

The protocol was decided with the owner before any adaptation run (README,
"E1 adaptation protocol"): these tests pin it, so a config edit or a code
change that quietly alters what E1 measures fails here first.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml

from minime.config import load_config
from minime.training.adaptation import (
    ARMS,
    BASES,
    CONTROL,
    GENERAL_RATIO,
    PERSONAL_SOURCES,
    PILOT_BASE,
    PILOT_DATA_SEED,
    PILOT_EXTENSIONS,
    PILOT_GRID,
    T_BONFERRONI,
    E1Error,
    lr_text,
    main_jobs,
    paired_analysis,
    personal_share,
    pick_pilot_lr,
    pilot_jobs,
    train_command,
)
from minime.training.lr_sweep import flatten
from minime.training.noise_floor import EVAL_METRICS, PRIMARY_EVAL_METRICS, config_fingerprint

EXPERIMENTS = "configs/experiments"
PILOT_REPORT = "reports/e1/pilot.json"
SOURCES = {*GENERAL_RATIO, *PERSONAL_SOURCES}
ALL_PILOT_LRS = (*PILOT_GRID, *PILOT_EXTENSIONS.values())


def _pick(repo_root):
    path = repo_root / PILOT_REPORT
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    return report["pick"] if report.get("status") == "picked" else None


def _arm(repo_root, config, lr=3.0e-4, **overrides):
    """An arm's config at ``lr``: the synthetic runs never depend on the real pick."""
    over = {"training": {"learning_rate": lr}}
    for dotted, value in overrides.items():
        section, _, key = dotted.partition("__")
        over.setdefault(section, {})[key] = value
    return load_config(repo_root / EXPERIMENTS / config, overrides=over)


def _pilot_config(repo_root, lr):
    return load_config(repo_root / EXPERIMENTS / f"e1_pilot_{lr_text(lr)}.yaml")


# ------------------------------------------------------------ the configs


def test_the_recipe_is_2_1m_tokens_of_8192_from_the_probes_model(repo_root):
    probe = load_config(repo_root / "configs" / "probe.yaml")
    cfg = _pilot_config(repo_root, 3e-4)
    assert cfg.tokens_per_step == 8_192
    assert (cfg.training.batch_size, cfg.training.gradient_accumulation) == (4, 4)
    assert cfg.training.max_steps == 256
    assert cfg.token_budget == 2_097_152
    assert cfg.training.warmup_steps == 16
    assert cfg.training.min_lr_ratio == 0.1
    assert cfg.training.optimizer == "adamw"
    # --init-from refuses any model difference; the configs must not make one.
    assert cfg.to_dict()["model"] == probe.to_dict()["model"]


def test_every_e1_config_writes_every_mixture_key(repo_root):
    """`extends` merges dicts: an arm that left a key out would inherit it."""
    for path in sorted((repo_root / EXPERIMENTS).glob("e1_*.yaml")):
        raw = yaml.safe_load(path.read_text())
        mixture = (raw.get("data") or {}).get("mixture")
        if path.name.startswith("e1_pilot_"):
            assert mixture is None, f"{path.name}: a pilot takes the 50/50 arm's mixture as it is"
            continue
        assert set(mixture) == SOURCES, path.name
        assert mixture["comment"] == 0, path.name


def test_the_arms_are_the_protocols(repo_root):
    for share, config in ARMS.items():
        cfg = _arm(repo_root, config)
        assert cfg.role == "main"
        assert personal_share(cfg.data.mixture) == share


def test_the_main_configs_wait_for_the_pilots_pick(repo_root):
    """Before the pick the main configs cannot load; after it, they either
    still cannot (the LR not applied yet: nothing can run) or run at the pick."""
    pick = _pick(repo_root)
    for config in ARMS.values():
        try:
            lr = load_config(repo_root / EXPERIMENTS / config).training.learning_rate
        except ValueError as exc:
            assert "learning_rate is null" in str(exc)
            continue
        assert pick is not None, f"{config} has an LR, but the pilot has not picked one"
        assert lr == pick, f"{config} runs at {lr}, the pilot picked {pick}"


def test_the_pilots_differ_only_in_their_lr(repo_root):
    configs = {lr: _pilot_config(repo_root, lr) for lr in ALL_PILOT_LRS}
    views = {lr: flatten(cfg.to_dict()) for lr, cfg in configs.items()}
    first = next(iter(views.values()))
    for lr, view in views.items():
        differing = {k for k in first.keys() | view.keys() if first.get(k) != view.get(k)}
        assert differing <= {"name", "notes", "training.learning_rate"}
        assert isinstance(view["training.learning_rate"], float)
        assert view["training.learning_rate"] == pytest.approx(lr)
        assert configs[lr].role == "pilot"
        assert configs[lr].data.seed == PILOT_DATA_SEED
        assert personal_share(configs[lr].data.mixture) == 0.5


def test_the_pilot_and_main_seeds_never_coincide():
    assert len(set(BASES.values())) == len(BASES)
    assert PILOT_DATA_SEED not in BASES.values()
    assert PILOT_BASE in BASES


def test_bonferroni_quantiles_match_scipy():
    stats = pytest.importorskip("scipy.stats")
    for df, t in T_BONFERRONI.items():
        assert t == pytest.approx(stats.t.ppf(1 - 0.05 / 12, df), abs=1e-4)
    # The shipped design: four bases, ±3.12 s_d.
    assert T_BONFERRONI[3] / 2 == pytest.approx(3.116, abs=1e-3)


@pytest.mark.parametrize("mixture,problem", [
    ({"web": 7, "dialogue": 2, "code": 1, "chat": 1, "comment": 1}, "comment must be 0"),
    ({"web": 6, "dialogue": 2, "code": 1, "chat": 1, "comment": 0}, "general part"),
    ({"web": 7, "dialogue": 2, "code": 1, "chat": 1}, "must name exactly"),
])
def test_personal_share_refuses_a_mixture_unlike_an_arm(mixture, problem):
    with pytest.raises(E1Error, match=problem):
        personal_share(mixture)


# --------------------------------------------------------------- the plan


def test_the_pilot_plan(repo_root):
    jobs = pilot_jobs()
    assert [j.run_name for j in jobs] == ["e1-pilot-0.0001", "e1-pilot-0.0003", "e1-pilot-0.001"]
    assert all(j.base == PILOT_BASE and j.role == "pilot" and not j.overrides for j in jobs)
    assert all((repo_root / j.config).exists() for j in jobs)
    assert pilot_jobs(3e-3)[-1].run_name == "e1-pilot-0.003"
    assert pilot_jobs(3e-5)[-1].run_name == "e1-pilot-0.00003"
    with pytest.raises(E1Error, match="not an extension point"):
        pilot_jobs(2e-3)


def test_the_main_plan_runs_every_arm_from_every_base_base_by_base(repo_root):
    jobs = main_jobs()
    assert len(jobs) == len(ARMS) * len(BASES) == 16
    assert len({j.run_name for j in jobs}) == 16
    assert [j.base for j in jobs[:4]] == [PILOT_BASE] * 4
    assert jobs[1].run_name == "e1-m25-p1s1337"
    for job in jobs:
        assert job.overrides == (f"data.seed={BASES[job.base]}",)
        assert (repo_root / job.config).exists()


def test_the_train_command(tmp_path):
    job = main_jobs()[1]
    cmd = train_command(job, repo=Path("/repo"), root=Path("/data"), run_dir=Path("/data/runs/x"), python="py")
    at = cmd.index("--data")
    assert cmd[at + 1:at + 3] == ["/data/shards/general", "/data/shards/personal"]
    at = cmd.index("--val-shard")
    # General first: it is the run's val_loss; the pilot ranks on "personal".
    assert cmd[at + 1:at + 3] == ["general=/data/shards/general/val", "personal=/data/shards/personal/val"]
    assert cmd[cmd.index("--init-from") + 1] == "/data/checkpoints/p1-s1337/last.pt"
    assert cmd[cmd.index("--run-name") + 1] == "e1-m25-p1s1337"
    assert cmd[-2:] == ["--set", "data.seed=2001"]


# ------------------------------------------------------- synthetic runs


def _subdir(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.mkdir()
    return path


def _sha(base: int) -> str:
    return f"{base:02d}" * 32


def _fake_run(tmp_path, cfg, *, base=0, name=None, status="finished", personal=5.0, general=3.8,
              init_fp=None, step=None):
    """A run directory as train.py writes it, for one E1 config."""
    config = cfg.to_dict()
    run_dir = tmp_path / (name or f"{config['name']}-b{base}-{len(list(tmp_path.iterdir()))}")
    run_dir.mkdir()
    manifest = {
        "config": config,
        "data": {"data_dir": ["/g", "/p"], "tokenizer": "minime-16k@abc", "tokenizer_dev": False,
                 "shards": {"web": 10, "dialogue": 10, "code": 10, "chat": 10, "comment": 10},
                 "val_shard": {"general": 100, "personal": 50}},
        "device": {"device": "mps", "precision": "fp32"},
        "init_from": {"path": f"/ckpt/{base}/last.pt", "sha256": _sha(base), "step": 2300,
                      "config_name": "p1", "config_fingerprint": init_fp or f"fp{base}"},
    }
    records = [{"step": 1, "phase": "train", "train_loss": 5.0}]
    if status == "finished":
        manifest["summary"] = {"steps": config["training"]["max_steps"], "final_val_loss": general,
                               "final_val_losses": {"general": general, "personal": personal}}
    else:
        records.append({"step": 2, "phase": "diverged", "train_loss": float("nan")})
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    (run_dir / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return run_dir


def _eval_record(config: dict, values: dict[str, float], *, step=None, eval_fp="evalfp") -> dict:
    record = {
        "eval_version": 1, "eval_config_fingerprint": eval_fp, "split": "val",
        "checkpoint": {"config_fingerprint": config_fingerprint(config), "step": step or config["training"]["max_steps"]},
        "tokenizer": {"id": "minime-16k@abc"},
        "device": {"device": "mps", "precision": "fp32", "torch": "2.11.0"},
        "eval_sets": {"personal": "d1", "general": "d2"},
    }
    for metric in EVAL_METRICS:
        node = record
        *parents, leaf = metric.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = values.get(metric, 0.5)
    return record


def _write_eval(run_dir: Path, values: dict[str, float], **kwargs) -> None:
    config = json.loads((run_dir / "manifest.json").read_text())["config"]
    (run_dir / "eval-last.json").write_text(json.dumps(_eval_record(config, values, **kwargs)))


PERSONAL, GENERAL = PRIMARY_EVAL_METRICS


# ---------------------------------------------------------------- the pilot


def _pilots(tmp_path, repo_root, personal: dict[float, float | None], **kwargs):
    """One pilot run per LR; ``None`` is a diverged run."""
    return [
        _fake_run(tmp_path, _pilot_config(repo_root, lr), status="diverged" if loss is None else "finished",
                  personal=loss if loss is not None else 0.0, name=f"e1-pilot-{lr_text(lr)}", **kwargs)
        for lr, loss in personal.items()
    ]


def test_the_middle_of_the_grid_wins_without_an_extension(tmp_path, repo_root):
    report = pick_pilot_lr(_pilots(tmp_path, repo_root, {1e-4: 4.9, 3e-4: 4.7, 1e-3: 4.8}))
    assert report["status"] == "picked"
    assert report["pick"] == 3e-4
    assert report["at_edge"] is False
    assert report["token_budget"] == 2_097_152


@pytest.mark.parametrize("edge,beyond,losses", [
    (1e-3, 3e-3, {1e-4: 4.9, 3e-4: 4.8, 1e-3: 4.7}),
    (1e-4, 3e-5, {1e-4: 4.6, 3e-4: 4.8, 1e-3: 4.9}),
])
def test_an_edge_winner_asks_for_one_point_beyond_it(tmp_path, repo_root, edge, beyond, losses):
    report = pick_pilot_lr(_pilots(tmp_path, repo_root, losses))
    assert report["status"] == "extend"
    assert report["next"] == beyond
    assert report["pick"] is None


@pytest.mark.parametrize("beyond_loss,pick,at_edge", [
    (4.6, 3e-3, True),     # the extension wins too: the best tested LR, still at the edge
    (4.75, 1e-3, False),   # the edge winner holds
    (None, 1e-3, False),   # the extension diverged: reported, out of the ranking
])
def test_the_search_stops_after_the_extension(tmp_path, repo_root, beyond_loss, pick, at_edge):
    report = pick_pilot_lr(_pilots(tmp_path, repo_root, {1e-4: 4.9, 3e-4: 4.8, 1e-3: 4.7, 3e-3: beyond_loss}))
    assert report["status"] == "picked"
    assert report["pick"] == pick
    assert report["at_edge"] is at_edge
    assert report["diverged"] == ([] if beyond_loss is not None else ["e1-pilot-0.003"])


def test_an_exact_tie_goes_to_the_lower_lr(tmp_path, repo_root):
    report = pick_pilot_lr(_pilots(tmp_path, repo_root, {1e-4: 4.9, 3e-4: 4.7, 1e-3: 4.7}))
    assert report["pick"] == 3e-4


def test_a_diverged_pilot_drops_out(tmp_path, repo_root):
    report = pick_pilot_lr(_pilots(tmp_path, repo_root, {1e-4: 4.9, 3e-4: 4.8, 1e-3: None}))
    assert report["pick"] == 3e-4
    assert report["diverged"] == ["e1-pilot-0.001"]


def test_an_extension_nobody_called_for_is_refused(tmp_path, repo_root):
    with pytest.raises(E1Error, match="was not called for"):
        pick_pilot_lr(_pilots(tmp_path, repo_root, {1e-4: 4.9, 3e-4: 4.7, 1e-3: 4.8, 3e-3: 4.6}))
    # 1e-3 won the grid, so only 3e-3 may be added.
    with pytest.raises(E1Error, match="was not called for"):
        pick_pilot_lr(_pilots(_subdir(tmp_path, "b"), repo_root, {1e-4: 5.0, 3e-4: 4.9, 1e-3: 4.7, 3e-5: 4.6}))


def test_a_missing_grid_point_is_refused(tmp_path, repo_root):
    with pytest.raises(E1Error, match="have no run"):
        pick_pilot_lr(_pilots(tmp_path, repo_root, {1e-4: 4.9, 3e-4: 4.7}))


def test_the_pilot_refuses_runs_that_are_not_the_protocols(tmp_path, repo_root):
    good = {1e-4: 4.9, 3e-4: 4.7}
    runs = _pilots(tmp_path, repo_root, good)
    main = _arm(repo_root, "e1_m50.yaml", lr=1e-3, data__seed=PILOT_DATA_SEED)
    with pytest.raises(E1Error, match="role 'main'"):
        pick_pilot_lr([*runs, _fake_run(tmp_path, main)])

    other_seed = load_config(repo_root / EXPERIMENTS / "e1_pilot_0.001.yaml", overrides={"data": {"seed": 7}})
    with pytest.raises(E1Error, match="data.seed 7"):
        pick_pilot_lr([*runs, _fake_run(tmp_path, other_seed)])

    longer = load_config(repo_root / EXPERIMENTS / "e1_pilot_0.001.yaml", overrides={"training": {"warmup_steps": 32}})
    with pytest.raises(E1Error, match="warmup_steps"):
        pick_pilot_lr([*runs, _fake_run(tmp_path, longer)])

    with pytest.raises(E1Error, match="different checkpoints"):
        pick_pilot_lr([*runs, _fake_run(tmp_path, _pilot_config(repo_root, 1e-3), base=1)])


def test_the_pilots_must_start_from_the_pilot_base(tmp_path, repo_root):
    reference = _p1_reference(tmp_path, repo_root, PILOT_BASE, seed=1337)
    fingerprint = config_fingerprint(json.loads((reference / "manifest.json").read_text())["config"])
    runs = _pilots(tmp_path, repo_root, {1e-4: 4.9, 3e-4: 4.7, 1e-3: 4.8}, init_fp=fingerprint)
    assert pick_pilot_lr(runs, references={PILOT_BASE: reference})["pick"] == 3e-4
    others = _pilots(_subdir(tmp_path, "other"), repo_root, {1e-4: 4.9, 3e-4: 4.7, 1e-3: 4.8})
    with pytest.raises(E1Error, match="not p1-s1337"):
        pick_pilot_lr(others, references={PILOT_BASE: reference})


# ------------------------------------------------------ the paired analysis


def _p1_reference(tmp_path, repo_root, name, seed, values=None):
    """An unchanged P1 run: its manifest and its eval file."""
    config_name = "p1.yaml" if seed == 1337 else f"p1_s{seed}.yaml"
    cfg = load_config(repo_root / EXPERIMENTS / config_name).to_dict()
    run_dir = tmp_path / name
    run_dir.mkdir()
    manifest = {"config": cfg, "data": {"tokenizer": "minime-16k@abc"}, "device": {"precision": "fp16"},
                "summary": {"steps": cfg["training"]["max_steps"], "final_val_loss": 3.7}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    (run_dir / "metrics.jsonl").write_text("")
    (run_dir / "eval-last.json").write_text(json.dumps(_eval_record(cfg, values or {PERSONAL: 7.4, GENERAL: 3.72})))
    return run_dir


def _design(tmp_path, repo_root, values, *, lr=3.0e-4, status=None, references=False, names=None):
    """The 16 main runs; ``values(base, share)`` -> eval metrics, ``status(base, share)``."""
    runs, refs = [], {}
    for base, (p1_name, seed) in enumerate(BASES.items()):
        fingerprint = f"fp{base}"
        if references:
            ref = _p1_reference(tmp_path, repo_root, p1_name, int(p1_name.split("s")[-1]))
            refs[p1_name] = ref
            fingerprint = config_fingerprint(json.loads((ref / "manifest.json").read_text())["config"])
        for share, config in ARMS.items():
            cfg = _arm(repo_root, config, lr=lr, data__seed=seed)
            state = status(base, share) if status else "finished"
            name = names(base, share) if names else None
            run = _fake_run(tmp_path, cfg, base=base, status=state, init_fp=fingerprint, name=name)
            if state == "finished":
                _write_eval(run, values(base, share))
            runs.append(run)
    return runs, refs


def _values(personal_drop, general_cost):
    """Personal CE falls by ``personal_drop[share][base]``; general CE rises by ``general_cost``."""
    def values(base, share):
        return {PERSONAL: 7.4 - personal_drop[share][base], GENERAL: 3.72 + general_cost[share][base]}
    return values


ZERO = [0.0, 0.0, 0.0, 0.0]
DROP = {0.0: ZERO, 0.25: [1.0, 1.1, 0.9, 1.0], 0.5: [1.5, 1.6, 1.4, 1.5], 1.0: [2.0, 2.2, 1.8, 2.1]}
COST = {0.0: ZERO, 0.25: [0.01, -0.01, 0.02, 0.0], 0.5: [0.05, 0.06, 0.04, 0.05], 1.0: [0.3, 0.32, 0.29, 0.31]}


def test_six_bonferroni_verdicts_on_the_paired_differences(tmp_path, repo_root):
    runs, _ = _design(tmp_path, repo_root, _values(DROP, COST))
    report = paired_analysis(runs)
    assert report["n_bases"] == 4
    assert report["multiplier"] == pytest.approx(6.2315 / 2)

    strong = report["contrasts"]["25"]["metrics"]
    diffs = [-1.0, -1.1, -0.9, -1.0]
    assert strong[PERSONAL]["differences"] == pytest.approx(diffs)
    sd = math.sqrt(sum((d + 1.0) ** 2 for d in diffs) / 3)
    assert strong[PERSONAL]["half_width"] == pytest.approx(6.2315 * sd / 2)
    assert strong[PERSONAL]["verdict"] == "lower than the control"
    # +0.005 ± 3.12 s_d contains zero: unresolved, not "no cost".
    assert strong[GENERAL]["verdict"] == "unresolved"
    assert report["contrasts"]["100"]["metrics"][GENERAL]["verdict"] == "higher than the control"
    verdicts = [m for c in report["contrasts"].values() for m in c["metrics"].values() if "half_width" in m]
    assert len(verdicts) == 6
    exploratory = strong[EVAL_METRICS[2]]
    assert exploratory["role"] == "exploratory" and "verdict" not in exploratory


def test_runs_pair_by_their_starting_checkpoint_not_their_names(tmp_path, repo_root):
    """Names that lie about the base change nothing: the init sha256 pairs."""
    honest, _ = _design(_subdir(tmp_path, "a"), repo_root, _values(DROP, COST))
    liars, _ = _design(_subdir(tmp_path, "b"), repo_root, _values(DROP, COST),
                       names=lambda base, share: f"e1-m{round(share * 100)}-p1s{1340 - base}")
    a, b = paired_analysis(honest), paired_analysis(liars)
    assert a["contrasts"] == b["contrasts"]


def test_a_diverged_run_is_its_strategy_failing_there(tmp_path, repo_root):
    runs, _ = _design(tmp_path, repo_root, _values(DROP, COST),
                      status=lambda base, share: "diverged" if (base, share) == (2, 1.0) else "finished")
    report = paired_analysis(runs)
    failed = report["contrasts"]["100"]
    assert failed["failed_bases"] and "no verdict" in failed["metrics"][PERSONAL]["verdict"]
    assert "half_width" not in failed["metrics"][PERSONAL]
    assert report["contrasts"]["25"]["metrics"][PERSONAL]["verdict"] == "lower than the control"


def test_a_diverged_control_leaves_its_base_without_contrasts(tmp_path, repo_root):
    runs, _ = _design(tmp_path, repo_root, _values(DROP, COST),
                      status=lambda base, share: "diverged" if (base, share) == (0, CONTROL) else "finished")
    report = paired_analysis(runs)
    assert all(c["failed_bases"] for c in report["contrasts"].values())


def test_the_unchanged_base_is_descriptive(tmp_path, repo_root):
    runs, refs = _design(tmp_path, repo_root, _values(DROP, COST), references=True)
    report = paired_analysis(runs, references=refs)
    assert [b["base"] for b in report["bases"]] == list(BASES)
    change = report["from_base"]["100"][GENERAL]
    assert change["changes"] == pytest.approx([0.3, 0.32, 0.29, 0.31])
    assert "verdict" not in change


def test_a_base_without_a_reference_is_refused(tmp_path, repo_root):
    runs, refs = _design(tmp_path, repo_root, _values(DROP, COST), references=True)
    refs.pop("p1-s1339")
    with pytest.raises(E1Error, match="no reference run"):
        paired_analysis(runs, references=refs)


def test_an_exploratory_metric_an_adapted_model_cannot_produce_does_not_block(tmp_path, repo_root):
    runs, _ = _design(tmp_path, repo_root, _values(DROP, COST))
    record = json.loads((runs[1] / "eval-last.json").read_text())
    record["voice"]["authorship"]["discriminator"]["auc"] = None
    (runs[1] / "eval-last.json").write_text(json.dumps(record))
    report = paired_analysis(runs)
    assert "unavailable" in report["contrasts"]["25"]["metrics"]["voice.authorship.discriminator.auc"]


def test_the_analysis_follows_the_pilots_pick(tmp_path, repo_root):
    runs, _ = _design(tmp_path, repo_root, _values(DROP, COST), lr=3.0e-4)
    pilot = {"status": "picked", "pick": 3e-4, "data_seed": PILOT_DATA_SEED}
    assert paired_analysis(runs, pilot=pilot)["learning_rate"] == 3e-4
    with pytest.raises(E1Error, match="not at the pilot's pick"):
        paired_analysis(runs, pilot={**pilot, "pick": 1e-3})
    with pytest.raises(E1Error, match="has not picked"):
        paired_analysis(runs, pilot={"status": "extend", "next": 3e-3})
    with pytest.raises(E1Error, match="pilot's data.seed"):
        paired_analysis(runs, pilot={**pilot, "data_seed": BASES[PILOT_BASE]})


def test_the_design_must_be_complete_and_seeded_by_base(tmp_path, repo_root):
    runs, _ = _design(tmp_path, repo_root, _values(DROP, COST))
    with pytest.raises(E1Error, match="lacks arm"):
        paired_analysis(runs[:-1])

    extra = _fake_run(tmp_path, _arm(repo_root, "e1_m25.yaml", data__seed=BASES[PILOT_BASE]), base=0)
    with pytest.raises(E1Error, match="two 0.25 runs"):
        paired_analysis([*runs, extra])


def test_arms_of_one_base_share_a_seed_and_bases_do_not(tmp_path, repo_root):
    seeds = iter(range(100, 200))
    runs = []
    for base in range(4):
        for share, config in ARMS.items():
            run = _fake_run(tmp_path, _arm(repo_root, config, data__seed=next(seeds)), base=base)
            _write_eval(run, _values(DROP, COST)(base, share))
            runs.append(run)
    with pytest.raises(E1Error, match="they share one"):
        paired_analysis(runs)

    shared = []
    for base in range(4):
        for share, config in ARMS.items():
            run = _fake_run(tmp_path, _arm(repo_root, config, data__seed=2001), base=base)
            _write_eval(run, _values(DROP, COST)(base, share))
            shared.append(run)
    with pytest.raises(E1Error, match="bases share a data.seed"):
        paired_analysis(shared)


def test_a_pilot_stays_out_of_the_analysis(tmp_path, repo_root):
    runs, _ = _design(tmp_path, repo_root, _values(DROP, COST))
    pilot = _fake_run(tmp_path, _pilot_config(repo_root, 3e-4))
    with pytest.raises(E1Error, match="pilots stay out"):
        paired_analysis([*runs, pilot])


def test_runs_that_differ_in_more_than_their_mixture_are_refused(tmp_path, repo_root):
    runs, _ = _design(tmp_path, repo_root, _values(DROP, COST))
    odd = _fake_run(tmp_path, _arm(repo_root, "e1_m25.yaml", data__seed=2001, training__warmup_steps=32), base=9)
    with pytest.raises(E1Error, match="warmup_steps"):
        paired_analysis([*runs, odd])


def test_each_eval_file_must_be_its_own_runs_final_checkpoint(tmp_path, repo_root):
    runs, _ = _design(tmp_path, repo_root, _values(DROP, COST))
    _write_eval(runs[5], _values(DROP, COST)(1, 0.25), step=128)
    with pytest.raises(E1Error, match="not the final step"):
        paired_analysis(runs)

    runs, _ = _design(_subdir(tmp_path, "x"), repo_root, _values(DROP, COST))
    (runs[5] / "eval-last.json").write_text((runs[6] / "eval-last.json").read_text())
    with pytest.raises(E1Error, match="another config or seed"):
        paired_analysis(runs)

    runs, _ = _design(_subdir(tmp_path, "y"), repo_root, _values(DROP, COST))
    _write_eval(runs[5], _values(DROP, COST)(1, 0.25), eval_fp="other")
    with pytest.raises(E1Error, match="not evaluated like the baseline"):
        paired_analysis(runs)


# ------------------------------------------------------- the 3e-4 follow-up

from minime.training.adaptation import (  # noqa: E402
    FOLLOWUP_ARMS,
    FOLLOWUP_LR,
    T_BONFERRONI_FOLLOWUP,
    against_main,
    followup_jobs,
)
from minime.training.noise_floor import eval_summary  # noqa: E402
from minime.training.p4 import T_P4, P4Error, p4_against_followup  # noqa: E402


def test_the_followup_configs_change_the_lr_alone(repo_root):
    for share, config in FOLLOWUP_ARMS.items():
        followup = load_config(repo_root / EXPERIMENTS / config)
        main = load_config(repo_root / EXPERIMENTS / ARMS[share], overrides={"training": {"learning_rate": 1.0e-3}})
        a, b = flatten(main.to_dict()), flatten(followup.to_dict())
        assert {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)} == {
            "name", "notes", "role", "training.learning_rate"}
        assert followup.training.learning_rate == FOLLOWUP_LR and followup.role == "followup"
        assert personal_share(followup.data.mixture) == share
    assert set(FOLLOWUP_ARMS) == {CONTROL, 0.5} and FOLLOWUP_LR in PILOT_GRID


def test_the_followup_plan_reuses_the_main_runs_seeds_under_other_names(repo_root):
    jobs = followup_jobs()
    assert len(jobs) == 8 and len({j.run_name for j in jobs}) == 8
    assert [j.run_name for j in jobs[:2]] == ["e1-lr0.0003-m0-p1s1337", "e1-lr0.0003-m50-p1s1337"]
    mains = {j.run_name for j in main_jobs()}
    for job in jobs:
        assert job.role == "followup" and job.overrides == (f"data.seed={BASES[job.base]}",)
        assert (repo_root / job.config).exists()
        # Never caught by the main runs' glob, experiments/e1-m*-p1s*.
        assert not Path(job.run_name).match("e1-m*-p1s*") and job.run_name not in mains


def test_followup_bonferroni_quantiles_match_scipy():
    stats = pytest.importorskip("scipy.stats")
    for df, t in T_BONFERRONI_FOLLOWUP.items():
        assert t == pytest.approx(stats.t.ppf(1 - 0.05 / 4, df), abs=1e-4)
    assert T_BONFERRONI_FOLLOWUP[3] / 2 == pytest.approx(2.088, abs=1e-3)


def _followup_design(tmp_path, repo_root, values, *, main_values=None, references=False, lr=FOLLOWUP_LR):
    """The follow-up's eight runs; with ``main_values``, the main runs of its arms from the same bases."""
    runs, mains, refs = [], [], {}
    for base, (p1_name, seed) in enumerate(BASES.items()):
        fingerprint = f"fp{base}"
        if references:
            refs[p1_name] = _p1_reference(tmp_path, repo_root, p1_name, int(p1_name.split("s")[-1]))
            fingerprint = config_fingerprint(json.loads((refs[p1_name] / "manifest.json").read_text())["config"])
        for share, config in FOLLOWUP_ARMS.items():
            run = _fake_run(tmp_path, _arm(repo_root, config, lr=lr, data__seed=seed), base=base,
                            init_fp=fingerprint)
            _write_eval(run, values(base, share))
            runs.append(run)
            if main_values:
                main = _fake_run(tmp_path, _arm(repo_root, ARMS[share], lr=1e-3, data__seed=seed), base=base,
                                 init_fp=fingerprint)
                _write_eval(main, main_values(base, share))
                mains.append(main)
    return runs, mains, refs


def _rerun_at(tmp_path, repo_root, base, share, lr):
    run = _fake_run(tmp_path, _arm(repo_root, FOLLOWUP_ARMS[share], lr=lr, data__seed=list(BASES.values())[base]),
                    base=base)
    _write_eval(run, _values(GENTLE_DROP, GENTLE_COST)(base, share))
    return run


GENTLE_COST = {0.0: ZERO, 0.5: [0.02, 0.03, 0.01, 0.02]}
GENTLE_DROP = {0.0: ZERO, 0.5: [1.2, 1.3, 1.1, 1.2]}


def test_the_followup_has_two_verdicts(tmp_path, repo_root):
    runs, _, refs = _followup_design(tmp_path, repo_root, _values(GENTLE_DROP, GENTLE_COST), references=True)
    report = paired_analysis(runs, references=refs, followup=True)
    assert report["learning_rate"] == FOLLOWUP_LR and list(report["contrasts"]) == ["50"]
    assert report["multiplier"] == pytest.approx(4.1765 / 2) and "over 2 verdicts" in report["rule"]
    verdicts = [m for c in report["contrasts"].values() for m in c["metrics"].values() if "half_width" in m]
    assert len(verdicts) == 2
    personal = report["contrasts"]["50"]["metrics"][PERSONAL]
    assert personal["mean"] == pytest.approx(-1.2) and personal["verdict"] == "lower than the control"
    assert list(report["from_base"]) == ["0", "50"]


def test_the_followup_refuses_main_runs_and_another_lr(tmp_path, repo_root):
    runs, mains, _ = _followup_design(tmp_path, repo_root, _values(GENTLE_DROP, GENTLE_COST),
                                      main_values=_values(DROP, COST))
    with pytest.raises(E1Error, match="not 'followup'"):
        paired_analysis([*runs[:-1], mains[-1]], followup=True)
    with pytest.raises(E1Error, match="not 'main'"):
        paired_analysis(runs)
    with pytest.raises(E1Error, match="the runs differ in"):   # one run at another LR
        paired_analysis([*runs[:-1], _rerun_at(tmp_path, repo_root, 3, 0.5, 1e-4)], followup=True)
    elsewhere, _, _ = _followup_design(_subdir(tmp_path, "lr"), repo_root, _values(GENTLE_DROP, GENTLE_COST), lr=1e-4)
    with pytest.raises(E1Error, match="not at the follow-up's"):
        paired_analysis(elsewhere, followup=True)
    pilot = {"status": "picked", "pick": 1e-3, "data_seed": BASES[PILOT_BASE]}
    with pytest.raises(E1Error, match="pilot's data.seed"):
        paired_analysis(runs, pilot=pilot, followup=True)


def test_each_followup_run_is_set_beside_its_main_run(tmp_path, repo_root):
    runs, mains, _ = _followup_design(tmp_path, repo_root, _values(GENTLE_DROP, GENTLE_COST),
                                      main_values=_values(DROP, COST))
    result = against_main(runs, mains)
    general = result["arms"]["0"]["metrics"][GENERAL]
    assert general["changes"] == pytest.approx(ZERO)
    gentler = result["arms"]["50"]["metrics"][GENERAL]
    assert gentler["changes"] == pytest.approx([g - c for g, c in zip(GENTLE_COST[0.5], COST[0.5])])
    assert gentler["followup"] == pytest.approx([3.72 + c for c in GENTLE_COST[0.5]])
    assert result["arms"]["50"]["main_lr"] == [1e-3]
    assert [p["data_seed"] for p in result["arms"]["50"]["pairs"]] == sorted(BASES.values())
    with pytest.raises(E1Error, match="no main run of its arm"):
        against_main(runs, mains[:-1])
    with pytest.raises(E1Error, match="not in the 'main' role"):
        against_main(runs, [*mains, runs[0]])


def test_p4_against_the_followup_is_described_not_judged(tmp_path, repo_root):
    runs, _, _ = _followup_design(tmp_path, repo_root, _values(GENTLE_DROP, GENTLE_COST))
    record = json.loads((runs[1] / "eval-last.json").read_text())
    summary = eval_summary(record, "eval-last.json")
    p4 = [3.80, 3.79, 3.81, 3.82]
    p4_report = {"eval": summary, "diverged": [],
                 "contrasts": {"e1-50": {PERSONAL: {"p4": p4}, GENERAL: {"p4": [3.72] * 3 + [3.73]}}}}
    e1 = {PERSONAL: [6.2, 6.1, 6.3, 6.2], GENERAL: [3.74, 3.75, 3.73, 3.74]}
    described = p4_against_followup(p4_report, e1, summary)
    entry = described[PERSONAL]
    assert entry["difference"] == pytest.approx(sum(p4) / 4 - 6.2) and "verdict" not in entry
    assert entry["half_width"] == pytest.approx(T_P4 * entry["se"])
    with pytest.raises(P4Error, match="not alike"):
        p4_against_followup(p4_report, e1, {**summary, "eval_config_fingerprint": "other"})
    assert "nothing" in p4_against_followup(p4_report, {**e1, GENERAL: [3.7, None, 3.7, 3.7]}, summary)[GENERAL][
        "described"]


def test_followup_script(tmp_path, repo_root):
    import subprocess
    import sys

    runs, mains, refs = _followup_design(_subdir(tmp_path, "runs"), repo_root, _values(GENTLE_DROP, GENTLE_COST),
                                         main_values=_values(DROP, COST), references=True)
    summary = eval_summary(json.loads((runs[1] / "eval-last.json").read_text()), "eval-last.json")
    p4 = tmp_path / "verdicts.json"
    p4.write_text(json.dumps({"eval": summary, "diverged": [], "contrasts": {"e1-50": {
        PERSONAL: {"p4": [3.80, 3.79, 3.81, 3.82]}, GENERAL: {"p4": [3.72, 3.72, 3.73, 3.72]}}}}))
    pilot = tmp_path / "pilot.json"
    pilot.write_text(json.dumps({"status": "picked", "pick": 1e-3, "data_seed": PILOT_DATA_SEED}))
    out = tmp_path / "followup.json"
    cmd = [sys.executable, str(repo_root / "scripts" / "e1_followup.py"), *map(str, runs),
           "--main", *map(str, mains), "--pilot", str(pilot), "--p4", str(p4),
           "--reference", *map(str, refs.values()), "--out", str(out)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    report = json.loads(out.read_text())
    assert set(report) >= {"contrasts", "from_base", "against_main", "p4_described"}
    assert "not a verdict" in report["p4_described"]["reading"] and "Bonferroni over 2" in done.stdout
    done = subprocess.run([*cmd[:2], *map(str, mains), *cmd[2 + len(runs):]], capture_output=True, text=True)
    assert done.returncode == 1 and "not 'followup'" in done.stderr
