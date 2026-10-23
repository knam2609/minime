"""M10 (HANDOFF §4a): the flagship's recipe, the LR sweep's configs and its pick.

The rule is tested before any sweep run, on synthetic run directories built
from the real sweep configs: the pick, its tie-break, the edge's one
extension, divergence, the cases that go back to the owner, and the runs it
refuses.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

from minime.config import load_config
from minime.data.sources import GENERAL_SLICES
from minime.training.lr_sweep import SweepError, flatten
from minime.training.m10 import (
    CHAT_TOKENS,
    FLAGSHIP_CONFIG,
    FLAGSHIP_MIXTURE,
    FLAGSHIP_STEPS,
    FLAGSHIP_TOKENS_PER_STEP,
    SWEEP_EXTENSIONS,
    SWEEP_GRID,
    SWEEP_STEPS,
    SWEEP_TOKENS,
    next_sweep_job,
    pick_sweep_lr,
    sweep_config,
)

REPORT = "reports/m10/sweep.json"


# ------------------------------------------------------------------ recipe


def test_the_chat_is_p4s_exposure_and_the_general_part_7_2_1():
    total = sum(FLAGSHIP_MIXTURE.values())
    assert total == 17_250
    assert FLAGSHIP_STEPS * FLAGSHIP_TOKENS_PER_STEP * FLAGSHIP_MIXTURE["chat"] / total == CHAT_TOKENS
    general = {k: v for k, v in FLAGSHIP_MIXTURE.items() if k != "chat"}
    assert set(general) == set(GENERAL_SLICES)
    assert general["web"] * 2 == general["dialogue"] * 7 and general["web"] == general["code"] * 7
    assert FLAGSHIP_MIXTURE["chat"] / total == pytest.approx(0.00116, abs=1e-5)


def _flagship(repo_root, lr: float = 1.6e-3):
    from minime.config import parse_cli_overrides

    return load_config(repo_root / FLAGSHIP_CONFIG, overrides=parse_cli_overrides([f"training.learning_rate={lr}"]))


def test_the_flagship_config_is_the_decided_recipe(repo_root):
    cfg = _flagship(repo_root)
    assert cfg.data.mixture == FLAGSHIP_MIXTURE, "every key written out, comment absent"
    assert cfg.training.max_steps == FLAGSHIP_STEPS and cfg.tokens_per_step == FLAGSHIP_TOKENS_PER_STEP
    assert cfg.training.optimizer == "adamw" and cfg.model.pos == "rope"


def test_the_flagship_waits_for_the_sweeps_pick(repo_root):
    """Null until reports/m10/sweep.json picks; then exactly the pick."""
    raw = (repo_root / FLAGSHIP_CONFIG).read_text()
    report_path = repo_root / REPORT
    if not report_path.exists():
        with pytest.raises(ValueError, match="M10 sweep's pick"):
            load_config(repo_root / FLAGSHIP_CONFIG)
        return
    report = json.loads(report_path.read_text())
    assert report["status"] == "picked", "the flagship needs a picked LR, not an extension or a revisit"
    assert load_config(repo_root / FLAGSHIP_CONFIG).training.learning_rate == report["pick"]
    assert re.search(r"learning_rate: \d\.\d+e-\d+", raw), "write the LR as 1.6e-3: YAML 1.1 reads 1e-3 as a string"


# ----------------------------------------------------------------- configs


def _point(repo_root, lr: float):
    return load_config(repo_root / sweep_config(lr))


def test_each_sweep_point_is_the_flagship_at_an_eighth_of_its_budget(repo_root):
    flagship = flatten(_flagship(repo_root).to_dict())
    for lr in [*SWEEP_GRID, *SWEEP_EXTENSIONS.values()]:
        cfg = _point(repo_root, lr)
        assert cfg.training.learning_rate == lr and isinstance(cfg.training.learning_rate, float)
        assert cfg.token_budget == SWEEP_TOKENS == 112_984_064
        assert cfg.training.max_steps == SWEEP_STEPS == FLAGSHIP_STEPS // 8
        assert cfg.training.warmup_steps == round(300 * SWEEP_STEPS / FLAGSHIP_STEPS)
        assert cfg.run_name == f"m10-lr-{lr:.4f}-s1337"
        differing = {k for k, v in flatten(cfg.to_dict()).items() if flagship.get(k) != v}
        assert differing - {"training.learning_rate"} == {"name", "notes", "training.max_steps",
                                                          "training.warmup_steps"}, differing


def test_the_extensions_are_a_factor_of_two_beyond_the_edges():
    assert SWEEP_GRID == (8e-4, 1.6e-3, 3.2e-3, 6.4e-3)
    assert set(SWEEP_EXTENSIONS) == {min(SWEEP_GRID), max(SWEEP_GRID)}
    assert SWEEP_EXTENSIONS[8e-4] == pytest.approx(8e-4 / 2) and SWEEP_EXTENSIONS[6.4e-3] == pytest.approx(6.4e-3 * 2)


# -------------------------------------------------------------------- pick


@pytest.fixture
def sweep(repo_root, tmp_path):
    """Writes a run directory per {lr: final val loss | "diverged" | "paused"} and returns them;
    ``changes`` maps an LR to dotted config edits for its run."""
    made = []

    def make(points: dict[float, float | str], changes: dict | None = None) -> list[Path]:
        made.append(tmp_path / f"sweep{len(made)}")
        dirs = []
        for lr, outcome in points.items():
            cfg = json.loads(json.dumps(_point(repo_root, lr).to_dict()))
            for dotted, value in (changes or {}).get(lr, {}).items():
                *path, last = dotted.split(".")
                node = cfg
                for key in path:
                    node = node[key]
                node[last] = value
            manifest = {"run": f"m10-lr-{lr:.4f}-s1337", "config": cfg, "device": {"precision": "fp16"},
                        "data": {"tokenizer": "minime-16k@92bb9c71bf6e2da5", "shards": {"chat": 293534}}}
            records = []
            if isinstance(outcome, float):
                manifest["summary"] = {"steps": cfg["training"]["max_steps"], "final_val_loss": outcome}
            elif outcome == "diverged":
                records.append({"step": 12, "train_loss": float("nan"), "phase": "diverged"})
            elif outcome == "paused":
                manifest["paused"] = {"step": 200}
            run = made[-1] / manifest["run"]
            run.mkdir(parents=True)
            (run / "manifest.json").write_text(json.dumps(manifest))
            (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
            dirs.append(run)
        return dirs

    return make


def test_the_lowest_final_general_loss_wins_inside_the_grid(sweep):
    report = pick_sweep_lr(sweep({8e-4: 3.40, 1.6e-3: 3.31, 3.2e-3: 3.30, 6.4e-3: 3.36}))
    assert report["status"] == "picked" and report["pick"] == 3.2e-3
    assert report["pick_run"] == "m10-lr-0.0032-s1337"
    assert report["margin"] == pytest.approx(0.01)
    assert report["token_budget"] == SWEEP_TOKENS and report["precision"] == "fp16"


def test_the_pick_reads_the_unrounded_loss_and_a_tie_goes_to_the_lower_lr(sweep):
    # metrics.jsonl would round both to 3.30000.
    assert pick_sweep_lr(sweep({8e-4: 3.5, 1.6e-3: 3.300004, 3.2e-3: 3.300001, 6.4e-3: 3.4}))["pick"] == 3.2e-3


def test_an_exact_tie_goes_to_the_lower_lr(sweep):
    report = pick_sweep_lr(sweep({8e-4: 3.5, 1.6e-3: 3.3, 3.2e-3: 3.3, 6.4e-3: 3.4}))
    assert report["pick"] == 1.6e-3 and report["margin"] == 0.0


@pytest.mark.parametrize("edge, loss_at", [(8e-4, {8e-4: 3.2}), (6.4e-3, {6.4e-3: 3.2})])
def test_an_edge_winner_earns_one_extension(sweep, edge, loss_at):
    grid = {lr: 3.4 for lr in SWEEP_GRID} | loss_at
    report = pick_sweep_lr(sweep(grid))
    assert report["status"] == "extend" and report["pick"] is None
    assert report["next"] == SWEEP_EXTENSIONS[edge]


def test_an_extension_that_loses_leaves_the_edge_point_picked(sweep):
    report = pick_sweep_lr(sweep({4e-4: 3.25, 8e-4: 3.2, 1.6e-3: 3.3, 3.2e-3: 3.4, 6.4e-3: 3.5}))
    assert report["status"] == "picked" and report["pick"] == 8e-4 and report["extension"] == 4e-4


def test_an_extension_that_wins_goes_back_to_the_owner(sweep):
    report = pick_sweep_lr(sweep({8e-4: 3.5, 1.6e-3: 3.4, 3.2e-3: 3.3, 6.4e-3: 3.2, 1.28e-2: 3.1}))
    assert report["status"] == "revisit" and report["pick"] is None and "extension 0.0128 won" in report["reason"]


def test_a_diverged_edge_is_still_the_edge_and_is_excluded(sweep):
    """6.4e-3 diverges and 3.2e-3 wins: 3.2e-3 is the pick, with no extension."""
    report = pick_sweep_lr(sweep({8e-4: 3.5, 1.6e-3: 3.4, 3.2e-3: 3.3, 6.4e-3: "diverged"}))
    assert report["status"] == "picked" and report["pick"] == 3.2e-3
    assert report["diverged"] == ["m10-lr-0.0064-s1337"]


def test_no_finite_point_goes_back_to_the_owner(sweep):
    report = pick_sweep_lr(sweep({lr: "diverged" for lr in SWEEP_GRID}))
    assert report["status"] == "revisit" and "no point on the grid finished" in report["reason"]


def test_an_incomplete_point_is_completed_never_dropped(sweep):
    with pytest.raises(SweepError, match="paused at step 200 of 431: resume it"):
        pick_sweep_lr(sweep({8e-4: 3.5, 1.6e-3: 3.4, 3.2e-3: "paused", 6.4e-3: 3.6}))
    with pytest.raises(SweepError, match="the grid's \\[0.0064\\] have no run"):
        pick_sweep_lr(sweep({8e-4: 3.5, 1.6e-3: 3.4, 3.2e-3: 3.3}))


@pytest.mark.parametrize("points, changes, message", [
    ({8e-4: 3.5, 1.6e-3: 3.4, 3.2e-3: 3.3, 6.4e-3: 3.6, 1.28e-2: 3.7}, {},
     "the extension 0.0128 was not called for"),
    ({8e-4: 3.5, 1.6e-3: 3.4, 3.2e-3: 3.3, 6.4e-3: 3.6}, {3.2e-3: {"data.seed": 99}},
     "not a controlled comparison.*data.seed"),
    # Every point alike, so only the protocol's own checks can see it.
    ({8e-4: 3.5, 1.6e-3: 3.4, 3.2e-3: 3.3, 6.4e-3: 3.6}, {lr: {"data.mixture.chat": 0} for lr in SWEEP_GRID},
     "the sweep runs the flagship's"),
    ({8e-4: 3.5, 1.6e-3: 3.4, 3.2e-3: 3.3, 6.4e-3: 3.6}, {lr: {"training.batch_size": 8} for lr in SWEEP_GRID},
     "56,492,032 tokens a run; the sweep's budget is 112,984,064"),
    ({8e-4: 3.5, 1.6e-3: 3.4, 3.2e-3: 3.3, 6.4e-3: 3.6}, {lr: {"training.max_steps": 3450} for lr in SWEEP_GRID},
     "3450 steps; the sweep runs 431"),
    ({8e-4: 3.5, 1.6e-3: 3.4, 3.2e-3: 3.3, 6.4e-3: 3.6}, {8e-4: {"training.optimizer": "muon"}},
     "the M10 sweep is AdamW"),
])
def test_runs_that_are_not_the_sweep_are_refused(sweep, points, changes, message):
    with pytest.raises(SweepError, match=message):
        pick_sweep_lr(sweep(points, changes))


def test_the_notebook_queues_the_extension_and_nothing_else(sweep):
    config, why = next_sweep_job(sweep({8e-4: 3.2, 1.6e-3: 3.3, 3.2e-3: 3.4, 6.4e-3: 3.5}))
    assert config == "configs/experiments/m10_lr_0.0004.yaml" and "extension 0.0004" in why
    assert next_sweep_job(sweep({8e-4: 3.5, 1.6e-3: 3.2, 3.2e-3: 3.4, 6.4e-3: 3.5}))[0] is None


def test_the_notebook_never_crashes_on_an_unrankable_sweep(sweep):
    config, why = next_sweep_job(sweep({8e-4: 3.5, 1.6e-3: "paused", 3.2e-3: 3.3, 6.4e-3: 3.6}))
    assert config is None and "cannot be ranked" in why


def test_every_extension_config_exists(repo_root):
    for lr in SWEEP_EXTENSIONS.values():
        assert (repo_root / sweep_config(lr)).exists()


def test_pick_script_writes_the_report_and_exits_by_status(repo_root, sweep, tmp_path):
    script = repo_root / "scripts" / "pick_m10_lr.py"
    out = tmp_path / "report" / "sweep.json"
    runs = sweep({8e-4: 3.5, 1.6e-3: 3.4, 3.2e-3: 3.3, 6.4e-3: 3.2})
    done = subprocess.run([sys.executable, str(script), *map(str, runs), "--out", str(out)],
                          capture_output=True, text=True)
    assert done.returncode == 2 and "EXTEND" in done.stderr
    assert json.loads(out.read_text())["next"] == 1.28e-2
    runs.append(runs[0].parent / "elsewhere")
    done = subprocess.run([sys.executable, str(script), *map(str, runs), "--out", str(out)],
                          capture_output=True, text=True)
    assert done.returncode == 1 and "no manifest.json" in done.stderr


# ------------------------------------------------------------- adaptation

from minime.training.m10 import (  # noqa: E402
    ADAPT_TOKENS,
    CANDIDATES,
    FLAGSHIP_RUN,
    RERUN_PREFIX,
    AdaptationError,
    adapt_eval_command,
    adapt_train_command,
    adaptation_jobs,
    candidate_lr,
    candidate_of,
    final_lr,
    gate_adaptation,
)
from minime.training.noise_floor import EVAL_METRICS, PRIMARY_EVAL_METRICS, config_fingerprint  # noqa: E402

PERSONAL, GENERAL = PRIMARY_EVAL_METRICS
FLAGSHIP_SHA = "ab" * 32


def _candidate_config(repo_root, job) -> dict:
    return json.loads(json.dumps(load_config(repo_root / job.config).to_dict()))


def test_the_candidates_are_instantiated_from_the_flagships_final_lr(repo_root):
    f = final_lr(repo_root / "experiments" / FLAGSHIP_RUN)
    assert f == pytest.approx(8e-5, rel=1e-5), "the flagship's schedule ends at 10% of 8e-4"
    assert candidate_lr(f, 1) == pytest.approx(2e-5, rel=1e-5)
    configs = {}
    for job in adaptation_jobs():
        cfg = load_config(repo_root / job.config)
        assert candidate_of(cfg.to_dict(), f, job.config) == job.candidate
        assert cfg.tokens_per_step * cfg.training.max_steps == ADAPT_TOKENS
        assert set(cfg.data.mixture) == {"web", "dialogue", "code", "chat"}, "every key written, no comment"
        assert re.search(r"learning_rate: \d\.\d+e-\d+", (repo_root / job.config).read_text())
        configs[Path(job.config)] = flatten(cfg.to_dict())
    from minime.training.lr_sweep import check_controlled

    varied = {"name", "notes", "training.learning_rate", *(f"data.mixture.{s}" for s in ("web", "dialogue", "code", "chat"))}
    check_controlled(configs, varied)  # identical seeds, batch layout, schedule


def test_a_config_off_its_candidate_is_refused(repo_root):
    job = adaptation_jobs()[0]
    f = final_lr(repo_root / "experiments" / FLAGSHIP_RUN)
    for dotted, value, message in [
        ("training.learning_rate", 3e-5, "none of the candidates"),
        ("data.mixture.chat", 0.2, "none of the candidates"),
        ("data.mixture.web", 0.7, "general part"),
        ("training.max_steps", 256, "not the adaptation recipe"),
        ("training.batch_size", 4, "tokens"),
    ]:
        cfg = _candidate_config(repo_root, job)
        *path, last = dotted.split(".")
        node = cfg
        for key in path:
            node = node[key]
        node[last] = value
        with pytest.raises(AdaptationError, match=message):
            candidate_of(cfg, f, job.config)


def test_the_adaptation_commands(repo_root):
    job = adaptation_jobs()[1]
    cmd = adapt_train_command(job, repo=Path("/repo"), root=Path("/data"), run_dir=Path("/data/runs/x"), python="py")
    assert cmd[cmd.index("--init-from") + 1] == "/data/checkpoints/flagship-s1337/last.pt"
    at = cmd.index("--val-shard")
    assert cmd[at + 1:at + 3] == ["general=/data/shards/general/val", "personal=/data/shards/personal/val"]
    assert cmd[cmd.index("--run-name") + 1] == "m10-adapt-b" and "--set" not in cmd
    ev = adapt_eval_command(Path("/data/runs/x/last.pt"), repo=Path("/repo"), root=Path("/data"), python="py")
    assert ev[ev.index("--eval-config") + 1] == "/repo/configs/eval_flagship.yaml"
    assert cmd[cmd.index("--experiment") + 1] == "M10-adaptation"


def test_the_reruns_are_the_candidates_under_their_own_names():
    """Retrained after the MPS batch fix, beside the originals: same configs, never the same run directories."""
    originals, reruns = adaptation_jobs(), adaptation_jobs(rerun=True)
    assert [(j.candidate, j.config) for j in reruns] == [(j.candidate, j.config) for j in originals]
    assert [j.run_name for j in reruns] == ["m10-rerun-a", "m10-rerun-b", "m10-rerun-c"]
    assert not any(j.run_name.startswith("m10-adapt-") for j in reruns), "outside the originals' glob"
    cmd = adapt_train_command(reruns[2], repo=Path("/repo"), root=Path("/data"), run_dir=Path("/data/runs/x"),
                              python="py")
    assert cmd[cmd.index("--run-name") + 1] == "m10-rerun-c"
    assert cmd[cmd.index("--experiment") + 1] == "M10-rerun"
    assert cmd[cmd.index("--init-from") + 1] == "/data/checkpoints/flagship-s1337/last.pt"


def _eval_record(config: dict, values: dict, *, context=1024, step=None, eval_fp="flagshipfp", split="val") -> dict:
    record = {
        "eval_version": 1, "eval_config_fingerprint": eval_fp, "split": split,
        "checkpoint": {"config_fingerprint": config_fingerprint(config),
                       "step": step or config["training"]["max_steps"]},
        "tokenizer": {"id": "minime-16k@92bb9c71bf6e2da5"},
        "device": {"device": "mps", "precision": "fp32", "torch": "2.11.0"},
        "eval_sets": {f"ce/personal/{split}": {"digest": "d1"}},
        "perplexity": {"context_length": context, "native_context_length": 1024},
    }
    for metric in EVAL_METRICS:
        node = record
        *parents, leaf = metric.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = values.get(metric, 0.5)
    return record


@pytest.fixture
def adapt(repo_root, tmp_path):
    """The flagship's directory (its real manifest and final LR) and one per candidate.

    ``make({"a": (personal, general) | "diverged", ...}, base=(personal,
    general), changes={"a": {dotted: value}}, init={"a": sha256},
    evals={"a": {kwarg: value}})`` returns ``(flagship_dir, candidate_dirs)``.
    """
    made = []

    def make(outcomes, *, base=(4.0, 3.40), changes=None, init=None, evals=None, rerun=False):
        root = tmp_path / f"gate{len(made)}"
        made.append(root)
        flagship = json.loads((repo_root / "experiments" / FLAGSHIP_RUN / "manifest.json").read_text())
        f_dir = root / FLAGSHIP_RUN
        f_dir.mkdir(parents=True)
        (f_dir / "manifest.json").write_text(json.dumps(flagship))
        (f_dir / "metrics.jsonl").write_text(json.dumps(
            {"step": 3450, "phase": "train", "lr": 8.000017904042781e-05}) + "\n")
        (f_dir / "SHA256SUMS").write_text(f"{'cd' * 32}  best.pt\n{FLAGSHIP_SHA}  last.pt\n")
        (f_dir / "eval-last.json").write_text(json.dumps(
            _eval_record(flagship["config"], {PERSONAL: base[0], GENERAL: base[1]})))

        dirs = []
        for job in adaptation_jobs(rerun):
            if job.candidate not in outcomes:
                continue
            outcome = outcomes[job.candidate]
            cfg = _candidate_config(repo_root, job)
            for dotted, value in (changes or {}).get(job.candidate, {}).items():
                *path, last = dotted.split(".")
                node = cfg
                for key in path:
                    node = node[key]
                node[last] = value
            manifest = {
                "run": job.run_name, "config": cfg, "device": {"precision": "fp32"},
                "data": {"tokenizer": "minime-16k@92bb9c71bf6e2da5", "shards": {"chat": 293534}},
                "init_from": {"path": "/ssd/flagship-s1337/last.pt",
                              "sha256": (init or {}).get(job.candidate, FLAGSHIP_SHA), "step": 3450,
                              "config_name": "flagship", "config_fingerprint": config_fingerprint(flagship["config"])},
            }
            records = []
            if outcome == "diverged":
                records.append({"step": 40, "train_loss": float("nan"), "phase": "diverged"})
            else:
                manifest["summary"] = {"steps": 128, "final_val_loss": 3.4,
                                       "final_val_losses": {"general": 3.4, "personal": 3.7}}
            run = root / f"{job.run_name}-{len(dirs)}"
            run.mkdir()
            (run / "manifest.json").write_text(json.dumps(manifest))
            (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
            if outcome != "diverged":
                personal, general = outcome
                (run / "eval-last.json").write_text(json.dumps(_eval_record(
                    cfg, {PERSONAL: personal, GENERAL: general}, **(evals or {}).get(job.candidate, {}))))
            dirs.append(run)
        return f_dir, dirs

    return make


def test_the_largest_personal_fall_within_the_general_cap_is_picked(adapt):
    report = gate_adaptation(*adapt({"a": (3.85, 3.41), "b": (3.75, 3.415), "c": (3.60, 3.43)}))
    assert report["status"] == "accepted" and report["pick"] == "b" and report["pick_run"].startswith("m10-adapt-b")
    c = report["candidates"]
    assert c["a"]["accepted"] and c["b"]["accepted"]
    assert not c["c"]["accepted"] and "general-weighted CE +0.0300" in c["c"]["reason"]
    assert c["b"]["change"][PERSONAL] == pytest.approx(-0.25)
    assert report["final_choice_frozen"] is False
    assert report["l"] == pytest.approx(2e-5, rel=1e-5) and report["data_seed"] == 3000


@pytest.mark.parametrize("personal, general, accepted", [
    (3.8999, 3.4199, True),     # just inside both thresholds
    (3.9001, 3.41, False),      # personal falls by less than 0.10
    (3.80, 3.4201, False),      # general rises by more than 0.02
])
def test_the_thresholds(adapt, personal, general, accepted):
    report = gate_adaptation(*adapt({"a": (personal, general), "b": (4.1, 3.5), "c": (4.1, 3.5)}))
    assert report["candidates"]["a"]["accepted"] is accepted
    assert report["pick"] == ("a" if accepted else None)


def test_none_accepted_keeps_the_unadapted_flagship(adapt):
    report = gate_adaptation(*adapt({"a": (3.95, 3.40), "b": (3.70, 3.45), "c": "diverged"}))
    assert report["status"] == "none accepted" and report["pick"] is None
    assert report["pick_run"] == FLAGSHIP_RUN
    assert report["candidates"]["c"] == {**report["candidates"]["c"], "status": "diverged", "accepted": False}


def test_an_exact_personal_tie_goes_to_the_lower_general_ce(adapt):
    report = gate_adaptation(*adapt({"a": (3.80, 3.41), "b": (3.80, 3.405), "c": (3.80, 3.415)}))
    assert report["pick"] == "b"


def test_a_diverged_candidate_fails_and_the_others_are_judged(adapt):
    report = gate_adaptation(*adapt({"a": "diverged", "b": (3.80, 3.41), "c": (3.70, 3.43)}))
    assert report["pick"] == "b" and report["candidates"]["a"]["reason"] == "diverged"


@pytest.mark.parametrize("kwargs, message", [
    ({"outcomes": {"a": (3.8, 3.4), "b": (3.8, 3.4)}}, r"\['C'\] have no run"),
    ({"init": {"b": "ef" * 32}}, "not the flagship's last.pt"),
    ({"changes": {"c": {"data.seed": 3001}}}, "not a controlled comparison"),
    ({"changes": {"a": {"training.learning_rate": 2.5e-5}}}, "none of the candidates"),
    ({"evals": {"b": {"eval_fp": "other"}}}, "not evaluated like the baseline"),
    ({"evals": {"b": {"step": 64}}}, "not the final step"),
])
def test_runs_that_are_not_the_protocol_are_refused(adapt, kwargs, message):
    outcomes = kwargs.pop("outcomes", {"a": (3.8, 3.4), "b": (3.8, 3.4), "c": (3.8, 3.4)})
    with pytest.raises(AdaptationError, match=message):
        gate_adaptation(*adapt(outcomes, **kwargs))


def test_a_candidate_run_twice_is_refused(adapt):
    flagship, dirs = adapt({"a": (3.8, 3.4), "b": (3.8, 3.4), "c": (3.8, 3.4)})
    with pytest.raises(AdaptationError, match="two runs of candidate A"):
        gate_adaptation(flagship, [*dirs, dirs[0]])


def test_the_gate_reads_native_context_evaluations_only(adapt):
    with pytest.raises(AdaptationError, match="not a native-context evaluation"):
        gate_adaptation(*adapt({"a": (3.8, 3.4), "b": (3.8, 3.4), "c": (3.8, 3.4)},
                               evals={n: {"context": 512} for n in CANDIDATES}))


def test_gate_script_writes_the_report(repo_root, adapt, tmp_path):
    script = repo_root / "scripts" / "pick_m10_adaptation.py"
    out = tmp_path / "report" / "adaptation.json"
    flagship, dirs = adapt({"a": (3.85, 3.41), "b": (3.75, 3.415), "c": (3.60, 3.43)})
    done = subprocess.run([sys.executable, str(script), str(flagship), *map(str, dirs), "--out", str(out)],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "<- pick" in done.stdout and json.loads(out.read_text())["pick"] == "b"
    done = subprocess.run([sys.executable, str(script), str(flagship), *map(str, dirs[:2]), "--out", str(out)],
                          capture_output=True, text=True)
    assert done.returncode == 1 and "have no run" in done.stderr


def test_gate_script_describes_the_reruns_and_never_writes_the_frozen_report(repo_root, adapt, tmp_path):
    script = repo_root / "scripts" / "pick_m10_adaptation.py"
    out = tmp_path / "rerun.json"
    outcomes = {"a": (3.85, 3.41), "b": (3.75, 3.415), "c": (3.60, 3.43)}
    flagship, reruns = adapt(outcomes, rerun=True)
    _, originals = adapt(outcomes)
    base = [sys.executable, str(script), str(flagship)]
    done = subprocess.run([*base, *map(str, reruns), "--out", str(out), "--rerun"], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    report = json.loads(out.read_text())
    assert report["pick"] == "b" and report["final_choice_frozen"] is False
    assert report["reading"].startswith("description") and "adaptation.json" in report["reading"]
    assert "a description" in done.stdout and RERUN_PREFIX in report["pick_run"]
    out.unlink()
    for runs, flags, message in [(originals, ["--rerun"], "not a rerun"), (reruns, [], "a rerun: pass --rerun"),
                                 ([*originals[:2], reruns[2]], [], "a rerun: pass --rerun")]:
        done = subprocess.run([*base, *map(str, runs), "--out", str(out), *flags], capture_output=True, text=True)
        assert done.returncode == 1 and message in done.stderr and not out.exists()
    done = subprocess.run([*base, *map(str, reruns), "--out", str(out), "--rerun", "--owner-choice", "b",
                           "--chosen-on", "2026-10-20"], capture_output=True, text=True)
    assert done.returncode == 2 and "final choice stays" in done.stderr and not out.exists()


# ---------------------------------------------------- common-context CE

from minime.training.m10 import (  # noqa: E402
    COMMON_EVAL_FILE,
    COMMON_EVAL_FILES,
    COMMON_GROUPS,
    checkpoint_path,
    common_context_table,
    common_rows,
)


def _row(experiments: Path, run: str, values: tuple[float, float], *, native=None, context=512, fp="commonfp",
         split="val", exist_ok=False):
    """A finished run with its common-context eval file (and, for a 512 model on val, its native one)."""
    own = 1024 if run.startswith(("flagship", "m10-adapt")) else 512
    config = {"name": run, "model": {"context_length": own}, "training": {"max_steps": 10}}
    run_dir = experiments / run
    run_dir.mkdir(parents=True, exist_ok=exist_ok)
    (run_dir / "manifest.json").write_text(json.dumps({
        "config": config, "data": {"tokenizer": "minime-16k@92bb9c71bf6e2da5"},
        "summary": {"steps": 10, "final_val_loss": 3.0}}))
    (run_dir / "metrics.jsonl").write_text("")

    def record(vals, ctx, eval_fp):
        r = _eval_record(config, {PERSONAL: vals[0], GENERAL: vals[1]}, context=ctx, eval_fp=eval_fp, split=split)
        r["perplexity"]["native_context_length"] = own
        r["perplexity"]["general"] = {s: {"loss": vals[1]} for s in ("web", "dialogue", "code")}
        return r

    (run_dir / COMMON_EVAL_FILES[split]).write_text(json.dumps(record(values, context, fp)))
    if own == 512 and split == "val":
        (run_dir / "eval-last.json").write_text(json.dumps(record(native or values, 512, "nativefp")))


ACCEPTED = {"status": "accepted", "pick_run": "m10-adapt-c", "final_choice_frozen": False}


@pytest.fixture
def common(tmp_path):
    def make(values=None, accepted=True, split="val", experiments=None, **row_kwargs):
        experiments = experiments or tmp_path / f"exp{len(list(tmp_path.iterdir()))}"
        for group, runs in common_rows(ACCEPTED if accepted else None).items():
            for i, run in enumerate(runs):
                default = (3.5 + i / 100, 3.7 + i / 100)
                _row(experiments, run, (values or {}).get(run, default), split=split,
                     **{"exist_ok": split == "test", **row_kwargs.get(run, {})})
        return experiments
    return make


def test_the_rows_are_the_probes_the_flagship_and_the_accepted_adaptation():
    assert [len(r) for r in COMMON_GROUPS.values()] == [4, 4, 4, 4, 1]
    assert "adaptation" not in common_rows(None)
    assert "adaptation" not in common_rows({"status": "none accepted", "pick_run": FLAGSHIP_RUN})
    assert common_rows(ACCEPTED)["adaptation"] == ("m10-adapt-c",)
    root = Path("/data")
    assert checkpoint_path(root, "p4-s1337") == root / "checkpoints" / "p4-s1337" / "last.pt"
    assert checkpoint_path(root, FLAGSHIP_RUN) == root / "checkpoints" / FLAGSHIP_RUN / "last.pt"
    assert checkpoint_path(root, "e1-m50-p1s1340") == root / "runs" / "e1-m50-p1s1340" / "last.pt"
    assert checkpoint_path(root, "m10-adapt-c") == root / "runs" / "m10-adapt-c" / "last.pt"


def test_the_table_describes_each_group(common):
    table = common_context_table(common(), ACCEPTED)
    assert list(table["groups"]) == ["P1", "P4", "E1 25/75", "E1 50/50", "flagship", "adaptation"]
    p1 = table["groups"]["P1"]
    assert p1["mean"][PERSONAL] == pytest.approx(3.515) and p1["sd"][PERSONAL] == pytest.approx(0.0129, abs=1e-4)
    assert table["groups"]["flagship"]["sd"] is None
    assert table["native_agreement"]["max_abs_difference"] == 0.0
    assert table["adaptation_final"] is False
    assert "adaptation" not in common_context_table(common(accepted=False), None)["groups"]


def test_a_probe_scored_differently_on_its_own_windows_is_refused(common):
    experiments = common(**{"p4-s1338": {"native": (3.51 + 2e-4, 3.71)}})
    with pytest.raises(AdaptationError, match="the same windows scored differently"):
        common_context_table(experiments, ACCEPTED)
    experiments = common(**{"p4-s1338": {"native": (3.51 + 5e-5, 3.71)}})
    gap = common_context_table(experiments, ACCEPTED)["native_agreement"]["max_abs_difference"]
    assert gap == pytest.approx(5e-5, rel=1e-3)


@pytest.mark.parametrize("row, message", [
    ({FLAGSHIP_RUN: {"context": 1024}}, "not 512"),
    ({"e1-m25-p1s1339": {"fp": "otherfp"}}, "not evaluated like the baseline"),
])
def test_rows_scored_unlike_the_others_are_refused(common, row, message):
    with pytest.raises(AdaptationError, match=message):
        common_context_table(common(**row), ACCEPTED)


def test_a_row_without_its_evaluation_is_refused(common):
    experiments = common()
    (experiments / "p1-s1340" / COMMON_EVAL_FILE).unlink()
    with pytest.raises(AdaptationError, match=f"no {COMMON_EVAL_FILE}"):
        common_context_table(experiments, ACCEPTED)


# ------------------------------------------- the owner's choice, the test split

from minime.training.m10 import (  # noqa: E402
    TEST_EVAL_FILE,
    adapt_eval_command,
    chosen_adaptation,
    common_eval_command,
    final_report,
    final_report_runs,
    freeze_choice,
    full_eval_rows,
    p4_described,
)
from minime.training.p4 import T_P4  # noqa: E402

FROZEN_C = {**ACCEPTED, "final_choice_frozen": True, "final_choice": "c", "final_run": "m10-adapt-c",
            "final_choice_made": "2026-10-19, by the owner, after reading the voice samples"}
FROZEN_FLAGSHIP = {**FROZEN_C, "final_choice": "flagship", "final_run": FLAGSHIP_RUN}


def test_the_owner_chooses_the_pick_or_the_flagship(adapt):
    report = gate_adaptation(*adapt({"a": (3.85, 3.41), "b": (3.75, 3.415), "c": (3.60, 3.43)}))
    frozen = freeze_choice(report, "b", "2026-10-19")
    assert frozen["final_choice_frozen"] is True and frozen["final_run"] == report["pick_run"]
    assert frozen["final_choice_made"].startswith("2026-10-19, by the owner")
    keys = list(frozen)
    assert keys[keys.index("final_choice_frozen"):][:4] == [
        "final_choice_frozen", "final_choice", "final_run", "final_choice_made"]
    assert report["final_choice_frozen"] is False, "the gate's own report is not changed"
    assert chosen_adaptation(report) == chosen_adaptation(frozen) == report["pick_run"]
    kept = freeze_choice(report, "flagship", "2026-10-19")
    assert kept["final_run"] == FLAGSHIP_RUN and chosen_adaptation(kept) is None
    for choice in ("a", "c", "B"):
        with pytest.raises(AdaptationError, match="the gate's pick and the unadapted flagship"):
            freeze_choice(report, choice, "2026-10-19")
    none = gate_adaptation(*adapt({"a": (3.95, 3.40), "b": (3.95, 3.40), "c": (3.95, 3.40)}))
    with pytest.raises(AdaptationError, match=r"\['flagship'\]"):
        freeze_choice(none, "a", "2026-10-19")
    assert freeze_choice(none, "flagship", "2026-10-19")["final_run"] == FLAGSHIP_RUN


def test_gate_script_freezes_the_owners_choice(repo_root, adapt, tmp_path):
    script = repo_root / "scripts" / "pick_m10_adaptation.py"
    out = tmp_path / "adaptation.json"
    flagship, dirs = adapt({"a": (3.85, 3.41), "b": (3.75, 3.415), "c": (3.60, 3.43)})
    base = [sys.executable, str(script), str(flagship), *map(str, dirs), "--out", str(out)]
    done = subprocess.run([*base, "--owner-choice", "b", "--chosen-on", "2026-10-19"], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "final choice, frozen" in done.stdout and json.loads(out.read_text())["final_choice"] == "b"
    done = subprocess.run([*base, "--owner-choice", "c", "--chosen-on", "2026-10-19"], capture_output=True, text=True)
    assert done.returncode == 1 and "the gate's pick" in done.stderr
    assert json.loads(out.read_text())["final_choice"] == "b", "a refused choice writes nothing"
    done = subprocess.run([*base, "--owner-choice", "b"], capture_output=True, text=True)
    assert done.returncode == 2 and "go together" in done.stderr


def test_the_test_split_commands():
    checkpoint, repo, root = Path("/data/runs/x/last.pt"), Path("/repo"), Path("/data")
    for command in (adapt_eval_command, common_eval_command):
        val = command(checkpoint, repo=repo, root=root, python="py")
        test = command(checkpoint, repo=repo, root=root, python="py", split="test")
        assert "--split" not in val and test == [*val, "--split", "test", "--final-report"]
        with pytest.raises(ValueError, match="split"):
            command(checkpoint, repo=repo, root=root, python="py", split="train")


def test_the_test_phase_waits_for_the_frozen_choice():
    for unfrozen in (None, ACCEPTED, {"status": "none accepted", "pick_run": FLAGSHIP_RUN,
                                      "final_choice_frozen": False}):
        with pytest.raises(AdaptationError, match="not frozen"):
            final_report_runs(unfrozen)
    runs = final_report_runs(FROZEN_C)
    assert runs[:2] == [(FLAGSHIP_RUN, TEST_EVAL_FILE), ("m10-adapt-c", TEST_EVAL_FILE)]
    assert len(runs) == 20 and {f for _, f in runs[2:]} == {COMMON_EVAL_FILES["test"]}
    assert ("m10-adapt-c", COMMON_EVAL_FILES["test"]) in runs
    kept = final_report_runs(FROZEN_FLAGSHIP)
    assert len(kept) == 18 and not any(run.startswith("m10-adapt") for run, _ in kept)
    assert "adaptation" not in common_rows(FROZEN_FLAGSHIP) and full_eval_rows(FROZEN_FLAGSHIP) == (FLAGSHIP_RUN,)


@pytest.fixture
def final(common):
    """An experiments directory with every val and test file of the final report."""
    def make(adaptation=FROZEN_C, *, test_fp="flagshipfp"):
        accepted = chosen_adaptation(adaptation) is not None
        experiments = common(accepted=accepted)
        common(accepted=accepted, split="test", experiments=experiments)
        for run in full_eval_rows(adaptation):
            config = json.loads((experiments / run / "manifest.json").read_text())["config"]
            values = ({"val": (3.7, 3.40), "test": (3.8, 3.41)} if run == FLAGSHIP_RUN
                      else {"val": (3.6, 3.41), "test": (3.6, 3.43)})
            for split, name, fp in (("val", "eval-last.json", "flagshipfp"), ("test", TEST_EVAL_FILE, test_fp)):
                record = _eval_record(config, {PERSONAL: values[split][0], GENERAL: values[split][1]},
                                      split=split, eval_fp=fp)
                (experiments / run / name).write_text(json.dumps(record))
        return experiments
    return make


def test_the_final_report_sets_test_beside_val(final):
    result = final_report(final(), FROZEN_C)
    runs = result["full"]["runs"]
    assert list(runs) == [FLAGSHIP_RUN, "m10-adapt-c"]
    assert runs[FLAGSHIP_RUN]["test"][PERSONAL] == 3.8 and runs["m10-adapt-c"]["val"][PERSONAL] == 3.6
    change = result["full"]["adaptation_minus_flagship"]
    assert change["test"][PERSONAL] == pytest.approx(-0.2) and change["val"][GENERAL] == pytest.approx(0.01)
    assert result["full"]["eval"]["test"]["split"] == "test" and result["full"]["eval"]["val"]["split"] == "val"
    table = result["common_context"]
    assert table["split"] == "test" and table["eval_file"] == COMMON_EVAL_FILES["test"]
    assert table["native_agreement"] is None
    assert list(table["groups"]) == ["P1", "P4", "E1 25/75", "E1 50/50", "flagship", "adaptation"]
    assert "no rule" in result["reading"] and result["final_run"] == "m10-adapt-c"


def test_the_final_report_describes_p4s_contrasts_on_both_splits(final):
    described = final_report(final(), FROZEN_C)["p4_described"]
    assert "not a verdict" in described["reading"] and list(described) == ["reading", "val", "test"]
    for split in ("val", "test"):
        assert list(described[split]) == ["p1", "e1-50"]
        for entry in (described[split][name][m] for name in ("p1", "e1-50") for m in (PERSONAL, GENERAL)):
            assert "verdict" not in entry
            assert entry["half_width"] == pytest.approx(T_P4 * entry["se"])
    # The fixture scores every group alike: no difference, the floor's spread on both sides.
    assert described["test"]["p1"][PERSONAL]["difference"] == pytest.approx(0.0)


def test_p4_described_is_the_verdicts_arithmetic():
    def group(rows):
        return {"runs": {f"r{i}": {"values": {PERSONAL: p, GENERAL: g}} for i, (p, g) in enumerate(rows)}}
    p4, e1 = [3.80, 3.79, 3.81, 3.82], [3.50, 3.51, 3.53, 3.52]
    table = {"groups": {"P4": group([(p, 3.72) for p in p4]),
                        "P1": group([(7.5, 3.72), (7.4, 3.71), (7.5, 3.73), (7.2, 3.72)]),
                        "E1 50/50": group([(p, 4.0 + p / 100) for p in e1])}}
    entry = p4_described(table)["e1-50"][PERSONAL]
    se = math.sqrt(statistics.variance(p4) / 4 + statistics.variance(e1) / 4)
    assert entry["difference"] == pytest.approx(statistics.fmean(p4) - statistics.fmean(e1))
    assert entry["se"] == pytest.approx(se) and entry["low"] == pytest.approx(entry["difference"] - T_P4 * se)


def test_the_final_report_without_an_adaptation(final):
    result = final_report(final(FROZEN_FLAGSHIP), FROZEN_FLAGSHIP)
    assert list(result["full"]["runs"]) == [FLAGSHIP_RUN] and result["full"]["adaptation_minus_flagship"] is None
    assert "adaptation" not in result["common_context"]["groups"]


def test_the_final_report_refuses_a_changed_setting_or_split(final):
    with pytest.raises(AdaptationError, match="changes no setting"):
        final_report(final(test_fp="otherfp"), FROZEN_C)
    experiments = final()
    c = experiments / "m10-adapt-c"
    (c / TEST_EVAL_FILE).write_text((c / "eval-last.json").read_text())   # a val file under the test name
    with pytest.raises(AdaptationError, match="'val' split; this reads 'test'"):
        final_report(experiments, FROZEN_C)
    experiments = final()
    p1 = experiments / "p1-s1338"
    (p1 / COMMON_EVAL_FILES["test"]).write_text((p1 / COMMON_EVAL_FILE).read_text())
    with pytest.raises(AdaptationError, match="'val' split; this reads 'test'"):
        final_report(experiments, FROZEN_C)
    with pytest.raises(AdaptationError, match="not frozen"):
        final_report(final(), ACCEPTED)


def test_final_eval_script(repo_root, final, tmp_path):
    script = repo_root / "scripts" / "final_eval.py"
    adaptation, root = tmp_path / "adaptation.json", tmp_path / "data"
    adaptation.write_text(json.dumps(ACCEPTED))
    cmd = [sys.executable, str(script), "eval", "--dry-run", "--root", str(root), "--adaptation", str(adaptation)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    assert done.returncode == 1 and "not frozen" in done.stderr
    adaptation.write_text(json.dumps(FROZEN_C))
    done = subprocess.run(cmd, capture_output=True, text=True)
    assert done.returncode == 1 and "no checkpoint" in done.stderr
    for run, _ in final_report_runs(FROZEN_C):
        checkpoint_path(root, run).parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path(root, run).touch()
    (checkpoint_path(root, "p1-s1337").parent / COMMON_EVAL_FILES["test"]).write_text("{}")
    done = subprocess.run(cmd, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines()
    assert "p1-s1337: eval-last-ctx512-test.json read already, never redone" in lines
    commands = [line for line in lines if "evaluate.py" in line]
    assert len(commands) == 19 and all(line.endswith("--split test --final-report") for line in commands)
    assert "eval_flagship.yaml" in commands[0] and "eval_common512.yaml" in commands[-1]

    # report: copies each test file from beside its checkpoint, refuses one that differs
    experiments = final()
    moved = experiments / FLAGSHIP_RUN / TEST_EVAL_FILE
    shutil.move(moved, checkpoint_path(root, FLAGSHIP_RUN).parent / TEST_EVAL_FILE)
    (checkpoint_path(root, "p1-s1337").parent / COMMON_EVAL_FILES["test"]).unlink()
    out = tmp_path / "test.json"
    report = [sys.executable, str(script), "report", "--root", str(root), "--adaptation", str(adaptation),
              "--experiments", str(experiments), "--out", str(out)]
    done = subprocess.run(report, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert moved.exists() and json.loads(out.read_text())["final_run"] == "m10-adapt-c"
    (checkpoint_path(root, "p4-s1339").parent / COMMON_EVAL_FILES["test"]).write_text("{}")
    done = subprocess.run(report, capture_output=True, text=True)
    assert done.returncode == 1 and "the test split is read once" in done.stderr


from minime.training.m10 import non_dominated  # noqa: E402


def test_non_dominated_means():
    points = {"flagship": (3.44, 3.84), "c": (3.45, 3.68), "p4": (3.72, 3.81), "e25": (3.92, 3.57),
              "e50": (4.00, 3.49), "p1": (3.72, 7.38)}
    assert non_dominated(points) == ["flagship", "c", "e25", "e50"]
    assert non_dominated({"x": (1.0, 1.0), "y": (1.0, 1.0), "z": (1.0, 2.0), "w": (2.0, 1.0)}) == ["x", "y"]


def test_plot_frontier_script(repo_root, final, tmp_path):
    pytest.importorskip("matplotlib")
    experiments = final()
    val, test, out = tmp_path / "common512.json", tmp_path / "test.json", tmp_path / "frontier.png"
    val.write_text(json.dumps(common_context_table(experiments, FROZEN_C)))
    test.write_text(json.dumps(final_report(experiments, FROZEN_C)))
    cmd = [sys.executable, str(repo_root / "scripts" / "plot_frontier.py"),
           "--val", str(val), "--test", str(test), "--out", str(out)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("val, test ->") and out.read_bytes()[:4] == b"\x89PNG"
    test.unlink()   # before the test phase: the val panel alone
    done = subprocess.run(cmd, capture_output=True, text=True)
    assert done.returncode == 0 and done.stdout.startswith("val ->"), done.stderr
