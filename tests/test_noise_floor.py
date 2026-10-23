"""The seed-noise floor (M8): the P1 seed configs, the arms and the interval."""

from __future__ import annotations

import json
import math
import subprocess
import sys

import pytest

from minime.config import load_config
from minime.training.lr_sweep import flatten
from minime.training.noise_floor import (
    EVAL_FILE,
    EVAL_METRICS,
    PRIMARY_EVAL_METRICS,
    SEED_KEYS,
    T_975,
    FloorError,
    check_comparable,
    check_eval_comparable,
    compare,
    config_fingerprint,
    measure_floor,
)

EXPERIMENTS = "configs/experiments"
FLOOR = ["p1.yaml", "p1_s1338.yaml", "p1_s1339.yaml", "p1_s1340.yaml"]


# ------------------------------------------------------ shipped configs


def test_p1_is_the_probe(repo_root):
    probe = flatten(load_config(repo_root / "configs" / "probe.yaml").to_dict())
    p1 = flatten(load_config(repo_root / EXPERIMENTS / "p1.yaml").to_dict())
    assert {k for k in probe if probe[k] != p1[k]} == {"name", "notes"}


def test_the_floor_configs_change_both_seeds_and_nothing_else(repo_root):
    flat = [flatten(load_config(repo_root / EXPERIMENTS / name).to_dict()) for name in FLOOR]
    for key in SEED_KEYS:
        assert len({f[key] for f in flat}) == len(flat), f"two floor runs share {key}"
    for k in set().union(*flat) - set(SEED_KEYS):
        assert len({json.dumps(f[k]) for f in flat}) == 1, f"the floor configs differ in {k}"
    names = [f"{f['name']}-s{f['training.seed']}" for f in flat]
    assert names == ["p1-s1337", "p1-s1338", "p1-s1339", "p1-s1340"]


@pytest.mark.parametrize("arm,variable", [
    ("p2.yaml", {"model.pos": "learned"}),
    ("p3.yaml", {"training.optimizer": "muon"}),
])
def test_each_arm_is_p1_with_one_variable_changed(repo_root, arm, variable):
    p1 = flatten(load_config(repo_root / EXPERIMENTS / "p1.yaml").to_dict())
    flat = flatten(load_config(repo_root / EXPERIMENTS / arm).to_dict())
    assert {k: flat[k] for k in p1 if p1[k] != flat[k]} == {
        "name": arm.removesuffix(".yaml"), "notes": flat["notes"], **variable
    }


P4_SEEDS = ["p4.yaml", "p4_s1338.yaml", "p4_s1339.yaml", "p4_s1340.yaml"]


def test_p4_is_p1_with_the_owners_chat_mixed_in_and_nothing_else(repo_root):
    p1 = flatten(load_config(repo_root / EXPERIMENTS / "p1.yaml").to_dict())
    p4 = flatten(load_config(repo_root / EXPERIMENTS / "p4.yaml").to_dict())
    changed = {k for k in p1.keys() | p4.keys() if p1.get(k) != p4.get(k)}
    assert changed == {"name", "notes", "data.mixture.web", "data.mixture.dialogue", "data.mixture.code",
                       "data.mixture.chat"}


def test_p4_expects_the_50_50_arms_personal_tokens(repo_root):
    """Decided with the owner 2026-10-13: P4's chat exposure is E1 50/50's, 1,048,576 tokens."""
    cfg = load_config(repo_root / EXPERIMENTS / "p4.yaml")
    mixture = cfg.data.normalised_mixture()
    budget = cfg.training.max_steps * cfg.training.batch_size * cfg.training.gradient_accumulation \
        * cfg.model.context_length
    assert budget == 301_465_600
    assert mixture["chat"] * budget == pytest.approx(1_048_576, abs=1e-3)
    assert mixture["chat"] * budget / 293_534 == pytest.approx(3.57, abs=0.005)   # passes over the chat
    general = {k: v for k, v in mixture.items() if k != "chat"}
    assert general["web"] / general["code"] == pytest.approx(7)
    assert general["dialogue"] / general["code"] == pytest.approx(2)
    # Only chat goes up to Kaggle: no other personal source may be named, not even at zero.
    assert set(mixture) == {"web", "dialogue", "code", "chat"}


def test_the_p4_seeds_are_the_floors_seed_pairs(repo_root):
    flat = [flatten(load_config(repo_root / EXPERIMENTS / name).to_dict()) for name in P4_SEEDS]
    floor = [flatten(load_config(repo_root / EXPERIMENTS / name).to_dict()) for name in FLOOR]
    assert [(f["training.seed"], f["data.seed"]) for f in flat] == \
        [(f["training.seed"], f["data.seed"]) for f in floor]
    for k in set().union(*flat) - set(SEED_KEYS):
        assert len({json.dumps(f[k]) for f in flat}) == 1, f"the P4 configs differ in {k}"
    assert [f"{f['name']}-s{f['training.seed']}" for f in flat] == ["p4-s1337", "p4-s1338", "p4-s1339", "p4-s1340"]


def test_p3_runs_muon_at_the_muon_sweeps_pick(repo_root):
    report = json.loads((repo_root / "reports" / "lr_sweep" / "muon.json").read_text())
    assert not report["at_edge"]
    p3 = load_config(repo_root / EXPERIMENTS / "p3.yaml")
    assert p3.training.muon_lr == report["winner"]
    assert p3.training.learning_rate == report["anchor"]["value"]


# ------------------------------------------------------------ the floor


def _fake_run(tmp_path, repo_root, config_name, *, loss=3.0, status="finished", precision="fp16",
              **edits):
    cfg = load_config(repo_root / EXPERIMENTS / config_name).to_dict()
    for dotted, value in edits.items():
        section, _, key = dotted.partition("__")
        cfg[section][key] = value
    run_dir = tmp_path / f"{cfg['name']}-s{cfg['training']['seed']}-{len(list(tmp_path.iterdir()))}"
    run_dir.mkdir()
    manifest = {
        "config": cfg,
        "data": {"data_dir": str(tmp_path), "tokenizer": "minime-16k@abc", "shards": {"web": 10}},
        "device": {"accelerator": "Tesla T4", "precision": precision},
    }
    if status == "finished":
        manifest["summary"] = {"steps": cfg["training"]["max_steps"], "final_val_loss": loss}
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    records = [{"step": 1, "phase": "train", "train_loss": 9.0}]
    if status == "diverged":
        records.append({"step": 2, "phase": "diverged", "train_loss": float("nan")})
    (run_dir / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return run_dir


def test_the_floor_is_the_95_percent_prediction_interval(tmp_path, repo_root):
    losses = [3.40, 3.42, 3.47]
    floor = measure_floor([_fake_run(tmp_path, repo_root, n, loss=l) for n, l in zip(FLOOR, losses)])
    std = math.sqrt(sum((x - 3.43) ** 2 for x in losses) / 2)
    assert floor["n"] == 3
    assert floor["mean"] == pytest.approx(3.43)
    assert floor["std"] == pytest.approx(std)
    assert floor["range"] == pytest.approx(0.07)
    assert floor["half_width"] == pytest.approx(4.3027 * std * math.sqrt(4 / 3))
    assert floor["config"] == "p1"
    assert floor["precision"] == "fp16"
    assert floor["token_budget"] == 2300 * 131_072
    assert [(r["training_seed"], r["data_seed"]) for r in floor["runs"]] == [
        (1337, 1234), (1338, 1235), (1339, 1236)
    ]


def test_the_shipped_floor_is_four_seeds_at_plus_minus_3_56_s(tmp_path, repo_root):
    losses = [3.40, 3.42, 3.47, 3.45]
    floor = measure_floor([_fake_run(tmp_path, repo_root, n, loss=l) for n, l in zip(FLOOR, losses)])
    assert floor["n"] == len(FLOOR) == 4
    assert floor["half_width"] / floor["std"] == pytest.approx(3.56, abs=0.005)


def test_a_floor_without_spread_judges_nothing(tmp_path, repo_root):
    runs = _evaluated_floor(tmp_path, repo_root, blimp=(0.6, 0.6, 0.6))
    floor = measure_floor(runs, "blimp.accuracy")
    assert floor["std"] == 0.0
    assert compare(floor, 0.61)["verdict"].startswith("not judged")


def test_compare_calls_only_a_run_outside_the_interval_different():
    floor = {"mean": 3.43, "half_width": 0.05}
    assert compare(floor, 3.40)["verdict"] == "within the floor"
    assert compare(floor, 3.48)["verdict"] == "within the floor"
    assert compare(floor, 3.37)["verdict"] == "lower than the baseline"
    assert compare(floor, 3.49)["verdict"] == "higher than the baseline"
    assert compare(floor, 3.37)["difference"] == pytest.approx(-0.06)


def _manifest(run_dir):
    return json.loads((run_dir / "manifest.json").read_text())


@pytest.mark.parametrize("arm", ["p2.yaml", "p3.yaml", "p4.yaml"])
def test_the_arms_are_comparable_with_the_baseline(tmp_path, repo_root, arm):
    check_comparable(_manifest(_fake_run(tmp_path, repo_root, "p1.yaml")),
                     _manifest(_fake_run(tmp_path, repo_root, arm)))


@pytest.mark.parametrize("edits,match", [
    ({"training__max_steps": 575}, "training.max_steps"),
    ({"training__validation_batches": 10}, "training.validation_batches"),
    ({"training__batch_size": 16, "training__gradient_accumulation": 16}, "training.batch_size"),
    ({"model__context_length": 256}, "model.context_length"),
    ({"precision": "bf16"}, "manifest.device.precision"),
])
def test_refuses_an_arm_validated_or_budgeted_differently(tmp_path, repo_root, edits, match):
    base = _manifest(_fake_run(tmp_path, repo_root, "p1.yaml"))
    arm = _manifest(_fake_run(tmp_path, repo_root, "p2.yaml", **edits))
    with pytest.raises(FloorError, match=match):
        check_comparable(base, arm)


def test_refuses_an_arm_validated_on_another_val_shard(tmp_path, repo_root):
    base = _manifest(_fake_run(tmp_path, repo_root, "p1.yaml"))
    arm = _manifest(_fake_run(tmp_path, repo_root, "p2.yaml"))
    base["data"]["val_shard"] = {"val": 4_537_964}
    arm["data"]["val_shard"] = {"val": 1_000_000}
    with pytest.raises(FloorError, match="manifest.data.val_shard.val"):
        check_comparable(base, arm)
    arm["data"]["val_shard"] = {"personal_val": 4_537_964}
    with pytest.raises(FloorError, match="val_shard"):
        check_comparable(base, arm)


def test_refuses_runs_that_share_a_data_order(tmp_path, repo_root):
    """Option A (only training.seed varies) is not the floor this project chose."""
    runs = [_fake_run(tmp_path, repo_root, n, data__seed=1234) for n in FLOOR]
    with pytest.raises(FloorError, match="share a data.seed"):
        measure_floor(runs)


def test_refuses_runs_that_share_an_initialisation(tmp_path, repo_root):
    runs = [_fake_run(tmp_path, repo_root, n, training__seed=1337) for n in FLOOR]
    with pytest.raises(FloorError, match="share a training.seed"):
        measure_floor(runs)


def test_refuses_runs_that_differ_in_anything_else(tmp_path, repo_root):
    runs = [_fake_run(tmp_path, repo_root, n) for n in FLOOR[:2]]
    runs.append(_fake_run(tmp_path, repo_root, FLOOR[2], training__learning_rate=1.2e-3))
    with pytest.raises(FloorError, match="training.learning_rate"):
        measure_floor(runs)


def test_refuses_runs_at_different_precisions(tmp_path, repo_root):
    runs = [_fake_run(tmp_path, repo_root, n) for n in FLOOR[:2]]
    runs.append(_fake_run(tmp_path, repo_root, FLOOR[2], precision="bf16"))
    with pytest.raises(FloorError, match="manifest.device.precision"):
        measure_floor(runs)


@pytest.mark.parametrize("status,match", [("diverged", "diverged"), ("killed", "neither finished")])
def test_refuses_a_baseline_that_did_not_finish(tmp_path, repo_root, status, match):
    runs = [_fake_run(tmp_path, repo_root, n) for n in FLOOR[:2]]
    runs.append(_fake_run(tmp_path, repo_root, FLOOR[2], status=status))
    with pytest.raises(FloorError, match=match):
        measure_floor(runs)


def test_refuses_fewer_than_three_runs(tmp_path, repo_root):
    with pytest.raises(FloorError, match="at least 3"):
        measure_floor([_fake_run(tmp_path, repo_root, n) for n in FLOOR[:2]])


def test_t_quantiles_match_scipy():
    stats = pytest.importorskip("scipy.stats")
    for df, t in T_975.items():
        assert t == pytest.approx(stats.t.ppf(0.975, df), abs=1e-4)


def test_cli_writes_the_report_and_judges_an_arm(tmp_path, repo_root):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    runs = [_fake_run(runs_dir, repo_root, n, loss=l) for n, l in zip(FLOOR, [3.40, 3.42, 3.47])]
    arm = _fake_run(runs_dir, repo_root, "p1.yaml", loss=3.10)
    out = tmp_path / "reports"
    cmd = [sys.executable, str(repo_root / "scripts" / "noise_floor.py"), *map(str, runs),
           "--compare", str(arm), "--out-dir", str(out)]

    done = subprocess.run(cmd, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert json.loads((out / "p1.json").read_text())["n"] == 3
    assert "lower than the baseline" in done.stdout

    short = _fake_run(runs_dir, repo_root, "p1.yaml", loss=3.10, training__max_steps=575)
    done = subprocess.run(cmd[:-4] + ["--compare", str(short), "--out-dir", str(out)],
                          capture_output=True, text=True)
    assert done.returncode == 1
    assert "not comparable" in done.stderr and "training.max_steps" in done.stderr

    done = subprocess.run(cmd[:2] + [str(r) for r in runs[:2]], capture_output=True, text=True)
    assert done.returncode == 1
    assert "at least 3" in done.stderr


# ------------------------------------------------------- eval metrics


def _fake_eval(run_dir, *, values=None, eval_file=EVAL_FILE, **edits):
    """An eval file of ``run_dir``'s final checkpoint; ``edits`` replace top-level keys."""
    config = _manifest(run_dir)["config"]
    values = values or {}
    record = {
        "eval_version": 1,
        "eval_config_fingerprint": "7c9503161d6f",
        "split": "val",
        "checkpoint": {"path": str(run_dir / "last.pt"), "step": config["training"]["max_steps"],
                       "config_name": config["name"], "config_fingerprint": config_fingerprint(config)},
        "tokenizer": {"id": "minime-16k@abc", "dev": False},
        "device": {"device": "mps", "accelerator": "Apple M1", "precision": "fp32", "torch": "2.11.0"},
        "perplexity": {"personal": {"loss": values.get("personal", 4.0)}},
        "blimp": {"accuracy": values.get("blimp", 0.6)},
        "eval_sets": {"ce/personal/val": {"digest": "aaaa", "tokens": 100}, "blimp": {"sha256": "bbbb"}},
        **edits,
    }
    (run_dir / eval_file).write_text(json.dumps(record))
    return record


def _evaluated_floor(tmp_path, repo_root, personal=(4.10, 4.12, 4.17), blimp=(0.60, 0.61, 0.59)):
    runs = [_fake_run(tmp_path, repo_root, n) for n in FLOOR[: len(personal)]]
    for run, p, b in zip(runs, personal, blimp):
        _fake_eval(run, values={"personal": p, "blimp": b})
    return runs


def test_an_eval_metric_gets_the_same_interval(tmp_path, repo_root):
    runs = _evaluated_floor(tmp_path, repo_root)
    floor = measure_floor(runs, "perplexity.personal.loss")
    std = math.sqrt(sum((x - 4.13) ** 2 for x in (4.10, 4.12, 4.17)) / 2)
    assert floor["metric"] == "perplexity.personal.loss"
    assert floor["mean"] == pytest.approx(4.13)
    assert floor["half_width"] == pytest.approx(4.3027 * std * math.sqrt(4 / 3))
    assert [r["perplexity.personal.loss"] for r in floor["runs"]] == [4.10, 4.12, 4.17]
    assert floor["eval"]["split"] == "val"
    assert floor["eval"]["eval_config_fingerprint"] == "7c9503161d6f"
    assert len(floor["eval"]["eval_sets_digest"]) == 16
    assert "eval" not in measure_floor(runs)  # the validation-loss floor is unchanged


def test_the_config_fingerprint_is_the_one_the_harness_records(tmp_path, repo_root):
    """A run's manifest and its checkpoint, as train.py writes them, agree."""
    from minime.config import ModelConfig
    from minime.evaluation.harness import load_model
    from minime.model import build_model
    from minime.training.checkpoint import save_checkpoint
    from minime.training.logging import RunLogger

    cfg = load_config(repo_root / "configs" / "tiny.yaml").to_dict()
    model = build_model(ModelConfig(**cfg["model"]))
    ckpt = save_checkpoint(tmp_path / "run" / "last.pt", model=model, step=3, config=cfg)
    RunLogger(tmp_path / "run", cfg).close()
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert load_model(ckpt)[1]["config_fingerprint"] == config_fingerprint(manifest["config"])
    cfg["training"]["seed"] += 1
    assert config_fingerprint(cfg) != config_fingerprint(manifest["config"])


def test_refuses_a_run_without_an_eval_file(tmp_path, repo_root):
    runs = _evaluated_floor(tmp_path, repo_root)
    (runs[1] / EVAL_FILE).unlink()
    with pytest.raises(FloorError, match="no eval-last.json"):
        measure_floor(runs, "perplexity.personal.loss")


def test_refuses_another_seeds_eval_file(tmp_path, repo_root):
    """Same config name, same step: only the fingerprint tells two seeds apart."""
    runs = _evaluated_floor(tmp_path, repo_root)
    (runs[1] / EVAL_FILE).write_text((runs[0] / EVAL_FILE).read_text())
    with pytest.raises(FloorError, match="another config or seed"):
        measure_floor(runs, "perplexity.personal.loss")


def test_refuses_an_eval_of_an_earlier_checkpoint(tmp_path, repo_root):
    runs = _evaluated_floor(tmp_path, repo_root)
    record = json.loads((runs[2] / EVAL_FILE).read_text())
    record["checkpoint"]["step"] = 500
    (runs[2] / EVAL_FILE).write_text(json.dumps(record))
    with pytest.raises(FloorError, match="step 500, not the final step 2300"):
        measure_floor(runs, "perplexity.personal.loss")


def test_refuses_the_test_split(tmp_path, repo_root):
    runs = _evaluated_floor(tmp_path, repo_root)
    _fake_eval(runs[0], split="test")
    with pytest.raises(FloorError, match="'test' split"):
        measure_floor(runs, "perplexity.personal.loss")


def test_refuses_an_eval_with_another_tokenizer(tmp_path, repo_root):
    runs = _evaluated_floor(tmp_path, repo_root)
    _fake_eval(runs[0], tokenizer={"id": "minime-16k-dev@def", "dev": True})
    with pytest.raises(FloorError, match="tokenizer"):
        measure_floor(runs, "perplexity.personal.loss")


@pytest.mark.parametrize("edits,match", [
    ({"eval_config_fingerprint": "000000000000"}, "eval_config_fingerprint"),
    ({"eval_sets": {"ce/personal/val": {"digest": "cccc", "tokens": 100}}}, "eval_sets"),
    ({"device": {"device": "cuda", "precision": "fp32", "torch": "2.11.0"}}, "device.device"),
    ({"device": {"device": "mps", "precision": "fp32", "torch": "2.12.0"}}, "device.torch"),
    ({"eval_version": 2}, "eval_version"),
])
def test_refuses_evals_made_differently(tmp_path, repo_root, edits, match):
    runs = _evaluated_floor(tmp_path, repo_root)
    _fake_eval(runs[2], **edits)
    with pytest.raises(FloorError, match=match):
        measure_floor(runs, "perplexity.personal.loss")
    base = json.loads((runs[0] / EVAL_FILE).read_text())
    with pytest.raises(FloorError, match="not evaluated like the baseline"):
        check_eval_comparable(base, json.loads((runs[2] / EVAL_FILE).read_text()))


@pytest.mark.parametrize("metric,match", [
    ("perplexity.general_weighted.loss", "no 'general_weighted'"),
    ("perplexity.personal", "not a finite number"),
    ("blimp.accuracy", "not a finite number"),
])
def test_refuses_a_metric_that_is_not_a_number_in_every_file(tmp_path, repo_root, metric, match):
    runs = _evaluated_floor(tmp_path, repo_root)
    record = json.loads((runs[1] / EVAL_FILE).read_text())
    record["blimp"]["accuracy"] = None
    (runs[1] / EVAL_FILE).write_text(json.dumps(record))
    with pytest.raises(FloorError, match=match):
        measure_floor(runs, metric)


def test_eval_comparisons_skip_the_validation_windows_but_not_the_budget(tmp_path, repo_root):
    base = _manifest(_fake_run(tmp_path, repo_root, "p1.yaml"))
    arm = _manifest(_fake_run(tmp_path, repo_root, "p2.yaml", training__validation_batches=10))
    base["data"]["val_shard"] = {"val": 4_537_964}
    arm["data"]["val_shard"] = {"personal_val": 1_000_000}
    with pytest.raises(FloorError):
        check_comparable(base, arm)
    check_comparable(base, arm, validation=False)
    short = _manifest(_fake_run(tmp_path, repo_root, "p2.yaml", training__max_steps=575))
    with pytest.raises(FloorError, match="training.max_steps"):
        check_comparable(base, short, validation=False)


def test_cli_floors_eval_metrics_and_judges_an_arm(tmp_path, repo_root):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    runs = _evaluated_floor(runs_dir, repo_root)
    arm = _fake_run(runs_dir, repo_root, "p2.yaml", loss=3.10)
    _fake_eval(arm, values={"personal": 3.80, "blimp": 0.70})
    out = tmp_path / "reports"
    cmd = [sys.executable, str(repo_root / "scripts" / "noise_floor.py"), *map(str, runs),
           "--eval", "perplexity.personal.loss", "blimp.accuracy", "--compare", str(arm),
           "--out-dir", str(out)]

    done = subprocess.run(cmd, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    report = json.loads((out / "p1-eval.json").read_text())
    assert set(report["metrics"]) == {"perplexity.personal.loss", "blimp.accuracy"}
    assert report["eval"]["file"] == EVAL_FILE
    judged = report["comparisons"][arm.name]
    assert judged["perplexity.personal.loss"]["verdict"] == "lower than the baseline"
    assert judged["blimp.accuracy"]["verdict"] == "higher than the baseline"
    assert judged["perplexity.personal.loss"]["role"] == "primary"
    assert judged["blimp.accuracy"]["role"] == report["metrics"]["blimp.accuracy"]["role"] == "exploratory"
    assert "blimp.accuracy 0.7000 (+0.1000) -> higher than the baseline  [exploratory]" in done.stdout
    assert not (out / "p1.json").exists()  # the validation-loss report is its own

    _fake_eval(arm, values={"personal": 3.80}, eval_config_fingerprint="000000000000")
    done = subprocess.run(cmd, capture_output=True, text=True)
    assert done.returncode == 1
    assert "not evaluated like the baseline" in done.stderr


def test_only_the_frontiers_two_axes_are_primary():
    """Decided with the owner (2026-10-05) before any floor checkpoint's eval was read."""
    assert PRIMARY_EVAL_METRICS == ("perplexity.personal.loss", "perplexity.general_weighted.loss")
    assert EVAL_METRICS[:2] == PRIMARY_EVAL_METRICS and len(set(EVAL_METRICS)) == len(EVAL_METRICS)
