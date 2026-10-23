"""The E4 learning-rate protocol (M7): the sweep configs and the pick."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from minime.config import load_config
from minime.training.lr_sweep import SWEEPS, SweepError, flatten, pick_lr

EXPERIMENTS = "configs/experiments"


def _sweep_configs(repo_root, sweep: str):
    paths = sorted((repo_root / EXPERIMENTS).glob(f"e4_lr_{sweep}_*.yaml"))
    return {p.name: load_config(p) for p in paths if not p.name.endswith("_base.yaml")}


# ------------------------------------------------------ shipped configs


@pytest.mark.parametrize("sweep", sorted(SWEEPS))
def test_each_sweep_varies_its_learning_rate_and_nothing_else(repo_root, sweep):
    configs = _sweep_configs(repo_root, sweep)
    assert len(configs) >= 3

    key = SWEEPS[sweep].key
    flat = {name: flatten(cfg.to_dict()) for name, cfg in configs.items()}
    values = [f[key] for f in flat.values()]
    assert len(set(values)) == len(values), "two sweep points share an LR"
    assert all(isinstance(v, float) for v in values), "YAML 1.1 reads 6e-4 as a string"

    for k in set().union(*flat.values()) - {key, "name", "notes"}:
        assert len({json.dumps(f[k]) for f in flat.values()}) == 1, f"{sweep} sweep varies {k}"
    assert all(cfg.training.optimizer == SWEEPS[sweep].optimizer for cfg in configs.values())
    assert len({cfg.name for cfg in configs.values()}) == len(configs), "run names would collide"


def test_the_sweep_is_the_probe_at_a_quarter_budget(repo_root):
    probe = load_config(repo_root / "configs" / "probe.yaml")
    for sweep in SWEEPS:
        for cfg in _sweep_configs(repo_root, sweep).values():
            assert cfg.model == probe.model
            assert cfg.tokens_per_step == probe.tokens_per_step
            assert cfg.token_budget == pytest.approx(probe.token_budget / 4, rel=0.01)
            assert cfg.training.warmup_steps / cfg.training.max_steps == pytest.approx(
                probe.training.warmup_steps / probe.training.max_steps, rel=0.02
            )


def test_the_two_sweeps_share_everything_but_the_optimiser(repo_root):
    """The Muon sweep differs from an AdamW point only in the optimiser and Muon's LR."""
    adamw = next(iter(_sweep_configs(repo_root, "adamw").values()))
    muon = next(iter(_sweep_configs(repo_root, "muon").values()))
    a, m = flatten(adamw.to_dict()), flatten(muon.to_dict())
    differing = {k for k in a if a[k] != m[k]} - {"name", "notes"}
    assert differing <= {"training.optimizer", "training.muon_lr", "training.learning_rate"}


def test_the_probe_and_the_muon_sweep_use_the_adamw_pick(repo_root):
    report = json.loads((repo_root / "reports" / "lr_sweep" / "adamw.json").read_text())
    assert not report["at_edge"]
    probe = load_config(repo_root / "configs" / "probe.yaml")
    assert probe.training.learning_rate == report["winner"]
    for cfg in _sweep_configs(repo_root, "muon").values():
        assert cfg.training.learning_rate == report["winner"]


# ------------------------------------------------------------- the pick


def _fake_run(tmp_path, repo_root, config_name, *, loss=3.0, status="finished", data=None, precision=None,
              **edits):
    cfg = load_config(repo_root / EXPERIMENTS / config_name).to_dict()
    for dotted, value in edits.items():
        section, _, key = dotted.partition("__")
        cfg[section][key] = value
    run_dir = tmp_path / f"{cfg['name']}-{len(list(tmp_path.iterdir()))}"
    run_dir.mkdir()
    manifest = {
        "config": cfg,
        "data": data or {"data_dir": str(tmp_path), "tokenizer": "minime-16k@abc", "shards": {"general": 10}},
    }
    if precision:
        manifest["device"] = {"accelerator": "Tesla T4", "precision": precision}
    if status == "finished":
        manifest["summary"] = {"steps": cfg["training"]["max_steps"], "final_val_loss": loss}
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    records = [{"step": 1, "phase": "train", "train_loss": 9.0}]
    if status == "diverged":
        records.append({"step": 2, "phase": "diverged", "train_loss": float("nan")})
    (run_dir / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return run_dir


ADAMW = ["e4_lr_adamw_0.0006.yaml", "e4_lr_adamw_0.0012.yaml", "e4_lr_adamw_0.0024.yaml", "e4_lr_adamw_0.0048.yaml"]
MUON = ["e4_lr_muon_0.01.yaml", "e4_lr_muon_0.02.yaml", "e4_lr_muon_0.04.yaml", "e4_lr_muon_0.08.yaml"]


def test_picks_the_lowest_final_validation_loss(tmp_path, repo_root):
    runs = [
        _fake_run(tmp_path, repo_root, name, loss=loss)
        for name, loss in zip(ADAMW, [3.40, 3.21, 3.25, 3.60])
    ]
    report = pick_lr("adamw", runs)
    assert report["winner"] == 1.2e-3
    assert report["winner_final_val_loss"] == 3.21
    assert report["margin"] == pytest.approx(0.04)
    assert not report["at_edge"]
    assert [p["value"] for p in report["points"]] == [6e-4, 1.2e-3, 2.4e-3, 4.8e-3]
    assert report["token_budget"] == 575 * 131_072


def test_a_diverged_run_is_a_data_point_not_an_error(tmp_path, repo_root):
    runs = [_fake_run(tmp_path, repo_root, n, loss=l) for n, l in zip(ADAMW[:3], [3.4, 3.2, 3.3])]
    runs.append(_fake_run(tmp_path, repo_root, ADAMW[3], status="diverged"))
    report = pick_lr("adamw", runs)
    assert report["winner"] == 1.2e-3
    assert report["points"][-1]["status"] == "diverged"


@pytest.mark.parametrize("best", [0, 3])
def test_a_winner_at_the_edge_is_flagged(tmp_path, repo_root, best):
    losses = [3.5, 3.4, 3.3, 3.2] if best == 3 else [3.2, 3.3, 3.4, 3.5]
    runs = [_fake_run(tmp_path, repo_root, n, loss=l) for n, l in zip(ADAMW, losses)]
    assert pick_lr("adamw", runs)["at_edge"]


def test_refuses_a_sweep_that_varies_anything_else(tmp_path, repo_root):
    runs = [_fake_run(tmp_path, repo_root, n) for n in ADAMW[:2]]
    runs.append(_fake_run(tmp_path, repo_root, ADAMW[2], training__seed=7))
    with pytest.raises(SweepError, match="training.seed"):
        pick_lr("adamw", runs)


def test_refuses_runs_on_different_data(tmp_path, repo_root):
    runs = [_fake_run(tmp_path, repo_root, n) for n in ADAMW[:2]]
    other = {"data_dir": "/kaggle/input", "tokenizer": "minime-16k@other", "shards": {"general": 10}}
    runs.append(_fake_run(tmp_path, repo_root, ADAMW[2], data=other))
    with pytest.raises(SweepError, match="manifest.data.tokenizer"):
        pick_lr("adamw", runs)


def test_refuses_runs_at_different_precisions(tmp_path, repo_root):
    """`precision: auto` resolves per GPU, so only the manifest shows what ran."""
    runs = [_fake_run(tmp_path, repo_root, n, precision="bf16") for n in ADAMW[:2]]
    runs.append(_fake_run(tmp_path, repo_root, ADAMW[2], precision="fp16"))
    with pytest.raises(SweepError, match="manifest.device.precision"):
        pick_lr("adamw", runs)


def test_the_data_directory_itself_may_differ(tmp_path, repo_root):
    moved = {"data_dir": "/kaggle/input/shards", "tokenizer": "minime-16k@abc", "shards": {"general": 10}}
    runs = [_fake_run(tmp_path, repo_root, n, loss=l) for n, l in zip(ADAMW[:2], [3.4, 3.2])]
    runs.append(_fake_run(tmp_path, repo_root, ADAMW[2], loss=3.3, data=moved))
    assert pick_lr("adamw", runs)["winner"] == 1.2e-3


def test_refuses_an_unfinished_run(tmp_path, repo_root):
    runs = [_fake_run(tmp_path, repo_root, n) for n in ADAMW[:2]]
    runs.append(_fake_run(tmp_path, repo_root, ADAMW[2], status="killed"))
    with pytest.raises(SweepError, match="neither finished nor diverged"):
        pick_lr("adamw", runs)


def test_refuses_a_run_of_the_other_optimiser(tmp_path, repo_root):
    runs = [_fake_run(tmp_path, repo_root, n) for n in ADAMW[:3]]
    with pytest.raises(SweepError, match="optimizer 'adamw' in the muon sweep"):
        pick_lr("muon", runs)


def test_refuses_too_few_points_and_repeated_points(tmp_path, repo_root):
    with pytest.raises(SweepError, match="at least 3"):
        pick_lr("adamw", [_fake_run(tmp_path, repo_root, n) for n in ADAMW[:2]])
    runs = [_fake_run(tmp_path, repo_root, n) for n in (ADAMW[0], ADAMW[0], ADAMW[1])]
    with pytest.raises(SweepError, match="share"):
        pick_lr("adamw", runs)


def _adamw_report(winner, at_edge=False):
    return {"sweep": "adamw", "key": "training.learning_rate", "winner": winner, "at_edge": at_edge}


def test_the_muon_sweep_is_anchored_to_the_adamw_winner(tmp_path, repo_root):
    runs = [_fake_run(tmp_path, repo_root, n, loss=l) for n, l in zip(MUON, [3.3, 3.1, 3.2, 3.4])]

    shipped = load_config(repo_root / EXPERIMENTS / MUON[0]).training.learning_rate

    with pytest.raises(SweepError, match="pick adamw first"):
        pick_lr("muon", runs)
    with pytest.raises(SweepError, match="edge"):
        pick_lr("muon", runs, anchor_report=_adamw_report(winner=shipped, at_edge=True))
    with pytest.raises(SweepError, match=f"learning_rate={shipped * 2}"):
        pick_lr("muon", runs, anchor_report=_adamw_report(winner=shipped * 2))

    report = pick_lr("muon", runs, anchor_report=_adamw_report(winner=shipped))
    assert report["winner"] == 0.02
    assert report["anchor"] == {"sweep": "adamw", "key": "training.learning_rate", "value": shipped}


def test_cli_writes_the_report_and_exits_2_at_the_edge(tmp_path, repo_root):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    runs = [_fake_run(runs_dir, repo_root, n, loss=l) for n, l in zip(ADAMW, [3.5, 3.4, 3.3, 3.2])]
    out = tmp_path / "reports"
    cmd = [sys.executable, str(repo_root / "scripts" / "pick_lr.py"), "adamw", *map(str, runs),
           "--out-dir", str(out)]

    done = subprocess.run(cmd, capture_output=True, text=True)
    assert done.returncode == 2, done.stderr
    assert "EDGE" in done.stderr
    assert json.loads((out / "adamw.json").read_text())["winner"] == 4.8e-3

    muon = [_fake_run(runs_dir, repo_root, n) for n in MUON]
    cmd = [sys.executable, str(repo_root / "scripts" / "pick_lr.py"), "muon", *map(str, muon),
           "--out-dir", str(out)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    assert done.returncode == 1 and "edge" in done.stderr
