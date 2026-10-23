"""Tests for the training system.

The resume test is the important one.  A checkpoint that restores weights but
not the optimizer moments, the RNG or the sampler position produces a run that
looks healthy and is not the experiment it claims to be.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import torch

from minime.config import Config, DataConfig, ModelConfig, TrainingConfig
from minime.data.binidx import ShardWriter, TokenShard
from minime.data.mixture import MixtureSampler, make_val_batches
from minime.model import build_model
from minime.training.checkpoint import (
    file_sha256,
    load_checkpoint,
    load_weights,
    rotate_checkpoints,
    save_checkpoint,
    summarise,
)
from minime.training.inputs import parse_val_shards, resolve_shard
from minime.training.noise_floor import config_fingerprint
from minime.training.logging import RunLogger, read_metrics
from minime.training.schedule import apply_schedule, lr_at_step
from minime.training.launch import PAUSED_EXIT
from minime.training.trainer import Trainer
from minime.utils.costs import append_cost_row, estimate_cost, read_costs, total_spend
from minime.utils.device import resolve
from minime.utils.throughput import ThroughputMeter, format_duration, peak_flops_for


# ---------------------------------------------------------------- schedule


def test_warmup_rises_linearly_to_the_peak():
    lrs = [lr_at_step(s, 1.0, warmup_steps=10, max_steps=100) for s in range(10)]
    assert lrs[0] == pytest.approx(0.1)
    assert lrs[-1] == pytest.approx(1.0)
    assert all(b > a for a, b in zip(lrs, lrs[1:]))


def test_first_step_has_a_nonzero_learning_rate():
    """An lr of exactly 0 on step 0 wastes a step and hides bugs."""
    assert lr_at_step(0, 1.0, warmup_steps=100, max_steps=1000) > 0


def test_cosine_decays_to_the_floor_and_stays_there():
    kwargs = dict(base_lr=1.0, warmup_steps=10, max_steps=100, min_lr_ratio=0.1)
    assert lr_at_step(10, **kwargs) == pytest.approx(1.0)
    assert lr_at_step(55, **kwargs) == pytest.approx(0.55, abs=0.02)
    assert lr_at_step(100, **kwargs) == pytest.approx(0.1)
    assert lr_at_step(500, **kwargs) == pytest.approx(0.1), "past max_steps must clamp, not go negative"


def test_schedule_is_monotonic_after_warmup():
    lrs = [lr_at_step(s, 1.0, 10, 100) for s in range(10, 101)]
    assert all(b <= a + 1e-12 for a, b in zip(lrs, lrs[1:]))


def test_zero_warmup_starts_at_peak():
    assert lr_at_step(0, 1.0, warmup_steps=0, max_steps=100) == pytest.approx(1.0)


def test_schedule_scales_each_group_to_its_own_peak():
    model = build_model(ModelConfig(vocab_size=32, d_model=16, n_heads=2, n_layers=1, context_length=8))
    opt = torch.optim.AdamW(model.param_groups(0.1), lr=1.0)
    lrs = apply_schedule(opt, [0.02, 6e-4], step=60, warmup_steps=10, max_steps=100)
    assert [g["lr"] for g in opt.param_groups] == lrs
    assert lrs == [lr_at_step(60, peak, 10, 100) for peak in (0.02, 6e-4)]
    assert lrs[0] / lrs[1] == pytest.approx(0.02 / 6e-4)


def test_schedule_needs_one_peak_per_group():
    model = build_model(ModelConfig(vocab_size=32, d_model=16, n_heads=2, n_layers=1, context_length=8))
    opt = torch.optim.AdamW(model.param_groups(0.1), lr=1.0)
    with pytest.raises(ValueError, match="peak LRs"):
        apply_schedule(opt, [1.0], step=0, warmup_steps=0, max_steps=10)


# -------------------------------------------------------------- checkpoint


@pytest.fixture
def small_model():
    return build_model(
        ModelConfig(vocab_size=64, d_model=32, n_heads=4, n_layers=2, context_length=16)
    )


def test_checkpoint_restores_weights_exactly(tmp_path, small_model):
    idx = torch.randint(0, 64, (2, 16))
    small_model.eval()
    with torch.no_grad():
        before, _ = small_model(idx)

    save_checkpoint(tmp_path / "c.pt", model=small_model, step=7)

    restored = build_model(small_model.cfg)
    load_checkpoint(tmp_path / "c.pt", model=restored)
    restored.eval()
    with torch.no_grad():
        after, _ = restored(idx)

    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_checkpoint_restores_optimizer_moments(tmp_path, small_model):
    opt = torch.optim.AdamW(small_model.param_groups(0.1), lr=1e-3)
    idx = torch.randint(0, 64, (2, 16))
    for _ in range(3):
        _, loss = small_model(idx, targets=idx)
        opt.zero_grad()
        loss.backward()
        opt.step()

    save_checkpoint(tmp_path / "c.pt", model=small_model, optimizer=opt, step=3)

    fresh = build_model(small_model.cfg)
    fresh_opt = torch.optim.AdamW(fresh.param_groups(0.1), lr=1e-3)
    payload = load_checkpoint(tmp_path / "c.pt", model=fresh, optimizer=fresh_opt)

    assert payload["step"] == 3
    original = opt.state_dict()["state"]
    loaded = fresh_opt.state_dict()["state"]
    assert set(original) == set(loaded) and len(loaded) > 0
    for key in original:
        torch.testing.assert_close(original[key]["exp_avg"], loaded[key]["exp_avg"])
        assert original[key]["step"] == loaded[key]["step"]


def test_checkpoint_carries_sampler_position(tmp_path, small_model):
    shard = _make_shard(tmp_path / "s", vocab=64, n_tokens=4000)
    sampler = MixtureSampler({"a": shard}, {"a": 1.0}, context_length=16, batch_size=4, seed=5)
    for _ in range(3):
        sampler.batch()

    save_checkpoint(tmp_path / "c.pt", model=small_model, sampler_state=sampler.state_dict())
    expected = sampler.batch()[0]

    other = MixtureSampler({"a": shard}, {"a": 1.0}, context_length=16, batch_size=4, seed=5)
    load_checkpoint(tmp_path / "c.pt", sampler=other)
    assert torch.equal(other.batch()[0], expected)


def test_checkpoint_restores_rng_streams(tmp_path, small_model):
    save_checkpoint(tmp_path / "c.pt", model=small_model)
    expected = (torch.rand(3), np.random.rand(3))

    load_checkpoint(tmp_path / "c.pt")
    torch.testing.assert_close(torch.rand(3), expected[0])
    np.testing.assert_allclose(np.random.rand(3), expected[1])


def test_partial_write_does_not_clobber_a_good_checkpoint(tmp_path, small_model):
    path = save_checkpoint(tmp_path / "c.pt", model=small_model, step=1)
    assert path.exists()
    assert not list(tmp_path.glob("*.tmp")), "temporary file must be renamed away"


def test_rotate_keeps_only_the_newest_checkpoints(tmp_path, small_model):
    for step in (100, 200, 300, 400):
        save_checkpoint(tmp_path / f"step_{step}.pt", model=small_model, step=step)
    save_checkpoint(tmp_path / "last.pt", model=small_model, step=400)
    save_checkpoint(tmp_path / "best.pt", model=small_model, step=300)

    removed = rotate_checkpoints(tmp_path, keep=2)

    assert {p.name for p in removed} == {"step_100.pt", "step_200.pt"}
    remaining = {p.name for p in tmp_path.glob("*.pt")}
    assert remaining == {"step_300.pt", "step_400.pt", "last.pt", "best.pt"}


def test_summarise_reads_metadata(tmp_path, small_model):
    save_checkpoint(
        tmp_path / "c.pt", model=small_model, step=42,
        config={"name": "probe"}, metrics={"best_val_loss": 3.2},
    )
    info = summarise(tmp_path / "c.pt")
    assert info["step"] == 42 and info["config_name"] == "probe"
    assert info["metrics"]["best_val_loss"] == 3.2


# ----------------------------------------------------------------- trainer


def _make_shard(path, vocab: int = 64, n_tokens: int = 8000, seed: int = 0) -> TokenShard:
    rng = np.random.default_rng(seed)
    with ShardWriter(path, vocab_size=vocab) as writer:
        for _ in range(n_tokens // 200):
            writer.add_document(rng.integers(0, vocab, size=200))
    return TokenShard(path)


def _make_trainer(tmp_path, run_name="run", named_val=False, init_from=None, **training_kwargs):
    model_cfg = ModelConfig(vocab_size=64, d_model=32, n_heads=4, n_layers=2, context_length=16)
    defaults = dict(
        batch_size=4, gradient_accumulation=2, max_steps=20, warmup_steps=4,
        validation_interval=10, validation_batches=2, log_interval=5,
        checkpoint_interval=10, device="cpu", precision="fp32", seed=3,
    )
    training = TrainingConfig(**{**defaults, **training_kwargs})
    cfg = Config(name="unit", model=model_cfg, training=training, data=DataConfig(mixture={"a": 1.0}))

    train_shard = _make_shard(tmp_path / "train", seed=1)
    val_shard = _make_shard(tmp_path / "val", n_tokens=2000, seed=2)
    sampler = MixtureSampler({"a": train_shard}, {"a": 1.0}, context_length=16, batch_size=4, seed=9)
    val_batches = make_val_batches(val_shard, context_length=16, batch_size=4, n_batches=2)
    if named_val:
        other = _make_shard(tmp_path / "val2", n_tokens=2000, seed=5)
        val_batches = {"general": val_batches,
                       "personal": make_val_batches(other, context_length=16, batch_size=4, n_batches=2)}

    trainer = Trainer(
        config=cfg, model=build_model(model_cfg), sampler=sampler, val_batches=val_batches,
        run_dir=tmp_path / run_name, experiment="unit-test", venue="local",
        costs_path=tmp_path / "costs.csv", init_from=init_from,
    )
    return trainer, cfg


@pytest.mark.slow
def test_training_run_produces_every_artifact(tmp_path):
    trainer, cfg = _make_trainer(tmp_path)
    summary = trainer.train()

    run_dir = tmp_path / "run"
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "last.pt").exists()
    assert (run_dir / "best.pt").exists()

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["config"]["name"] == "unit"
    assert manifest["device"]["precision"] == "fp32"
    assert "git_revision" in manifest
    assert manifest["summary"]["steps"] == 20

    assert summary["tokens_seen"] == cfg.tokens_per_step * 20
    assert summary["final_val_loss"] is not None
    assert summary["tokens_per_second"] > 0


def _train_script(data_dir, run_dir, seed, cwd):
    """scripts/train.py in a fresh process, as Kaggle runs it: one step of
    the tiny model on CPU.  ``cwd`` keeps its cost row out of the tracked
    experiments/costs.csv."""
    repo = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(repo / "scripts" / "train.py"), "--config", str(repo / "configs" / "tiny.yaml"),
           "--data", str(data_dir), "--mixture", "a=1", "--run-dir", str(run_dir),
           "--set", "training.max_steps=1", "training.warmup_steps=0", "training.batch_size=2",
           "training.validation_batches=2", "training.device=cpu", "training.precision=fp32",
           "model.vocab_size=64", f"training.seed={seed}"]
    done = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=240)
    assert done.returncode == 0, done.stderr
    initial = [r for r in read_metrics(run_dir / "metrics.jsonl") if r.get("phase") == "init"]
    return initial[0]["val_loss"]


@pytest.mark.slow
def test_training_seed_sets_the_initial_weights_in_a_fresh_process(tmp_path):
    """torch seeds every process randomly at start-up, and the conftest seeds
    every test, so only a fresh process shows whether the model was built
    before the seed was set.  The validation windows are fixed, so the
    step-0 validation loss is a function of the initial weights alone."""
    data = tmp_path / "data"
    data.mkdir()
    _make_shard(data / "a.train", seed=1)
    _make_shard(data / "val", n_tokens=2000, seed=2)
    first = _train_script(data, tmp_path / "s3-a", 3, tmp_path)
    again = _train_script(data, tmp_path / "s3-b", 3, tmp_path)
    other = _train_script(data, tmp_path / "s4", 4, tmp_path)
    assert first == again
    assert first != other


# ----------------------------------------------- adaptation runs (M9, E1)


@pytest.mark.slow
def test_named_validation_sets_are_each_logged(tmp_path):
    """An adaptation run traces both axes of the frontier at every validation."""
    trainer, _ = _make_trainer(tmp_path, named_val=True)
    summary = trainer.train()
    records = [r for r in read_metrics(tmp_path / "run" / "metrics.jsonl")
               if r.get("phase") in ("init", "validation")]
    assert records[0]["step"] == 0
    for record in records:
        assert set(record["val_losses"]) == {"general", "personal"}
        # The first set is the run's val_loss.
        assert record["val_loss"] == pytest.approx(record["val_losses"]["general"], abs=1e-5)
        assert record["val_losses"]["general"] != record["val_losses"]["personal"]
    assert set(summary["final_val_losses"]) == {"general", "personal"}
    assert summary["final_val_loss"] == summary["final_val_losses"]["general"]


@pytest.mark.slow
def test_one_validation_set_logs_as_before(tmp_path):
    trainer, _ = _make_trainer(tmp_path)
    summary = trainer.train()
    records = read_metrics(tmp_path / "run" / "metrics.jsonl")
    assert not any("val_losses" in r for r in records)
    assert "final_val_losses" not in summary


def _source_checkpoint(tmp_path, **model_changes):
    model_cfg = {"vocab_size": 64, "d_model": 32, "n_heads": 4, "n_layers": 2, "context_length": 16,
                 **model_changes}
    model = build_model(ModelConfig(**model_cfg))
    config = {"name": "p1", "model": ModelConfig(**model_cfg).__dict__, "training": {"seed": 1}}
    path = save_checkpoint(tmp_path / "source.pt", model=model, step=2300, config=config,
                           optimizer=torch.optim.AdamW(model.parameters()))
    return path, model, config


def test_load_weights_starts_from_the_weights_and_nothing_else(tmp_path):
    path, source, config = _source_checkpoint(tmp_path)
    torch.manual_seed(99)
    fresh = build_model(ModelConfig(vocab_size=64, d_model=32, n_heads=4, n_layers=2, context_length=16))
    info = load_weights(path, fresh, dict(config["model"]))
    for name, tensor in source.state_dict().items():
        assert torch.equal(fresh.state_dict()[name], tensor), name
    assert info["sha256"] == file_sha256(path) and len(info["sha256"]) == 64
    assert info["step"] == 2300
    assert info["config_name"] == "p1"
    # The fingerprint evaluate.py records for the source checkpoint, so a run
    # can be traced to the P1 run it started from.
    assert info["config_fingerprint"] == config_fingerprint(config)


def test_load_weights_refuses_another_model(tmp_path):
    path, _, _ = _source_checkpoint(tmp_path, n_layers=3)
    model = build_model(ModelConfig(vocab_size=64, d_model=32, n_heads=4, n_layers=2, context_length=16))
    with pytest.raises(ValueError, match="n_layers"):
        load_weights(path, model, ModelConfig(vocab_size=64, d_model=32, n_heads=4, n_layers=2,
                                              context_length=16).__dict__)


@pytest.mark.slow
def test_the_start_is_recorded_and_survives_a_resume(tmp_path):
    start = {"path": "/ckpt/p1/last.pt", "sha256": "ab" * 32, "step": 2300,
             "config_name": "p1", "config_fingerprint": "887b94ede8a4"}
    trainer, _ = _make_trainer(tmp_path, init_from=start)
    trainer.train()
    assert json.loads((tmp_path / "run" / "manifest.json").read_text())["init_from"] == start
    assert load_checkpoint(tmp_path / "run" / "step_10.pt", restore_rng=False)["init_from"] == start

    resumed, _ = _make_trainer(tmp_path, run_name="resumed")
    resumed.resume(tmp_path / "run" / "step_10.pt")
    resumed.train()
    assert json.loads((tmp_path / "resumed" / "manifest.json").read_text())["init_from"] == start

    other, _ = _make_trainer(tmp_path, run_name="other", init_from={**start, "sha256": "cd" * 32})
    with pytest.raises(ValueError, match="started from"):
        other.resume(tmp_path / "run" / "step_10.pt")


def test_a_source_must_be_in_exactly_one_data_directory(tmp_path):
    first, second = tmp_path / "general", tmp_path / "personal"
    first.mkdir(), second.mkdir()
    _make_shard(first / "web.train", n_tokens=400)
    _make_shard(second / "chat.train", n_tokens=400)
    assert resolve_shard([first, second], "chat").n_tokens == 400
    assert resolve_shard(first, "web").n_tokens == 400
    with pytest.raises(FileNotFoundError, match="no shard for source 'code'"):
        resolve_shard([first, second], "code")
    _make_shard(second / "web.train", n_tokens=400)
    with pytest.raises(ValueError, match="more than one data directory"):
        resolve_shard([first, second], "web")


def test_validation_sets_are_one_shard_or_several_named_ones(tmp_path):
    assert parse_val_shards(["val"], [tmp_path]) == {"val": tmp_path / "val"}
    named = parse_val_shards(["general=/g/val", "personal=/p/val"], [tmp_path])
    assert list(named) == ["general", "personal"]
    assert named["personal"] == Path("/p/val")
    with pytest.raises(ValueError, match="two validation sets named"):
        parse_val_shards(["a=/x", "a=/y"], [tmp_path])
    with pytest.raises(ValueError, match="NAME=PATH"):
        parse_val_shards(["=/x"], [tmp_path])


def _train(args: list[str], sets: list[str], cwd) -> subprocess.CompletedProcess:
    repo = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(repo / "scripts" / "train.py"), "--config", str(repo / "configs" / "tiny.yaml"),
           *args, "--set", "training.max_steps=1", "training.warmup_steps=0", "training.batch_size=2",
           "training.validation_batches=2", "training.device=cpu", "training.precision=fp32",
           "model.vocab_size=64", *sets]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=240)


@pytest.mark.slow
def test_train_py_adapts_a_checkpoint_on_two_data_directories(tmp_path):
    """In a fresh process, as scripts/run_e1.py runs it: the weights come from
    the checkpoint (its last validation and the new run's first agree to the
    digit on the same windows), the mixture spans both directories, and both
    validation sets are logged."""
    general, personal = tmp_path / "general", tmp_path / "personal"
    general.mkdir(), personal.mkdir()
    _make_shard(general / "a.train", seed=1)
    _make_shard(general / "val", n_tokens=2000, seed=2)
    _make_shard(personal / "b.train", seed=3)
    _make_shard(personal / "val", n_tokens=2000, seed=4)

    done = _train(["--data", str(general), "--mixture", "a=1", "--run-dir", str(tmp_path / "source")],
                  ["training.seed=3"], tmp_path)
    assert done.returncode == 0, done.stderr
    source = json.loads((tmp_path / "source" / "manifest.json").read_text())
    checkpoint = tmp_path / "source" / "last.pt"

    adapt = ["--data", str(general), str(personal), "--mixture", "a=1,b=1",
             "--val-shard", f"first={general / 'val'}", f"second={personal / 'val'}",
             "--init-from", str(checkpoint)]
    done = _train([*adapt, "--run-dir", str(tmp_path / "adapted")], ["training.seed=4"], tmp_path)
    assert done.returncode == 0, done.stderr
    manifest = json.loads((tmp_path / "adapted" / "manifest.json").read_text())
    assert manifest["init_from"]["sha256"] == file_sha256(checkpoint)
    assert manifest["init_from"]["step"] == 1
    assert manifest["init_from"]["config_fingerprint"] == config_fingerprint(source["config"])
    assert manifest["data"]["data_dir"] == [str(general), str(personal)]
    assert set(manifest["data"]["shards"]) == {"a", "b"}
    assert set(manifest["data"]["val_shard"]) == {"first", "second"}
    initial = next(r for r in read_metrics(tmp_path / "adapted" / "metrics.jsonl") if r["phase"] == "init")
    assert initial["val_losses"]["first"] == pytest.approx(source["summary"]["final_val_loss"], abs=1e-5)
    assert load_checkpoint(tmp_path / "adapted" / "last.pt", restore_rng=False)["init_from"] == manifest["init_from"]

    done = _train([*adapt, "--run-dir", str(tmp_path / "wider")], ["model.n_layers=2"], tmp_path)
    assert done.returncode != 0
    assert "its model differs from this run's" in done.stderr

    done = _train([*adapt, "--resume", str(checkpoint), "--run-dir", str(tmp_path / "both")], [], tmp_path)
    assert done.returncode == 2
    assert "use one" in done.stderr


@pytest.mark.slow
def test_metrics_stream_records_grad_norm_and_validation(tmp_path):
    trainer, _ = _make_trainer(tmp_path)
    trainer.train()

    records = read_metrics(tmp_path / "run" / "metrics.jsonl")
    train_records = [r for r in records if r.get("phase") == "train"]
    val_records = [r for r in records if r.get("phase") in ("validation", "init")]

    assert train_records and val_records
    assert all("grad_norm" in r and r["grad_norm"] >= 0 for r in train_records)
    assert all("lr" in r and "tokens_seen" in r for r in train_records)
    assert all(r["val_ppl"] == pytest.approx(math.exp(r["val_loss"]), rel=1e-3) for r in val_records)
    assert val_records[0]["step"] == 0, "there must be a pre-training baseline point"


@pytest.mark.slow
def test_cost_row_is_written(tmp_path):
    trainer, _ = _make_trainer(tmp_path)
    trainer.train()

    rows = read_costs(tmp_path / "costs.csv")
    assert len(rows) == 1
    assert rows[0]["experiment"] == "unit-test"
    assert int(rows[0]["tokens_processed"]) > 0
    assert float(rows[0]["est_cost_usd"]) == 0.0


@pytest.mark.slow
@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_resume_continues_the_same_trajectory(tmp_path, optimizer):
    """On CPU this is exact, so any drift means state was dropped."""
    full, _ = _make_trainer(tmp_path, max_steps=20, optimizer=optimizer)
    full.train()
    uninterrupted = {
        r["step"]: r["train_loss"]
        for r in read_metrics(tmp_path / "run" / "metrics.jsonl")
        if r.get("phase") == "train"
    }

    resumed_trainer, _ = _make_trainer(tmp_path, run_name="resumed", max_steps=20, optimizer=optimizer)
    resumed_trainer.resume(tmp_path / "run" / "step_10.pt")
    resumed_trainer.train()
    resumed = {
        r["step"]: r["train_loss"]
        for r in read_metrics(tmp_path / "resumed" / "metrics.jsonl")
        if r.get("phase") == "train"
    }

    shared = sorted(set(uninterrupted) & set(resumed))
    assert shared, "no overlapping steps to compare"
    for step in shared:
        assert resumed[step] == pytest.approx(uninterrupted[step], abs=1e-4), (
            f"step {step} diverged after resume: "
            f"{uninterrupted[step]} vs {resumed[step]}"
        )


# ------------------------------------------- fp16 resume and the clean stop (M10)


def _scaler(**kwargs) -> torch.amp.GradScaler:
    """An enabled GradScaler on CPU.  At a loss scale on the edge of fp32
    overflow, some steps' gradients overflow and are skipped, in a pattern
    that depends on the scale and its growth tracker -- so a resume that
    restarted them skips other steps and leaves the uninterrupted curve."""
    return torch.amp.GradScaler("cpu", enabled=True, **{"init_scale": 2.0**127, "growth_interval": 3, **kwargs})


def test_checkpoint_restores_the_grad_scaler(tmp_path, small_model):
    scaler = _scaler(init_scale=2.0**10)
    opt = torch.optim.AdamW(small_model.parameters(), lr=1e-3)
    idx = torch.randint(0, 64, (2, 16))
    for _ in range(4):
        opt.zero_grad()
        _, loss = small_model(idx, targets=idx)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    state = scaler.state_dict()
    assert state["scale"] == 2.0**11 and state["_growth_tracker"] == 1, "a state a fresh scaler does not have"
    save_checkpoint(tmp_path / "c.pt", model=small_model, scaler_state=state, precision="fp16")

    fresh = _scaler(init_scale=2.0**10)
    payload = load_checkpoint(tmp_path / "c.pt", scaler=fresh)
    assert fresh.state_dict() == state
    assert payload["precision"] == "fp16"


def test_an_enabled_scaler_needs_the_saved_loss_scale(tmp_path, small_model):
    save_checkpoint(tmp_path / "none.pt", model=small_model)
    with pytest.raises(ValueError, match="restart the loss scale"):
        load_checkpoint(tmp_path / "none.pt", scaler=_scaler())
    # A disabled scaler would ignore the state silently: fp16 resumed as fp32.
    save_checkpoint(tmp_path / "fp16.pt", model=small_model, scaler_state=_scaler().state_dict())
    with pytest.raises(ValueError, match="enabled GradScaler"):
        load_checkpoint(tmp_path / "fp16.pt", scaler=torch.amp.GradScaler("cpu", enabled=False))
    # fp32 and bf16 save a disabled scaler's empty state, and resume without one.
    save_checkpoint(tmp_path / "fp32.pt", model=small_model,
                    scaler_state=torch.amp.GradScaler("cpu", enabled=False).state_dict())
    load_checkpoint(tmp_path / "fp32.pt", scaler=torch.amp.GradScaler("cpu", enabled=False))


def _losses(run_dir) -> dict[int, tuple[float, bool]]:
    """Each logged step's train loss, and whether its gradients were finite (not skipped)."""
    return {r["step"]: (r["train_loss"], math.isfinite(r["grad_norm"]))
            for r in read_metrics(Path(run_dir) / "metrics.jsonl") if r.get("phase") == "train"}


@pytest.mark.slow
def test_an_fp16_resume_keeps_the_loss_scale(tmp_path):
    full, _ = _make_trainer(tmp_path, max_steps=20, log_interval=1)
    full.scaler = _scaler()
    full.train()
    uninterrupted = _losses(tmp_path / "run")
    skipped = sorted(step for step, (_, finite) in uninterrupted.items() if not finite)
    assert any(s <= 10 for s in skipped) and any(s > 10 for s in skipped), skipped

    resumed, _ = _make_trainer(tmp_path, run_name="resumed", max_steps=20, log_interval=1)
    resumed.scaler = _scaler()
    resumed.resume(tmp_path / "run" / "step_10.pt")
    resumed.train()
    after = _losses(tmp_path / "resumed")
    assert sorted(after) == list(range(11, 21))
    for step, (loss, finite) in after.items():
        assert finite == uninterrupted[step][1], f"step {step}: skipped in one run and not the other"
        assert loss == pytest.approx(uninterrupted[step][0], abs=1e-4), f"step {step}"


@pytest.mark.slow
def test_a_resume_refuses_another_config_or_precision(tmp_path):
    first, _ = _make_trainer(tmp_path, max_steps=20)
    first.train()
    longer, _ = _make_trainer(tmp_path, run_name="longer", max_steps=30)
    with pytest.raises(ValueError, match=r"another config: training\.max_steps \(20 there, 30 here\)"):
        longer.resume(tmp_path / "run" / "step_10.pt")

    payload = torch.load(tmp_path / "run" / "step_10.pt", weights_only=False)
    assert payload["precision"] == "fp32"
    payload["precision"] = "fp16"
    torch.save(payload, tmp_path / "fp16.pt")
    other, _ = _make_trainer(tmp_path, run_name="other", max_steps=20)
    with pytest.raises(ValueError, match="trained in fp16"):
        other.resume(tmp_path / "fp16.pt")


@pytest.mark.slow
def test_a_clean_stop_pauses_and_a_resume_finishes_the_same_run(tmp_path):
    torch.manual_seed(0)   # _make_trainer draws the initial weights: the same for both runs
    full, cfg = _make_trainer(tmp_path, max_steps=20, log_interval=1)
    done = full.train()
    uninterrupted = _losses(tmp_path / "run")

    # Off the checkpoint and validation steps (10, 20), so last.pt is the only copy.
    torch.manual_seed(0)
    first, _ = _make_trainer(tmp_path, run_name="split", max_steps=20, log_interval=1)
    paused = first.train(stop_at_step=7)
    assert paused["paused"] and paused["steps"] == 7 and paused["reason"] == "stop_at_step 7"
    run = tmp_path / "split"
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["paused"]["step"] == 7
    assert "summary" not in manifest and "finished" not in manifest, "a paused run is not finished"
    assert [(s["start_step"], s["end_step"], s["outcome"]) for s in manifest["sessions"]] == [(0, 7, "paused")]
    assert load_checkpoint(run / "last.pt", restore_rng=False)["step"] == 7
    assert [r["phase"] for r in read_metrics(run / "metrics.jsonl")][-1] == "paused"

    second, _ = _make_trainer(tmp_path, run_name="split", max_steps=20, log_interval=1)
    second.resume(run / "last.pt")
    summary = second.train()
    assert summary["steps"] == 20 and not summary.get("paused")
    assert summary["final_val_loss"] == pytest.approx(done["final_val_loss"], abs=1e-5)
    assert summary["best_val_loss"] == pytest.approx(done["best_val_loss"], abs=1e-5)
    manifest = json.loads((run / "manifest.json").read_text())
    assert "paused" not in manifest and manifest["summary"]["steps"] == 20
    assert [(s["start_step"], s["end_step"], s["outcome"]) for s in manifest["sessions"]] == [
        (0, 7, "paused"), (7, 20, "finished")]
    assert all(s["git_revision"] and s["device"]["precision"] == "fp32" for s in manifest["sessions"])

    # One curve: every step once, the uninterrupted run's losses.
    records = read_metrics(run / "metrics.jsonl")
    steps = [r["step"] for r in records if r.get("phase") == "train"]
    assert steps == list(range(1, 21))
    assert [r["phase"] for r in records if r.get("phase") in ("paused", "resumed")] == ["paused", "resumed"]
    split = _losses(run)
    for step in range(1, 21):
        assert split[step][0] == pytest.approx(uninterrupted[step][0], abs=1e-4), f"step {step}"

    # Each session its own cost row: its own tokens, not the run's so far.
    rows = [r for r in read_costs(tmp_path / "costs.csv") if r["run_id"] == "split"]
    assert [int(r["tokens_processed"]) for r in rows] == [7 * cfg.tokens_per_step, 13 * cfg.tokens_per_step]
    assert rows[0]["final_val_loss"] == "" and "[paused at step 7]" in rows[0]["notes"]
    assert "[resumed at step 7]" in rows[1]["notes"]


@pytest.mark.slow
def test_a_deadline_pauses_after_the_step_that_passes_it(tmp_path):
    trainer, _ = _make_trainer(tmp_path, max_steps=20)
    paused = trainer.train(stop_by=datetime.now(timezone.utc) - timedelta(seconds=1))
    assert paused["paused"] and paused["steps"] == 1 and paused["reason"].startswith("stop_by ")


def test_a_stop_must_be_ahead_and_a_deadline_must_say_whose_clock(tmp_path):
    trainer, _ = _make_trainer(tmp_path, max_steps=20)
    with pytest.raises(ValueError, match="timezone-aware"):
        trainer.train(stop_by=datetime.now())
    trainer.state.step = 5
    with pytest.raises(ValueError, match="not after this session's first step"):
        trainer.train(stop_at_step=5)


def test_a_resumed_session_must_be_the_same_run(tmp_path):
    config = {"name": "flagship", "training": {"max_steps": 3450}}
    data = {"data_dir": "/kaggle/input/a", "tokenizer": "t@1", "shards": {"web": 10}}
    RunLogger(tmp_path, config, data=data).close()
    # Where the shards live may change between sessions; what they are may not.
    moved = RunLogger(tmp_path, config, data={**data, "data_dir": "/kaggle/input/b"}, resume=True, start_step=9)
    moved.close()
    sessions = json.loads((tmp_path / "manifest.json").read_text())["sessions"]
    assert [s.get("data_dir") for s in sessions] == ["/kaggle/input/a", "/kaggle/input/b"]
    assert sessions[1]["start_step"] == 9
    with pytest.raises(ValueError, match=r"other data; differs in \['shards'\]"):
        RunLogger(tmp_path, config, data={**data, "shards": {"web": 11}}, resume=True)
    with pytest.raises(ValueError, match=r"another config; differs in \['training.max_steps'\]"):
        RunLogger(tmp_path, {**config, "training": {"max_steps": 4000}}, data=data, resume=True)


def test_a_manifest_from_before_sessions_keeps_its_first_session(tmp_path):
    old = {"run": "p1-s1337", "started": "2026-10-05T01:00:00+00:00", "git_revision": "03afb51",
           "python": "3.11.13", "device": {"precision": "fp16"}, "config": {"name": "p1"}}
    (tmp_path / "manifest.json").write_text(json.dumps(old))
    RunLogger(tmp_path, {"name": "p1"}, resume=True, start_step=500).close()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["sessions"][0] == {k: old[k] for k in ("started", "git_revision", "python", "device")}
    assert manifest["resumed_from"] == old["started"]


def test_the_logged_train_loss_is_the_mean_over_micro_batches(tmp_path):
    """The loss is summed on the device and read once a step (the sync fix)."""
    trainer, cfg = _make_trainer(tmp_path, gradient_accumulation=3)
    state = trainer.sampler.state_dict()
    with torch.no_grad():
        expected = sum(float(trainer.model(*trainer.sampler.batch(trainer.spec.device))[1])
                       for _ in range(cfg.training.gradient_accumulation)) / cfg.training.gradient_accumulation
    trainer.sampler.load_state_dict(state)
    loss, _ = trainer._train_step()
    assert loss == pytest.approx(expected, rel=1e-6)


@pytest.mark.slow
def test_train_py_pauses_with_its_own_exit_status_and_resumes(tmp_path):
    """In fresh processes, as a Kaggle session runs it."""
    data = tmp_path / "data"
    data.mkdir()
    _make_shard(data / "a.train", seed=1)
    _make_shard(data / "val", n_tokens=2000, seed=2)
    repo = Path(__file__).resolve().parents[1]
    run = tmp_path / "run"
    cmd = [sys.executable, str(repo / "scripts" / "train.py"), "--config", str(repo / "configs" / "tiny.yaml"),
           "--data", str(data), "--mixture", "a=1", "--run-dir", str(run)]
    sets = ["--set", "training.max_steps=3", "training.warmup_steps=0", "training.batch_size=2",
            "training.validation_batches=2", "training.device=cpu", "training.precision=fp32",
            "model.vocab_size=64"]

    done = subprocess.run([*cmd, "--stop-at-step", "1", *sets], capture_output=True, text=True, cwd=tmp_path,
                          timeout=240)
    assert done.returncode == PAUSED_EXIT, done.stderr
    assert json.loads((run / "manifest.json").read_text())["paused"]["step"] == 1

    done = subprocess.run([*cmd, "--resume", str(run / "last.pt"), *sets[:1], "training.max_steps=4", *sets[2:]],
                          capture_output=True, text=True, cwd=tmp_path, timeout=240)
    assert done.returncode != 0 and "another config" in done.stderr

    done = subprocess.run([*cmd, "--resume", str(run / "last.pt"), *sets], capture_output=True, text=True,
                          cwd=tmp_path, timeout=240)
    assert done.returncode == 0, done.stderr
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["summary"]["steps"] == 3
    assert [s["outcome"] for s in manifest["sessions"]] == ["paused", "finished"]

    done = subprocess.run([*cmd, "--stop-by", "2026-10-20T09:30:00", *sets], capture_output=True, text=True,
                          cwd=tmp_path, timeout=240)
    assert done.returncode == 2 and "no UTC offset" in done.stderr


@pytest.mark.slow
def test_divergence_stops_the_run(tmp_path):
    """A NaN loss must abort rather than keep spending GPU hours."""
    trainer, _ = _make_trainer(tmp_path, learning_rate=1e9)
    with pytest.raises(RuntimeError, match="stopping rather than"):
        trainer.train()


# ------------------------------------------------------------- throughput


def test_throughput_meter_ignores_the_warmup_step():
    meter = ThroughputMeter()
    for _ in range(3):
        meter.start()
        meter.stop(1000)
    assert meter.total_tokens == 2000, "the first step must be excluded"
    assert meter.tokens_per_second > 0


def test_throughput_meter_requires_start_before_stop():
    with pytest.raises(RuntimeError, match="before start"):
        ThroughputMeter().stop(10)


def test_mfu_is_none_for_unknown_hardware():
    meter = ThroughputMeter()
    for _ in range(3):
        meter.start()
        meter.stop(1000)
    assert meter.mfu(1e6, "Some Unreleased GPU") is None
    assert meter.mfu(1e6, "Tesla T4") is not None


def test_peak_flops_matches_substrings():
    assert peak_flops_for("Tesla T4") == peak_flops_for("Tesla T4 (kaggle)")


@pytest.mark.parametrize("seconds,expected", [(45, "45s"), (125, "2m05s"), (3725, "1h02m")])
def test_format_duration(seconds, expected):
    assert format_duration(seconds) == expected


# ------------------------------------------------------------------ device


@pytest.fixture
def fake_gpu(monkeypatch):
    """A CUDA device of a chosen compute capability, on a machine without one."""

    def install(capability):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "Fake GPU")
        # What torch answers on a T4: emulated bf16 counts as supported.
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda including_emulation=True: True)

    return install


@pytest.mark.parametrize("capability,precision", [((7, 5), "fp16"), ((8, 0), "bf16"), ((8, 9), "bf16")])
def test_auto_precision_is_bf16_only_where_the_hardware_has_it(fake_gpu, capability, precision):
    fake_gpu(capability)
    spec = resolve("cuda", "auto")
    assert spec.precision == precision
    assert spec.use_scaler == (precision == "fp16")


def test_an_explicit_precision_is_kept_on_cuda(fake_gpu):
    fake_gpu((7, 5))
    assert resolve("cuda", "bf16").precision == "bf16"


def test_precision_is_fp32_off_cuda():
    assert resolve("cpu", "auto").precision == "fp32"
    assert resolve("cpu", "fp16").precision == "fp32"


# ------------------------------------------------------------------ costs


def test_cost_scales_with_runtime_and_rate():
    assert estimate_cost("vast-4090", 3600) == pytest.approx(0.35)
    assert estimate_cost("vast-4090", 1800) == pytest.approx(0.175)
    assert estimate_cost("kaggle-free", 36000) == 0.0
    assert estimate_cost("anything", 3600, rate_override=2.0) == pytest.approx(2.0)


def test_unknown_venue_is_rejected_with_a_hint():
    with pytest.raises(KeyError, match="--cost-per-hour"):
        estimate_cost("my-gpu", 60)


def test_total_spend_aggregates_runs(tmp_path):
    path = tmp_path / "costs.csv"
    for venue, seconds in (("kaggle-free", 3600), ("vast-4090", 7200)):
        append_cost_row(
            path, experiment="e", run_id="r", venue=venue, device="cuda", gpu_type="x",
            runtime_seconds=seconds, tokens_processed=1, model_params=1, tokens_per_second=1,
        )
    usd, hours = total_spend(path)
    assert usd == pytest.approx(0.70)
    assert hours == pytest.approx(3.0)
