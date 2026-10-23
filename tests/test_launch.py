"""Remote-run plumbing (M8): shard checksums, finding the shards, the GPU queue."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import torch

from minime.config import DataConfig, load_config
from minime.data.binidx import ShardWriter
from minime.training import launch
from minime.training.checkpoint import save_checkpoint
from minime.training.launch import (
    CHECKSUMS,
    PAUSED_EXIT,
    Job,
    LaunchError,
    device_ids,
    find_shard_dir,
    plan_runs,
    check_sources,
    run_queue,
    stage_upload,
    train_command,
    verify_checksums,
    verify_download,
    write_checksums,
    write_run_checksums,
)

FROZEN_ID = "minime-16k@92bb9c71bf6e2da5"


# ------------------------------------------------------------ checksums


@pytest.fixture
def shard_dir(tmp_path):
    d = tmp_path / "general"
    d.mkdir()
    (d / "web.train.bin").write_bytes(bytes(range(256)) * 64)
    (d / "web.train.idx.npy").write_bytes(b"\x93NUMPY index")
    (d / "manifest.json").write_text(json.dumps({"kind": "general", "tokenizer_id": FROZEN_ID}))
    return d


def test_checksums_round_trip_in_sha256sum_format(shard_dir):
    (shard_dir / "._web.train.bin").write_bytes(b"macOS metadata")
    path = write_checksums(shard_dir)
    lines = path.read_text().splitlines()
    assert [line.split("  ")[1] for line in lines] == [
        "manifest.json", "web.train.bin", "web.train.idx.npy"
    ]
    assert all(len(line.split("  ")[0]) == 64 for line in lines)
    assert verify_checksums(shard_dir) == 3


def test_checksums_match_the_sha256sum_tool(shard_dir):
    tool = subprocess.run(["which", "sha256sum"], capture_output=True, text=True).stdout.strip()
    tool = tool or subprocess.run(["which", "shasum"], capture_output=True, text=True).stdout.strip()
    if not tool:
        pytest.skip("no sha256sum or shasum")
    write_checksums(shard_dir)
    cmd = [tool, "-c", CHECKSUMS] if tool.endswith("sha256sum") else [tool, "-a", "256", "-c", CHECKSUMS]
    assert subprocess.run(cmd, cwd=shard_dir, capture_output=True).returncode == 0


@pytest.mark.parametrize("damage, message", [
    (lambda d: (d / "web.train.bin").write_bytes(b"truncated"), "changed: web.train.bin"),
    (lambda d: (d / "web.train.idx.npy").unlink(), "missing: web.train.idx.npy"),
    (lambda d: (d / "code.train.bin").write_bytes(b"extra"), "not in SHA256SUMS: code.train.bin"),
])
def test_verify_refuses_a_directory_that_is_not_what_was_hashed(shard_dir, damage, message):
    write_checksums(shard_dir)
    damage(shard_dir)
    with pytest.raises(LaunchError, match=message):
        verify_checksums(shard_dir)


def test_verify_refuses_unhashed_shards(shard_dir):
    with pytest.raises(LaunchError, match="checksum_shards.py"):
        verify_checksums(shard_dir)


def test_checksum_script_writes_then_verifies(repo_root, shard_dir):
    script = str(repo_root / "scripts" / "checksum_shards.py")
    assert subprocess.run([sys.executable, script, str(shard_dir)], capture_output=True).returncode == 0
    ok = subprocess.run([sys.executable, script, "--verify", str(shard_dir)], capture_output=True, text=True)
    assert ok.returncode == 0 and "3 files match" in ok.stdout
    (shard_dir / "web.train.bin").write_bytes(b"x")
    bad = subprocess.run([sys.executable, script, "--verify", str(shard_dir)], capture_output=True, text=True)
    assert bad.returncode == 1 and "changed: web.train.bin" in bad.stderr


# ------------------------------------------------------------ find_shard_dir


def test_find_shard_dir_searches_below_the_dataset_root(tmp_path, shard_dir):
    # Kaggle nests an uploaded folder one level down, or not, depending on the upload.
    personal = tmp_path / "personal"
    personal.mkdir()
    (personal / "manifest.json").write_text(json.dumps({"kind": "personal", "tokenizer_id": FROZEN_ID}))
    assert find_shard_dir(tmp_path, "general", FROZEN_ID) == shard_dir


def test_find_shard_dir_refuses_ambiguity_absence_and_the_wrong_tokenizer(tmp_path, shard_dir):
    with pytest.raises(LaunchError, match="tokenised with"):
        find_shard_dir(tmp_path, "general", "minime-16k-dev@0000000000000000")
    with pytest.raises(LaunchError, match="found 0"):
        find_shard_dir(tmp_path, "code-only", FROZEN_ID)
    twin = tmp_path / "copy"
    twin.mkdir()
    (twin / "manifest.json").write_text((shard_dir / "manifest.json").read_text())
    with pytest.raises(LaunchError, match="found 2"):
        find_shard_dir(tmp_path, "general", FROZEN_ID)


# ------------------------------------------------------------ planning


SWEEP = [f"configs/experiments/e4_lr_adamw_{lr}.yaml" for lr in ("0.0006", "0.0012", "0.0024", "0.0048")]


def test_plan_names_runs_as_train_py_does(repo_root, tmp_path):
    names = plan_runs([Job(c) for c in SWEEP], repo=repo_root, work_dir=tmp_path)
    assert names == [load_config(repo_root / c).run_name for c in SWEEP]
    assert names[1] == "e4-lr-adamw-0.0012-s1337"


def test_plan_gives_each_seed_its_own_run(repo_root, tmp_path):
    jobs = [Job("configs/probe.yaml", (f"training.seed={s}",)) for s in (1337, 1338, 1339)]
    assert plan_runs(jobs, repo=repo_root, work_dir=tmp_path) == [
        "probe-s1337", "probe-s1338", "probe-s1339"
    ]


def test_plan_refuses_jobs_that_would_share_or_reuse_a_run_directory(repo_root, tmp_path):
    with pytest.raises(LaunchError, match="share a run directory"):
        plan_runs([Job(SWEEP[0]), Job(SWEEP[1]), Job(SWEEP[0])], repo=repo_root, work_dir=tmp_path)
    # An override that does not change the run name still collides.
    with pytest.raises(LaunchError, match="share a run directory"):
        plan_runs([Job("configs/probe.yaml"), Job("configs/probe.yaml", ("training.log_interval=5",))],
                  repo=repo_root, work_dir=tmp_path)
    (tmp_path / "experiments" / "e4-lr-adamw-0.0006-s1337").mkdir(parents=True)
    with pytest.raises(LaunchError, match="already exist"):
        plan_runs([Job(SWEEP[0])], repo=repo_root, work_dir=tmp_path)


def test_plan_refuses_a_missing_config_or_no_jobs(repo_root, tmp_path):
    with pytest.raises(LaunchError, match="no config"):
        plan_runs([Job("configs/experiments/e4_lr_adamw_0.1.yaml")], repo=repo_root, work_dir=tmp_path)
    with pytest.raises(LaunchError, match="no jobs"):
        plan_runs([], repo=repo_root, work_dir=tmp_path)


def test_train_command_puts_overrides_last(tmp_path):
    cmd = train_command(Job("configs/probe.yaml", ("training.seed=7", "training.max_steps=3")),
                        repo=tmp_path, data="/data", venue="kaggle-free", experiment="p1", python="py")
    assert cmd[:3] == ["py", "-u", str(tmp_path / "scripts" / "train.py")]
    assert cmd[cmd.index("--config") + 1] == str(tmp_path / "configs" / "probe.yaml")
    assert cmd[-3:] == ["--set", "training.seed=7", "training.max_steps=3"]
    assert "--set" not in train_command(Job("c.yaml"), repo=tmp_path, data="/d", venue="v", experiment="e")


def test_train_command_passes_every_data_directory_in_order(tmp_path):
    cmd = train_command(Job("configs/experiments/p4.yaml"), repo=tmp_path,
                        data=["/in/general", "/in/personal"], venue="kaggle-free", experiment="pretrain")
    at = cmd.index("--data")
    assert cmd[at + 1:at + 3] == ["/in/general", "/in/personal"]
    assert cmd[at + 3] == "--venue"
    with pytest.raises(LaunchError, match="no data directory"):
        train_command(Job("c.yaml"), repo=tmp_path, data=[], venue="v", experiment="e")


def test_device_ids_respect_an_existing_mask(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert device_ids(2) == ["0", "1"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5, 7")
    assert device_ids(2) == ["5", "7"]
    with pytest.raises(LaunchError, match="allows 2"):
        device_ids(3)
    with pytest.raises(LaunchError, match="at least 1"):
        device_ids(0)


# ------------------------------------------------------------ the queue

# Stands in for scripts/train.py: records what it was given, then exits with
# the code named in its config (``exit: 3``), after ``sleep`` seconds.
FAKE_TRAIN = textwrap.dedent("""
    import json, os, sys, time
    from pathlib import Path
    args = sys.argv[1:]
    config = Path(args[args.index("--config") + 1])
    spec = dict(line.lstrip("# ").split(": ", 1) for line in config.read_text().splitlines())
    record = {"args": args, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "cwd": os.getcwd(),
              "start": time.time()}
    print("fake train", spec["name"], flush=True)
    time.sleep(float(spec.get("sleep", 0.2)))
    record["end"] = time.time()
    Path("records").mkdir(exist_ok=True)
    Path("records", spec["name"] + ".json").write_text(json.dumps(record))
    code = int(spec.get("exit", 0))
    if code:
        print(f"RuntimeError: loss became nan at step 3", flush=True)
    sys.exit(code)
""")


@pytest.fixture
def fake_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "configs").mkdir()
    (repo / "scripts" / "train.py").write_text(FAKE_TRAIN)

    def add(name: str, **extra) -> Job:
        lines = [f"name: {name}"] + [f"# {k}: {v}" for k, v in extra.items()]
        # The fake reads "# key: value" comments; load_config ignores them.
        (repo / "configs" / f"{name}.yaml").write_text("\n".join(lines) + "\n")
        return Job(f"configs/{name}.yaml")

    return repo, add


@pytest.fixture
def fake_data(tmp_path):
    """A shard directory holding the fake configs' sources (empty ``.bin`` files: only names are checked)."""
    d = tmp_path / "shards"
    d.mkdir()
    for name in DataConfig().mixture:
        (d / f"{name}.train.bin").write_bytes(b"")
    return d


def _records(work_dir: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in (work_dir / "records").glob("*.json")}


def test_queue_runs_every_job_one_per_gpu(fake_repo, fake_data, tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    repo, add = fake_repo
    jobs = [add(f"job{i}", sleep=0.3) for i in range(5)]
    work = tmp_path / "work"
    work.mkdir()
    echoed: list[str] = []

    results = run_queue(jobs, repo=repo, data=fake_data, venue="kaggle-free", experiment="e4-lr",
                        gpus=2, work_dir=work, poll_seconds=0.02, progress_seconds=0.0,
                        echo=echoed.append)

    assert [r.run_name for r in results] == [f"job{i}-s1337" for i in range(5)]
    assert all(r.ok for r in results)
    records = _records(work)
    assert len(records) == 5
    assert {rec["gpu"] for rec in records.values()} == {"0", "1"}
    assert all(Path(rec["cwd"]).resolve() == work.resolve() for rec in records.values())
    # A GPU starts its next job only after its last one has exited.
    for gpu in ("0", "1"):
        spans = sorted((rec["start"], rec["end"]) for rec in records.values() if rec["gpu"] == gpu)
        assert all(prev[1] <= nxt[0] for prev, nxt in zip(spans, spans[1:]))
    args = records["job3"]["args"]
    assert args[args.index("--venue") + 1] == "kaggle-free"
    assert args[args.index("--experiment") + 1] == "e4-lr"

    log = (work / "logs" / "job3-s1337.log").read_text()
    assert log.startswith("$ ") and "fake train job3" in log
    assert any("start job0-s1337" in line for line in echoed)
    assert echoed[-1] == "5/5 finished"


def test_a_failed_job_does_not_stop_the_queue(fake_repo, fake_data, tmp_path):
    repo, add = fake_repo
    jobs = [add("ok1"), add("diverges", exit=3), add("ok2")]
    echoed: list[str] = []
    results = run_queue(jobs, repo=repo, data=fake_data, venue="local", experiment="e",
                        gpus=1, work_dir=tmp_path, poll_seconds=0.02, echo=echoed.append)
    assert [r.returncode for r in results] == [0, 3, 0]
    assert [r.ok for r in results] == [True, False, True]
    assert any("FAILED (exit 3): RuntimeError: loss became nan" in line for line in echoed)
    assert echoed[-1] == "2/3 finished; failed: ['diverges-s1337']"


def test_queue_refuses_before_any_job_starts(fake_repo, fake_data, tmp_path):
    repo, add = fake_repo
    job = add("a")
    with pytest.raises(LaunchError, match="unknown venue"):
        run_queue([job], repo=repo, data=fake_data, venue="kaggle-paid", experiment="e",
                  gpus=1, work_dir=tmp_path)
    with pytest.raises(LaunchError, match="share a run directory"):
        run_queue([job, job], repo=repo, data=fake_data, venue="local", experiment="e",
                  gpus=1, work_dir=tmp_path)
    with pytest.raises(LaunchError, match="no shard for source"):
        run_queue([job], repo=repo, data=tmp_path / "elsewhere", venue="local", experiment="e",
                  gpus=1, work_dir=tmp_path)
    assert not (tmp_path / "records").exists() and not (tmp_path / "logs").exists()


def test_an_interrupt_stops_running_jobs(fake_repo, fake_data, tmp_path):
    repo, add = fake_repo
    job = add("slow", sleep=60)

    def interrupt(line: str) -> None:
        if "start" in line:
            raise KeyboardInterrupt

    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        run_queue([job], repo=repo, data=fake_data, venue="local", experiment="e",
                  gpus=1, work_dir=tmp_path, poll_seconds=0.02, echo=interrupt)
    assert time.monotonic() - started < 30
    assert not (tmp_path / "records").exists(), "the job ran to completion"


# ------------------------------------------------------------ check_sources


def test_check_sources_wants_each_source_in_exactly_one_directory(repo_root, tmp_path):
    general, personal = tmp_path / "general", tmp_path / "personal"
    general.mkdir(), personal.mkdir()
    for name in ("web", "dialogue", "code"):
        (general / f"{name}.train.bin").write_bytes(b"")
    p4 = Job("configs/experiments/p4.yaml")
    check_sources([Job("configs/experiments/p1.yaml")], repo=repo_root, data=general)
    with pytest.raises(LaunchError, match="no shard for source 'chat'"):
        check_sources([p4], repo=repo_root, data=general)
    (personal / "chat.train.bin").write_bytes(b"")
    check_sources([p4], repo=repo_root, data=[general, personal])
    (personal / "web.train.bin").write_bytes(b"")
    with pytest.raises(LaunchError, match="more than one data directory"):
        check_sources([p4], repo=repo_root, data=[general, personal])


# ------------------------------------------------------------ stage_upload


@pytest.fixture
def personal_dir(tmp_path):
    """The personal shard directory's layout: train shards per source, val and test merged."""
    d = tmp_path / "shards" / "personal"
    d.mkdir(parents=True)
    entries = {}
    for name in ("chat.train", "comment.train", "val", "test"):
        (d / f"{name}.bin").write_bytes(name.encode() * 50)
        (d / f"{name}.idx.npy").write_bytes(b"\x93NUMPY " + name.encode())
        (d / f"{name}.meta.json").write_text(json.dumps({"name": name, "tokenizer": FROZEN_ID}))
        entries[name] = {"name": name, "n_tokens": 100, "tokenizer": FROZEN_ID}
    (d / "._chat.train.bin").write_bytes(b"macOS metadata")
    (d / "manifest.json").write_text(json.dumps({
        "tokenizer": "minime-16k", "tokenizer_id": FROZEN_ID, "tokenizer_dev": False,
        "vocab_size": 16384, "eot_id": 0, "sources": ["/private/corpus.jsonl"],
        "splits": {"train": {"documents": 3}}, "shards": entries,
    }))
    return d


def test_stage_upload_copies_the_named_shard_and_nothing_else(personal_dir, tmp_path):
    out = stage_upload(personal_dir, tmp_path / "upload", shards=["chat.train"], kind="personal")
    assert sorted(p.name for p in out.iterdir()) == [
        CHECKSUMS, "chat.train.bin", "chat.train.idx.npy", "chat.train.meta.json", "manifest.json",
    ]
    for name in ("chat.train.bin", "chat.train.idx.npy", "chat.train.meta.json"):
        assert (out / name).read_bytes() == (personal_dir / name).read_bytes()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["kind"] == "personal"
    assert manifest["tokenizer_id"] == FROZEN_ID and manifest["vocab_size"] == 16384
    assert list(manifest["shards"]) == ["chat.train"]
    # The source manifest's corpus path and split counts stay on the laptop.
    assert "sources" not in manifest and "splits" not in manifest
    assert "/private/" not in (out / "manifest.json").read_text()
    assert verify_checksums(out) == 4
    assert find_shard_dir(tmp_path, "personal", FROZEN_ID) == out


def test_stage_upload_refuses_an_existing_directory_and_an_unknown_shard(personal_dir, tmp_path):
    out = tmp_path / "upload"
    out.mkdir()
    with pytest.raises(LaunchError, match="exists"):
        stage_upload(personal_dir, out, shards=["chat.train"], kind="personal")
    out.rmdir()
    with pytest.raises(LaunchError, match="no shard"):
        stage_upload(personal_dir, out, shards=["longform.train"], kind="personal")
    (personal_dir / "chat.train.meta.json").unlink()
    with pytest.raises(LaunchError, match="no chat.train.meta.json"):
        stage_upload(personal_dir, out, shards=["chat.train"], kind="personal")
    assert not out.exists()


def test_stage_script_writes_then_the_launcher_verifies(repo_root, personal_dir, tmp_path):
    out = tmp_path / "upload" / "personal-chat-train"
    script = repo_root / "scripts" / "stage_personal_upload.py"
    done = subprocess.run([sys.executable, str(script), "--source", str(personal_dir), "--out", str(out)],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "private Kaggle dataset" in done.stdout
    assert verify_checksums(out) == 4
    again = subprocess.run([sys.executable, str(script), "--source", str(personal_dir), "--out", str(out)],
                           capture_output=True, text=True)
    assert again.returncode == 1 and "exists" in again.stderr


# ------------------------------------------------- resuming across sessions (M10)


def test_train_command_resumes_and_stops_by_a_deadline(tmp_path):
    deadline = datetime(2026, 10, 20, 9, 30, tzinfo=timezone.utc)
    cmd = train_command(Job("configs/flagship.yaml", ("training.seed=1",), resume=True), repo=tmp_path,
                        data=tmp_path / "d", venue="kaggle-free", experiment="flagship",
                        resume=tmp_path / "experiments" / "flagship-s1" / "last.pt", stop_by=deadline)
    assert cmd[cmd.index("--resume") + 1].endswith("experiments/flagship-s1/last.pt")
    assert cmd[cmd.index("--stop-by") + 1] == "2026-10-20T09:30:00+00:00"
    assert cmd[-2:] == ["--set", "training.seed=1"], "--set takes every argument after it"


def _fake_run(root: Path, name: str, config: dict, *, end: str = "paused", revision: str = "abc1234",
              accelerator: str = "Tesla T4") -> Path:
    """A run directory as a session leaves it in its output, hashed."""
    run = root / "experiments" / name
    run.mkdir(parents=True)
    manifest = {"run": name, "git_revision": revision, "config": config,
                "device": {"accelerator": accelerator, "precision": "fp16"}}
    if end == "paused":
        manifest["paused"] = {"step": 1800, "at": "2026-10-20T09:30:00+00:00", "reason": "stop_by"}
    elif end == "finished":
        manifest["summary"] = {"steps": 3450}
    (run / "manifest.json").write_text(json.dumps(manifest))
    (run / "metrics.jsonl").write_text('{"step": 1800, "phase": "paused"}\n')
    for ckpt in ("last.pt", "best.pt", "step_1500.pt"):
        (run / ckpt).write_bytes(ckpt.encode() * 100)
    write_checksums(run)
    return run


@pytest.fixture
def paused(fake_repo, tmp_path, monkeypatch):
    """A fake flagship job and its paused run in an earlier session's output, at this commit."""
    repo, add = fake_repo
    job = Job(add("flagship").config, resume=True)
    config = load_config(repo / job.config).to_dict()
    earlier = tmp_path / "input" / "minime-kaggle-train"
    _fake_run(earlier, "flagship-s1337", config)
    monkeypatch.setattr(launch, "git_revision", lambda repo=None: "abc1234")
    return repo, job, config, earlier


def test_a_resumed_job_continues_the_paused_run(paused, fake_data, tmp_path):
    repo, job, _, earlier = paused
    work = tmp_path / "work"
    deadline = datetime.now(timezone.utc) + timedelta(hours=11)
    echoed: list[str] = []
    results = run_queue([job], repo=repo, data=fake_data, venue="kaggle-free", experiment="flagship", gpus=1,
                        work_dir=work, resume_root=tmp_path / "input", stop_by=deadline,
                        accelerator="Tesla T4", poll_seconds=0.02, echo=echoed.append)
    assert results[0].ok
    run = work / "experiments" / "flagship-s1337"
    assert sorted(p.name for p in run.iterdir()) == ["best.pt", "last.pt", "manifest.json", "metrics.jsonl"]
    for name in ("manifest.json", "metrics.jsonl", "last.pt", "best.pt"):
        assert (run / name).read_bytes() == (earlier / "experiments" / "flagship-s1337" / name).read_bytes()
    args = _records(work)["flagship"]["args"]
    assert args[args.index("--resume") + 1] == str(run / "last.pt")
    assert args[args.index("--stop-by") + 1] == deadline.isoformat(timespec="seconds")
    assert "[gpu 0] resume flagship-s1337" in echoed


def _damage_manifest(change):
    def damage(run: Path, config: dict) -> None:
        manifest = json.loads((run / "manifest.json").read_text())
        change(manifest, config)
        (run / "manifest.json").write_text(json.dumps(manifest))
        write_checksums(run)
    return damage


@pytest.mark.parametrize("damage, message", [
    (_damage_manifest(lambda m, c: (m.pop("paused"), m.update(summary={"steps": 3450}))),
     "is finished at step 3450: nothing to resume"),
    (_damage_manifest(lambda m, c: m.pop("paused")), "neither finished nor paused"),
    (_damage_manifest(lambda m, c: m.update(config={**c, "training": {**c["training"], "max_steps": 4000}})),
     "another config"),
    (_damage_manifest(lambda m, c: m.update(git_revision="eab19cb")), "trained at commit eab19cb"),
    (_damage_manifest(lambda m, c: m.update(device={"accelerator": "Tesla P100-PCIE-16GB"})), "ran on"),
    (lambda run, config: (run / "last.pt").write_bytes(b"truncated"), "changed: last.pt"),
    (lambda run, config: (run / "metrics.jsonl").unlink(), "missing: metrics.jsonl"),
    (lambda run, config: shutil.copytree(run, run.parents[2] / "again" / "experiments" / run.name),
     "expected one run directory"),
])
def test_a_resume_is_refused_before_any_gpu_time(paused, fake_data, tmp_path, damage, message):
    repo, job, config, earlier = paused
    damage(earlier / "experiments" / "flagship-s1337", config)
    work = tmp_path / "work"
    with pytest.raises(LaunchError, match=message):
        run_queue([job], repo=repo, data=fake_data, venue="kaggle-free", experiment="flagship", gpus=1,
                  work_dir=work, resume_root=tmp_path / "input", accelerator="Tesla T4")
    assert not work.exists(), "nothing copied, nothing started"


def test_a_resume_needs_the_earlier_output_and_a_clean_checkout(paused, fake_data, tmp_path, monkeypatch):
    repo, job, _, _ = paused
    kwargs = dict(repo=repo, data=fake_data, venue="kaggle-free", experiment="flagship", gpus=1,
                  work_dir=tmp_path / "work")
    with pytest.raises(LaunchError, match="needs resume_root"):
        run_queue([job], **kwargs)
    monkeypatch.setattr(launch, "git_revision", lambda repo=None: "abc1234-dirty")
    with pytest.raises(LaunchError, match="this checkout is abc1234-dirty"):
        run_queue([job], resume_root=tmp_path / "input", **kwargs)


def test_a_paused_job_is_reported_as_paused_not_failed(fake_repo, fake_data, tmp_path):
    repo, add = fake_repo
    jobs = [add("long", exit=PAUSED_EXIT), add("diverges", exit=1), add("short")]
    echoed: list[str] = []
    results = run_queue(jobs, repo=repo, data=fake_data, venue="local", experiment="e", gpus=1,
                        work_dir=tmp_path, poll_seconds=0.02, echo=echoed.append)
    assert [(r.ok, r.paused) for r in results] == [(False, True), (False, False), (True, False)]
    assert "[gpu 0] paused (resumable) long-s1337" in " ".join(echoed)
    assert echoed[-1] == "1/3 finished; paused: ['long-s1337']; failed: ['diverges-s1337']"


# ---------------------------------------------- checkpoints going home (M10)


def _checkpoint_run(tmp_path, *, step: int = 431, end: str = "finished") -> tuple[Path, dict]:
    config = {"name": "m10-lr-0.0016", "training": {"seed": 1337, "max_steps": 431}, "data": {"seed": 1234}}
    run = tmp_path / "experiments" / "m10-lr-0.0016-s1337"
    run.mkdir(parents=True)
    model = torch.nn.Linear(2, 2)
    save_checkpoint(run / "last.pt", model=model, step=step, config=config)
    save_checkpoint(run / "step_200.pt", model=model, step=200, config=config)
    manifest = {"run": run.name, "config": config}
    manifest.update({"summary": {"steps": 431}} if end == "finished" else {"paused": {"step": step}})
    (run / "manifest.json").write_text(json.dumps(manifest))
    (run / "metrics.jsonl").write_text("")
    return run, config


def test_run_checksums_hash_every_run_directory(tmp_path):
    run, _ = _checkpoint_run(tmp_path)
    (tmp_path / "experiments" / "costs.csv").write_text("date\n")
    (tmp_path / "experiments" / "not-a-run").mkdir()
    assert write_run_checksums(tmp_path) == [run / CHECKSUMS]
    listed = [line.split("  ")[1] for line in (run / CHECKSUMS).read_text().splitlines()]
    assert listed == ["last.pt", "manifest.json", "metrics.jsonl", "step_200.pt"]
    assert write_run_checksums(tmp_path / "elsewhere") == []


def test_verify_download_accepts_the_file_the_session_wrote(tmp_path):
    run, _ = _checkpoint_run(tmp_path)
    write_run_checksums(tmp_path)
    home = tmp_path / "ssd"
    home.mkdir()
    # The browser saves last.pt as last.zip: the same bytes under another name.
    (home / "last.zip").write_bytes((run / "last.pt").read_bytes())
    report = verify_download(home / "last.zip", run, name="last.pt")
    assert report["step"] == 431 and report["seeds"] == [1337, 1234] and report["run_status"] == "finished"
    assert verify_download(run / "step_200.pt", run)["step"] == 200


def test_verify_download_refuses_what_the_session_did_not_write(tmp_path):
    run, config = _checkpoint_run(tmp_path)
    write_run_checksums(tmp_path)
    with pytest.raises(LaunchError, match="lists no last.zip"):
        verify_download(run / "last.pt", run, name="last.zip")
    damaged = tmp_path / "last.pt"
    damaged.write_bytes((run / "last.pt").read_bytes()[:-1] + b"x")
    with pytest.raises(LaunchError, match="sha256 differs"):
        verify_download(damaged, run, name="last.pt")

    # Hashed and intact, but not this run's last checkpoint.
    other, _ = _checkpoint_run(tmp_path / "other", step=300)
    write_run_checksums(tmp_path / "other")
    with pytest.raises(LaunchError, match="step 300, but the run's last.pt is at step 431"):
        verify_download(other / "last.pt", other)
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["config"]["training"]["seed"] = 1338
    (run / "manifest.json").write_text(json.dumps(manifest))
    write_run_checksums(tmp_path)
    with pytest.raises(LaunchError, match="its config is not"):
        verify_download(run / "last.pt", run)


def test_verify_download_reads_a_paused_runs_last_step(tmp_path):
    run, _ = _checkpoint_run(tmp_path, step=1800, end="paused")
    write_run_checksums(tmp_path)
    assert verify_download(run / "last.pt", run)["run_status"] == "paused"


def test_verify_download_script(repo_root, tmp_path):
    run, _ = _checkpoint_run(tmp_path)
    write_run_checksums(tmp_path)
    script = repo_root / "scripts" / "verify_download.py"
    done = subprocess.run([sys.executable, str(script), str(run / "last.pt"), "--run-dir", str(run)],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "verified" in done.stdout
    (run / "step_200.pt").write_bytes(b"x")
    done = subprocess.run([sys.executable, str(script), str(run / "step_200.pt"), "--run-dir", str(run)],
                          capture_output=True, text=True)
    assert done.returncode == 1 and "sha256 differs" in done.stderr


# ------------------------------------- the whole path, with the real train.py (M10)


def _shard(path: Path, seed: int, n_tokens: int = 4000) -> None:
    rng = np.random.default_rng(seed)
    with ShardWriter(path, vocab_size=64) as writer:
        for _ in range(n_tokens // 200):
            writer.add_document(rng.integers(0, 64, size=200))


@pytest.mark.slow
def test_a_run_paused_in_one_session_finishes_in_the_next(repo_root, tmp_path, monkeypatch):
    """Two sessions of run_queue around the real scripts/train.py, on CPU: the
    first pauses at its deadline (exit 75), hashes its output; the second
    finds the run in that output, copies it in and finishes it as one run."""
    data = tmp_path / "shards"
    data.mkdir()
    for i, name in enumerate(("web.train", "dialogue.train", "code.train", "val")):
        _shard(data / name, seed=i)
    job = Job("configs/tiny.yaml", ("training.max_steps=3", "training.warmup_steps=0", "training.batch_size=2",
                                    "training.validation_batches=2", "training.device=cpu",
                                    "training.precision=fp32", "model.vocab_size=64"))
    kwargs = dict(repo=repo_root, data=data, venue="kaggle-free", experiment="flagship", gpus=1,
                  poll_seconds=0.05, echo=lambda line: None)

    first = tmp_path / "session1"
    results = run_queue([job], work_dir=first, stop_by=datetime.now(timezone.utc) - timedelta(seconds=1), **kwargs)
    assert results[0].paused, (first / "logs" / "tiny-s1337.log").read_text()[-2000:]
    run = first / "experiments" / "tiny-s1337"
    # This test's tree may be uncommitted: stand in for a session at a clean commit.
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["git_revision"] = "abc1234"
    (run / "manifest.json").write_text(json.dumps(manifest))
    write_run_checksums(first)
    monkeypatch.setattr(launch, "git_revision", lambda repo=None: "abc1234")

    second = tmp_path / "session2"
    resumed = Job(job.config, job.overrides, resume=True)
    results = run_queue([resumed], work_dir=second, resume_root=first, **kwargs)
    assert results[0].ok, (second / "logs" / "tiny-s1337.log").read_text()[-2000:]
    run = second / "experiments" / "tiny-s1337"
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["summary"]["steps"] == 3
    assert [(s["start_step"], s.get("end_step"), s["outcome"]) for s in manifest["sessions"]] == [
        (0, 1, "paused"), (1, 3, "finished")]
    phases = [json.loads(line)["phase"] for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert phases.count("init") == 1 and phases.index("paused") < phases.index("resumed")
    write_run_checksums(second)
    assert verify_download(run / "last.pt", run)["step"] == 3
