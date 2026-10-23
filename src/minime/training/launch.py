"""Remote-run plumbing: find and verify the shards, then keep every GPU busy.

Kaggle's free "GPU T4 x2" machine has two GPUs and the trainer uses one, so
running the jobs one after another leaves half of the weekly quota idle.
``run_queue`` runs each job as a ``scripts/train.py`` subprocess pinned to one
device with ``CUDA_VISIBLE_DEVICES``, each with a log file of its own, and
starts the next job on whichever GPU frees first.

A failed job does not stop the queue: a diverged sweep point is a result, and
the next job may be fine.  Results come back in job order with their exit
codes, and ``pick_lr.py`` refuses unfinished runs, so nothing is lost
silently.  What *can* be checked before a GPU is used is checked first --
configs that do not load, two jobs that would write one run directory, an
unknown venue, shards that are not the ones hashed on the laptop.

A run longer than a session (the flagship) stops cleanly before Kaggle's 12 h
-- ``train.py --stop-by``, exit status ``PAUSED_EXIT`` -- and a later session
resumes it: ``run_queue`` finds the paused run in the earlier session's
output, checks it is this job at this commit and that its files are the ones
that session hashed, copies it in and runs ``train.py --resume``.  Every
session hashes its run directories (``write_run_checksums``), so a download
can be checked against the source (``verify_download``).

The logic lives here, under test, so ``notebooks/kaggle_train.ipynb`` stays a
launcher.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

from ..config import load_config, parse_cli_overrides
from ..data.binidx import shard_stem
from ..utils.costs import hourly_rate
from ..utils.throughput import format_duration
from .inputs import find_shard
from .logging import git_revision

CHECKSUMS = "SHA256SUMS"
# scripts/train.py's exit status for a run that stopped cleanly before its
# budget (--stop-at-step / --stop-by): resumable, not failed.  sysexits'
# EX_TEMPFAIL, "try again later"; a crash exits 1.
PAUSED_EXIT = 75
# What a resumed session needs from the paused one: the checkpoint it
# continues, the best one so far, and the manifest and metrics it extends.
# The rotating step_*.pt stay in the earlier session's output.
RESUME_FILES = ("manifest.json", "metrics.jsonl", "last.pt", "best.pt")


class LaunchError(RuntimeError):
    """The jobs or the data are not fit to spend GPU time on."""


# ------------------------------------------------------------------ data


def _is_hashed(path: Path) -> bool:
    # ``._name`` files are macOS metadata that exFAT grows beside real files.
    return path.is_file() and path.name != CHECKSUMS and not path.name.startswith("._")


def _sha256(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def write_checksums(data_dir: str | Path) -> Path:
    """Hash every file in a shard directory into ``SHA256SUMS`` (``sha256sum`` format)."""
    data_dir = Path(data_dir)
    files = sorted(p for p in data_dir.iterdir() if _is_hashed(p))
    if not files:
        raise LaunchError(f"{data_dir} has no files to hash")
    lines = [f"{_sha256(p)}  {p.name}" for p in files]
    out = data_dir / CHECKSUMS
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _read_checksums(data_dir: Path) -> dict[str, str]:
    sums_path = data_dir / CHECKSUMS
    if not sums_path.exists():
        raise LaunchError(
            f"{sums_path} is missing; run scripts/checksum_shards.py on the shards before uploading"
        )
    expected = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, _, name = line.partition("  ")
            expected[name] = digest
    return expected


def verify_checksums(data_dir: str | Path) -> int:
    """Check a shard directory is exactly what ``write_checksums`` hashed.

    A missing file, a changed byte and an unlisted file are all refusals: an
    upload that dropped or added a shard is not the data the laptop evaluates
    against.  Returns the number of files verified.
    """
    data_dir = Path(data_dir)
    expected = _read_checksums(data_dir)
    present = {p.name for p in data_dir.iterdir() if _is_hashed(p)}

    problems = [f"missing: {name}" for name in sorted(set(expected) - present)]
    problems += [f"not in {CHECKSUMS}: {name}" for name in sorted(present - set(expected))]
    problems += [
        f"changed: {name}"
        for name in sorted(set(expected) & present)
        if _sha256(data_dir / name) != expected[name]
    ]
    if problems:
        raise LaunchError(f"{data_dir} does not match {CHECKSUMS}:\n  " + "\n  ".join(problems))
    return len(expected)


def stage_upload(source: str | Path, out: str | Path, *, shards: Sequence[str], kind: str) -> Path:
    """Copy ``shards`` alone from a shard directory into a new one, fit to upload.

    An upload carries what its jobs read and nothing more.  P4 needs the
    owner's chat train split, so the personal val and test splits, the comment
    shard and the source manifest -- its corpus path and split counts -- stay
    on the laptop.  The new manifest keeps the tokenizer, the vocabulary and
    each staged shard's entry, and names its ``kind`` for ``find_shard_dir``.
    Every copy is checked against its source before ``SHA256SUMS`` is written.
    """
    source, out = Path(source), Path(out)
    if out.exists():
        raise LaunchError(f"{out} exists: staging writes a new directory, never into an old one")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    missing = [s for s in shards if s not in manifest.get("shards", {})]
    if missing:
        raise LaunchError(f"{source}/manifest.json lists no shard {missing}")
    files = sorted(p for p in source.iterdir() if _is_hashed(p) and str(shard_stem(p.name)) in shards)
    for name in shards:
        for suffix in (".bin", ".meta.json"):
            if not (source / f"{name}{suffix}").is_file():
                raise LaunchError(f"{source} has no {name}{suffix}")

    out.mkdir(parents=True)
    try:
        for path in files:
            copy = out / path.name
            copy.write_bytes(path.read_bytes())
            if _sha256(copy) != _sha256(path):
                raise LaunchError(f"{copy} differs from {path} after copying")
        staged = {
            "kind": kind,
            "tokenizer": manifest.get("tokenizer"),
            "tokenizer_id": manifest.get("tokenizer_id"),
            "tokenizer_dev": manifest.get("tokenizer_dev", False),
            "vocab_size": manifest.get("vocab_size"),
            "eot_id": manifest.get("eot_id"),
            "shards": {name: manifest["shards"][name] for name in shards},
            "staged_from": {"manifest_sha256": _sha256(source / "manifest.json")},
        }
        (out / "manifest.json").write_text(json.dumps(staged, indent=2) + "\n", encoding="utf-8")
        write_checksums(out)
    except BaseException:
        for path in out.iterdir():
            path.unlink()
        out.rmdir()
        raise
    return out


def find_shard_dir(root: str | Path, kind: str, tokenizer_id: str) -> Path:
    """The one directory under ``root`` holding a ``kind`` corpus made by ``tokenizer_id``.

    A Kaggle dataset lands at ``/kaggle/input/<slug>/`` or one folder deeper,
    depending on how it was uploaded; searching for the manifest avoids
    guessing.
    """
    root = Path(root)
    found = []
    for manifest_path in sorted(root.rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("kind") == kind:
            found.append((manifest_path.parent, manifest.get("tokenizer_id")))
    if len(found) != 1:
        where = ", ".join(str(d) for d, _ in found) or "none"
        raise LaunchError(f"expected one '{kind}' shard directory under {root}, found {len(found)}: {where}")
    data_dir, found_id = found[0]
    if found_id != tokenizer_id:
        raise LaunchError(f"{data_dir} was tokenised with {found_id!r}, not {tokenizer_id!r}")
    return data_dir


# ---------------------------------------------------- checkpoints going home


def write_run_checksums(work_dir: str | Path) -> list[Path]:
    """``SHA256SUMS`` in every run directory under ``work_dir/experiments``.

    Written on Kaggle at the end of a session, beside the checkpoints, so each
    downloaded file can be checked against the bytes the session wrote.  The
    file travels in ``results.tar.gz`` (which leaves the ``.pt`` files out).
    """
    root = Path(work_dir) / "experiments"
    if not root.is_dir():
        return []
    return [write_checksums(d) for d in sorted(root.iterdir()) if (d / "manifest.json").is_file()]


def _run_end(manifest: dict) -> tuple[str, int]:
    """``("finished" | "paused", step)`` for a run directory's manifest."""
    if "summary" in manifest:
        return "finished", manifest["summary"]["steps"]
    if "paused" in manifest:
        return "paused", manifest["paused"]["step"]
    raise LaunchError(f"run {manifest.get('run')!r} neither finished nor paused (killed mid-session?)")


def _config_fingerprint(config: dict) -> str:
    # As noise_floor.config_fingerprint and the eval harness compute it.
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:12]


def verify_download(checkpoint: str | Path, run_dir: str | Path, name: str | None = None) -> dict:
    """Check a downloaded checkpoint against the session that wrote it.

    ``run_dir`` is the run directory from ``results.tar.gz``: its
    ``SHA256SUMS`` and ``manifest.json``.  ``name`` is the file's name there,
    when the download was renamed (a browser saves ``last.pt`` as
    ``last.zip``).  Verified = the sha256 matches the source's, the file
    loads, its config is the manifest's (seeds included), and its step is the
    run's last (``last.pt``) or the one in its name (``step_<N>.pt``).
    """
    import torch  # here, not at the top: the notebook's kernel never needs it

    checkpoint, run_dir = Path(checkpoint), Path(run_dir)
    name = name or checkpoint.name
    expected = _read_checksums(run_dir)
    if name not in expected:
        raise LaunchError(f"{run_dir / CHECKSUMS} lists no {name}")
    if _sha256(checkpoint) != expected[name]:
        raise LaunchError(f"{checkpoint} is not the {name} the session wrote: sha256 differs")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("config") or {}
    if _config_fingerprint(config) != _config_fingerprint(manifest["config"]):
        raise LaunchError(f"{checkpoint}: its config is not {run_dir.name}'s")
    seeds = (config["training"]["seed"], config["data"]["seed"])
    if seeds != (manifest["config"]["training"]["seed"], manifest["config"]["data"]["seed"]):
        raise LaunchError(f"{checkpoint}: seeds {seeds} are not the run's")
    status, end = _run_end(manifest)
    step = payload.get("step")
    wanted = end if name == "last.pt" else int(name[5:-3]) if name.startswith("step_") else None
    if (wanted is not None and step != wanted) or step > end:
        raise LaunchError(f"{checkpoint}: step {step}, but the run's {name} is at step {wanted or end}")
    return {"file": str(checkpoint), "name": name, "sha256": expected[name], "step": step,
            "run": run_dir.name, "run_status": status, "config_fingerprint": _config_fingerprint(config),
            "seeds": list(seeds)}


# ------------------------------------------------------------------ jobs


@dataclass(frozen=True)
class Job:
    config: str                        # relative to the repository root
    overrides: tuple[str, ...] = ()    # ``--set`` arguments, e.g. "training.seed=1338"
    # Continue this run from where an earlier session paused it (run_queue's
    # ``resume_root`` holds that session's output), rather than start it.
    resume: bool = False


@dataclass(frozen=True)
class JobResult:
    job: Job
    run_name: str
    gpu: str
    returncode: int
    seconds: float
    log: Path

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def paused(self) -> bool:
        return self.returncode == PAUSED_EXIT


def plan_runs(jobs: Sequence[Job], *, repo: str | Path, work_dir: str | Path) -> list[str]:
    """Each job's run name, refusing jobs that would share or reuse a run directory.

    The run logger appends to ``metrics.jsonl``, so two jobs writing one run
    directory -- or one job writing into a finished run's -- would interleave
    two curves without an error.
    """
    repo, work_dir = Path(repo), Path(work_dir)
    if not jobs:
        raise LaunchError("no jobs")
    names = []
    for job in jobs:
        path = repo / job.config
        if not path.exists():
            raise LaunchError(f"no config {path}")
        cfg = load_config(path, overrides=parse_cli_overrides(list(job.overrides)))
        names.append(cfg.run_name)

    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise LaunchError(f"jobs share a run directory: {repeated}")
    existing = [n for n in names if (work_dir / "experiments" / n).exists()]
    if existing:
        raise LaunchError(f"run directories already exist under {work_dir / 'experiments'}: {existing}")
    return names


def data_dirs(data: str | Path | Sequence[str | Path]) -> list[Path]:
    """One shard directory or several, as a list; the first holds the val set."""
    dirs = [Path(data)] if isinstance(data, (str, Path)) else [Path(d) for d in data]
    if not dirs:
        raise LaunchError("no data directory")
    return dirs


def check_sources(jobs: Sequence[Job], *, repo: str | Path, data: str | Path | Sequence[str | Path]) -> None:
    """Refuse a job whose mixture names a source the data directories do not hold once.

    ``train.py`` would find out too, but only after the queue has started --
    a session that forgot the personal dataset should fail before any GPU time.
    """
    dirs = data_dirs(data)
    problems = []
    for job in jobs:
        cfg = load_config(Path(repo) / job.config, overrides=parse_cli_overrides(list(job.overrides)))
        for name in cfg.data.mixture:
            try:
                find_shard(dirs, name)
            except (FileNotFoundError, ValueError) as exc:
                problems.append(f"{job.config}: {exc}")
    if problems:
        raise LaunchError("the data does not hold the jobs' sources:\n  " + "\n  ".join(problems))


def find_paused_run(root: str | Path, run_name: str) -> Path:
    """The one run directory named ``run_name`` under ``root`` (an earlier session's attached output)."""
    found = sorted(p.parent for p in Path(root).rglob("manifest.json") if p.parent.name == run_name)
    if len(found) != 1:
        where = ", ".join(map(str, found)) or "none"
        raise LaunchError(f"expected one run directory {run_name!r} under {root}, found {len(found)}: {where}")
    return found[0]


def check_resume(job: Job, run_name: str, *, repo: str | Path, resume_root: str | Path,
                 accelerator: str | None = None) -> Path:
    """The paused run ``job`` continues, refused unless it is this job at this commit.

    Checks, before any GPU time: the run paused cleanly (not finished, not
    killed); its config is the job's, field for field; its last session ran
    this checkout's commit, with a clean tree, on ``accelerator`` (the same
    GPU type, so the same precision); and its files are the ones that session
    hashed.  ``train.py --resume`` then checks the checkpoint's own config and
    precision again.
    """
    repo = Path(repo)
    source = find_paused_run(resume_root, run_name)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    status, step = _run_end(manifest)
    if status != "paused":
        raise LaunchError(f"{source} is {status} at step {step}: nothing to resume")
    cfg = load_config(repo / job.config, overrides=parse_cli_overrides(list(job.overrides)))
    if _config_fingerprint(manifest["config"]) != _config_fingerprint(cfg.to_dict()):
        raise LaunchError(f"{source} was trained under another config than {job.config} {list(job.overrides)}")
    here, there = git_revision(repo), manifest.get("git_revision")
    if here != there or here.endswith("-dirty") or here == "unknown":
        raise LaunchError(f"{source} was trained at commit {there}; this checkout is {here}")
    if accelerator is not None and manifest.get("device", {}).get("accelerator") != accelerator:
        raise LaunchError(f"{source} ran on {manifest.get('device', {}).get('accelerator')!r}, "
                          f"not {accelerator!r}")
    verify_checksums(source)
    if not (source / "last.pt").is_file():
        raise LaunchError(f"{source} has no last.pt")
    return source


def copy_run(source: str | Path, dest: str | Path) -> Path:
    """Copy a paused run's ``RESUME_FILES`` into a new run directory; returns its ``last.pt``.

    Each copy is checked against the source session's ``SHA256SUMS``.
    """
    source, dest = Path(source), Path(dest)
    if dest.exists():
        raise LaunchError(f"{dest} exists: a resumed run is copied into a new directory")
    expected = _read_checksums(source)
    dest.mkdir(parents=True)
    for name in RESUME_FILES:
        if (source / name).is_file():
            if name not in expected:
                raise LaunchError(f"{source / CHECKSUMS} does not list {name}")
            shutil.copyfile(source / name, dest / name)
            if _sha256(dest / name) != expected[name]:
                raise LaunchError(f"{dest / name} differs from {source / name} after copying")
    return dest / "last.pt"


def train_command(
    job: Job, *, repo: str | Path, data: str | Path | Sequence[str | Path], venue: str, experiment: str,
    python: str = sys.executable, resume: str | Path | None = None, stop_by: datetime | None = None,
) -> list[str]:
    repo = Path(repo)
    cmd = [
        python, "-u", str(repo / "scripts" / "train.py"),
        "--config", str(repo / job.config),
        "--data", *map(str, data_dirs(data)),
        "--venue", venue,
        "--experiment", experiment,
    ]
    if resume is not None:
        cmd += ["--resume", str(resume)]
    if stop_by is not None:
        cmd += ["--stop-by", stop_by.isoformat(timespec="seconds")]
    if job.overrides:
        cmd += ["--set", *job.overrides]   # last: --set takes every argument after it
    return cmd


def device_ids(gpus: int) -> list[str]:
    """Physical ids for ``gpus`` devices, respecting an existing ``CUDA_VISIBLE_DEVICES``."""
    if gpus < 1:
        raise LaunchError("gpus must be at least 1")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    ids = [d.strip() for d in visible.split(",") if d.strip()] if visible else [str(i) for i in range(gpus)]
    if len(ids) < gpus:
        raise LaunchError(f"asked for {gpus} GPUs, CUDA_VISIBLE_DEVICES={visible!r} allows {len(ids)}")
    return ids[:gpus]


@dataclass
class _Running:
    index: int
    job: Job
    run_name: str
    log: Path
    proc: subprocess.Popen
    handle: object
    started: float


def _last_line(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    except OSError:
        return ""
    return lines[-1] if lines else "(no output yet)"


def run_queue(
    jobs: Sequence[Job],
    *,
    repo: str | Path,
    data: str | Path | Sequence[str | Path],
    venue: str,
    experiment: str,
    gpus: int,
    work_dir: str | Path,
    resume_root: str | Path | None = None,
    stop_by: datetime | None = None,
    accelerator: str | None = None,
    python: str = sys.executable,
    poll_seconds: float = 5.0,
    progress_seconds: float = 300.0,
    echo: Callable[[str], None] = print,
) -> list[JobResult]:
    """Run ``jobs`` at most ``gpus`` at a time, one per GPU; results in job order.

    Every job runs with ``work_dir`` as its working directory, so its run
    directory and ``experiments/costs.csv`` land under it; its output goes
    to ``work_dir/logs/<run name>.log``.  A ``Job(resume=True)`` continues its
    paused run from ``resume_root`` (``check_resume``, ``copy_run``).  With
    ``stop_by``, every job pauses cleanly after the first step to end past
    it (``PAUSED_EXIT``), resumable in a later session.
    """
    repo, work_dir = Path(repo), Path(work_dir)
    jobs = list(jobs)
    names = plan_runs(jobs, repo=repo, work_dir=work_dir)
    check_sources(jobs, repo=repo, data=data)
    try:
        hourly_rate(venue)
    except KeyError as exc:
        # The trainer would only skip the cost row -- after the GPU time is spent.
        raise LaunchError(exc.args[0]) from None
    if any(job.resume for job in jobs) and resume_root is None:
        raise LaunchError("a resumed job needs resume_root: the earlier session's output")
    sources = {i: check_resume(job, names[i], repo=repo, resume_root=resume_root, accelerator=accelerator)
               for i, job in enumerate(jobs) if job.resume}
    devices = device_ids(gpus)
    resumes = {i: copy_run(source, work_dir / "experiments" / names[i]) for i, source in sources.items()}
    log_dir = work_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    pending = deque(enumerate(jobs))
    running: dict[str, _Running] = {}
    results: dict[int, JobResult] = {}
    last_progress = time.monotonic()
    echo(f"{len(jobs)} job(s) on {len(devices)} GPU(s): {', '.join(names)}")
    try:
        while pending or running:
            for gpu in devices:
                if gpu not in running and pending:
                    index, job = pending.popleft()
                    log = log_dir / f"{names[index]}.log"
                    handle = log.open("w", encoding="utf-8")
                    cmd = train_command(job, repo=repo, data=data, venue=venue, experiment=experiment,
                                        python=python, resume=resumes.get(index), stop_by=stop_by)
                    handle.write("$ " + " ".join(cmd) + "\n")
                    handle.flush()
                    proc = subprocess.Popen(
                        cmd, cwd=work_dir, stdout=handle, stderr=subprocess.STDOUT,
                        env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu},
                    )
                    running[gpu] = _Running(index, job, names[index], log, proc, handle, time.monotonic())
                    echo(f"[gpu {gpu}] {'resume' if index in resumes else 'start'} {names[index]}")

            for gpu, run in list(running.items()):
                code = run.proc.poll()
                if code is None:
                    continue
                run.handle.close()
                seconds = time.monotonic() - run.started
                results[run.index] = JobResult(run.job, run.run_name, gpu, code, seconds, run.log)
                del running[gpu]
                status = ("done" if code == 0 else "paused (resumable)" if code == PAUSED_EXIT
                          else f"FAILED (exit {code}): {_last_line(run.log)}")
                echo(f"[gpu {gpu}] {status} {run.run_name} in {format_duration(seconds)}")

            if running and time.monotonic() - last_progress >= progress_seconds:
                for gpu, run in sorted(running.items()):
                    echo(f"[gpu {gpu}] {run.run_name}: {_last_line(run.log)}")
                last_progress = time.monotonic()
            if running:
                time.sleep(poll_seconds)
    finally:
        # An interrupt (the notebook's stop button) must not leave jobs holding the GPUs.
        for run in running.values():
            run.proc.terminate()
            try:
                run.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                run.proc.kill()
            run.handle.close()

    ordered = [results[i] for i in range(len(jobs))]
    paused = [r.run_name for r in ordered if r.paused]
    failed = [r.run_name for r in ordered if not r.ok and not r.paused]
    echo(f"{len(ordered) - len(failed) - len(paused)}/{len(ordered)} finished"
         + (f"; paused: {paused}" if paused else "") + (f"; failed: {failed}" if failed else ""))
    return ordered
