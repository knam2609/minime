"""Run logging: a JSONL metrics stream plus a manifest describing the run.

JSONL is the source of truth rather than a tracking service, so every plot in
the report can be regenerated offline from files in the repository, and a run
is never lost because an API key expired.  W&B, when enabled, is a mirror.

A run may span several sessions (the flagship resumes across Kaggle sessions):
the manifest's ``sessions`` lists each one's start, commit, device and data
directory, and how it ended -- ``paused`` (a clean stop, resumable) or
``finished``.  The top-level ``started``, ``git_revision`` and ``device``
describe the latest session, as they always have.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def git_revision(repo: Path | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo or Path(__file__).resolve().parents[3],
            capture_output=True, text=True, timeout=5, check=True,
        )
        revision = out.stdout.strip()
        # --no-optional-locks: a plain `git status` takes .git/index.lock to
        # refresh the index, and when the timeout kills it mid-refresh (a
        # loaded machine, a slow disk) the lock is left behind and every later
        # git command in the repository fails until it is deleted by hand.
        dirty = subprocess.run(
            ["git", "--no-optional-locks", "status", "--porcelain"],
            cwd=repo or Path(__file__).resolve().parents[3],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        return f"{revision}-dirty" if dirty else revision
    except Exception:
        return "unknown"


class RunLogger:
    def __init__(
        self,
        run_dir: str | Path,
        config: dict,
        device_info: dict | None = None,
        wandb_project: str | None = None,
        resume: bool = False,
        data: dict | None = None,
        init_from: dict | None = None,
        start_step: int = 0,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.manifest_path = self.run_dir / "manifest.json"
        self.config = config

        session = {
            "started": _now(),
            "git_revision": git_revision(),
            "python": sys.version.split()[0],
            "device": device_info or {},
            "start_step": start_step,
        }
        if data and "data_dir" in data:
            session["data_dir"] = data["data_dir"]
        self.manifest = {
            "run": self.run_dir.name,
            "started": session["started"],
            "git_revision": session["git_revision"],
            "python": session["python"],
            "device": session["device"],
            "config": config,
        }
        if data:
            # Which shards, and which tokenizer made them: the config alone
            # does not say what the model was trained on.
            self.manifest["data"] = data
        if init_from:
            # The checkpoint whose weights the run started from (--init-from).
            self.manifest["init_from"] = init_from
        sessions = []
        if resume and self.manifest_path.exists():
            previous = json.loads(self.manifest_path.read_text())
            _check_same_run(previous, config, data, self.manifest_path)
            self.manifest["resumed_from"] = previous.get("started")
            # Runs from before ``sessions`` existed: their one session, as recorded.
            sessions = previous.get("sessions") or [
                {k: previous[k] for k in ("started", "git_revision", "python", "device") if k in previous}
            ]
        self.manifest["sessions"] = [*sessions, session]
        self._write()

        self._handle = self.metrics_path.open("a", encoding="utf-8")
        self._wandb = None
        if wandb_project:
            self._wandb = _init_wandb(wandb_project, self.run_dir.name, config)

    def log(self, record: dict[str, Any]) -> None:
        record = {"wall_time": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
        self._handle.write(json.dumps(record, default=str) + "\n")
        self._handle.flush()          # a killed run keeps every step it logged
        if self._wandb is not None:
            self._wandb.log({k: v for k, v in record.items() if k != "wall_time"})

    def finalise(self, summary: dict[str, Any]) -> None:
        self.manifest["finished"] = _now()
        self.manifest["summary"] = summary
        self._end_session("finished", summary["steps"], summary)
        self._write()
        if self._wandb is not None:
            self._wandb.summary.update(summary)
            self._wandb.finish()

    def pause(self, step: int, reason: str, stats: dict[str, Any]) -> None:
        """Record a clean stop: no ``finished``, no ``summary`` -- the run is not done."""
        self.manifest["paused"] = {"step": step, "at": _now(), "reason": reason}
        self._end_session("paused", step, stats)
        self._write()
        if self._wandb is not None:
            self._wandb.finish()

    def _end_session(self, outcome: str, step: int, stats: dict[str, Any]) -> None:
        self.manifest["sessions"][-1].update(
            ended=_now(), end_step=step, outcome=outcome,
            runtime_seconds=stats.get("runtime_seconds"), tokens_per_second=stats.get("tokens_per_second"),
        )

    def _write(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2, default=str))

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _check_same_run(previous: dict, config: dict, data: dict | None, path: Path) -> None:
    """Refuse to continue a run under another config or on other data.

    A resumed session appends to the earlier sessions' metrics: under another
    config (``max_steps`` changes the schedule) or other shards it would be a
    different experiment wearing the same name.  Where the shards live may
    differ (a Kaggle input path, a local one); what they are may not.
    """
    if _canonical(previous.get("config")) != _canonical(config):
        before, now = _flat(json.loads(_canonical(previous.get("config")))), _flat(json.loads(_canonical(config)))
        differing = sorted(k for k in before.keys() | now.keys() if before.get(k) != now.get(k))
        raise ValueError(f"{path}: the run was started under another config; differs in {differing}")
    if previous.get("data") is not None and data is not None:
        before = {k: v for k, v in previous["data"].items() if k != "data_dir"}
        now = {k: v for k, v in json.loads(_canonical(data)).items() if k != "data_dir"}
        if _canonical(before) != _canonical(now):
            differing = sorted(k for k in before.keys() | now.keys() if before.get(k) != now.get(k))
            raise ValueError(f"{path}: the run was started on other data; differs in {differing}")


def _flat(tree: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(tree, dict):
        return {prefix.rstrip("."): tree}
    out: dict[str, Any] = {}
    for key, value in tree.items():
        out.update(_flat(value, f"{prefix}{key}."))
    return out


def _init_wandb(project: str, name: str, config: dict):
    try:
        import wandb
    except ImportError:
        print("wandb requested but not installed; continuing with JSONL only")
        return None
    return wandb.init(project=project, name=name, config=config)


def read_metrics(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
