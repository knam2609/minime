"""Checkpoint save/resume.

A checkpoint carries everything needed to make the resumed run indistinguishable
from an uninterrupted one: weights, optimizer moments, the step counter, every
RNG stream, the data sampler's position and, for fp16, the GradScaler's loss
scale.  Dropping any one of them produces a run that looks fine and is quietly
not the experiment it claims to be -- most insidiously the sampler position,
which silently re-feeds data the model has already seen.  (A lost loss scale
restarts at 65,536 with a fresh growth tracker: skipped steps after the
resume, and a curve that is not the uninterrupted one.)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from ..utils.seed import load_rng_state, rng_state


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    config: dict | None = None,
    sampler_state: dict | None = None,
    metrics: dict | None = None,
    extra: dict | None = None,
    init_from: dict | None = None,
    scaler_state: dict | None = None,
    precision: str | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "step": step,
        "config": config or {},
        "rng": rng_state(),
        "metrics": metrics or {},
        "format_version": 1,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if sampler_state is not None:
        payload["sampler"] = sampler_state
    if scaler_state is not None:
        # Empty for a disabled scaler (fp32, bf16): only fp16 has a loss scale.
        payload["scaler"] = scaler_state
    if precision is not None:
        # The realised precision: the config usually says "auto".
        payload["precision"] = precision
    if init_from is not None:
        # Which checkpoint the run started from (``load_weights``): every
        # checkpoint carries it, so a resumed run still knows.
        payload["init_from"] = init_from
    if extra:
        payload["extra"] = extra

    # Write to a temporary file first: a run killed mid-save must not leave a
    # truncated checkpoint where the last good one used to be.
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    sampler=None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
    strict: bool = True,
    scaler: torch.amp.GradScaler | None = None,
) -> dict:
    # weights_only=False because the payload intentionally contains RNG state
    # objects.  These are our own checkpoints; never point this at an untrusted
    # file.
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    restore_checkpoint(payload, model=model, optimizer=optimizer, sampler=sampler,
                       scaler=scaler, restore_rng=restore_rng, strict=strict, source=path)
    return payload


def restore_checkpoint(
    payload: dict,
    *,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    sampler=None,
    scaler: torch.amp.GradScaler | None = None,
    restore_rng: bool = True,
    strict: bool = True,
    source: str | Path = "checkpoint",
) -> None:
    """Put a loaded payload's state into the objects given (``load_checkpoint``'s second half).

    The scaler must agree with the payload: an enabled scaler (fp16) needs the
    saved loss scale, and a saved loss scale needs an enabled scaler --
    ``GradScaler.load_state_dict`` on a disabled scaler would drop it silently.
    Both are checked before anything is restored.
    """
    if scaler is not None:
        saved = payload.get("scaler") or {}
        if scaler.is_enabled() and not saved:
            raise ValueError(
                f"{source} holds no GradScaler state, and this run uses one (fp16): resuming would "
                "restart the loss scale and its growth tracker, so the run would not continue the same "
                "trajectory"
            )
        if saved and not scaler.is_enabled():
            raise ValueError(f"{source} was saved with an enabled GradScaler (fp16); this run has none")

    if model is not None:
        model.load_state_dict(payload["model"], strict=strict)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if sampler is not None and "sampler" in payload:
        sampler.load_state_dict(payload["sampler"])
    if scaler is not None and scaler.is_enabled():
        scaler.load_state_dict(payload["scaler"])
    if restore_rng and "rng" in payload:
        load_rng_state(payload["rng"])


def file_sha256(path: str | Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def load_weights(path: str | Path, model: torch.nn.Module, model_config: dict) -> dict:
    """Start ``model`` from a checkpoint's weights, and nothing else (``train.py --init-from``).

    No optimiser moments, step, sampler position or RNG: an adaptation run
    restarts its optimiser and schedule from a pretrained model.  Refuses a
    checkpoint whose model config differs from ``model_config`` in anything.
    Returns what identifies the start -- the file's sha256 (runs are paired
    on it), its step, and its config's name and fingerprint (the one
    ``evaluation.harness.load_model`` records) -- for the manifest and every
    checkpoint the run writes.
    """
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("config") or {}
    saved = config.get("model")
    if saved is None:
        raise ValueError(f"{path} has no model config; it was not written by this project's trainer")
    differing = sorted(k for k in saved.keys() | model_config.keys() if saved.get(k) != model_config.get(k))
    if differing:
        shown = ", ".join(f"{k} ({saved.get(k)!r} there, {model_config.get(k)!r} here)" for k in differing)
        raise ValueError(f"{path}: its model differs from this run's: {shown}")
    model.load_state_dict(payload["model"], strict=True)
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "step": payload.get("step"),
        "config_name": config.get("name"),
        "config_fingerprint": hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:12],
    }


def rotate_checkpoints(directory: str | Path, keep: int, pattern: str = "step_*.pt") -> list[Path]:
    """Delete all but the most recent ``keep`` step checkpoints.

    ``last.pt`` and ``best.pt`` are named separately and never rotated.
    """
    directory = Path(directory)
    if keep <= 0 or not directory.exists():
        return []
    checkpoints = sorted(directory.glob(pattern), key=lambda p: _step_of(p))
    doomed = checkpoints[:-keep] if len(checkpoints) > keep else []
    for path in doomed:
        path.unlink(missing_ok=True)
    return doomed


def _step_of(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except ValueError:
        return -1


def summarise(path: str | Path) -> dict:
    """Read a checkpoint's metadata without materialising the weights on GPU."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return {
        "step": payload.get("step"),
        "metrics": payload.get("metrics", {}),
        "config_name": payload.get("config", {}).get("name"),
        "n_tensors": len(payload.get("model", {})),
        "has_optimizer": "optimizer" in payload,
        "has_sampler": "sampler" in payload,
    }
