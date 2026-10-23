"""Typed configuration objects loaded from YAML.

Design notes
------------
* Every experiment arm is a YAML file that ``extends`` a base config and
  overrides one thing.  That is what makes the ablations *controlled*: the
  diff between two arms is literally the diff between two files.
* Unknown keys are a hard error.  A silently-ignored typo in ``n_layers``
  would invalidate an experiment without ever looking wrong.
* ``Config.fingerprint()`` goes into every run manifest so a metrics file can
  always be traced back to the exact settings that produced it.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

NormKind = Literal["rmsnorm", "layernorm"]
PosKind = Literal["rope", "learned"]
FFNKind = Literal["swiglu", "gelu"]
OptimKind = Literal["adamw", "muon"]
PrecisionKind = Literal["auto", "fp32", "fp16", "bf16"]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """Where corpora, tokenised shards and checkpoints live.

    Defaults to ``<repo>/data`` but is redirected to an external drive with
    ``MINIME_DATA_ROOT`` so the internal disk never holds a tokenised corpus.
    """
    return Path(os.environ.get("MINIME_DATA_ROOT", _REPO_ROOT / "data")).expanduser()


@dataclass
class ModelConfig:
    vocab_size: int = 16384
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 6
    context_length: int = 512
    dropout: float = 0.0

    # --- ablation switches: one implementation, no code forks ---
    norm: NormKind = "rmsnorm"
    pos: PosKind = "rope"
    ffn: FFNKind = "swiglu"

    bias: bool = False
    tie_embeddings: bool = True
    rope_theta: float = 10_000.0
    ffn_mult: float = 4.0
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError(
                f"d_model={self.d_model} is not divisible by n_heads={self.n_heads}"
            )
        if self.norm not in ("rmsnorm", "layernorm"):
            raise ValueError(f"unknown norm: {self.norm!r}")
        if self.pos not in ("rope", "learned"):
            raise ValueError(f"unknown pos: {self.pos!r}")
        if self.ffn not in ("swiglu", "gelu"):
            raise ValueError(f"unknown ffn: {self.ffn!r}")
        if self.pos == "rope" and (self.d_model // self.n_heads) % 2:
            raise ValueError("RoPE requires an even head dimension")
        if self.vocab_size > 65535:
            raise ValueError("vocab_size must fit in uint16 for the .bin storage format")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def ffn_hidden(self) -> int:
        """Hidden width of the feed-forward block.

        SwiGLU has three matrices instead of two, so the usual 4x width is
        scaled by 2/3 to keep the parameter count comparable to GELU.  Without
        this the architecture ablation would also be a size ablation.
        """
        if self.ffn == "swiglu":
            hidden = int(2 * self.ffn_mult * self.d_model / 3)
            return 64 * ((hidden + 63) // 64)  # round up to a multiple of 64
        return int(self.ffn_mult * self.d_model)


@dataclass
class TrainingConfig:
    batch_size: int = 32
    gradient_accumulation: int = 1
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 100
    max_steps: int = 1000

    optimizer: OptimKind = "adamw"
    # Muon settings apply to the hidden 2D weights only; everything else stays
    # on AdamW with the settings above.  No decay on Muon's weights by default,
    # as in the reference implementation: at lr 0.02, decay 0.1 would shrink
    # them ~33x faster than AdamW's lr * weight_decay does.
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.0

    validation_interval: int = 100
    validation_batches: int = 20
    log_interval: int = 10
    checkpoint_interval: int = 500
    keep_last_n_checkpoints: int = 2

    seed: int = 1337
    precision: PrecisionKind = "auto"
    device: str = "auto"
    compile: bool = False
    num_workers: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.betas, list):
            self.betas = tuple(self.betas)  # YAML gives a list
        if self.learning_rate is None:
            # E1's main configs and the flagship leave it null until their
            # pilot or sweep picks it, so no run starts at an LR nobody chose.
            raise ValueError(
                "training.learning_rate is null: this config waits for it to be chosen "
                "(E1's main runs: the LR pilot's pick, reports/e1/pilot.json; the flagship: "
                "the M10 sweep's pick, reports/m10/sweep.json)"
            )
        if isinstance(self.learning_rate, bool) or not isinstance(self.learning_rate, (int, float)):
            raise ValueError(
                f"training.learning_rate must be a number, got {self.learning_rate!r} "
                "(YAML 1.1 reads 1e-4 as a string: write 1.0e-4)"
            )
        if self.warmup_steps > self.max_steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) exceeds max_steps ({self.max_steps}): "
                "the learning rate would never reach its peak and never decay. "
                "For a short smoke run, lower warmup_steps too."
            )
        if self.optimizer not in ("adamw", "muon"):
            raise ValueError(f"unknown optimizer: {self.optimizer!r}")


@dataclass
class DataConfig:
    """Which shards feed the run, and in what proportion.

    ``mixture`` maps a source name to a sampling weight.  Weights are
    normalised, so ``{prose: 7, chat: 2, code: 1}`` and ``{prose: 70, chat: 20,
    code: 10}`` describe the same mixture.  Experiment E2 varies only this dict.
    """

    dataset: str = "general"
    mixture: dict[str, float] = field(default_factory=lambda: {"general": 1.0})
    tokenizer_path: str = "tokenizer/minime-16k.json"
    seed: int = 1234

    def normalised_mixture(self) -> dict[str, float]:
        total = sum(self.mixture.values())
        if total <= 0:
            raise ValueError("mixture weights must sum to a positive number")
        return {k: v / total for k, v in self.mixture.items()}


@dataclass
class Config:
    name: str = "unnamed"
    notes: str = ""
    # What a run is for within its experiment ("pilot", "main"), so an
    # analysis can refuse a run in the wrong role rather than trust a file
    # name.  A label, like ``name``: not an experimental variable.
    role: str = ""
    out_dir: str = "artifacts"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @property
    def tokens_per_step(self) -> int:
        """Tokens consumed per *optimizer* step.

        Held constant across devices by trading batch size against gradient
        accumulation -- otherwise a Kaggle run and a laptop run are not the
        same experiment.
        """
        return (
            self.training.batch_size
            * self.training.gradient_accumulation
            * self.model.context_length
        )

    @property
    def token_budget(self) -> int:
        return self.tokens_per_step * self.training.max_steps

    @property
    def run_name(self) -> str:
        """The default run directory's name: two seeds of one config never share one."""
        return f"{self.name}-s{self.training.seed}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        """Stable short hash of the full config, for run manifests."""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _build(cls: type, payload: dict[str, Any], where: str):
    known = {f.name for f in fields(cls)}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(
            f"unknown key(s) {sorted(unknown)} in '{where}' section "
            f"(known: {sorted(known)})"
        )
    return cls(**payload)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config, resolving a chain of ``extends:`` parents."""
    raw = _load_raw(Path(path))
    if overrides:
        raw = _deep_merge(raw, overrides)

    sections = {"model": ModelConfig, "training": TrainingConfig, "data": DataConfig}
    built: dict[str, Any] = {}
    for key, cls in sections.items():
        built[key] = _build(cls, raw.pop(key, {}) or {}, key)

    top_known = {f.name for f in fields(Config)} - set(sections)
    unknown = set(raw) - top_known
    if unknown:
        raise ValueError(f"unknown top-level key(s) {sorted(unknown)}")
    return Config(**raw, **built)


def _load_raw(path: Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"circular 'extends' chain at {path}")
    _seen.add(path)

    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}

    parent_rel = raw.pop("extends", None)
    if parent_rel is None:
        return raw
    parent = (path.parent / parent_rel).resolve()
    return _deep_merge(_load_raw(parent, _seen), raw)


def parse_cli_overrides(pairs: list[str]) -> dict[str, Any]:
    """Turn ``["training.max_steps=50", "model.n_layers=2"]`` into a nested dict."""
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"override must look like key=value, got {pair!r}")
        dotted, _, value = pair.partition("=")
        cursor = out
        parts = dotted.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = _coerce_scalar(value)
    return out


def _coerce_scalar(value: str):
    """Parse a CLI scalar, working around YAML 1.1's exponent rule.

    ``yaml.safe_load("1e-4")`` returns the *string* "1e-4" because YAML 1.1
    requires a decimal point in an exponential float.  Left alone, that means
    ``--set training.learning_rate=1e-4`` silently hands a string to AdamW.
    """
    parsed = yaml.safe_load(value)
    if isinstance(parsed, str):
        try:
            return int(parsed)
        except ValueError:
            pass
        try:
            return float(parsed)
        except ValueError:
            pass
    return parsed


__all__ = [
    "Config",
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "load_config",
    "parse_cli_overrides",
    "data_root",
    "is_dataclass",
]
