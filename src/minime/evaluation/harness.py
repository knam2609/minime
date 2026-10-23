"""The evaluation harness: one checkpoint in, one ``eval.json`` out.

Every family measures a fixed, hashed eval set with fixed settings, so two
``eval.json`` files are comparable exactly when their ``eval_sets`` digests
and ``eval_config_fingerprint`` agree -- and the harness writes both.

Families (``EvalConfig.families``):

``perplexity``    CE / perplexity / bits-per-byte on the personal split (the
                  primary voice metric) and on each general slice, plus the
                  mixture-weighted general CE that detects forgetting.
``voice``         continuations of held-out personal conversations, judged by
                  stylometry (profiles, distances, Burrows' Delta), authorship
                  classifiers (impersonation AUC, attribution rate) and
                  diversity -- each beside the real messages and a control.
``degeneration``  repetition, diversity and entropy of long web continuations.
``code``          syntax validity of generated Python function bodies.
``blimp``         grammatical minimal pairs.

Held-out data is the ``val`` split.  The personal ``test`` split is carved out
for the final report and read only when the caller passes ``split="test"``
*and* ``unlock_test=True``; the CLI spells that ``--split test --final-report``.

``eval.json`` holds numbers and hashes only.  Generated text is personal-ish
(the model was trained on the owner's messages), so it goes to a separate
samples file under a gitignored ``samples/`` directory, or nowhere.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from ..config import ModelConfig, _build, _deep_merge, _load_raw
from ..data.binidx import TokenShard, common_tokenizer
from ..model import build_model
from ..tokenizer import is_dev_tokenizer
from ..training.logging import git_revision
from ..utils.device import DeviceSpec, autocast_ctx, describe, resolve
from . import authorship, codeeval, degeneration, stylometry
from .generation import Generation, sample
from .perplexity import cross_entropy, weighted_mean
from .sets import document_prompts, prefix_prompts
from .tasks import blimp

EVAL_VERSION = 1
FAMILIES = ("perplexity", "voice", "degeneration", "code", "blimp")
GENERAL_SLICES = ("web", "dialogue", "code")


# ------------------------------------------------------------------ config


@dataclass
class SamplingSpec:
    temperature: float = 1.0
    top_k: int | None = 50


@dataclass
class PerplexitySpec:
    max_tokens: int = 1_000_000  # per general slice; the personal split is always covered fully
    general_weights: dict[str, float] = field(
        default_factory=lambda: {"web": 0.7, "dialogue": 0.2, "code": 0.1}
    )
    # Added for the flagship (M10), after the probe-scale evaluations; see
    # LATER_SETTINGS.  context_length: the scoring windows' length, None = the
    # model's own (512 scores the flagship like the probes).  batch_size:
    # windows per forward pass, None = EvalConfig.batch_size; it changes the
    # summation order only, not the windows.
    context_length: int | None = None
    batch_size: int | None = None

    def __post_init__(self) -> None:
        for name in ("context_length", "batch_size"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"perplexity.{name} must be a positive integer or null, not {value!r}")


@dataclass
class VoiceSpec:
    prompts: int = 512
    prompt_tokens: int = 128
    max_new_tokens: int = 64  # also the length cap on every text the family compares
    profile_words: int = 30_000  # per reference author, for the style profiles and Delta
    chunk_words: int = 300
    mfw: int = 100
    folds: int = 5
    classifier_docs: int = 2000  # per class, for the attribution classifier


@dataclass
class DegenerationSpec:
    prompts: int = 128
    prompt_tokens: int = 64
    max_new_tokens: int = 128


@dataclass
class CodeSpec:
    prompts: int = 200
    prompt_tokens: int = 256
    max_new_tokens: int = 128
    per_file: int = 2


@dataclass
class BlimpSpec:
    pairs_per_paradigm: int = 100
    path: str = ""  # default: data_root()/eval/blimp.jsonl


@dataclass
class EvalConfig:
    version: int = EVAL_VERSION
    seed: int = 1234
    batch_size: int = 32
    families: list[str] = field(default_factory=lambda: list(FAMILIES))
    sampling: SamplingSpec = field(default_factory=SamplingSpec)
    perplexity: PerplexitySpec = field(default_factory=PerplexitySpec)
    voice: VoiceSpec = field(default_factory=VoiceSpec)
    degeneration: DegenerationSpec = field(default_factory=DegenerationSpec)
    code: CodeSpec = field(default_factory=CodeSpec)
    blimp: BlimpSpec = field(default_factory=BlimpSpec)

    def __post_init__(self) -> None:
        unknown = set(self.families) - set(FAMILIES)
        if unknown:
            raise ValueError(f"unknown eval family/families {sorted(unknown)}; known: {FAMILIES}")

    def to_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        """Hash of every setting except ``families`` (running fewer families
        does not change what each family measures).  A setting in
        ``LATER_SETTINGS`` counts only once it is set."""
        payload = {k: v for k, v in self.to_dict().items() if k != "families"}
        for section, names in LATER_SETTINGS.items():
            for name in names:
                if payload[section][name] is None:
                    del payload[section][name]
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


# Settings added after evaluations had been written with the fingerprint.  At
# their default (None) they are left out of the hash, so those files keep
# theirs (configs/eval.yaml: 7c9503161d6f, every probe evaluation); any value
# set changes it.
LATER_SETTINGS = {"perplexity": ("context_length", "batch_size")}


_SECTIONS = {
    "sampling": SamplingSpec, "perplexity": PerplexitySpec, "voice": VoiceSpec,
    "degeneration": DegenerationSpec, "code": CodeSpec, "blimp": BlimpSpec,
}


def load_eval_config(path: str | Path | None = None, overrides: dict | None = None) -> EvalConfig:
    """Defaults, then the YAML at ``path`` (``extends:`` allowed), then overrides.

    Unknown keys are errors, as in the training config.
    """
    raw = _load_raw(Path(path)) if path else {}
    if overrides:
        raw = _deep_merge(raw, overrides)
    built = {key: _build(cls, raw.pop(key, {}) or {}, key) for key, cls in _SECTIONS.items()}
    return _build(EvalConfig, {**raw, **built}, "eval")


# ------------------------------------------------------------------ inputs


def load_model(checkpoint: str | Path, device: torch.device | str = "cpu") -> tuple[torch.nn.Module, dict]:
    """Rebuild the model a checkpoint was trained as, with its weights."""
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    cfg = payload.get("config") or {}
    if "model" not in cfg:
        raise ValueError(f"{checkpoint} has no model config; it was not written by this project's trainer")
    model = build_model(ModelConfig(**cfg["model"]))
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    info = {
        "path": str(checkpoint),
        "step": payload.get("step"),
        "config_name": cfg.get("name"),
        "config_fingerprint": hashlib.sha256(
            json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12],
        "params": model.num_parameters(),
        "tokens_seen": (payload.get("metrics") or {}).get("tokens_seen"),
    }
    return model, info


@dataclass
class Shards:
    """The shards a run of the harness reads, resolved and checked up front."""

    personal: dict[str, TokenShard]  # "split" -> held-out, "train/<name>" -> training shards
    general: dict[str, TokenShard]  # "<slice>.<split>"

    def all(self) -> dict[str, TokenShard]:
        return {**{f"personal/{k}": v for k, v in self.personal.items()},
                **{f"general/{k}": v for k, v in self.general.items()}}


def resolve_shards(personal: Path | None, general: Path | None, split: str) -> Shards:
    found_personal: dict[str, TokenShard] = {}
    if personal is not None:
        found_personal["split"] = TokenShard(personal / split)
        for meta in sorted(personal.glob("*.train.meta.json")):
            name = meta.name[: -len(".meta.json")]
            found_personal[f"train/{name}"] = TokenShard(personal / name)
    found_general: dict[str, TokenShard] = {}
    if general is not None:
        for name in GENERAL_SLICES:
            for s in {split, "val"}:
                path = general / f"{name}.{s}"
                if Path(f"{path}.bin").exists():
                    found_general[f"{name}.{s}"] = TokenShard(path)
    return Shards(found_personal, found_general)


# ------------------------------------------------------------------ running


@dataclass
class _Context:
    model: torch.nn.Module
    tokenizer: Any
    shards: Shards
    cfg: EvalConfig
    split: str
    spec: DeviceSpec
    samples: list[dict]
    eval_sets: dict[str, dict]
    log: Callable[[str], None]

    def autocast(self):
        return autocast_ctx(self.spec)

    def generate(self, prompts: np.ndarray, max_new_tokens: int, seed_offset: int) -> Generation:
        return sample(
            self.model, prompts, max_new_tokens, self.tokenizer.eot_id,
            temperature=self.cfg.sampling.temperature, top_k=self.cfg.sampling.top_k,
            seed=self.cfg.seed + seed_offset, batch_size=self.cfg.batch_size,
            device=self.spec.device, autocast=self.autocast,
        )

    def decode(self, ids) -> str:
        return self.tokenizer.decode([int(i) for i in ids])

    def general(self, name: str, split: str | None = None) -> TokenShard:
        key = f"{name}.{split or self.split}"
        if key not in self.shards.general:
            raise FileNotFoundError(f"this family needs the general shard '{key}'; pass --general")
        return self.shards.general[key]

    def personal_split(self) -> TokenShard:
        if "split" not in self.shards.personal:
            raise FileNotFoundError("this family needs the personal shards; pass --personal")
        return self.shards.personal["split"]

    def record_samples(self, family: str, gen: Generation, doc_indices) -> None:
        for i, (tokens, ended, doc) in enumerate(zip(gen.tokens, gen.ended, doc_indices)):
            self.samples.append({"family": family, "prompt": i, "doc": int(doc),
                                 "ended": ended, "text": self.decode(tokens)})


def _texts(ctx: _Context, shards: list[TokenShard], wanted: int, max_tokens: int,
           words: int | None = None) -> list[str]:
    """Decoded documents drawn across ``shards``: ``wanted`` of them, or until ``words``.

    Documents are visited in a permutation seeded by the eval seed, so any
    prefix of the visit order is spread over the whole shard (whose own order
    is its source's stream order) and the draw is identical on every run.
    """
    refs = [(s, i) for s in shards for i in range(s.n_docs)]
    out, taken = [], 0
    eot = ctx.tokenizer.eot_id
    for k in np.random.default_rng(ctx.cfg.seed).permutation(len(refs)):
        shard, i = refs[k]
        doc = shard.document(i)
        doc = doc[:-1] if doc.size and doc[-1] == eot else doc
        if not doc.size:
            continue
        text = ctx.decode(doc[:max_tokens])
        out.append(text)
        taken += len(stylometry.words(text))
        if (words is not None and taken >= words) or (words is None and len(out) >= wanted):
            break
    return out


def _perplexity(ctx: _Context) -> dict:
    cfg, spec = ctx.cfg.perplexity, ctx.spec
    native = ctx.model.cfg.context_length
    context = cfg.context_length or native
    if context > native:
        raise ValueError(f"perplexity.context_length {context} is longer than the model's context ({native})")
    batch_size = cfg.batch_size or ctx.cfg.batch_size
    byte_lengths = ctx.tokenizer.byte_lengths()
    run = lambda shard, cap: cross_entropy(  # noqa: E731
        ctx.model, shard, context, batch_size, spec.device,
        max_tokens=cap, byte_lengths=byte_lengths, autocast=ctx.autocast,
    )
    # The scoring context, stated: the checkpoint's own config is untouched.
    out: dict[str, Any] = {"context_length": context, "native_context_length": native}
    if "split" in ctx.shards.personal:
        out["personal"] = run(ctx.personal_split(), None)
        ctx.eval_sets[f"ce/personal/{ctx.split}"] = {"digest": out["personal"]["digest"],
                                                     "tokens": out["personal"]["tokens"]}
    general = {}
    for name in cfg.general_weights:
        key = f"{name}.{ctx.split}"
        if key in ctx.shards.general:
            general[name] = run(ctx.shards.general[key], cfg.max_tokens)
            ctx.eval_sets[f"ce/general/{key}"] = {"digest": general[name]["digest"],
                                                  "tokens": general[name]["tokens"]}
    if general:
        out["general"] = general
        if set(cfg.general_weights) <= set(general):
            loss = weighted_mean(general, cfg.general_weights)
            out["general_weighted"] = {"loss": loss, "ppl": float(np.exp(min(loss, 50.0))),
                                       "weights": dict(cfg.general_weights)}
    if "personal" not in out and not general:
        raise FileNotFoundError("perplexity needs --personal and/or --general shards")
    return out


def _voice(ctx: _Context) -> dict:
    v = ctx.cfg.voice
    ps = document_prompts(ctx.personal_split(), v.prompts, v.prompt_tokens, v.max_new_tokens,
                          ctx.tokenizer.eot_id)
    ctx.eval_sets["voice/prompts"] = ps.describe()
    gen = ctx.generate(ps.prompts, v.max_new_tokens, seed_offset=0)
    ctx.record_samples("voice", gen, ps.doc_indices)
    real = [ctx.decode(t) for t in ps.targets]
    fake = [ctx.decode(t) for t in gen.tokens]

    train = [s for k, s in sorted(ctx.shards.personal.items()) if k.startswith("train/")]
    if not train:
        raise FileNotFoundError("the voice family needs the personal *.train shards (the owner's profile)")
    dialogue, web = ctx.general("dialogue", "val"), ctx.general("web", "val")
    budget = dict(wanted=0, max_tokens=v.max_new_tokens, words=v.profile_words)
    me_ref = _texts(ctx, train, **budget)
    dialogue_ref = _texts(ctx, [dialogue], **budget)
    web_ref = _texts(ctx, [web], **budget)
    me_cls = _texts(ctx, train, v.classifier_docs, v.max_new_tokens)
    control_cls = _texts(ctx, [dialogue], v.classifier_docs, v.max_new_tokens)
    for name, texts in {"me_train": me_ref, "dialogue": dialogue_ref, "web": web_ref,
                        "classifier_me": me_cls, "classifier_control": control_cls}.items():
        ctx.eval_sets[f"voice/reference/{name}"] = {
            "digest": hashlib.sha256("\x1e".join(texts).encode()).hexdigest()[:16], "texts": len(texts)}

    return {
        "prompts": ps.describe(),
        "generation": _generation_stats(gen),
        "stylometry": {
            "profiles": {name: stylometry.profile(texts).scalars() for name, texts in
                         {"generated": fake, "real": real, "me_train": me_ref,
                          "dialogue": dialogue_ref, "web": web_ref}.items()},
            # Distances to the real held-out messages.  me_train is the
            # same-author ceiling, dialogue the generic-chat control.
            "distance_to_real": {name: stylometry.compare(texts, real) for name, texts in
                                 {"generated": fake, "me_train": me_ref, "dialogue": dialogue_ref}.items()},
            "delta": stylometry.burrows_delta(
                {"me": me_ref, "dialogue": dialogue_ref, "web": web_ref},
                {"generated": fake, "real": real}, target="me",
                chunk_size=v.chunk_words, mfw=v.mfw,
            ),
        },
        "authorship": {
            "discriminator": authorship.discriminator_auc(
                real, fake, groups=list(range(len(real))) * 2, folds=v.folds, seed=ctx.cfg.seed),
            "attribution": authorship.attribution(
                me_cls, control_cls, {"real": real, "generated": fake}, seed=ctx.cfg.seed),
        },
        "diversity": {"generated": degeneration.summarise(fake), "real": degeneration.summarise(real)},
    }


def _generation_stats(gen: Generation) -> dict:
    return {"eot_rate": gen.eot_rate, "mean_tokens": gen.mean_length,
            "entropy": gen.entropy, "positions": gen.positions}


def _degeneration(ctx: _Context) -> dict:
    d = ctx.cfg.degeneration
    ps = prefix_prompts(ctx.general("web"), d.prompts, d.prompt_tokens, d.max_new_tokens,
                        ctx.tokenizer.eot_id)
    ctx.eval_sets["degeneration/prompts"] = ps.describe()
    gen = ctx.generate(ps.prompts, d.max_new_tokens, seed_offset=1)
    ctx.record_samples("degeneration", gen, ps.doc_indices)
    return {
        "prompts": ps.describe(),
        "generation": _generation_stats(gen),
        "generated": degeneration.summarise([ctx.decode(t) for t in gen.tokens]),
        "real": degeneration.summarise([ctx.decode(t) for t in ps.targets]),
    }


def _code(ctx: _Context) -> dict:
    c = ctx.cfg.code
    prompts, digest = codeeval.function_prompts(
        ctx.general("code"), np.asarray(ctx.tokenizer.byte_lengths()), ctx.decode,
        c.prompts, c.prompt_tokens, c.per_file, ctx.tokenizer.eot_id,
    )
    if not prompts:
        raise ValueError("no held-out code file yielded a function prompt")
    ctx.eval_sets["code/prompts"] = {"digest": digest, "prompts": len(prompts),
                                     "prompt_tokens": c.prompt_tokens}
    gen = ctx.generate(np.stack([p.prompt for p in prompts]), c.max_new_tokens, seed_offset=2)
    ctx.record_samples("code", gen, [p.doc_index for p in prompts])
    result = codeeval.syntax_validity(prompts, [ctx.decode(t) for t in gen.tokens], gen.ended)
    return {**result, "generation": _generation_stats(gen)}


def _blimp(ctx: _Context) -> dict:
    b = ctx.cfg.blimp
    pairs, digest = blimp.load_pairs(b.path or None, b.pairs_per_paradigm)
    ctx.eval_sets["blimp"] = {"sha256": digest[:16], "pairs": len(pairs),
                              "pairs_per_paradigm": b.pairs_per_paradigm}
    return blimp.evaluate(ctx.model, ctx.tokenizer, pairs, ctx.cfg.batch_size * 2,
                          ctx.spec.device, ctx.autocast)


_RUNNERS = {"perplexity": _perplexity, "voice": _voice, "degeneration": _degeneration,
            "code": _code, "blimp": _blimp}


def run_eval(
    model: torch.nn.Module,
    tokenizer,
    cfg: EvalConfig,
    personal: str | Path | None = None,
    general: str | Path | None = None,
    split: str = "val",
    unlock_test: bool = False,
    spec: DeviceSpec | None = None,
    checkpoint_info: dict | None = None,
    log: Callable[[str], None] = print,
) -> tuple[dict, list[dict]]:
    """Run ``cfg.families`` and return ``(eval_record, samples)``."""
    if split not in ("val", "test"):
        raise ValueError(f"split must be 'val' or 'test', not {split!r}")
    if split == "test" and not unlock_test:
        raise PermissionError(
            "the test split is reserved for the final report; pass unlock_test=True "
            "(--split test --final-report) only when writing it"
        )
    spec = spec or resolve("cpu")
    shards = resolve_shards(Path(personal) if personal else None, Path(general) if general else None, split)
    used = shards.all()
    if not used:
        raise FileNotFoundError("no shards found; pass --personal and/or --general")
    tokenizer_id = common_tokenizer(used)
    if tokenizer_id != tokenizer.id:
        raise ValueError(f"the shards were made by {tokenizer_id or '<unrecorded>'}, "
                         f"not by the tokenizer given ({tokenizer.id})")
    if model.cfg.vocab_size != tokenizer.vocab_size:
        raise ValueError(f"model vocab {model.cfg.vocab_size} != tokenizer vocab {tokenizer.vocab_size}")

    ctx = _Context(model.to(spec.device).eval(), tokenizer, shards, cfg, split, spec, [], {}, log)
    record: dict[str, Any] = {
        "eval_version": cfg.version,
        "eval_config_fingerprint": cfg.fingerprint(),
        "eval_config": cfg.to_dict(),
        "split": split,
        "checkpoint": checkpoint_info or {},
        "tokenizer": {"id": tokenizer.id, "dev": is_dev_tokenizer(tokenizer.id)},
        "git_revision": git_revision(),
        "device": describe(spec),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "shards": {name: {"tokens": s.n_tokens, "docs": s.n_docs} for name, s in sorted(used.items())},
        "seconds": {},
    }
    for family in cfg.families:
        started = time.perf_counter()
        log(f"  {family} ...")
        record[family] = _RUNNERS[family](ctx)
        record["seconds"][family] = round(time.perf_counter() - started, 1)
    record["eval_sets"] = ctx.eval_sets
    return record, ctx.samples


def write_outputs(record: dict, samples: list[dict], out: str | Path,
                  samples_path: str | Path | None) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, default=_jsonable) + "\n")
    if samples_path is not None and samples:
        samples_path = Path(samples_path)
        samples_path.parent.mkdir(parents=True, exist_ok=True)
        with samples_path.open("w", encoding="utf-8") as handle:
            for row in samples:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    raise TypeError(f"not JSON serialisable: {type(x).__name__}")
