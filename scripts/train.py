#!/usr/bin/env python3
"""Train a MiniMe model.

    # local smoke test
    python scripts/train.py --config configs/tiny.yaml --data data/shards/sample \
        --mixture chat=7,longform=2,code=1 --set training.max_steps=200

    # stop cleanly before the budget (a Kaggle session ends at 12 h): last.pt,
    # the manifest marked paused, exit status 75 -- by step, or by a deadline
    python scripts/train.py --config configs/tiny.yaml --data data/shards/sample \
        --stop-at-step 1000          # or --stop-by 2026-10-20T09:30:00+00:00

    # resume exactly where it stopped: the same config and overrides, or it
    # refuses (max_steps alone would change the schedule)
    python scripts/train.py --config configs/tiny.yaml --data data/shards/sample \
        --resume experiments/tiny-s1337/last.pt

    # adapt a pretrained checkpoint (weights only; fresh optimiser and
    # schedule) on general + personal shards, validating on both
    python scripts/train.py --config configs/experiments/e1_m50.yaml \
        --data $ROOT/shards/general $ROOT/shards/personal \
        --val-shard general=$ROOT/shards/general/val personal=$ROOT/shards/personal/val \
        --init-from $ROOT/checkpoints/p1-s1337/last.pt --run-name e1-m50-p1s1337

``scripts/run_e1.py`` writes those commands for E1's pilot and main runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minime.config import load_config, parse_cli_overrides  # noqa: E402
from minime.data.binidx import TokenShard, common_tokenizer  # noqa: E402
from minime.data.mixture import MixtureSampler, make_val_batches  # noqa: E402
from minime.model import build_model  # noqa: E402
from minime.tokenizer import is_dev_tokenizer  # noqa: E402
from minime.training.checkpoint import load_weights  # noqa: E402
from minime.training.inputs import parse_val_shards, resolve_shard  # noqa: E402
from minime.training.launch import PAUSED_EXIT  # noqa: E402
from minime.training.trainer import Trainer  # noqa: E402
from minime.utils.seed import seed_everything  # noqa: E402


def parse_mixture(text: str | None) -> dict[str, float] | None:
    """``"chat=7,longform=2,code=1"`` -> weights dict."""
    if not text:
        return None
    weights = {}
    for part in text.split(","):
        name, _, value = part.partition("=")
        weights[name.strip()] = float(value or 1.0)
    return weights


def parse_deadline(text: str) -> datetime:
    """An ISO 8601 time with its offset, e.g. ``2026-10-20T09:30:00+00:00``."""
    try:
        deadline = datetime.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an ISO 8601 time: {text!r}") from None
    if deadline.tzinfo is None:
        raise argparse.ArgumentTypeError(f"{text!r} has no UTC offset: a deadline must say whose clock")
    return deadline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, nargs="+", required=True,
                        help="directories of token shards; each mixture source must be in exactly one")
    parser.add_argument("--mixture", help="override the config mixture, e.g. chat=7,code=3")
    parser.add_argument("--val-shard", nargs="+", default=["val"], metavar="SHARD|NAME=PATH",
                        help="validation sets: a shard in the first data directory, or NAME=PATH "
                             "for several (the first is the run's val_loss)")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--experiment", default="", help="experiment label for costs.csv")
    parser.add_argument("--venue", default="local", help="local | kaggle-free | colab-pro | vast-4090 ...")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--init-from", type=Path, default=None,
                        help="start from this checkpoint's weights only: fresh optimiser, schedule "
                             "and sampler (an adaptation run); refused if its model config differs")
    parser.add_argument("--stop-at-step", type=int, default=None,
                        help="pause cleanly after this step (last.pt, manifest marked paused, exit 75)")
    parser.add_argument("--stop-by", type=parse_deadline, default=None, metavar="ISO_TIME",
                        help="pause cleanly after the first step to end past this time")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--set", dest="overrides", nargs="*", default=[],
                        help="config overrides, e.g. training.max_steps=200 model.n_layers=2")
    args = parser.parse_args()
    if args.init_from and args.resume:
        parser.error("--init-from starts a new run from a checkpoint's weights; --resume continues "
                     "a run, which keeps the start it was given: use one")

    cfg = load_config(args.config, overrides=parse_cli_overrides(args.overrides))
    if mixture := parse_mixture(args.mixture):
        cfg.data.mixture = mixture

    data_vocabs = {}
    for data_dir in args.data:
        manifest_path = data_dir / "manifest.json"
        if manifest_path.exists():
            if vocab := json.loads(manifest_path.read_text()).get("vocab_size"):
                data_vocabs[str(data_dir)] = vocab
    if len(set(data_vocabs.values())) > 1:
        raise SystemExit(f"error: the data directories disagree on vocab_size: {data_vocabs}")
    if data_vocabs:
        data_vocab = next(iter(data_vocabs.values()))
        if data_vocab != cfg.model.vocab_size:
            # The vocabulary is a property of the data, not a free parameter.
            print(f"vocab_size {cfg.model.vocab_size} -> {data_vocab} (from the data manifest)")
            cfg.model.vocab_size = data_vocab

    shards = {name: resolve_shard(args.data, name) for name in cfg.data.mixture}
    for name, shard in shards.items():
        print(f"  {name:12s} {shard.n_tokens:12,d} tokens")

    sampler = MixtureSampler(
        shards=shards,
        weights=cfg.data.normalised_mixture(),
        context_length=cfg.model.context_length,
        batch_size=cfg.training.batch_size,
        seed=cfg.data.seed,
    )

    val_shards = {name: TokenShard(path) for name, path in parse_val_shards(args.val_shard, args.data).items()}
    tokenizer_id = common_tokenizer({**shards, **{f"{name} (val)": s for name, s in val_shards.items()}})
    tokenizer_dev = is_dev_tokenizer(tokenizer_id)
    print(f"tokenizer {tokenizer_id or '<unrecorded>'}")
    if tokenizer_dev:
        print("NOTE: dev tokenizer -- this run is pipeline work, not an experiment result")
    val_batches = {
        name: make_val_batches(
            shard,
            context_length=cfg.model.context_length,
            batch_size=cfg.training.batch_size,
            n_batches=cfg.training.validation_batches,
        )
        for name, shard in val_shards.items()
    }

    run_name = args.run_name or cfg.run_name
    run_dir = args.run_dir or Path("experiments") / run_name

    # Seed before the model exists: its initial weights are the first random
    # numbers drawn, and torch seeds every process randomly at start-up.
    # (Until 2026-10-05 the Trainer seeded only after this line, so no run
    # before then had its initial weights set by training.seed.)
    seed_everything(cfg.training.seed)
    model = build_model(cfg.model)
    init_from = None
    if args.init_from:
        init_from = load_weights(args.init_from, model, cfg.to_dict()["model"])
        print(f"weights from {args.init_from} (step {init_from['step']}, "
              f"sha256 {init_from['sha256'][:12]}); fresh optimiser and schedule")
    trainer = Trainer(
        config=cfg,
        model=model,
        sampler=sampler,
        val_batches=val_batches,
        run_dir=run_dir,
        experiment=args.experiment or cfg.name,
        venue=args.venue,
        wandb_project=args.wandb_project,
        data_info={
            # One directory as before; a list when the mixture spans several.
            "data_dir": str(args.data[0]) if len(args.data) == 1 else [str(d) for d in args.data],
            "tokenizer": tokenizer_id,
            "tokenizer_dev": tokenizer_dev,
            "shards": {name: shard.n_tokens for name, shard in shards.items()},
            "val_shard": {name: shard.n_tokens for name, shard in val_shards.items()},
        },
        init_from=init_from,
    )
    if args.resume:
        trainer.resume(args.resume)

    summary = trainer.train(stop_at_step=args.stop_at_step, stop_by=args.stop_by)
    print(json.dumps(summary, indent=2, default=str))
    return PAUSED_EXIT if summary.get("paused") else 0


if __name__ == "__main__":
    raise SystemExit(main())
