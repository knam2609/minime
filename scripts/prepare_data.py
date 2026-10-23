#!/usr/bin/env python3
"""Tokenise a corpus into training shards.

Two inputs, one output format.

**Documents** -- JSONL files of ``Document`` records, whatever produced them:
the Meta export parser, the long-form collector, the GitHub walker, or the
synthetic fixture.  Splits are by document, never by line.

    python scripts/prepare_data.py \\
        --docs data/sample_synthetic/sam.jsonl \\
        --out data/shards/sample --tokenizer char

    python scripts/prepare_data.py --docs data/personal/*.jsonl --by-source \\
        --out $MINIME_DATA_ROOT/shards/personal \\
        --tokenizer bpe --tokenizer-path tokenizer/minime-16k.json

**The general corpus** -- streamed from pinned Hugging Face datasets straight
into ``<slice>.{train,val,test}`` shards plus a combined ``val`` and ``test``.
Nothing raw is kept.  Finished slices are skipped on a re-run.

    python -u scripts/prepare_data.py --general \\
        --out $MINIME_DATA_ROOT/shards/general \\
        --tokenizer bpe --tokenizer-path tokenizer/minime-16k-dev.json

    # a smoke-sized run: any subset, any targets
    python -u scripts/prepare_data.py --general web=2M,dialogue=1M,code=1M ...
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minime.data.binidx import ShardWriter  # noqa: E402
from minime.data.document import read_jsonl  # noqa: E402
from minime.data.splits import check_no_leakage, describe_splits, split_documents  # noqa: E402
from minime.tokenizer import CharTokenizer, is_dev_tokenizer, load_tokenizer  # noqa: E402

log = functools.partial(print, flush=True)


def tokenizer_record(tokenizer, path: Path | None) -> dict:
    return {
        "tokenizer": tokenizer.name,
        "tokenizer_id": tokenizer.id,
        "tokenizer_fingerprint": tokenizer.fingerprint(),
        "tokenizer_dev": is_dev_tokenizer(tokenizer.id),
        "tokenizer_path": str(path) if path else None,
        "vocab_size": tokenizer.vocab_size,
        "eot_id": tokenizer.eot_id,
    }


def dev_banner(tokenizer) -> None:
    if is_dev_tokenizer(tokenizer.id):
        log(f"NOTE: {tokenizer.id} is a DEV tokenizer. Shards made with it are for pipeline work "
            "only and must be regenerated with the frozen tokenizer before any experiment.")


# ------------------------------------------------------------- documents


def prepare_documents(args: argparse.Namespace) -> int:
    docs = [doc for path in args.docs for doc in read_jsonl(path)]
    if not docs:
        log("no documents found")
        return 1
    log(f"loaded {len(docs):,} documents from {len(args.docs)} file(s)")

    ratios = {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio}
    splits = split_documents(docs, ratios, salt=args.salt)
    check_no_leakage(splits)

    args.out.mkdir(parents=True, exist_ok=True)
    if args.tokenizer == "char":
        tokenizer = CharTokenizer.fit(d.text for d in docs)
        tokenizer.save(args.out / "tokenizer.json")
        log(f"char tokenizer: {tokenizer.vocab_size} symbols")
    else:
        tokenizer = load_tokenizer(args.tokenizer_path)
        log(f"tokenizer: {tokenizer!r}")
        dev_banner(tokenizer)

    manifest: dict = {
        **tokenizer_record(tokenizer, args.tokenizer_path),
        "sources": [str(p) for p in args.docs],
        "salt": args.salt,
        "ratios": ratios,
        "splits": describe_splits(splits),
        "shards": {},
    }

    for split_name, split_docs in splits.items():
        groups: dict[str, list] = defaultdict(list)
        for doc in split_docs:
            key = doc.source if (args.by_source and split_name == "train") else split_name
            groups[key].append(doc)

        for key, group in groups.items():
            name = f"{key}.{split_name}" if args.by_source and split_name == "train" else key
            with ShardWriter(
                args.out / name,
                vocab_size=tokenizer.vocab_size,
                tokenizer=tokenizer.id,
                eot_id=tokenizer.eot_id,
                extra={"split": split_name, "source": key},
            ) as writer:
                writer.add_documents(tokenizer.encode_batch([doc.text for doc in group]))
                meta = writer.close()
            manifest["shards"][name] = meta.to_dict()
            log(f"  {name:24s} {meta.n_tokens:10,d} tokens  {meta.n_docs:6,d} docs")

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"manifest -> {args.out / 'manifest.json'}")
    return 0


# -------------------------------------------------------- general corpus


def prepare_general(args: argparse.Namespace) -> int:
    # Imported here so the documents path keeps working without `datasets`.
    from minime.data.ingest import (
        GENERAL_RATIOS, check_free_space, combine_shards, completed_slice, estimate_bytes,
        ingest_slice, slice_settings,
    )
    from minime.data.sources import GENERAL_SLICES, parse_targets

    targets = parse_targets(args.general)
    need = estimate_bytes(targets)
    check_free_space(args.out, need)
    args.out.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.tokenizer_path)
    log(f"tokenizer: {tokenizer!r}")
    dev_banner(tokenizer)
    log(f"general corpus -> {args.out}  (~{need / 2**30:.2f} GiB)")

    manifest: dict = {
        "kind": "general",
        **tokenizer_record(tokenizer, args.tokenizer_path),
        "ratios": GENERAL_RATIOS,
        "salt": "minime",
        "slices": {},
        "shards": {},
    }

    for name, target in targets.items():
        spec = GENERAL_SLICES[name]
        source = spec.source.describe()
        settings = slice_settings(source, tokenizer.id, target)
        summary = None if args.force else completed_slice(args.out, name, settings)
        if summary is not None:
            log(f"{name}: already complete with identical settings, skipping")
        else:
            log(f"{name}: streaming {spec.source.dataset}@{spec.source.revision[:10]} "
                f"to {target:,} train tokens")
            stats: Counter = Counter()
            summary = ingest_slice(
                spec.iter_documents(stats=stats), tokenizer, args.out, name, target,
                source=source, stats=stats, log=log,
            )
        manifest["slices"][name] = {**source, "purpose": spec.purpose, **summary}
        split_tokens = "  ".join(f"{s} {v['tokens']:,}" for s, v in summary["splits"].items())
        log(f"{name}: {split_tokens}  ({summary['seconds']:.0f}s)")

    for split in ("val", "test"):
        meta = combine_shards(
            [args.out / f"{name}.{split}" for name in targets], args.out / split, tokenizer.id,
            extra={"split": split, "slices": list(targets)},
        )
        log(f"combined {split}: {meta.n_tokens:,} tokens, {meta.n_docs:,} docs")

    stems = [f"{name}.{split}" for name in targets for split in ("train", "val", "test")]
    for stem in stems + ["val", "test"]:
        manifest["shards"][stem] = json.loads((args.out / f"{stem}.meta.json").read_text())
    manifest["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"manifest -> {args.out / 'manifest.json'}")

    short = [n for n, s in manifest["slices"].items() if s["exhausted"]]
    if short:
        log(f"WARNING: slice(s) {short} fell short of their targets; see the manifest.")
    return 0


# ------------------------------------------------------------------ main


def build_parser() -> argparse.ArgumentParser:
    from minime.data.sources import DEFAULT_TARGETS

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--docs", nargs="+", type=Path, help="JSONL files of Document records")
    what.add_argument("--general", nargs="?", const=DEFAULT_TARGETS, metavar="SLICE=TOKENS,...",
                      help=f"stream the general corpus (default {DEFAULT_TARGETS})")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    parser.add_argument("--tokenizer-path", type=Path, default=None,
                        help="saved BPE tokenizer (required with --tokenizer bpe)")
    parser.add_argument("--by-source", action="store_true",
                        help="write one training shard per document source, so the "
                             "mixture experiment can reweight them")
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--salt", default="minime", help="changes the split assignment")
    parser.add_argument("--force", action="store_true",
                        help="--general: redo slices even if already complete")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.tokenizer == "bpe" and args.tokenizer_path is None:
        parser.error("--tokenizer bpe needs --tokenizer-path")
    if args.general is not None:
        if args.tokenizer != "bpe":
            parser.error("--general needs --tokenizer bpe: a char vocabulary cannot be fitted to a stream")
        return prepare_general(args)
    return prepare_documents(args)


if __name__ == "__main__":
    raise SystemExit(main())
