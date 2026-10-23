#!/usr/bin/env python3
"""Train the byte-level BPE tokenizer.

The tokenizer is a controlled variable: trained once, frozen, shared by every
experiment.  The personal corpus does not exist yet, so there are two stages,
and this script refuses to blur them:

1. **Dev** (now) -- general corpus + synthetic fixture, to unblock the
   pipeline.  Must be named ``*-dev.json``.

       python -u scripts/train_tokenizer.py --out tokenizer/minime-16k-dev.json \\
           --docs data/sample_synthetic/sam.jsonl data/sample_synthetic/control.jsonl

2. **Frozen** (once, when the personal export lands, before any experiment) --
   the same plus the personal corpus.  ``--final`` requires personal documents
   and refuses to overwrite an existing frozen tokenizer.

       python -u scripts/train_tokenizer.py --final --out tokenizer/minime-16k.json \\
           --docs data/personal/*.jsonl

The sample is deterministic given (sources, seed): only train-split documents
are eligible -- the tokenizer never sees held-out text -- and the seed picks a
subset of them by hashing.
"""

from __future__ import annotations

import argparse
import functools
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minime.data.document import read_jsonl  # noqa: E402
from minime.data.ingest import sample_documents  # noqa: E402
from minime.data.splits import DEFAULT_RATIOS, assign_split, split_key  # noqa: E402
from minime.tokenizer.bpe import DEFAULT_VOCAB_SIZE, BPETokenizer  # noqa: E402

log = functools.partial(print, flush=True)


def parse_chars(text: str) -> int:
    scale = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    text = text.strip().upper()
    return int(float(text[:-1]) * scale[text[-1]]) if text[-1:] in scale else int(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--docs", nargs="*", type=Path, default=[],
                        help="JSONL Document files: fixture now, personal corpus for --final")
    parser.add_argument("--general-chars", default="100M",
                        help="characters to stream from the general slices, split by their "
                             "token targets (0 to skip)")
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-fraction", type=float, default=0.5,
                        help="share of eligible general documents the seed keeps")
    parser.add_argument("--final", action="store_true",
                        help="train the frozen tokenizer (requires personal documents)")
    parser.add_argument("--overwrite", action="store_true",
                        help="allow replacing an existing frozen tokenizer")
    args = parser.parse_args(argv)

    name = args.out.name.removesuffix(".json")
    if args.final and name.endswith("-dev"):
        parser.error("a frozen tokenizer must not be named *-dev")
    if not args.final and not name.endswith("-dev"):
        parser.error(f"a dev tokenizer must be named *-dev.json (got {args.out.name}); "
                     "pass --final only for the frozen tokenizer")
    if args.out.exists() and not args.overwrite:
        existing = BPETokenizer.load(args.out)
        if not existing.dev:
            parser.error(f"{args.out} is a frozen tokenizer. Replacing it invalidates every "
                         "shard and run made with it; pass --overwrite if that is intended.")
        log(f"replacing dev tokenizer {existing.id}; shards made with it are now stale")

    texts: list[str] = []
    groups: dict[str, list[str]] = {}
    provenance: dict = {
        "seed": args.seed,
        "vocab_size": args.vocab_size,
        "keep_fraction": args.keep_fraction,
        "general": {},
        "docs": {},
    }

    # Local documents: every train-split document, under the personal split rule
    # (by split group, so no conversation shapes merges and is then held out).
    has_personal = False
    for path in args.docs:
        kept = [d for d in read_jsonl(path) if assign_split(split_key(d), DEFAULT_RATIOS) == "train"]
        has_personal |= any(d.author == "me" for d in kept)
        groups[str(path)] = [d.text for d in kept]
        provenance["docs"][str(path)] = {"docs": len(kept), "chars": sum(len(d.text) for d in kept)}
        log(f"{path}: {len(kept):,} train documents, {provenance['docs'][str(path)]['chars']:,} chars")
    if args.final and not has_personal:
        parser.error("--final needs personal documents (author 'me') in --docs; "
                     "the frozen tokenizer exists to include them")

    general_chars = parse_chars(args.general_chars)
    if general_chars > 0:
        from minime.data.sources import GENERAL_SLICES

        total = sum(s.target_tokens for s in GENERAL_SLICES.values())
        for spec in GENERAL_SLICES.values():
            budget = general_chars * spec.target_tokens // total
            started = time.monotonic()
            log(f"{spec.name}: streaming {budget:,} chars from {spec.source.dataset}")
            sample = [d.text for d in sample_documents(
                spec.iter_documents(), budget, seed=args.seed, keep_fraction=args.keep_fraction)]
            groups[spec.name] = sample
            chars = sum(map(len, sample))
            provenance["general"][spec.name] = {
                **spec.source.describe(), "docs": len(sample), "chars": chars,
            }
            log(f"  {len(sample):,} documents, {chars:,} chars ({time.monotonic() - started:.0f}s)")

    for group in groups.values():
        texts.extend(group)
    if not texts:
        parser.error("no training text: pass --docs and/or a positive --general-chars")

    log(f"training {args.vocab_size:,}-token BPE on {len(texts):,} documents, "
        f"{sum(map(len, texts)):,} chars")
    started = time.monotonic()
    provenance["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tokenizer = BPETokenizer.train(
        texts, vocab_size=args.vocab_size, name=name, dev=not args.final, provenance=provenance,
    )
    log(f"trained in {time.monotonic() - started:.0f}s: {tokenizer!r}")

    # Compression per source, measured on the training sample itself -- a
    # sanity check (code and chat should compress worse than web prose), not
    # a held-out result.
    for label, group in groups.items():
        if not group:
            continue
        chars = sum(map(len, group))
        tokens = sum(map(len, tokenizer.encode_batch(group)))
        log(f"  {label:48s} {chars / max(tokens, 1):5.2f} chars/token")

    tokenizer.save(args.out)
    log(f"saved -> {args.out}  (fingerprint {tokenizer.fingerprint()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
