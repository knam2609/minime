#!/usr/bin/env python3
"""Sample from a checkpoint, for looking at what a model writes.

    python scripts/generate.py --checkpoint experiments/tiny-s1337/last.pt \\
        --tokenizer tokenizer/minime-16k.json --prompt "honestly" --n 5

The prompt starts a new document (it is preceded by ``<|endoftext|>``), and
each sample stops at the next one.  Sampling is the harness's: seeded per
sample, so ``--seed`` reproduces a set of samples exactly.  A model trained on
the personal corpus can reproduce it, so treat what this prints as personal
data.  Metrics come from ``scripts/evaluate.py``, never from reading samples.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minime.evaluation.generation import sample  # noqa: E402
from minime.evaluation.harness import load_model  # noqa: E402
from minime.tokenizer import load_tokenizer  # noqa: E402
from minime.utils.device import autocast_ctx, resolve  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    spec = resolve(args.device)
    tokenizer = load_tokenizer(args.tokenizer)
    model, info = load_model(args.checkpoint, spec.device)
    if model.cfg.vocab_size != tokenizer.vocab_size:
        parser.error(f"model vocab {model.cfg.vocab_size} != tokenizer vocab {tokenizer.vocab_size}")

    prompt = [tokenizer.eot_id, *tokenizer.encode(args.prompt)]
    gen = sample(
        model, np.array([prompt] * args.n), args.max_new_tokens, tokenizer.eot_id,
        temperature=args.temperature, top_k=args.top_k or None, seed=args.seed,
        device=spec.device, autocast=lambda: autocast_ctx(spec),
    )
    print(f"# {args.checkpoint} step {info['step']} | T={args.temperature} top-k={args.top_k} seed={args.seed}")
    for i, (tokens, ended) in enumerate(zip(gen.tokens, gen.ended)):
        print(f"--- {i}{'' if ended else ' (cut at --max-new-tokens)'}")
        print(args.prompt + tokenizer.decode(tokens))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
