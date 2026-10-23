#!/usr/bin/env python3
"""The showcase gate: may these chosen samples go on the public page?

    python scripts/showcase_gate.py chosen.jsonl     # one JSON object per line, with "text"

For each sample, in file order: the longest token run it shares with
``chat.train`` and what the corpus's scrubber (with ``personal_names.txt``)
finds in it, against the rule fixed before the memorisation check ran
(README "Memorisation check -- protocol"): a run no longer than the real
held-out continuations' 95th percentile (``reports/memorisation/
memorisation.json``), and nothing for the scrubber.  Prints numbers and
pass/fail only, never text.  Exit 0 when every sample passes, 1 otherwise.
The gate is a floor, not a judgement of what a sample says (README *The page*).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from minime.config import data_root  # noqa: E402
from minime.data.binidx import TokenShard  # noqa: E402
from minime.evaluation.memorisation import MemorisationError, RunIndex, gate  # noqa: E402
from minime.tokenizer import load_tokenizer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("samples", type=Path, help="JSONL, one chosen sample per line, each with a 'text'")
    parser.add_argument("--report", type=Path, default=REPO / "reports" / "memorisation" / "memorisation.json")
    parser.add_argument("--names", type=Path, default=REPO / "personal_names.txt")
    parser.add_argument("--root", type=Path, default=None, help="data root (default: $MINIME_DATA_ROOT)")
    args = parser.parse_args()
    if not args.report.exists():
        print(f"error: no {args.report}: run the memorisation check first", file=sys.stderr)
        return 1
    threshold = json.loads(args.report.read_text())["gate"]["threshold"]
    texts = [json.loads(line)["text"] for line in args.samples.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    tokenizer = load_tokenizer(REPO / "tokenizer" / "minime-16k.json")
    index = RunIndex(TokenShard((args.root or data_root()) / "shards" / "personal" / "chat.train").tokens)
    try:
        results = gate(texts, tokenizer.encode, index, threshold, args.names)
    except MemorisationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"threshold: a shared run of at most {threshold:.0f} tokens; the scrubber finds nothing")
    for r in results:
        found = ", ".join(f"{k} {v}" for k, v in sorted(r["scrubber"].items())) or "nothing"
        print(f"  sample {r['sample']}: {'PASS' if r['pass'] else 'FAIL'}  longest shared run "
              f"{r['longest_run']}  scrubber: {found}")
    passed = sum(r["pass"] for r in results)
    print(f"{passed} of {len(results)} pass")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
