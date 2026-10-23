#!/usr/bin/env python3
"""Build the public ``samples.json`` from the chosen samples, if every one passes the gate.

    python scripts/showcase_samples.py chosen.json
    python scripts/showcase_samples.py chosen.json --out /Volumes/BIWIN/minime/showcase/samples.json

``chosen.json`` is kept outside the repository: a list of
``{"file": <run>/samples/eval-last.jsonl or eval-last-test.jsonl, "row": <line index>}``,
from the flagship's and the final model's samples files only.  Each row's
generated text is run through the showcase gate (``scripts/showcase_gate.py``:
a run shared with ``chat.train`` of at most the threshold, nothing for the
scrubber); one failure and nothing is written.  The output holds the model,
split, family and generated text of each, never a prompt or a document index,
and goes outside the repository (default ``$MINIME_DATA_ROOT/showcase/
samples.json``), to be uploaded as the page's public object.  Prints numbers
and pass/fail only, never text.  The gate is a floor, not a judgement of
what a sample says (README *The page*).
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
from minime.evaluation.showcase import chosen_rows, samples_payload  # noqa: E402
from minime.tokenizer import load_tokenizer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("chosen", type=Path, help="the choices: [{\"file\": ..., \"row\": ...}, ...]")
    parser.add_argument("--out", type=Path, default=None,
                        help="default: $MINIME_DATA_ROOT/showcase/samples.json")
    parser.add_argument("--report", type=Path, default=REPO / "reports" / "memorisation" / "memorisation.json")
    parser.add_argument("--names", type=Path, default=REPO / "personal_names.txt")
    parser.add_argument("--root", type=Path, default=None, help="data root (default: $MINIME_DATA_ROOT)")
    args = parser.parse_args()
    root = args.root or data_root()
    out = (args.out or root / "showcase" / "samples.json").resolve()
    if out.is_relative_to(REPO):
        print(f"error: {out} is inside the repository: samples are never committed", file=sys.stderr)
        return 1
    threshold = json.loads(args.report.read_text())["gate"]["threshold"]
    try:
        rows = chosen_rows(json.loads(args.chosen.read_text(encoding="utf-8")))
        tokenizer = load_tokenizer(REPO / "tokenizer" / "minime-16k.json")
        index = RunIndex(TokenShard(root / "shards" / "personal" / "chat.train").tokens)
        results = gate([r["text"] for r in rows], tokenizer.encode, index, threshold, args.names)
        print(f"threshold: a shared run of at most {threshold:.0f} tokens; the scrubber finds nothing")
        for r, row in zip(results, rows):
            found = ", ".join(f"{k} {v}" for k, v in sorted(r["scrubber"].items())) or "nothing"
            print(f"  sample {r['sample']} ({row['model']}, {row['split']}, {row['family']}): "
                  f"{'PASS' if r['pass'] else 'FAIL'}  longest shared run {r['longest_run']}  scrubber: {found}")
        payload = samples_payload(rows, results, threshold)
    except MemorisationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(rows)} of {len(rows)} pass -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
