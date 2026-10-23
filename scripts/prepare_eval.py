#!/usr/bin/env python3
"""Fetch the external eval data (BLiMP) once, from a pinned revision.

    MINIME_DATA_ROOT=/Volumes/BIWIN/minime python scripts/prepare_eval.py

Writes ``$MINIME_DATA_ROOT/eval/blimp.jsonl`` and prints its sha256, which is
pinned in ``minime.evaluation.tasks.blimp.PINNED_SHA256`` so the harness
refuses any other file.  Every other eval set is derived from the token
shards and needs no preparation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minime.evaluation.tasks import blimp  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=None, help="default: $MINIME_DATA_ROOT/eval/blimp.jsonl")
    args = parser.parse_args()

    out = args.out or blimp.default_path()
    summary = blimp.download(out)
    print(f"sha256 {summary['sha256']}")
    if blimp.PINNED_SHA256 is None:
        print("PINNED_SHA256 is unset: pin this hash in src/minime/evaluation/tasks/blimp.py")
    elif summary["sha256"] != blimp.PINNED_SHA256:
        print(f"ERROR: expected {blimp.PINNED_SHA256}; the pinned revision produced different data")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
