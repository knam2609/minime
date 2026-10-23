#!/usr/bin/env python3
"""Stage the owner's chat train split for P4's private Kaggle dataset.

    # on the laptop, once, before the P4 session
    python scripts/stage_personal_upload.py
    # -> $MINIME_DATA_ROOT/upload/personal-chat-train: upload that directory,
    #    SHA256SUMS included, as a *private* Kaggle dataset

P4 mixes the owner's chat into a pretrain on Kaggle (decided with the owner
2026-10-13), so ``chat.train`` -- and only it -- leaves the laptop: the
personal val and test splits, the comment shard and the tokenizer stay.  The
staged directory holds the shard's three files, a manifest of kind
``personal`` (the notebook finds it by that) and ``SHA256SUMS``, which the
launcher verifies before any GPU time.  Tokenised text is still recoverable
text: delete the dataset once the session's checkpoints are home.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minime.config import data_root  # noqa: E402
from minime.training.launch import LaunchError, stage_upload  # noqa: E402

SHARDS = ("chat.train",)
KIND = "personal"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=None, help="default: $MINIME_DATA_ROOT/shards/personal")
    parser.add_argument("--out", type=Path, default=None,
                        help="a new directory; default: $MINIME_DATA_ROOT/upload/personal-chat-train")
    args = parser.parse_args()
    source = args.source or data_root() / "shards" / "personal"
    out = args.out or data_root() / "upload" / "personal-chat-train"

    try:
        stage_upload(source, out, shards=SHARDS, kind=KIND)
    except LaunchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for path in sorted(out.iterdir()):
        print(f"  {path.name:24s} {path.stat().st_size:>10,d} bytes")
    print(f"{out}: upload this directory as a private Kaggle dataset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
