#!/usr/bin/env python3
"""Hash a shard directory before it leaves the machine, or check it after.

    # on the laptop, before uploading shards/general as a Kaggle dataset
    python scripts/checksum_shards.py $MINIME_DATA_ROOT/shards/general

    # anywhere, later: is this directory still exactly what was hashed?
    python scripts/checksum_shards.py --verify /kaggle/input/minime-general

Writes ``SHA256SUMS`` into the directory, in ``sha256sum`` format.  The Kaggle
launcher verifies it before any GPU time is spent, so a truncated or partial
upload fails at the start rather than training on different data from the
data the laptop evaluates against.  Re-run it whenever the shards change.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minime.training.launch import LaunchError, verify_checksums, write_checksums  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--verify", action="store_true", help="check instead of writing")
    args = parser.parse_args()

    try:
        if args.verify:
            print(f"{verify_checksums(args.data_dir)} files match {args.data_dir}/SHA256SUMS")
        else:
            path = write_checksums(args.data_dir)
            n = sum(1 for line in path.read_text().splitlines() if line.strip())
            print(f"{n} files hashed -> {path}")
    except LaunchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
