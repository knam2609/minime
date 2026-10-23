#!/usr/bin/env python3
"""Check downloaded checkpoints against the Kaggle session that wrote them.

    # results.tar.gz unpacked in the scratchpad: its run directory holds the
    # session's SHA256SUMS and manifest.json
    python scripts/verify_download.py /Volumes/BIWIN/minime/checkpoints/<run>/last.pt \
        --run-dir <scratch>/experiments/<run>

    # the browser saved last.pt as "last (1).zip": the same bytes
    python scripts/verify_download.py "/Volumes/BIWIN/minime/checkpoints/last (1).zip" \
        --run-dir <scratch>/experiments/<run> --name last.pt

Verified = the sha256 matches the one the session wrote, the file loads, its
config (seeds included) is the run's, and its step is the run's last
(``last.pt``) or the one in its name (``step_<N>.pt``).  Only then may the
cloud copy of a chat-trained checkpoint be deleted (HANDOFF §4a).  Prints
numbers and hashes only; exits 1 on the first file that fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minime.training.launch import LaunchError, verify_download  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", type=Path, nargs="+", help="downloaded checkpoint files")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="the run's directory from results.tar.gz (SHA256SUMS, manifest.json)")
    parser.add_argument("--name", default=None,
                        help="the file's name in the session's output, when the download renamed it")
    args = parser.parse_args()
    if args.name and len(args.checkpoints) > 1:
        parser.error("--name names one file")

    for path in args.checkpoints:
        try:
            report = verify_download(path, args.run_dir, name=args.name)
        except (LaunchError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"verified {report['name']} of {report['run']} ({report['run_status']}): step {report['step']}, "
              f"sha256 {report['sha256'][:16]}, config {report['config_fingerprint']}, seeds {report['seeds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
