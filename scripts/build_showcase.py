#!/usr/bin/env python3
"""Write the showcase page's numbers, ``docs/data.js``, from the committed reports.

    python scripts/build_showcase.py            # -> docs/data.js
    python scripts/build_showcase.py --check    # exit 1 if docs/data.js is not what the reports give

The page (``docs/index.html``) reads only this file and, for the samples, a
public ``samples.json`` outside the repository (``scripts/showcase_samples.py``).
Numbers only: no text from any corpus or sample.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from minime.evaluation.showcase import page_data, render_data_js  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO / "docs" / "data.js")
    parser.add_argument("--check", action="store_true", help="compare with --out instead of writing it")
    args = parser.parse_args()
    text = render_data_js(page_data(REPO))
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.exists() else None
        print(f"{args.out}: {'up to date' if current == text else 'stale: run scripts/build_showcase.py'}")
        return 0 if current == text else 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"reports -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
