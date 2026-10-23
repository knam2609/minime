#!/usr/bin/env python3
"""Print the *structure* of a Meta export, never its contents.

The comment parser is written against remembered schemas, and Meta changes
those without notice.  This script makes validating it a two-minute job that
requires nobody to read anybody's messages:

    python scripts/inspect_export.py ~/path/to/facebook-export

For each JSON file it reports the key paths, the value *types*, record counts,
and which parser shape matched -- and nothing else.  Strings are reported as
`str(len=42)`.  No message text, name, email or URL is ever printed, so the
output is safe to paste into a chat or an issue.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minime.data.personal.meta_comments import (  # noqa: E402
    _extract, _records, find_comment_files, looks_like_comment_record,
)
from minime.data.personal.meta_export import find_thread_files, is_appledouble  # noqa: E402

MAX_DEPTH = 4


def shape(value, depth: int = 0) -> object:
    """A type-only description of a JSON value. Never returns a real string."""
    if depth >= MAX_DEPTH:
        return "..."
    if isinstance(value, dict):
        return {key: shape(val, depth + 1) for key, val in list(value.items())[:12]}
    if isinstance(value, list):
        if not value:
            return "[]"
        return [shape(value[0], depth + 1), f"...x{len(value)}"]
    if isinstance(value, str):
        return f"str(len={len(value)})"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return type(value).__name__


def inspect_comment_file(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"file": path.name, "error": type(exc).__name__}

    records = _records(payload)
    parsed = [r for r in records if _extract(r) is not None]
    # A reaction entry has a title and a timestamp and no text; it is a correct
    # skip, not a parse failure, and counting it as one produces a false
    # "schema has drifted" alarm.
    unrecognised = [r for r in records if _extract(r) is None and looks_like_comment_record(r)]
    no_text = len(records) - len(parsed) - len(unrecognised)
    root_keys = sorted(payload)[:12] if isinstance(payload, dict) else "[list]"

    report = {
        "file": str(path.name),
        "root": "list" if isinstance(payload, list) else "dict",
        "root_keys": root_keys,
        "records_found": len(records),
        "parsed_as_comments": len(parsed),
        "skipped_no_text": no_text,
        "unrecognised_with_text": len(unrecognised),
        "first_record_shape": shape(parsed[0] if parsed else records[0]) if records else None,
    }
    if unrecognised:
        report["unrecognised_shape"] = shape(unrecognised[0])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="root of the unzipped export")
    parser.add_argument("--all-json", action="store_true",
                        help="list every JSON file found, not just comment files")
    args = parser.parse_args()

    if not args.root.exists():
        print(f"not found: {args.root}", file=sys.stderr)
        return 1

    print(f"export root: {args.root}\n")

    threads = find_thread_files(args.root)
    print(f"MESSAGES: {len(threads)} thread file(s)")
    if threads:
        try:
            payload = json.loads(threads[0].read_text(encoding="utf-8"))
            messages = payload.get("messages") or []
            print(f"  keys: {sorted(payload)[:12]}")
            print(f"  messages in first thread: {len(messages)}")
            if messages:
                print(f"  message shape: {json.dumps(shape(messages[0]), indent=4)}")
        except Exception as exc:
            print(f"  could not read first thread: {type(exc).__name__}")
    else:
        print("  none found -- check the export is JSON, not HTML")

    comment_files = find_comment_files(args.root)
    print(f"\nCOMMENTS: {len(comment_files)} candidate file(s)")
    total, parsed, unrecognised = 0, 0, 0
    for path in comment_files:
        report = inspect_comment_file(path)
        total += report.get("records_found", 0) or 0
        parsed += report.get("parsed_as_comments", 0) or 0
        unrecognised += report.get("unrecognised_with_text", 0) or 0
        print(f"\n  {json.dumps(report, indent=4)}")

    if total:
        print(f"\n  {parsed} comment(s) parsed of {total} record(s); "
              f"{unrecognised} record(s) carry text but matched no known schema")
        if unrecognised:
            print("  -> schema drift: update meta_comments._extract using "
                  "'unrecognised_shape' above")
        else:
            print("  -> no drift detected; the remaining records are reactions "
                  "or metadata, which are correctly skipped")
    elif comment_files:
        print("\n  no records found in any candidate file")

    if args.all_json:
        counts: Counter = Counter()
        for path in sorted(args.root.rglob("*.json")):
            if is_appledouble(path):
                continue
            counts[path.parent.relative_to(args.root).as_posix()] += 1
        print("\nALL JSON, by directory:")
        for directory, count in sorted(counts.items()):
            print(f"  {count:5d}  {directory}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
