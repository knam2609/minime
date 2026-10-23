#!/usr/bin/env python3
"""Count what personal writing actually exists, before designing around it.

    # what am I working with?
    python scripts/inventory_personal.py \
        --meta-export ~/Downloads/facebook-export \
        --meta-export ~/Downloads/instagram-export \
        --me "Thang Nguyen" \
        --longform ~/Documents/uni ~/Documents/notes \
        --github ~/Developer

    # then write the scrubbed corpus (gitignored) once the census looks right
    ... --write data/personal/corpus.jsonl

    # not sure what name the export uses for you? omit --me and it says
    python scripts/inventory_personal.py --meta-export ~/Downloads/instagram-export

    # suggest a names file for the scrubber from the export's participants
    ... --suggest-names personal_names.txt

Prints aggregates only.  Document text is never echoed unless --show-samples
is passed, and samples are scrubbed and truncated even then.  Nobody else's
name is ever printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minime.data.document import write_jsonl  # noqa: E402
from minime.data.inventory import (  # noqa: E402
    format_table, inventory, pii_audit, split_preview, verdict,
)
from minime.data.personal import (  # noqa: E402
    CommentsConfig, GitHubConfig, LongformConfig, MetaExportConfig,
    collect_github, collect_longform, parse_comments, parse_export,
)
from minime.data.personal.github_repos import discover_authors  # noqa: E402
from minime.data.personal.language import UNACCENTED, describe  # noqa: E402
from minime.data.personal.meta_export import (  # noqa: E402
    detect_platform, discover_me, participant_names,
)
from minime.data.scrub import ScrubConfig, names_file_lines, scrub_documents  # noqa: E402

WORDS_FILE = Path("/usr/share/dict/words")


def share_or_none(text: str) -> float | None:
    return None if text.strip().lower() == "none" else float(text)


def common_words(path: Path = WORDS_FILE) -> set[str]:
    """Ordinary words a name part might collide with: lowercase dictionary
    entries (capitalised ones are proper nouns) plus unaccented Vietnamese chat
    words, which the scrubber would otherwise redact from code-switched text."""
    words = set(UNACCENTED)
    if path.exists():
        words |= {w for w in path.read_text(encoding="utf-8").split() if w == w.lower()}
    else:
        print(f"  note: {path} not found; only Vietnamese chat words are flagged as common",
              flush=True)
    return words


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--meta-export", type=Path, action="append", default=[],
                   help="root of a Meta 'Download Your Information' export (repeatable)")
    p.add_argument("--me", action="append", default=[],
                   help="your display name as it appears in the export (repeatable)")
    p.add_argument("--longform", type=Path, nargs="*", default=[],
                   help="directories of essays, reports and notes")
    p.add_argument("--github", type=Path, default=None, help="directory containing your repos")
    p.add_argument("--emails", nargs="*", default=[],
                   help="git author emails; omit to see the ones found in your repos")
    p.add_argument("--names-file", type=Path, default=Path("personal_names.txt"),
                   help="other people's names to redact, one per line (gitignored)")
    p.add_argument("--gap-seconds", type=int, default=300)
    p.add_argument("--write", type=Path, default=None, help="write the scrubbed corpus as JSONL")
    p.add_argument("--out", type=Path, default=None, help="write the census as JSON")
    p.add_argument("--show-samples", type=int, default=0,
                   help="print N scrubbed, truncated samples per source")
    p.add_argument("--redact-long-numbers", action="store_true")
    p.add_argument("--git-timeout", type=int, default=30,
                   help="per-repository seconds before skipping (slow disks need more)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--no-comments", action="store_true",
                   help="skip comments and read only messages from the export")
    p.add_argument("--max-vietnamese-share", type=share_or_none, default=0.5,
                   help="drop a message or comment when more than this share of its words "
                        "is Vietnamese ('none' keeps everything)")
    p.add_argument("--suggest-names", type=Path, default=None, metavar="PATH",
                   help="write the export's other participants' names to PATH for review "
                        "as a names file, then stop (refuses to overwrite)")
    return p


def main() -> int:
    args = build_parser().parse_args()
    docs = []

    # ---------------------------------------------------------------- chat
    if args.meta_export:
        missing = [root for root in args.meta_export if not root.exists()]
        for root in missing:
            print(f"  meta export not found: {root}", file=sys.stderr)
        roots = [root for root in args.meta_export if root.exists()]

        if not args.me:
            # Mirrors --emails: say what the export calls the owner rather than
            # fail. Only the top sender is named; nobody else's name is printed.
            print("--me not given. The sender present in the most threads:\n")
            for root in roots:
                guess = discover_me(root)
                if guess.name is None:
                    print(f"    {root}: no messages found")
                    continue
                print(f"    {guess.name!r}  in {guess.threads:,} of {guess.total:,} threads "
                      f"(next most common sender: {guess.runner_up:,})   {root}")
            print("\nIf that is you, re-run with --me \"<that name>\".")
            return 2

        if args.suggest_names:
            if args.suggest_names.exists():
                print(f"{args.suggest_names} already exists; not overwriting it. Move it "
                      "aside or edit it by hand.", file=sys.stderr)
                return 2
            names = [n for root in roots
                     for n in participant_names(root, MetaExportConfig(me=args.me))]
            lines = names_file_lines(names, common_words())
            args.suggest_names.write_text("\n".join(lines) + "\n", encoding="utf-8")
            active = sum(1 for line in lines if line and not line.startswith("#"))
            flagged = sum(1 for line in lines if line.startswith("# common word:"))
            print(f"wrote {active:,} names and name parts, plus {flagged:,} commented out as "
                  f"common words, from {len(set(names)):,} participants -> {args.suggest_names}")
            print("Review it (it is gitignored), then re-run without --suggest-names; the "
                  "scrubber reads it through --names-file.")
            return 0

        for root in roots:
            platform = detect_platform(root)
            chat_cfg = MetaExportConfig(
                me=args.me, gap_seconds=args.gap_seconds, platform=platform,
                max_vietnamese_share=args.max_vietnamese_share,
            )
            found = parse_export(root, chat_cfg)
            print(f"  {platform:11s} {len(found):7,d} utterances from {root}", flush=True)
            if chat_cfg.language_counts:
                print(f"    messages: {describe(chat_cfg.language_counts)}", flush=True)
            if not found:
                print(f"    no messages from {args.me} -- omit --me to see the name this "
                      "export uses for you.", flush=True)
            docs.extend(found)

            if not args.no_comments:
                comment_cfg = CommentsConfig(
                    me=args.me, platform=platform,
                    max_vietnamese_share=args.max_vietnamese_share,
                )
                comments = parse_comments(root, comment_cfg)
                print(f"  {platform:11s} {len(comments):7,d} comments", flush=True)
                if comment_cfg.language_counts:
                    print(f"    comments: {describe(comment_cfg.language_counts)}", flush=True)
                if not comments:
                    print("    no comments parsed. If the export contains them, run "
                          f"`python scripts/inspect_export.py {root}` -- it prints the "
                          "schema without printing any text.", flush=True)
                docs.extend(comments)

    # ------------------------------------------------------------ longform
    if args.longform:
        found = collect_longform(args.longform, LongformConfig())
        print(f"  longform    {len(found):7,d} documents")
        docs.extend(found)

    # -------------------------------------------------------------- github
    if args.github:
        if not args.emails:
            authors = discover_authors(
                args.github, timeout=args.git_timeout, verbose=args.verbose
            )
            print("\n--emails not given. Author addresses found in your repositories:\n")
            for email, count in authors:
                print(f"    {count:6,d} commits   {email}")
            print("\nNote: GitHub rewrites commit addresses to "
                  "<id>+<user>@users.noreply.github.com when email privacy is on,\n"
                  "so pick the address that actually appears above, then re-run with "
                  "--emails <address>.")
        else:
            found = collect_github(args.github, GitHubConfig(
                emails=args.emails, git_timeout=args.git_timeout, verbose=args.verbose,
            ))
            code = sum(1 for d in found if d.source == "code")
            print(f"  github      {code:7,d} files + {len(found) - code:,} commit messages")
            docs.extend(found)

    if not docs:
        print("\nno documents collected.", file=sys.stderr)
        return 1

    # --------------------------------------------------------------- scrub
    scrub_cfg = ScrubConfig.from_names_file(
        args.names_file, long_numbers=args.redact_long_numbers
    )
    if not scrub_cfg.names:
        print(f"\nnote: no names loaded from {args.names_file}; other people's names "
              "will not be redacted. Create that file (one name per line) to enable it.")
    docs, redactions = scrub_documents(docs, scrub_cfg)

    # -------------------------------------------------------------- report
    reports = inventory(docs)
    total_tokens = sum(r.est_tokens for r in reports.values())
    print("\n" + format_table(reports))
    print(f"\nredactions: {dict(redactions) or 'none'}")
    print(f"splits (by document): {split_preview(docs)}")
    print(f"splits (by group):    {split_preview(docs, groups=True)}  "
          "(a group is one month of one conversation, assigned as a unit)")
    audit = pii_audit(docs)
    print(f"post-scrub PII audit: {audit['documents_flagged']}/{audit['documents_checked']} "
          "documents still match an obvious-PII pattern")
    print(f"\n{verdict(total_tokens)}")

    if args.show_samples:
        for source in reports:
            print(f"\n--- {source} samples (scrubbed, truncated) ---")
            shown = 0
            for doc in docs:
                if doc.source != source:
                    continue
                print(f"  {doc.text[:160].replace(chr(10), ' / ')!r}")
                shown += 1
                if shown >= args.show_samples:
                    break

    if args.write:
        count = write_jsonl(args.write, docs)
        print(f"\nwrote {count:,} scrubbed documents -> {args.write}  (gitignored)")

    if args.out:
        payload = {
            "sources": {name: vars(r) | {"est_tokens": r.est_tokens} for name, r in reports.items()},
            "total_est_tokens": total_tokens,
            "redactions": dict(redactions),
            "splits": split_preview(docs),
            "split_groups": split_preview(docs, groups=True),
            "pii_audit": audit,
            "verdict": verdict(total_tokens),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"census -> {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
