"""Collector for long-form writing: essays, reports, notes, drafts.

The smallest source by volume and the most valuable per token: it is the only
place the corpus sees me writing in complete paragraphs.  A model trained only
on chat learns that nothing is ever longer than a line.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from ..document import Document, make_id

TEXT_EXTENSIONS = {".md", ".txt", ".markdown", ".tex", ".rst", ".org"}
SKIP_DIRS = {
    ".git", ".obsidian", "node_modules", "__pycache__", ".venv", "venv",
    ".trash", "templates", ".quarto",
}


@dataclass
class LongformConfig:
    extensions: set[str] = field(default_factory=lambda: set(TEXT_EXTENSIONS))
    min_chars: int = 200
    max_bytes: int = 2_000_000
    skip_dirs: set[str] = field(default_factory=lambda: set(SKIP_DIRS))


def _should_skip(path: Path, cfg: LongformConfig) -> bool:
    return any(part in cfg.skip_dirs or part.startswith(".") for part in path.parts[:-1])


def iter_longform(roots, cfg: LongformConfig | None = None) -> Iterator[Document]:
    cfg = cfg or LongformConfig()
    if isinstance(roots, (str, Path)):
        roots = [roots]

    for root in roots:
        root = Path(root).expanduser()
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in cfg.extensions:
                continue
            relative = path.relative_to(root)
            if _should_skip(relative, cfg):
                continue
            try:
                if path.stat().st_size > cfg.max_bytes:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore").strip()
            except OSError:
                continue
            if len(text) < cfg.min_chars:
                continue
            yield Document(
                id=make_id("longform", str(relative)),
                source="longform",
                text=text,
                author="me",
                # The path is kept as a stem only: directory names are often
                # course codes, employers or people.
                meta={"extension": path.suffix.lower(), "chars": len(text)},
            )


def collect_longform(roots, cfg: LongformConfig | None = None) -> list[Document]:
    return list(iter_longform(roots, cfg))
