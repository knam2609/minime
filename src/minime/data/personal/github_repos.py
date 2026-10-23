"""Collector for my own code and commit messages.

A repository directory is not the same thing as "code I wrote": it also holds
dependencies, generated files, other people's contributions and datasets.
Training on all of it would model the average of GitHub rather than me.

So authorship is established from git history -- a file is included only when
most of the commits touching it are mine -- and a series of filters removes
vendored, generated and data-shaped files.
"""

from __future__ import annotations

import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from ..document import Document, make_id

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cc", ".cpp", ".h", ".hpp",
    ".go", ".rs", ".rb", ".sql", ".sh", ".r", ".jl", ".scala", ".kt", ".swift", ".m",
}
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env", "dist", "build",
    "target", "vendor", "migrations", ".next", ".pytest_cache", "site-packages",
    "data", "datasets", "assets", "static", "fixtures", "third_party", ".ipynb_checkpoints",
}
SKIP_FILE_HINTS = ("lock", ".min.", "generated", "_pb2", "bundle")


@dataclass
class GitHubConfig:
    emails: list[str]
    extensions: set[str] = field(default_factory=lambda: set(CODE_EXTENSIONS))
    min_authorship: float = 0.5
    max_file_bytes: int = 100_000
    max_bytes_per_repo: int = 2_000_000
    min_chars: int = 80
    include_commits: bool = True
    min_commit_chars: int = 15
    skip_dirs: set[str] = field(default_factory=lambda: set(SKIP_DIRS))
    # Reading history is I/O bound, and a repo with many loose objects on a
    # nearly-full disk can take tens of seconds for a few dozen commits. A
    # bounded wait keeps one degraded repo from stalling the whole scan.
    git_timeout: int = 30
    verbose: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.emails, str):
            self.emails = [self.emails]
        self._emails_lower = {e.strip().lower() for e in self.emails}

    def is_me(self, email: str) -> bool:
        return email.strip().lower() in self._emails_lower


class GitTimeout(RuntimeError):
    """Raised internally when a repository takes too long to read."""


def _git(repo: Path, args: list[str], timeout: int = 30) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True,
            timeout=timeout, errors="ignore",
        )
        return result.stdout if result.returncode == 0 else ""
    except subprocess.TimeoutExpired:
        raise GitTimeout(f"git {args[0]} timed out after {timeout}s in {repo.name}") from None
    except OSError:
        return ""


def find_repos(root: str | Path, max_depth: int = 2) -> list[Path]:
    root = Path(root).expanduser()
    if (root / ".git").is_dir():
        return [root]
    repos = []
    for path in sorted(root.iterdir()) if root.exists() else []:
        if path.is_dir() and (path / ".git").is_dir():
            repos.append(path)
        elif path.is_dir() and max_depth > 1 and not path.name.startswith("."):
            repos.extend(find_repos(path, max_depth - 1))
    return repos


def file_authorship(repo: Path, cfg: GitHubConfig) -> dict[str, tuple[int, int]]:
    """Map each path to ``(commits_by_me, commits_total)``.

    One ``git log`` pass for the whole repository rather than one per file,
    which matters on repos with thousands of commits.
    """
    log = _git(repo, ["log", "--no-merges", "--format=%x01%ae", "--name-only"], cfg.git_timeout)
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    mine = False
    for line in log.splitlines():
        if line.startswith("\x01"):
            mine = cfg.is_me(line[1:])
        elif line.strip():
            entry = counts[line.strip()]
            entry[1] += 1
            if mine:
                entry[0] += 1
    return {path: (value[0], value[1]) for path, value in counts.items()}


def _looks_like_data(text: str) -> bool:
    """Reject minified bundles and CSV-ish blobs that happen to have a code suffix."""
    lines = text.splitlines() or [""]
    if max(len(line) for line in lines) > 1000:
        return True
    if len(lines) > 5 and sum(len(line) for line in lines) / len(lines) > 400:
        return True
    letters = sum(c.isalpha() for c in text)
    return letters / max(1, len(text)) < 0.35


def iter_repo_code(repo: Path, cfg: GitHubConfig) -> Iterator[Document]:
    authorship = file_authorship(repo, cfg)
    if not authorship:
        return
    repo_id = make_id("repo", repo.name)
    budget = cfg.max_bytes_per_repo

    for relative, (mine, total) in sorted(authorship.items()):
        if total == 0 or mine / total < cfg.min_authorship:
            continue
        path = repo / relative
        if path.suffix.lower() not in cfg.extensions or not path.is_file():
            continue
        parts = Path(relative).parts
        if any(part in cfg.skip_dirs for part in parts):
            continue
        if any(hint in Path(relative).name.lower() for hint in SKIP_FILE_HINTS):
            continue
        try:
            if path.stat().st_size > cfg.max_file_bytes:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if len(text) < cfg.min_chars or _looks_like_data(text):
            continue
        if budget - len(text) < 0:
            break
        budget -= len(text)

        yield Document(
            id=make_id(repo_id, relative),
            source="code",
            text=text,
            author="me",
            meta={
                "repo": repo_id,
                "extension": path.suffix.lower(),
                "authorship": round(mine / total, 3),
            },
        )


def iter_repo_commits(repo: Path, cfg: GitHubConfig) -> Iterator[Document]:
    repo_id = make_id("repo", repo.name)
    for email in cfg.emails:
        log = _git(repo, ["log", "--no-merges", f"--author={email}", "--format=%B%x00"],
                   cfg.git_timeout)
        for index, message in enumerate(log.split("\x00")):
            message = message.strip()
            if len(message) < cfg.min_commit_chars:
                continue
            yield Document(
                id=make_id(repo_id, "commit", email, index),
                source="commit",
                text=message,
                author="me",
                meta={"repo": repo_id},
            )


def iter_github(root: str | Path, cfg: GitHubConfig) -> Iterator[Document]:
    """Walk every repository, skipping (loudly) any that will not read in time."""
    repos = find_repos(root)
    for index, repo in enumerate(repos, start=1):
        try:
            documents = list(iter_repo_code(repo, cfg))
            if cfg.include_commits:
                documents.extend(iter_repo_commits(repo, cfg))
        except GitTimeout as exc:
            # Skipping is right, but silence is not: a quietly dropped repo
            # looks exactly like a repo with nothing of mine in it.
            print(f"  skipped {repo.name}: {exc}. Try `git -C {repo} gc` or raise --git-timeout.",
                  flush=True)
            continue
        if cfg.verbose:
            # flush: reading git history is slow enough that buffered progress
            # makes a working scan look like a hung one.
            print(f"  [{index}/{len(repos)}] {repo.name:28s} {len(documents):5,d} documents",
                  flush=True)
        yield from documents


def collect_github(root: str | Path, cfg: GitHubConfig) -> list[Document]:
    return list(iter_github(root, cfg))


def discover_authors(
    root: str | Path, top: int = 12, timeout: int = 30, verbose: bool = False
) -> list[tuple[str, int]]:
    """Author emails across all repos, most commits first.

    GitHub rewrites commit addresses to ``<id>+<user>@users.noreply.github.com``
    when the privacy setting is on, so the obvious guess -- the address you
    actually read mail at -- matches nothing and the collector silently returns
    an empty corpus.  The inventory prints this list instead of failing quietly.
    """
    tally: dict[str, int] = defaultdict(int)
    repos = find_repos(root)
    for index, repo in enumerate(repos, start=1):
        try:
            log = _git(repo, ["log", "--no-merges", "--format=%ae"], timeout)
        except GitTimeout as exc:
            print(f"  skipped {repo.name}: {exc}", flush=True)
            continue
        for line in log.splitlines():
            if line.strip():
                tally[line.strip()] += 1
        if verbose:
            print(f"  [{index}/{len(repos)}] scanned {repo.name}", flush=True)
    return sorted(tally.items(), key=lambda kv: -kv[1])[:top]
