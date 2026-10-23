"""Tests for the long-form and GitHub collectors, and the corpus census.

The GitHub tests build a real (tiny) git repository with two authors, because
the whole point of that collector is that it reads history rather than
trusting the filesystem.
"""

from __future__ import annotations

import subprocess

import pytest

from minime.data.document import Document
from minime.data.inventory import format_table, inventory, pii_audit, split_preview, verdict
from minime.data.personal.github_repos import (
    GitHubConfig, _looks_like_data, collect_github, discover_authors, file_authorship, find_repos,
)
from minime.data.personal.longform import LongformConfig, collect_longform

ME = "me@example.com"
THEM = "them@example.com"


# ---------------------------------------------------------------- longform


def test_longform_collects_text_files(tmp_path):
    (tmp_path / "essay.md").write_text("x" * 500)
    (tmp_path / "notes.txt").write_text("y" * 500)
    (tmp_path / "photo.png").write_bytes(b"\x89PNG")

    docs = collect_longform(tmp_path)
    assert len(docs) == 2
    assert {d.source for d in docs} == {"longform"}
    assert {d.meta["extension"] for d in docs} == {".md", ".txt"}


def test_longform_skips_short_files(tmp_path):
    (tmp_path / "stub.md").write_text("too short")
    (tmp_path / "real.md").write_text("x" * 500)
    assert len(collect_longform(tmp_path)) == 1


def test_longform_skips_hidden_and_excluded_directories(tmp_path):
    for folder in (".obsidian", "node_modules", ".git"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "f.md").write_text("z" * 500)
    (tmp_path / "keep.md").write_text("k" * 500)
    assert len(collect_longform(tmp_path)) == 1


def test_longform_does_not_leak_directory_names(tmp_path):
    """Folder names are often course codes, employers or people."""
    folder = tmp_path / "COMP90051-with-Anna"
    folder.mkdir()
    (folder / "report.md").write_text("q" * 500)

    doc = collect_longform(tmp_path)[0]
    assert "Anna" not in str(doc.meta) and "COMP90051" not in str(doc.meta)


def test_longform_ids_are_stable(tmp_path):
    (tmp_path / "a.md").write_text("m" * 500)
    assert [d.id for d in collect_longform(tmp_path)] == [d.id for d in collect_longform(tmp_path)]


def test_longform_respects_max_bytes(tmp_path):
    (tmp_path / "huge.md").write_text("x" * 5000)
    assert collect_longform(tmp_path, LongformConfig(max_bytes=1000)) == []


def test_longform_handles_a_missing_root(tmp_path):
    assert collect_longform(tmp_path / "nope") == []


# ------------------------------------------------------------------ github


def _run(repo, args, author_email=None, author_name="Someone"):
    env = {
        "GIT_AUTHOR_NAME": author_name, "GIT_AUTHOR_EMAIL": author_email or ME,
        "GIT_COMMITTER_NAME": author_name, "GIT_COMMITTER_EMAIL": author_email or ME,
        "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null",
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": str(repo),
    }
    subprocess.run(["git", *args], cwd=repo, env=env, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "myrepo"
    path.mkdir()
    _run(path, ["init", "-q", "-b", "main"])

    (path / "mine.py").write_text("def mine():\n    " + "# a real comment here\n    " * 6 + "return 1\n")
    _run(path, ["add", "mine.py"])
    _run(path, ["commit", "-q", "-m", "add my own module with a proper message"], ME)

    (path / "theirs.py").write_text("def theirs():\n    " + "# their comment here\n    " * 6 + "return 2\n")
    _run(path, ["add", "theirs.py"])
    _run(path, ["commit", "-q", "-m", "a contribution from somebody else entirely"], THEM)
    return path


def test_find_repos_detects_a_repository(repo):
    assert find_repos(repo) == [repo]
    assert find_repos(repo.parent) == [repo]


def test_file_authorship_counts_commits_per_author(repo):
    authorship = file_authorship(repo, GitHubConfig(emails=[ME]))
    assert authorship["mine.py"] == (1, 1)
    assert authorship["theirs.py"] == (0, 1)


def test_only_my_files_are_collected(repo):
    docs = collect_github(repo, GitHubConfig(emails=[ME]))
    names = {d.meta.get("extension") for d in docs if d.source == "code"}
    texts = " ".join(d.text for d in docs if d.source == "code")

    assert names == {".py"}
    assert "def mine" in texts
    assert "def theirs" not in texts, "another author's code must not enter the corpus"


def test_my_commit_messages_are_collected(repo):
    docs = collect_github(repo, GitHubConfig(emails=[ME]))
    messages = [d.text for d in docs if d.source == "commit"]
    assert messages == ["add my own module with a proper message"]


def test_commits_can_be_disabled(repo):
    docs = collect_github(repo, GitHubConfig(emails=[ME], include_commits=False))
    assert all(d.source != "commit" for d in docs)


def test_unknown_email_collects_nothing(repo):
    """The silent-zero case: a wrong address must not look like an empty repo."""
    assert collect_github(repo, GitHubConfig(emails=["nobody@example.com"])) == []


def test_discover_authors_lists_real_addresses(repo):
    authors = dict(discover_authors(repo))
    assert authors == {ME: 1, THEM: 1}


def test_repo_identity_is_hashed(repo):
    doc = collect_github(repo, GitHubConfig(emails=[ME]))[0]
    assert "myrepo" not in str(doc.meta)
    assert len(doc.meta["repo"]) == 16


def test_excluded_directories_are_skipped(repo):
    vendored = repo / "node_modules"
    vendored.mkdir()
    (vendored / "dep.py").write_text("def dep():\n    " + "# vendored code\n    " * 6 + "pass\n")
    _run(repo, ["add", "-f", "node_modules/dep.py"])
    _run(repo, ["commit", "-q", "-m", "vendor a dependency for the build"], ME)

    docs = collect_github(repo, GitHubConfig(emails=[ME]))
    assert all("dep" not in d.text for d in docs if d.source == "code")


@pytest.mark.parametrize("text,expected", [
    ("def f():\n    return 1\n" * 10, False),
    ("x" * 1500, True),                                   # minified bundle
    (",".join(str(i) for i in range(4000)), True),        # csv-shaped
    ("1,2,3\n" * 100, True),                              # numeric data
])
def test_data_shaped_files_are_rejected(text, expected):
    assert _looks_like_data(text) is expected


def test_authorship_threshold_is_configurable(repo):
    """A file the user barely touched should not count as theirs."""
    cfg = GitHubConfig(emails=[ME], min_authorship=1.01)
    assert [d for d in collect_github(repo, cfg) if d.source == "code"] == []


# --------------------------------------------------------------- inventory


def _docs():
    return [
        Document(id=f"c{i}", source="chat", text="ngl that was rough " * 3) for i in range(10)
    ] + [
        Document(id=f"l{i}", source="longform", text="A considered paragraph. " * 40) for i in range(3)
    ] + [
        Document(id=f"k{i}", source="code", text="def f():\n    return 1\n" * 5) for i in range(2)
    ]


def test_inventory_counts_by_source():
    reports = inventory(_docs())
    assert set(reports) == {"chat", "code", "longform"}
    assert reports["chat"].documents == 10
    assert reports["chat"].est_tokens > 0
    assert reports["longform"].mean_chars > reports["chat"].mean_chars


def test_inventory_table_renders_totals():
    table = format_table(inventory(_docs()))
    assert "TOTAL" in table and "source" in table
    assert "chat" in table and "longform" in table


def test_empty_inventory_is_handled():
    assert format_table(inventory([])) == "no documents found"


def test_split_preview_covers_every_document():
    docs = _docs()
    assert sum(split_preview(docs).values()) == len(docs)


def test_split_preview_reports_empty_buckets():
    """Observed on a real 25-document census: the val split was empty and the
    report simply omitted it, which reads as "fine" rather than "broken"."""
    preview = split_preview([Document(id="only", source="chat", text="hi")])
    assert set(preview) == {"train", "val", "test"}
    assert sum(preview.values()) == 1
    assert sorted(preview.values()) == [0, 0, 1]


def test_split_preview_counts_groups_as_the_unit_assigned():
    grouped = [Document(id=f"g{i}", source="chat", text="hi", meta={"split_group": "thread:2024-01"})
               for i in range(50)]
    loose = [Document(id=f"x{i}", source="comment", text="hi") for i in range(5)]
    by_docs = split_preview(grouped + loose)
    by_groups = split_preview(grouped + loose, groups=True)
    assert sum(by_docs.values()) == 55 and sum(by_groups.values()) == 6
    assert max(by_docs.values()) >= 50, "a group is never divided between splits"


def test_leakage_check_catches_a_divided_split_group():
    from minime.data.splits import check_no_leakage

    a = Document(id="a", source="chat", text="hi", meta={"split_group": "t:2024-01"})
    b = Document(id="b", source="chat", text="yo", meta={"split_group": "t:2024-01"})
    with pytest.raises(ValueError, match="split group t:2024-01"):
        check_no_leakage({"train": [a], "test": [b]})


def test_pii_audit_flags_residual_pii():
    docs = [
        Document(id="1", source="chat", text="clean text here"),
        Document(id="2", source="chat", text="reach me at a@b.com"),
    ]
    audit = pii_audit(docs)
    assert audit == {"documents_checked": 2, "documents_flagged": 1}


@pytest.mark.parametrize("tokens,expected", [
    (500_000, "ADAPTATION ONLY"),
    (3_000_000, "ADAPTATION + REPLAY"),
    (20_000_000, "FULL MIXTURE"),
])
def test_verdict_maps_token_count_to_a_decision(tokens, expected):
    assert verdict(tokens).startswith(expected)


# ------------------------------------------------------- degraded repositories


def test_a_slow_repository_is_skipped_loudly(repo, monkeypatch, capsys):
    """A repo that will not read in time must be reported, not silently dropped.

    Observed for real: a repository with many loose objects on a nearly-full
    disk took 22 seconds to list 38 commits. Silence there looks exactly like
    "this repo has nothing of mine in it".
    """
    import subprocess as sp

    from minime.data.personal import github_repos

    def slow(*args, **kwargs):
        raise sp.TimeoutExpired(cmd="git", timeout=kwargs.get("timeout", 1))

    monkeypatch.setattr(github_repos.subprocess, "run", slow)

    docs = collect_github(repo, GitHubConfig(emails=[ME], git_timeout=1))
    out = capsys.readouterr().out

    assert docs == []
    assert "skipped" in out and repo.name in out
    assert "gc" in out, "the message should say how to fix it"


def test_discover_authors_survives_a_slow_repository(repo, monkeypatch, capsys):
    import subprocess as sp

    from minime.data.personal import github_repos

    monkeypatch.setattr(
        github_repos.subprocess, "run",
        lambda *a, **k: (_ for _ in ()).throw(sp.TimeoutExpired(cmd="git", timeout=1)),
    )
    assert discover_authors(repo, timeout=1) == []
    assert "skipped" in capsys.readouterr().out


def test_git_failure_is_not_a_timeout(repo, monkeypatch):
    """A non-zero git exit means "nothing here", which is different from a hang."""
    import subprocess as sp

    from minime.data.personal import github_repos

    monkeypatch.setattr(
        github_repos.subprocess, "run",
        lambda *a, **k: sp.CompletedProcess(args=a, returncode=128, stdout="", stderr="fatal"),
    )
    assert collect_github(repo, GitHubConfig(emails=[ME])) == []
