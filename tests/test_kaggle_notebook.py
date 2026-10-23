"""The Kaggle launcher notebook (M8): a launcher whose parameters are checkable here.

The notebook itself cannot run off Kaggle, so what can go wrong before a GPU
is spent is tested from its source: that its jobs plan, that its pinned commit
holds the code it imports, and that it carries no outputs and no token.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess

import pytest

from minime.config import load_config, parse_cli_overrides
from minime.tokenizer import is_dev_tokenizer
from minime.training.launch import Job, plan_runs
from minime.utils.costs import VENUE_HOURLY_USD

NOTEBOOK = "notebooks/kaggle_train.ipynb"
# What each attachable dataset holds: shards/general, and the staged personal
# upload (scripts/stage_personal_upload.py: chat.train alone).
DATASET_SOURCES = {"general": {"web", "dialogue", "code"}, "personal": {"chat"}}


@pytest.fixture
def notebook(repo_root):
    return json.loads((repo_root / NOTEBOOK).read_text(encoding="utf-8"))


@pytest.fixture
def params(notebook):
    cell = next(c for c in notebook["cells"] if c.get("id") == "params")
    tree = ast.parse("".join(cell["source"]))
    return {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
    }


def _git(repo_root, *args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True)


def test_notebook_is_clean_nbformat_4(notebook):
    assert notebook["nbformat"] == 4
    code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert code, "no code cells"
    # Outputs would carry a run's log -- and whatever a failing cell printed -- into git.
    assert all(c["outputs"] == [] and c["execution_count"] is None for c in code)
    assert len({c["id"] for c in notebook["cells"]}) == len(notebook["cells"])


def test_notebook_holds_no_token(repo_root, notebook):
    raw = (repo_root / NOTEBOOK).read_text(encoding="utf-8")   # metadata included
    assert not re.search(r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}", raw)
    source = "".join("".join(c["source"]) for c in notebook["cells"])
    assert 'get_secret("GITHUB_TOKEN")' in source


def test_notebook_jobs_plan(repo_root, params, tmp_path):
    jobs = [Job(config, tuple(overrides)) for config, overrides in params["JOBS"]]
    names = plan_runs(jobs, repo=repo_root, work_dir=tmp_path)
    assert len(names) == len(jobs)
    assert params["VENUE"] in VENUE_HOURLY_USD


def test_notebook_trains_on_the_frozen_tokenizer(repo_root, params):
    assert not is_dev_tokenizer(params["TOKENIZER_ID"])
    frozen = repo_root / "tokenizer" / "minime-16k.json"
    if not frozen.exists():
        pytest.skip("frozen tokenizer not on this machine (gitignored)")
    pytest.importorskip("tokenizers")
    from minime.tokenizer import load_tokenizer

    assert params["TOKENIZER_ID"] == load_tokenizer(frozen).id


def test_notebook_pins_a_commit_that_holds_its_code(repo_root, params):
    commit = params["COMMIT"]
    assert re.fullmatch(r"[0-9a-f]{40}", commit), "pin a full sha: a branch moves"
    if _git(repo_root, "rev-parse", "--is-shallow-repository").stdout.strip() != "false":
        pytest.skip("not a full git checkout")
    needed = ["src/minime/training/launch.py", "scripts/train.py"]
    needed += [config for config, _ in params["JOBS"]]
    missing = [p for p in needed if _git(repo_root, "cat-file", "-e", f"{commit}:{p}").returncode]
    assert not missing, f"{commit[:12]} does not contain {missing}"


def test_notebook_attaches_the_datasets_its_jobs_read_and_no_others(repo_root, params):
    datasets = params["DATASETS"]
    assert datasets[0] == "general", "the first dataset holds the val set: P1's"
    assert len(set(datasets)) == len(datasets) and set(datasets) <= set(DATASET_SOURCES)
    held = set().union(*(DATASET_SOURCES[kind] for kind in datasets))
    read = set()
    for config, overrides in params["JOBS"]:
        cfg = load_config(repo_root / config, overrides=parse_cli_overrides(list(overrides)))
        sources = set(cfg.data.mixture)
        assert sources <= held, f"{config} reads {sorted(sources - held)}, which no dataset in DATASETS holds"
        read |= sources
    # The owner's chat goes up only for a session that trains on it.
    unused = [kind for kind in datasets if not DATASET_SOURCES[kind] & read]
    assert not unused, f"DATASETS attaches {unused}, which no job reads"


def test_every_name_the_notebook_imports_exists_at_the_pinned_commit(repo_root, notebook, params):
    """The notebook runs from the branch; its imports run from COMMIT.  A
    notebook that imports newer code than its pin fails on Kaggle, after the
    session has started."""
    if _git(repo_root, "rev-parse", "--is-shallow-repository").stdout.strip() != "false":
        pytest.skip("not a full git checkout")
    source = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    imports = [node for node in ast.walk(ast.parse(source))
               if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("minime.")]
    assert imports, "the notebook imports nothing from minime"
    for node in imports:
        path = "src/" + node.module.replace(".", "/") + ".py"
        shown = _git(repo_root, "show", f"{params['COMMIT']}:{path}")
        assert shown.returncode == 0, f"{params['COMMIT'][:12]} has no {path}"
        for alias in node.names:
            defined = re.search(rf"^(def|class) {alias.name}\b|^{alias.name}\s*[:=]", shown.stdout, re.M)
            assert defined, f"{path} at {params['COMMIT'][:12]} does not define {alias.name}"


def test_the_session_parameters_are_sane(params):
    assert isinstance(params["RESUME"], bool) and isinstance(params["SWEEP_EXTENSION"], bool)
    assert 0 < params["SESSION_HOURS"] < 12, "Kaggle stops a session at 12 h"


def test_a_sweep_extension_session_runs_exactly_m10s_grid(repo_root, params):
    """The notebook applies M10's edge rule only to M10's four points, and the
    extension it may queue must exist at the pinned commit."""
    if not params["SWEEP_EXTENSION"]:
        pytest.skip("not a sweep session")
    from minime.training.m10 import SWEEP_EXTENSIONS, SWEEP_GRID, sweep_config

    assert params["JOBS"] == [(sweep_config(lr), []) for lr in SWEEP_GRID]
    assert not params["RESUME"], "the extension follows four finished points, in a fresh session"
    if _git(repo_root, "rev-parse", "--is-shallow-repository").stdout.strip() != "false":
        pytest.skip("not a full git checkout")
    for lr in SWEEP_EXTENSIONS.values():
        assert _git(repo_root, "cat-file", "-e", f"{params['COMMIT']}:{sweep_config(lr)}").returncode == 0


# The commit that made training.seed set the initial weights (README, "the seed bug").
SEED_FIX = "192867ea991fe90f1951b6a674007a36ac9a7812"


def test_notebook_pins_a_commit_past_the_seed_fix(repo_root, params):
    """Before the fix, a run's initial weights were unseeded and unrecorded (HANDOFF §9)."""
    if _git(repo_root, "rev-parse", "--is-shallow-repository").stdout.strip() != "false":
        pytest.skip("not a full git checkout")
    assert _git(repo_root, "merge-base", "--is-ancestor", SEED_FIX, params["COMMIT"]).returncode == 0, \
        f"{params['COMMIT'][:12]} does not contain the seed fix {SEED_FIX[:12]}"
