"""The showcase page: its numbers come from the reports, its samples pass the gate and carry no prompt.

Decided with the owner on 2026-10-19 and 2026-10-20 (README "Privacy",
*Hosting*): a static page, the frontier and a few gated samples, no live
model.  These tests pin what may reach it.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pytest

from minime.evaluation.memorisation import MemorisationError
from minime.evaluation.showcase import (
    GROUPS,
    chosen_rows,
    frontier,
    page_data,
    render_data_js,
    samples_payload,
)
from minime.training.m10 import non_dominated

REPO = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _keys(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield k
            yield from _keys(v)
    elif isinstance(o, list):
        for v in o:
            yield from _keys(v)


# ------------------------------------------------------------ the numbers


def test_docs_data_js_is_what_the_reports_give():
    """Run scripts/build_showcase.py after any report changes; the page never holds a typed number."""
    assert (REPO / "docs" / "data.js").read_text(encoding="utf-8") == render_data_js(page_data(REPO))


def test_the_page_data_holds_numbers_and_labels_never_text():
    data = page_data(REPO)
    assert not {"text", "prompt", "doc", "samples"} & set(_keys(data))
    assert set(data["frontier"]) == {"val", "test"}
    assert data["final"]["final_run"] == "m10-adapt-c"
    assert data["memorisation"]["sensitive"] is False      # blind at this scale: the page must say so


def test_the_frontier_is_the_non_dominated_group_means():
    table = json.loads((REPO / "reports" / "m10" / "test.json").read_text())["common_context"]
    out = frontier(table)
    means = {g: (v["mean"]["general"], v["mean"]["personal"]) for g, v in out["groups"].items()}
    assert out["front"] == non_dominated(means)
    assert set(out["groups"]) == set(GROUPS)


def test_the_frontier_refuses_a_table_with_other_groups():
    table = json.loads((REPO / "reports" / "m10" / "common512.json").read_text())
    del table["groups"]["P1"]
    with pytest.raises(ValueError, match="groups"):
        frontier(table)


def test_the_page_may_fetch_from_the_samples_object_alone():
    html = (REPO / "docs" / "index.html").read_text(encoding="utf-8")
    csp = re.search(r'http-equiv="Content-Security-Policy" content="([^"]+)"', html).group(1)
    sources = dict(d.strip().split(" ", 1) for d in csp.split(";"))
    samples = urlsplit(re.search(r'name="minime:samples" content="([^"]*)"', html).group(1))
    assert samples.scheme == "https" and samples.path == "/samples.json"
    assert sources["connect-src"] == f"https://{samples.netloc}"
    assert sources["default-src"] == "'none'" and sources["script-src"] == "'self'"


# ------------------------------------------------------------ the samples


def _samples_file(root: Path, run: str, name: str = "eval-last.jsonl", n: int = 3) -> Path:
    path = root / run / "samples" / name
    path.parent.mkdir(parents=True)
    rows = [{"family": "voice", "prompt": i, "doc": 100 + i, "ended": True, "text": f"generated {i}"}
            for i in range(n)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_chosen_rows_keep_the_text_and_drop_the_prompt_and_document(tmp_path):
    final = _samples_file(tmp_path, "m10-adapt-c")
    flagship = _samples_file(tmp_path, "flagship-s1337", "eval-last-test.jsonl")
    rows = chosen_rows([{"file": str(final), "row": 2}, {"file": str(flagship), "row": 0}])
    assert rows == [{"model": "final", "split": "val", "family": "voice", "text": "generated 2"},
                    {"model": "flagship", "split": "test", "family": "voice", "text": "generated 0"}]


@pytest.mark.parametrize("run, name, row", [
    ("m10-adapt-a", "eval-last.jsonl", 0),          # a candidate the owner did not choose
    ("p4-s1337", "eval-last.jsonl", 0),             # a probe
    ("m10-adapt-c", "other.jsonl", 0),              # not an eval's samples file
    ("m10-adapt-c", "eval-last.jsonl", 3),          # past the end
    ("m10-adapt-c", "eval-last.jsonl", -1),
])
def test_chosen_rows_refuse_other_files_and_rows(tmp_path, run, name, row):
    path = _samples_file(tmp_path, run, name)
    with pytest.raises(MemorisationError):
        chosen_rows([{"file": str(path), "row": row}])


def test_chosen_rows_refuse_the_same_row_twice(tmp_path):
    path = _samples_file(tmp_path, "m10-adapt-c")
    with pytest.raises(MemorisationError, match="twice"):
        chosen_rows([{"file": str(path), "row": 1}, {"file": str(path), "row": 1}])


def test_the_payload_is_refused_when_any_sample_fails_the_gate():
    rows = [{"model": "final", "split": "val", "family": "voice", "text": "a"},
            {"model": "final", "split": "val", "family": "voice", "text": "b"}]
    results = [{"sample": 0, "pass": True}, {"sample": 1, "pass": False}]
    with pytest.raises(MemorisationError, match=r"\[1\]"):
        samples_payload(rows, results, 7.45)
    payload = samples_payload(rows, [{"sample": 0, "pass": True}, {"sample": 1, "pass": True}], 7.45)
    assert [set(s) for s in payload["samples"]] == [{"model", "split", "family", "text"}] * 2


def test_showcase_samples_never_prints_text_and_writes_outside_the_repo(tmp_path, monkeypatch, capsys):
    module = _load_script("showcase_samples")
    final = _samples_file(tmp_path / "runs", "m10-adapt-c")
    chosen = tmp_path / "chosen.json"
    chosen.write_text(json.dumps([{"file": str(final), "row": 0}, {"file": str(final), "row": 1}]))
    names = tmp_path / "names.txt"
    names.write_text("Somebody Else\n", encoding="utf-8")

    class Tokenizer:
        @staticmethod
        def encode(text):
            return [ord(c) for c in text]

    monkeypatch.setattr(module, "load_tokenizer", lambda path: Tokenizer())
    monkeypatch.setattr(module, "TokenShard", lambda path: type("S", (), {"tokens": np.arange(50) + 1000})())
    out = tmp_path / "public" / "samples.json"
    monkeypatch.setattr(sys, "argv", ["showcase_samples.py", str(chosen), "--out", str(out),
                                      "--names", str(names), "--root", str(tmp_path)])
    assert module.main() == 0
    printed = capsys.readouterr()
    assert "generated" not in printed.out + printed.err
    assert [s["text"] for s in json.loads(out.read_text())["samples"]] == ["generated 0", "generated 1"]

    monkeypatch.setattr(sys, "argv", ["showcase_samples.py", str(chosen), "--out", str(REPO / "docs" / "samples.json"),
                                      "--names", str(names), "--root", str(tmp_path)])
    assert module.main() == 1
    assert "inside the repository" in capsys.readouterr().err
    assert not (REPO / "docs" / "samples.json").exists()
