"""The showcase page: its numbers, and the samples chosen for it.

Decided with the owner on 2026-10-19 and 2026-10-20 (README "Privacy",
*Hosting*): MiniMe stays local; what others see is a static page, the
frontier and a few gated samples, with no live model behind it.

* **The numbers** (``page_data``) are read from the committed reports, never
  typed in: the frontier from the common-context tables (val,
  ``reports/m10/common512.json``; test, ``reports/m10/test.json``), the
  final model against the flagship from the test phase, the study's
  verdicts, the memorisation check and the compute.  ``scripts/
  build_showcase.py`` writes them to ``docs/data.js``, and a test keeps that
  file equal to what the reports give.
* **The samples** (``samples_payload``) are generated text only, from the
  flagship's and the final model's own samples files: never a prompt (a voice
  prompt is the owner's real messages), never a document index.  Every one
  must pass the memorisation check's gate (``gate``), or nothing is written.
  The file goes to a public object outside the repository, never into git.

Nothing here prints text.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ..training.m10 import CHAT_TOKENS, FLAGSHIP_MIXTURE, GENERAL_CAP, PERSONAL_GAIN, non_dominated
from .memorisation import MemorisationError

PERSONAL, GENERAL = "perplexity.personal.loss", "perplexity.general_weighted.loss"
# Common-context group -> (label, the model's size, what it is).
GROUPS = {
    "P1": ("P1", "17M", "general data only"),
    "P4": ("P4", "17M", "the owner's chat mixed in from step 0"),
    "E1 25/75": ("E1 25/75", "17M", "P1, then adapted on 25% chat, 75% general"),
    "E1 50/50": ("E1 50/50", "17M", "P1, then adapted on 50% chat, 50% general"),
    "flagship": ("Flagship", "49M", "the owner's chat mixed in from step 0"),
    "adaptation": ("Flagship + C", "49M", "the flagship, adapted by candidate C: the final model"),
}
# The final model against the flagship, in README's order (*The test split, read once*).
FULL_METRICS = {
    PERSONAL: ("personal CE", "lower", True),
    GENERAL: ("general-weighted CE", "lower", True),
    "voice.authorship.discriminator.auc": ("impersonation AUC", "lower", False),
    "voice.authorship.attribution.generated.attributed_to_target": ("attributed to the owner", "higher", False),
    "voice.stylometry.delta.questioned.generated.nearest_is_target": ("Delta: nearest is the owner", "higher", False),
    "voice.stylometry.distance_to_real.generated.function_word_jsd": ("function-word JSD to real", "lower", False),
    "code.valid": ("code syntax validity", "higher", False),
    "blimp.accuracy": ("BLiMP accuracy", "higher", False),
}
MEMO_GROUPS = ("P1", "P4", "E1 50/50", "E1 100/0", "flagship", "flagship + C")
# A samples file's run directory -> the model it shows; nothing else may reach the page.
SAMPLE_RUNS = {"m10-adapt-c": "final", "flagship-s1337": "flagship"}
SAMPLE_FILES = {"eval-last.jsonl": "val", "eval-last-test.jsonl": "test"}
PLACES = 6                       # the page rounds for display; never round twice


def _r(x: float | None) -> float | None:
    return None if x is None else round(float(x), PLACES)


def _point(values: dict) -> dict:
    return {"personal": _r(values[PERSONAL]), "general": _r(values[GENERAL])}


def frontier(table: dict) -> dict:
    """One split's common-context table: each group's runs and mean, and the non-dominated means."""
    if set(table["groups"]) != set(GROUPS):
        raise ValueError(f"common-context groups {sorted(table['groups'])}, expected {sorted(GROUPS)}")
    groups = {}
    for g, (label, size, what) in GROUPS.items():
        group = table["groups"][g]
        groups[g] = {"label": label, "size": size, "what": what, "mean": _point(group["mean"]),
                     "runs": {run: _point(r["values"]) for run, r in group["runs"].items()}}
    means = {g: (group["mean"][GENERAL], group["mean"][PERSONAL]) for g, group in table["groups"].items()}
    return {"split": table["split"], "context_length": table["context_length"], "groups": groups,
            "front": non_dominated(means)}


def final_model(test: dict) -> dict:
    """The flagship and the final model, every eval family, on val and test (read once)."""
    runs = test["full"]["runs"]
    if set(runs) != {"flagship-s1337", test["final_run"]}:
        raise ValueError(f"the test phase's full runs are {sorted(runs)}")
    rows = []
    for key, (label, better, primary) in FULL_METRICS.items():
        row = {"metric": label, "better": better, "primary": primary}
        for split in ("val", "test"):
            flagship, final = runs["flagship-s1337"][split][key], runs[test["final_run"]][split][key]
            row |= {f"flagship_{split}": _r(flagship), f"final_{split}": _r(final),
                    f"difference_{split}": _r(final - flagship)}
        rows.append(row)
    return {"final_run": test["final_run"], "chosen": test["final_choice_made"], "rows": rows}


def study(reports: Path) -> dict:
    """The controlled study's verdicts at the probe scale: E3, E4, E1 and P4."""
    floor = json.loads((reports / "noise_floor" / "p1.json").read_text())
    e1 = json.loads((reports / "e1" / "paired.json").read_text())
    p4 = json.loads((reports / "p4" / "verdicts.json").read_text())

    def pretrain(run: str) -> dict:
        c = floor["comparisons"][run]["final_val_loss"]
        return {"difference": _r(c["difference"]), "half_width": _r(c["half_width"]), "verdict": c["verdict"]}

    def interval(c: dict, diff: str) -> dict:
        return {"difference": _r(c[diff]), "half_width": _r(c["half_width"]), "verdict": c["verdict"]}

    return {
        "floor": {"n": floor["n"], "half_width": _r(floor["half_width"]), "metric": floor["metric"]},
        "e3": pretrain("p2-s1337"),
        "e4": pretrain("p3-s1337"),
        "e1": {arm: {axis: interval(e1["contrasts"][arm]["metrics"][key], "mean")
                     for axis, key in (("personal", PERSONAL), ("general", GENERAL))}
               for arm in ("25", "50", "100")},
        "p4": {other: {axis: interval(p4["contrasts"][other][key], "difference")
                       for axis, key in (("personal", PERSONAL), ("general", GENERAL))}
               for other in ("p1", "e1-50")},
    }


def memorisation(report: dict) -> dict:
    """The memorisation check, as it bears on the page: blind at this scale, and the gate's threshold."""
    groups = []                      # a list: the page keeps this order
    for g in MEMO_GROUPS:
        group = report["groups"][g]
        groups.append({"group": g, "n": len(group["runs"]),
                       **{k: _r(group[k]["mean"]) for k in ("train_rate", "val_rate", "gap", "overlap_mean",
                                                            "overlap_p95")}})
    real = report["real_continuations"]
    return {"sensitive": report["sensitivity"]["sensitive"], "groups": groups,
            "real": {"overlap_mean": _r(real["mean"]), "overlap_p95": _r(real["p95"]), "max": real["max"]},
            "threshold": _r(report["gate"]["threshold"])}


def compute(costs: Path, flagship_manifest: dict) -> dict:
    """GPU-hours on Kaggle and hours on the laptop, from every run's cost row; the flagship's size."""
    hours = {"kaggle-free": 0.0, "local": 0.0}
    with costs.open(newline="") as f:
        for row in csv.DictReader(f):
            hours[row["venue"]] = hours.get(row["venue"], 0.0) + float(row["runtime_hours"])
    summary = flagship_manifest["summary"]
    return {"kaggle_gpu_hours": round(hours["kaggle-free"], 1), "laptop_hours": round(hours["local"], 1),
            "params": summary["params"], "tokens": summary["tokens_seen"], "runtime": summary["runtime"]}


def recipe(adaptation: dict, flagship_manifest: dict) -> dict:
    """How the final model was made: the flagship's mixture, the chosen candidate and the gate it passed."""
    pick = adaptation["candidates"][adaptation["final_choice"]]
    if pick["run"] != adaptation["final_run"] or not pick["accepted"]:
        raise ValueError(f"the final choice {adaptation['final_choice']!r} is not an accepted candidate")
    tokens = flagship_manifest["summary"]["tokens_seen"]
    return {"chat_share_pretrain": _r(100 * CHAT_TOKENS / tokens), "chat_tokens": CHAT_TOKENS,
            "general_parts": {k: v for k, v in FLAGSHIP_MIXTURE.items() if k != "chat"},
            "candidate": adaptation["final_choice"].upper(), "candidates": len(adaptation["candidates"]),
            "chat_share_adapt": round(100 * pick["chat_share"]), "lr": pick["lr"],
            "adapt_tokens": adaptation["token_budget"], "personal_gain": PERSONAL_GAIN,
            "general_cap": GENERAL_CAP, "chosen": adaptation["final_choice_made"],
            "change_val": _point(pick["change"])}


def page_data(repo: Path) -> dict:
    """Every number the page shows, from the committed reports."""
    reports = repo / "reports"
    test = json.loads((reports / "m10" / "test.json").read_text())
    flagship = json.loads((repo / "experiments" / "flagship-s1337" / "manifest.json").read_text())
    return {
        "recipe": recipe(json.loads((reports / "m10" / "adaptation.json").read_text()), flagship),
        "frontier": {"val": frontier(json.loads((reports / "m10" / "common512.json").read_text())),
                     "test": frontier(test["common_context"])},
        "final": final_model(test),
        "study": study(reports),
        "memorisation": memorisation(json.loads((reports / "memorisation" / "memorisation.json").read_text())),
        "compute": compute(repo / "experiments" / "costs.csv", flagship),
    }


def render_data_js(data: dict) -> str:
    return ("// Written by scripts/build_showcase.py from the committed reports: do not edit.\n"
            f"window.MINIME = {json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False)};\n")


def chosen_rows(chosen: list[dict]) -> list[dict]:
    """The choices, ``{"file": <samples file>, "row": <line index>}``, read from their files.

    Only the flagship's and the final model's own samples files are accepted,
    and the model and split come from the file's place, never from the choice.
    """
    out, seen = [], set()
    for i, c in enumerate(chosen):
        path, row = Path(c["file"]), c["row"]
        model, split = SAMPLE_RUNS.get(path.parent.parent.name), SAMPLE_FILES.get(path.name)
        if model is None or split is None or path.parent.name != "samples":
            raise MemorisationError(f"choice {i}: not a samples file of the flagship or the final model "
                                    f"(<run>/samples/eval-last[-test].jsonl, run one of {sorted(SAMPLE_RUNS)})")
        if (path.resolve(), row) in seen:
            raise MemorisationError(f"choice {i}: the same row twice")
        seen.add((path.resolve(), row))
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not isinstance(row, int) or not 0 <= row < len(lines):
            raise MemorisationError(f"choice {i}: row {row!r}, the file has {len(lines)}")
        record = json.loads(lines[row])
        out.append({"model": model, "split": split, "family": record["family"], "text": record["text"]})
    return out


def samples_payload(rows: list[dict], results: list[dict], threshold: float) -> dict:
    """The public samples file: refused unless every row passed the gate."""
    if len(rows) != len(results) or not rows:
        raise MemorisationError(f"{len(rows)} samples, {len(results)} gate results")
    failed = [r["sample"] for r in results if not r["pass"]]
    if failed:
        raise MemorisationError(f"samples {failed} fail the gate: nothing is written")
    return {"version": 1, "gate": {"threshold": _r(threshold)},
            "samples": [{k: r[k] for k in ("model", "split", "family", "text")} for r in rows]}


__all__ = [
    "FULL_METRICS", "GROUPS", "SAMPLE_RUNS", "chosen_rows", "final_model", "frontier", "page_data",
    "recipe", "render_data_js", "samples_payload", "study",
]
