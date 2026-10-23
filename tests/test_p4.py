"""P4's verdicts: four P4 runs against P1's four and E1's 50/50 arm, unpaired.

The rule was decided with the owner before any P4 run (README, "P4
protocol"): these tests pin it on synthetic runs, so a code change that
quietly alters what P4 measures fails here first.
"""

from __future__ import annotations

import json
import math
import statistics
import subprocess
import sys

import pytest

from minime.config import load_config, parse_cli_overrides
from minime.training.adaptation import ARMS, BASES
from minime.training.noise_floor import EVAL_METRICS, PRIMARY_EVAL_METRICS, config_fingerprint
from minime.training.p4 import CONTRASTS, N_VERDICTS, T_P4, P4Error, p4_analysis

EXPERIMENTS = "configs/experiments"
PERSONAL, GENERAL = PRIMARY_EVAL_METRICS
P1_CONFIGS = ["p1.yaml", "p1_s1338.yaml", "p1_s1339.yaml", "p1_s1340.yaml"]
P4_CONFIGS = ["p4.yaml", "p4_s1338.yaml", "p4_s1339.yaml", "p4_s1340.yaml"]

# Shaped like the real numbers (README, E1 main runs): P1 and the 50/50 arm as
# measured; P4 an invented outcome.
P1_VALUES = {PERSONAL: [7.51, 7.36, 7.49, 7.15], GENERAL: [3.715, 3.723, 3.717, 3.719]}
P4_VALUES = {PERSONAL: [3.70, 3.72, 3.69, 3.71], GENERAL: [3.725, 3.731, 3.728, 3.722]}
E1_VALUES = {
    0.0: {PERSONAL: [7.02, 6.90, 6.94, 6.99], GENERAL: [3.863, 3.871, 3.865, 3.868]},
    0.25: {PERSONAL: [3.58, 3.56, 3.58, 3.57], GENERAL: [3.909, 3.921, 3.915, 3.921]},
    0.5: {PERSONAL: [3.490, 3.494, 3.494, 3.486], GENERAL: [3.989, 4.000, 3.999, 4.008]},
    1.0: {PERSONAL: [3.80, 3.79, 3.81, 3.80], GENERAL: [5.785, 5.795, 5.855, 5.782]},
}


def _eval_record(config: dict, values: dict[str, float], eval_fp="evalfp") -> dict:
    record = {
        "eval_version": 1, "eval_config_fingerprint": eval_fp, "split": "val",
        "checkpoint": {"config_fingerprint": config_fingerprint(config), "step": config["training"]["max_steps"]},
        "tokenizer": {"id": "minime-16k@abc"},
        "device": {"device": "mps", "precision": "fp32", "torch": "2.11.0"},
        "eval_sets": {"personal": "d1", "general": "d2"},
    }
    for metric in EVAL_METRICS:
        node = record
        *parents, leaf = metric.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = values.get(metric, 0.5)
    return record


def _write(run_dir, manifest, status, values, eval_fp="evalfp"):
    run_dir.mkdir()
    records = [{"step": 1, "phase": "train", "train_loss": 5.0}]
    if status != "finished":
        manifest.pop("summary", None)
        records.append({"step": 2, "phase": "diverged", "train_loss": float("nan")})
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    (run_dir / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    if status == "finished":
        (run_dir / "eval-last.json").write_text(json.dumps(_eval_record(manifest["config"], values, eval_fp)))
    return run_dir


def _pretrain(tmp_path, repo_root, config_name, values, *, status="finished", val_loss=3.70, precision="fp16",
              eval_fp="evalfp", **edits):
    """A pretrain's run directory as train.py writes it on Kaggle, with its eval file."""
    cfg = json.loads(json.dumps(load_config(repo_root / EXPERIMENTS / config_name).to_dict()))
    for dotted, value in edits.items():
        section, _, key = dotted.partition("__")
        cfg[section][key] = value
    shards = {"web": 630, "dialogue": 180, "code": 90}
    if "chat" in cfg["data"]["mixture"]:
        shards["chat"] = 293
    manifest = {
        "config": cfg,
        "data": {"data_dir": "/kaggle/input", "tokenizer": "minime-16k@abc", "tokenizer_dev": False,
                 "shards": shards, "val_shard": {"val": 4_537_964}},
        "device": {"precision": precision},
        "summary": {"steps": cfg["training"]["max_steps"], "final_val_loss": val_loss},
    }
    name = f"{cfg['name']}-s{cfg['training']['seed']}-{len(list(tmp_path.iterdir()))}"
    return _write(tmp_path / name, manifest, status, values, eval_fp)


def _group(tmp_path, repo_root, configs, values, **kwargs):
    return [_pretrain(tmp_path, repo_root, c, {m: v[i] for m, v in values.items()}, **kwargs)
            for i, c in enumerate(configs)]


def _e1(tmp_path, repo_root, values=E1_VALUES, role_of=None):
    """E1's 16 main runs as train.py --init-from writes them."""
    runs = []
    for base, (p1_name, seed) in enumerate(BASES.items()):
        for share, config in ARMS.items():
            cfg = json.loads(json.dumps(load_config(
                repo_root / EXPERIMENTS / config, overrides=parse_cli_overrides([f"data.seed={seed}"])).to_dict()))
            if role_of:
                cfg["role"] = role_of(base, share)
            manifest = {
                "config": cfg,
                "data": {"data_dir": ["/g", "/p"], "tokenizer": "minime-16k@abc", "tokenizer_dev": False,
                         "shards": {"web": 630, "dialogue": 180, "code": 90, "chat": 293, "comment": 1},
                         "val_shard": {"general": 100, "personal": 50}},
                "device": {"precision": "fp32"},
                "init_from": {"sha256": f"{base:02d}" * 32, "config_fingerprint": f"fp{base}", "step": 2300},
                "summary": {"steps": cfg["training"]["max_steps"], "final_val_loss": 3.9},
            }
            point = {m: v[base] for m, v in values[share].items()}
            runs.append(_write(tmp_path / f"e1-{share}-{base}", manifest, "finished", point))
    return runs


@pytest.fixture
def design(tmp_path, repo_root):
    return (_group(tmp_path, repo_root, P4_CONFIGS, P4_VALUES, val_loss=3.71),
            _group(tmp_path, repo_root, P1_CONFIGS, P1_VALUES),
            _e1(tmp_path, repo_root))


def test_t_is_welchs_conservative_quantile_bonferroni_over_four():
    stats = pytest.importorskip("scipy.stats")
    assert N_VERDICTS == 4
    assert T_P4 == pytest.approx(stats.t.ppf(1 - 0.05 / (2 * N_VERDICTS), 3), abs=1e-4)


def test_four_verdicts_by_welchs_interval(design):
    report = p4_analysis(*design)
    assert report["n_runs"] == 4 and report["t"] == T_P4
    assert report["chat_tokens_expected"] == 1_048_576
    assert report["token_budget"] == 301_465_600 and report["precision"] == "fp16"

    vs_p1, vs_e1 = (report["contrasts"][name] for name in CONTRASTS)
    personal = vs_p1[PERSONAL]
    se = math.sqrt(statistics.variance(P4_VALUES[PERSONAL]) / 4 + statistics.variance(P1_VALUES[PERSONAL]) / 4)
    assert personal["difference"] == pytest.approx(statistics.fmean(P4_VALUES[PERSONAL])
                                                   - statistics.fmean(P1_VALUES[PERSONAL]))
    assert personal["half_width"] == pytest.approx(5.3919 * se)
    assert personal["verdict"] == "lower than p1"
    # +0.008 ± ~0.014 contains zero: unresolved, never "no cost".
    assert vs_p1[GENERAL]["verdict"] == "unresolved"
    assert vs_e1[PERSONAL]["verdict"] == "higher than e1-50"
    assert vs_e1[GENERAL]["verdict"] == "lower than e1-50"
    verdicts = [m for c in report["contrasts"].values() for m in c.values() if "half_width" in m]
    assert len(verdicts) == N_VERDICTS
    # The exploratory metrics: differences, never a verdict.
    assert all("verdict" not in m for c in report["contrasts"].values() for m in c.values()
               if m["role"] == "exploratory")


def test_the_rest_is_descriptive(design):
    report = p4_analysis(*design)
    assert sorted(report["descriptive"]) == ["e1-0", "e1-100", "e1-25"]
    assert all("verdict" not in m for arm in report["descriptive"].values() for m in arm.values())
    control = report["descriptive"]["e1-0"][GENERAL]["difference"]
    assert control == pytest.approx(statistics.fmean(P4_VALUES[GENERAL]) - statistics.fmean(E1_VALUES[0.0][GENERAL]))
    val = report["val_loss"]
    assert val["role"] == "descriptive" and val["floor"]["n"] == 4
    assert [p["final_val_loss"] for p in val["p4"]] == [3.71] * 4


def test_a_diverged_p4_run_leaves_no_verdict(tmp_path, repo_root):
    p4 = _group(tmp_path, repo_root, P4_CONFIGS[:3], P4_VALUES)
    p4.append(_pretrain(tmp_path, repo_root, P4_CONFIGS[3], {}, status="diverged"))
    report = p4_analysis(p4, _group(tmp_path, repo_root, P1_CONFIGS, P1_VALUES), _e1(tmp_path, repo_root))
    assert report["diverged"] == [p4[3].name]
    for contrast in report["contrasts"].values():
        for metric in PRIMARY_EVAL_METRICS:
            assert "half_width" not in contrast[metric]
            assert contrast[metric]["verdict"].startswith("no verdict: P4 diverged")


@pytest.mark.parametrize("edits,match", [
    ({"precision": "fp32"}, "manifest.device.precision"),
    ({"training__max_steps": 2316}, "training.max_steps"),
    ({"training__learning_rate": 1e-3}, "training.learning_rate"),
    ({"model__pos": "learned"}, "model.pos"),
])
def test_refuses_a_p4_that_differs_from_p1_beyond_its_mixture(tmp_path, repo_root, edits, match):
    p4 = _group(tmp_path, repo_root, P4_CONFIGS, P4_VALUES, **edits)
    with pytest.raises(P4Error, match=match):
        p4_analysis(p4, _group(tmp_path, repo_root, P1_CONFIGS, P1_VALUES), _e1(tmp_path, repo_root))


def test_refuses_another_dose(tmp_path, repo_root):
    doubled = {"web": 4011, "dialogue": 1146, "code": 573, "chat": 40}
    p4 = _group(tmp_path, repo_root, P4_CONFIGS, P4_VALUES, data__mixture=doubled)
    with pytest.raises(P4Error, match="chat tokens expected"):
        p4_analysis(p4, _group(tmp_path, repo_root, P1_CONFIGS, P1_VALUES), _e1(tmp_path, repo_root))


def test_refuses_too_few_runs_and_shared_seeds(tmp_path, repo_root):
    p1 = _group(tmp_path, repo_root, P1_CONFIGS, P1_VALUES)
    e1 = _e1(tmp_path, repo_root)
    three = _group(tmp_path, repo_root, P4_CONFIGS[:3], P4_VALUES)
    with pytest.raises(P4Error, match="3 p4 run"):
        p4_analysis(three, p1, e1)
    same_seed = three + [_pretrain(tmp_path, repo_root, "p4.yaml", {})]
    with pytest.raises(P4Error, match="share a training.seed"):
        p4_analysis(same_seed, p1, e1)


def test_refuses_eval_files_not_alike_and_pilots(tmp_path, repo_root):
    p1 = _group(tmp_path, repo_root, P1_CONFIGS, P1_VALUES)
    e1 = _e1(tmp_path, repo_root)
    p4 = _group(tmp_path, repo_root, P4_CONFIGS[:3], P4_VALUES)
    p4.append(_pretrain(tmp_path, repo_root, P4_CONFIGS[3], {}, eval_fp="another-eval"))
    with pytest.raises(P4Error, match="not evaluated like the baseline"):
        p4_analysis(p4, p1, e1)
    other = tmp_path / "other"
    other.mkdir()
    p4 = _group(other, repo_root, P4_CONFIGS, P4_VALUES)
    pilots = _e1(other, repo_root, role_of=lambda base, share: "pilot" if base == 0 else "main")
    with pytest.raises(P4Error, match="pilots stay out"):
        p4_analysis(p4, p1, pilots)


def test_the_script_writes_the_report_or_refuses(repo_root, design, tmp_path):
    p4, p1, e1 = design
    out = tmp_path / "reports" / "verdicts.json"
    script = repo_root / "scripts" / "p4_compare.py"
    def run(p4_dirs):
        return subprocess.run([sys.executable, str(script), "--p4", *map(str, p4_dirs), "--p1", *map(str, p1),
                               "--e1", *map(str, e1), "--out", str(out)], capture_output=True, text=True)

    done = run(p4)
    assert done.returncode == 0, done.stderr
    assert "-> lower than p1" in done.stdout
    assert json.loads(out.read_text())["contrasts"]["e1-50"][GENERAL]["verdict"] == "lower than e1-50"
    refused = run(p4[:3])
    assert refused.returncode == 1 and "3 p4 run" in refused.stderr
