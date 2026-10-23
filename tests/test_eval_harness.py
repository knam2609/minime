"""The evaluation harness: perplexity, sampling, prompt sets, BLiMP, eval.json.

The end-to-end tests build a miniature of the real layout from the synthetic
fixture -- personal shards from ``sam``, general ``web``/``dialogue``/``code``
slices from ``control`` -- and run every family on an untrained model.  The
numbers are meaningless; the structure, the hashes, the test-split lock and
the absence of any document text in ``eval.json`` are what is checked.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from minime.config import ModelConfig
from minime.data.binidx import TokenShard, write_shard
from minime.data.document import read_jsonl
from minime.data.splits import assign_split, split_key
from minime.evaluation import harness
from minime.evaluation.generation import sample
from minime.evaluation.perplexity import cross_entropy, plan_windows, weighted_mean
from minime.evaluation.sets import document_prompts, evenly_spaced, prefix_prompts
from minime.evaluation.tasks import blimp
from minime.model import build_model
from minime.tokenizer import CharTokenizer

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "data" / "sample_synthetic"


# ----------------------------------------------------------------- perplexity


@pytest.mark.parametrize("n_tokens,ctx", [(2, 4), (17, 4), (16, 4), (100, 7), (1000, 64)])
def test_windows_score_every_target_exactly_once(n_tokens, ctx):
    windows = plan_windows(n_tokens, ctx)
    targets = [t for s, length in windows for t in range(s + 1, s + length + 1)]
    assert targets == list(range(1, n_tokens))
    assert all(length <= ctx for _, length in windows)


def test_capped_windows_are_full_disjoint_and_spread():
    windows = plan_windows(100_001, 100, max_tokens=1_000)
    assert len(windows) == 10 and all(length == 100 for _, length in windows)
    starts = [s for s, _ in windows]
    assert starts == sorted(set(starts)) and all(s % 100 == 0 for s in starts)
    assert starts[0] == 0 and starts[-1] >= 90_000
    assert starts[-1] + 100 <= 100_000


def test_capped_windows_never_undershoot_to_nothing():
    assert len(plan_windows(10_000, 512, max_tokens=10)) == 1


def test_a_single_token_shard_is_refused():
    with pytest.raises(ValueError):
        plan_windows(1, 8)


def _shard(tmp_path, tokens, name="s"):
    write_shard(tmp_path / name, [tokens], vocab_size=int(max(tokens)) + 1)
    return TokenShard(tmp_path / name)


def test_cross_entropy_matches_a_direct_computation(tmp_path, tiny_cfg):
    torch.manual_seed(0)
    model = build_model(tiny_cfg).eval()
    tokens = np.random.default_rng(0).integers(0, tiny_cfg.vocab_size, 40)
    shard = _shard(tmp_path, tokens)
    result = cross_entropy(model, shard, tiny_cfg.context_length, batch_size=2)

    nats, n = 0.0, 0
    for s, length in plan_windows(len(tokens), tiny_cfg.context_length):
        x = torch.tensor(tokens[s : s + length]).unsqueeze(0)
        y = torch.tensor(tokens[s + 1 : s + length + 1]).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(x)
        nats += float(F.cross_entropy(logits[0], y[0], reduction="sum"))
        n += length
    assert result["tokens"] == n == len(tokens) - 1
    assert result["loss"] == pytest.approx(nats / n, rel=1e-6)
    assert result["coverage"] == 1.0


def test_a_uniform_model_scores_ln_vocab_and_bits_per_byte_follows(tmp_path, tiny_cfg):
    model = build_model(tiny_cfg)
    model.norm_out.weight.data.zero_()  # every logit 0 -> uniform prediction
    shard = _shard(tmp_path, np.arange(50) % tiny_cfg.vocab_size)
    two_byte = [2] * tiny_cfg.vocab_size
    result = cross_entropy(model, shard, tiny_cfg.context_length, byte_lengths=two_byte)
    assert result["loss"] == pytest.approx(math.log(tiny_cfg.vocab_size), rel=1e-6)
    assert result["bits_per_byte"] == pytest.approx(math.log2(tiny_cfg.vocab_size) / 2, rel=1e-6)


def test_the_digest_names_the_text_scored(tmp_path, tiny_cfg):
    model = build_model(tiny_cfg)
    tokens = np.arange(60) % 50
    a = cross_entropy(model, _shard(tmp_path, tokens, "a"), 16)["digest"]
    assert a == cross_entropy(model, _shard(tmp_path, tokens, "b"), 16)["digest"]
    changed = tokens.copy()
    changed[30] += 1
    assert a != cross_entropy(model, _shard(tmp_path, changed, "c"), 16)["digest"]
    assert a != cross_entropy(model, _shard(tmp_path, tokens, "d"), 8)["digest"]


def test_weighted_mean_normalises_and_requires_every_slice():
    results = {"web": {"loss": 2.0}, "code": {"loss": 4.0}}
    assert weighted_mean(results, {"web": 3, "code": 1}) == pytest.approx(2.5)
    with pytest.raises(KeyError):
        weighted_mean(results, {"web": 1, "dialogue": 1})


# ------------------------------------------------------------------ sampling


class FixedLogits(nn.Module):
    """A 'model' whose next-token distribution never changes."""

    def __init__(self, logits, context_length=8):
        super().__init__()
        self.logits = torch.as_tensor(logits, dtype=torch.float32)
        self.cfg = SimpleNamespace(context_length=context_length, vocab_size=len(self.logits))

    def forward(self, idx, targets=None):
        b, t = idx.shape
        return self.logits.expand(b, t, -1).clone(), None


def test_rows_do_not_depend_on_batch_size(tiny_cfg):
    torch.manual_seed(0)
    model = build_model(tiny_cfg)
    prompts = np.random.default_rng(1).integers(1, tiny_cfg.vocab_size, (6, 5))
    runs = [sample(model, prompts, 10, eot_id=0, seed=7, batch_size=b).tokens for b in (1, 4, 6)]
    assert runs[0] == runs[1] == runs[2]
    assert sample(model, prompts, 10, eot_id=0, seed=8).tokens != runs[0]


def test_rows_stop_at_eot_and_exclude_it():
    always_eot = FixedLogits([10.0, 0.0, 0.0])
    gen = sample(always_eot, np.ones((3, 2), dtype=np.int64), 5, eot_id=0, temperature=0)
    assert gen.tokens == [[], [], []] and gen.ended == [True] * 3 and gen.eot_rate == 1.0
    never_eot = FixedLogits([-1e9, 5.0, 0.0])
    gen = sample(never_eot, np.ones((2, 2), dtype=np.int64), 4, eot_id=0)
    assert all(len(t) == 4 for t in gen.tokens) and gen.ended == [False, False]


def test_top_k_one_is_greedy_and_zero_probability_tokens_never_appear():
    logits = [0.0, 1.0, 3.0, 2.0]
    prompts = np.ones((20, 1), dtype=np.int64)
    greedy = sample(FixedLogits(logits), prompts, 3, eot_id=0, temperature=0).tokens
    top1 = sample(FixedLogits(logits), prompts, 3, eot_id=0, top_k=1).tokens
    assert greedy == top1 == [[2, 2, 2]] * 20
    top2 = sample(FixedLogits(logits), prompts, 30, eot_id=0, top_k=2).tokens
    assert {t for row in top2 for t in row} <= {2, 3}


def test_sampling_follows_the_distribution_and_entropy_is_the_models():
    logits = [-1e9, 0.0, 0.0]  # EOT impossible; tokens 1 and 2 at 50% each
    gen = sample(FixedLogits(logits), np.ones((400, 1), dtype=np.int64), 5, eot_id=0)
    share = np.mean([t == 1 for row in gen.tokens for t in row])
    assert 0.45 < share < 0.55
    assert gen.entropy == pytest.approx(math.log(2), rel=1e-4)
    assert gen.positions == 400 * 5


# ---------------------------------------------------------------- prompt sets


def test_evenly_spaced_covers_first_to_last():
    assert evenly_spaced(list(range(10)), 5) == [0, 2, 4, 6, 8]
    assert evenly_spaced([1, 2], 5) == [1, 2]
    assert evenly_spaced([], 3) == []


def _doc_shard(tmp_path, docs, eot=0):
    write_shard(tmp_path / "docs", docs, vocab_size=100, eot_id=eot)
    return TokenShard(tmp_path / "docs")


def test_document_prompts_end_with_the_previous_eot_and_target_the_document(tmp_path):
    docs = [[10 + i] * (3 + i % 4) for i in range(20)]
    shard = _doc_shard(tmp_path, docs)
    ps = document_prompts(shard, n=5, prompt_tokens=6, max_target_tokens=4, eot_id=0)
    assert ps.prompts.shape == (5, 6)
    for prompt, target, i in zip(ps.prompts, ps.targets, ps.doc_indices):
        assert prompt[-1] == 0  # the preceding document's EOT
        assert list(target) == docs[i][:4]
        assert int(shard.offsets[i]) >= 6
    again = document_prompts(shard, n=5, prompt_tokens=6, max_target_tokens=4, eot_id=0)
    assert again.digest == ps.digest and len(ps.digest) == 16


def test_prefix_prompts_split_long_documents(tmp_path):
    docs = [list(range(1, 3 + 5 * i)) for i in range(10)]
    shard = _doc_shard(tmp_path, docs)
    ps = prefix_prompts(shard, n=3, prompt_tokens=8, target_tokens=10, eot_id=0)
    for prompt, target, i in zip(ps.prompts, ps.targets, ps.doc_indices):
        assert list(prompt) == docs[i][:8] and list(target) == docs[i][8:18]
    with pytest.raises(ValueError):
        prefix_prompts(shard, n=3, prompt_tokens=100, target_tokens=100, eot_id=0)


# ---------------------------------------------------------------------- BLiMP


class Unigram(FixedLogits):
    pass


def test_sentence_logprobs_ignore_padding_and_sum_the_tokens():
    logp = torch.log_softmax(torch.tensor([0.0, 1.0, 2.0, 0.5]), -1)
    model = Unigram(torch.tensor([0.0, 1.0, 2.0, 0.5]), context_length=16)
    lists = [[1, 2], [3, 3, 3, 1, 2], [2]]
    together = blimp.sentence_logprobs(model, lists, bos_id=0, batch_size=8)
    alone = [blimp.sentence_logprobs(model, [ids], bos_id=0)[0] for ids in lists]
    assert together == pytest.approx(alone)
    assert together[0] == pytest.approx(float(logp[1] + logp[2]))


def _toy_pairs(path):
    rows = [
        {"paradigm": "p1", "field": "syntax", "phenomenon": "x", "pair_id": 0, "good": "cc", "bad": "ab"},
        {"paradigm": "p1", "field": "syntax", "phenomenon": "x", "pair_id": 1, "good": "ab", "bad": "cc"},
        {"paradigm": "p2", "field": "morphology", "phenomenon": "y", "pair_id": 0, "good": "cc", "bad": "aa"},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_blimp_accuracy_is_macro_over_paradigms(tmp_path):
    pairs, _ = blimp.load_pairs(_toy_pairs(tmp_path / "b.jsonl"), expected_sha256=None)
    tok = CharTokenizer(list("abc"))
    model = Unigram([0.0, 0.0, 0.0, 5.0])  # strongly prefers 'c' (id 3)
    result = blimp.evaluate(model, tok, pairs)
    assert result["by_paradigm"] == {"p1": 0.5, "p2": 1.0}
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["by_field"] == {"morphology": 1.0, "syntax": 0.5}


def test_blimp_refuses_a_file_that_is_not_the_pinned_one(tmp_path):
    path = _toy_pairs(tmp_path / "b.jsonl")
    with pytest.raises(ValueError, match="pinned"):
        blimp.load_pairs(path)
    pairs, _ = blimp.load_pairs(path, per_paradigm=1, expected_sha256=None)
    assert [(p["paradigm"], p["pair_id"]) for p in pairs] == [("p1", 0), ("p2", 0)]


# ------------------------------------------------------------ eval config


def test_eval_yaml_documents_the_defaults():
    assert harness.load_eval_config(REPO / "configs" / "eval.yaml") == harness.EvalConfig()


def test_eval_config_rejects_unknown_keys_and_families():
    with pytest.raises(ValueError, match="unknown"):
        harness.load_eval_config(None, {"voice": {"promts": 3}})
    with pytest.raises(ValueError, match="unknown eval family"):
        harness.load_eval_config(None, {"families": ["vibes"]})


def test_fingerprint_ignores_families_but_not_settings():
    base = harness.EvalConfig()
    assert harness.load_eval_config(None, {"families": ["blimp"]}).fingerprint() == base.fingerprint()
    assert harness.load_eval_config(None, {"voice": {"prompts": 3}}).fingerprint() != base.fingerprint()


# Every probe-scale evaluation (P1, P2, P3, P4, E1) carries it.
PROBE_EVAL_FINGERPRINT = "7c9503161d6f"


def test_eval_yaml_keeps_the_fingerprint_of_the_evaluations_already_written():
    assert harness.load_eval_config(REPO / "configs" / "eval.yaml").fingerprint() == PROBE_EVAL_FINGERPRINT
    # 512-token probes with eval.yaml; 1024-token flagship models with eval_flagship.yaml.
    by_context = {512: PROBE_EVAL_FINGERPRINT,
                  1024: harness.load_eval_config(REPO / "configs" / "eval_flagship.yaml").fingerprint()}
    written = sorted((REPO / "experiments").glob("*/eval-last.json"))
    assert written
    for path in written:
        context = json.loads((path.parent / "manifest.json").read_text())["config"]["model"]["context_length"]
        assert json.loads(path.read_text())["eval_config_fingerprint"] == by_context[context], path


@pytest.mark.parametrize("setting,value", [("context_length", 512), ("batch_size", 8)])
def test_a_later_setting_counts_in_the_fingerprint_once_set(setting, value):
    base = harness.EvalConfig()
    changed = harness.load_eval_config(None, {"perplexity": {setting: value}})
    assert changed.fingerprint() != base.fingerprint()
    assert changed.to_dict()["perplexity"][setting] == value


@pytest.mark.parametrize("value", [0, -512, 1.5, True, "512"])
def test_the_ce_settings_must_be_positive_integers(value):
    with pytest.raises(ValueError, match="positive integer"):
        harness.load_eval_config(None, {"perplexity": {"context_length": value}})


def test_the_flagship_and_common_context_configs_change_only_the_ce_settings():
    base = harness.load_eval_config(REPO / "configs" / "eval.yaml").to_dict()
    flagship = harness.load_eval_config(REPO / "configs" / "eval_flagship.yaml")
    common = harness.load_eval_config(REPO / "configs" / "eval_common512.yaml")
    assert flagship.to_dict() == {**base, "perplexity": {**base["perplexity"], "batch_size": 8}}
    assert common.to_dict() == {**base, "families": ["perplexity"],
                                "perplexity": {**base["perplexity"], "context_length": 512, "batch_size": 16}}
    fingerprints = {PROBE_EVAL_FINGERPRINT, flagship.fingerprint(), common.fingerprint()}
    assert len(fingerprints) == 3


# --------------------------------------------------------------- end to end

SMALL = {
    "batch_size": 16,
    "families": ["perplexity", "voice", "degeneration", "code", "blimp"],
    "perplexity": {"max_tokens": 2048},
    "voice": {"prompts": 30, "prompt_tokens": 32, "max_new_tokens": 40, "profile_words": 800,
              "chunk_words": 25, "mfw": 30, "folds": 3, "classifier_docs": 40},
    "degeneration": {"prompts": 6, "prompt_tokens": 24, "max_new_tokens": 40},
    "code": {"prompts": 4, "prompt_tokens": 48, "max_new_tokens": 40},
}


@pytest.fixture(scope="module")
def layout(tmp_path_factory):
    """Personal and general shard directories shaped like the real ones."""
    root = tmp_path_factory.mktemp("eval-layout")
    sam = list(read_jsonl(FIXTURE / "sam.jsonl"))
    ctrl = list(read_jsonl(FIXTURE / "control.jsonl"))
    tok = CharTokenizer.fit(d.text for d in sam + ctrl)
    ratios = {"train": 0.8, "val": 0.1, "test": 0.1}

    def write(path, docs):
        write_shard(path, tok.encode_batch([d.text for d in docs]), tok.vocab_size, tok.id, tok.eot_id)

    def split(docs):
        out = {"train": [], "val": [], "test": []}
        for d in docs:
            out[assign_split(split_key(d), ratios)].append(d)
        return out

    personal = split([d for d in sam if d.source == "chat"])
    write(root / "personal" / "chat.train", personal["train"])
    write(root / "personal" / "val", personal["val"])
    write(root / "personal" / "test", personal["test"])
    for name, source in (("web", "longform"), ("dialogue", "chat"), ("code", "code")):
        for s, docs in split([d for d in ctrl if d.source == source]).items():
            write(root / "general" / f"{name}.{s}", docs)
    tok.save(root / "tokenizer.json")
    blimp_path = _toy_pairs(root / "blimp.jsonl")
    val_texts = [d.text for d in personal["val"]]
    return SimpleNamespace(root=root, tok=tok, blimp=blimp_path, val_texts=val_texts)


def _model(tok, context=96):
    torch.manual_seed(0)
    return build_model(ModelConfig(vocab_size=tok.vocab_size, d_model=32, n_heads=4, n_layers=2,
                                   context_length=context))


def _run(layout, monkeypatch, **kwargs):
    monkeypatch.setattr(blimp, "PINNED_SHA256", None)
    cfg = harness.load_eval_config(None, {**SMALL, "blimp": {"path": str(layout.blimp)}})
    return harness.run_eval(_model(layout.tok), layout.tok, cfg, personal=layout.root / "personal",
                            general=layout.root / "general", log=lambda _: None, **kwargs)


def test_every_family_runs_and_eval_json_holds_numbers_not_text(layout, monkeypatch, tmp_path):
    record, samples = _run(layout, monkeypatch)
    for family in harness.FAMILIES:
        assert family in record and family in record["seconds"]
    assert record["perplexity"]["personal"]["coverage"] == 1.0
    assert set(record["perplexity"]["general"]) == {"web", "dialogue", "code"}
    assert "general_weighted" in record["perplexity"]
    assert 0.0 <= record["voice"]["authorship"]["discriminator"]["auc"] <= 1.0
    assert record["code"]["reference_valid"] == 1.0
    assert {"ce/personal/val", "voice/prompts", "degeneration/prompts", "code/prompts", "blimp"} <= set(
        record["eval_sets"])

    out = tmp_path / "eval.json"
    harness.write_outputs(record, samples, out, tmp_path / "samples" / "eval.jsonl")
    dumped = out.read_text()
    json.loads(dumped)
    # Privacy: no held-out message (or any long piece of one) is in eval.json.
    for text in layout.val_texts:
        for line in text.splitlines():
            if len(line) >= 12:
                assert line not in dumped
    assert all("text" in row for row in samples)
    assert (tmp_path / "samples" / "eval.jsonl").exists()


def test_the_noise_floors_eval_metrics_are_numbers_in_a_real_record(layout, monkeypatch):
    from minime.training.noise_floor import EVAL_METRICS, eval_metric

    record, _ = _run(layout, monkeypatch)
    for metric in EVAL_METRICS:
        eval_metric(record, metric)  # raises unless a finite number is at that path


def test_the_same_inputs_give_the_same_record(layout, monkeypatch):
    a, _ = _run(layout, monkeypatch)
    b, _ = _run(layout, monkeypatch)
    for key in ("eval_sets", "perplexity", "voice", "degeneration", "code", "blimp"):
        assert a[key] == b[key], key


def test_the_test_split_is_locked(layout, monkeypatch):
    with pytest.raises(PermissionError, match="final report"):
        _run(layout, monkeypatch, split="test")
    record, _ = _run(layout, monkeypatch, split="test", unlock_test=True)
    assert record["split"] == "test" and "ce/personal/test" in record["eval_sets"]


def _ce(layout, monkeypatch, **perplexity):
    monkeypatch.setattr(blimp, "PINNED_SHA256", None)
    cfg = harness.load_eval_config(None, {**SMALL, "families": ["perplexity"],
                                          "perplexity": {**SMALL["perplexity"], **perplexity}})
    record, _ = harness.run_eval(_model(layout.tok), layout.tok, cfg, personal=layout.root / "personal",
                                 general=layout.root / "general", log=lambda _: None)
    return record


def test_the_ce_settings_unset_score_on_the_models_own_windows(layout, monkeypatch):
    unset = _ce(layout, monkeypatch)
    native = _ce(layout, monkeypatch, context_length=96, batch_size=SMALL["batch_size"])
    assert unset["perplexity"]["context_length"] == unset["perplexity"]["native_context_length"] == 96
    assert unset["perplexity"] == native["perplexity"]
    assert unset["eval_sets"] == native["eval_sets"]
    assert unset["eval_config_fingerprint"] != native["eval_config_fingerprint"]


def test_the_ce_batch_size_changes_the_summation_not_the_windows(layout, monkeypatch):
    a = _ce(layout, monkeypatch)
    batches = []
    real = harness.cross_entropy
    monkeypatch.setattr(harness, "cross_entropy",
                        lambda *args, **kwargs: batches.append(args[3]) or real(*args, **kwargs))
    b = _ce(layout, monkeypatch, batch_size=3)
    assert batches and set(batches) == {3}
    assert a["eval_sets"] == b["eval_sets"]
    for key in ("personal", "general_weighted"):
        assert b["perplexity"][key]["loss"] == pytest.approx(a["perplexity"][key]["loss"], rel=1e-6)


def test_a_shorter_scoring_context_is_other_windows_and_a_longer_one_is_refused(layout, monkeypatch):
    native, short = _ce(layout, monkeypatch), _ce(layout, monkeypatch, context_length=48)
    assert short["perplexity"]["context_length"] == 48
    assert short["perplexity"]["native_context_length"] == 96
    assert short["eval_sets"]["ce/personal/val"]["digest"] != native["eval_sets"]["ce/personal/val"]["digest"]
    assert short["perplexity"]["personal"]["windows"] > native["perplexity"]["personal"]["windows"]
    assert short["perplexity"]["personal"]["coverage"] == 1.0
    with pytest.raises(ValueError, match="longer than the model's context"):
        _ce(layout, monkeypatch, context_length=97)


def test_a_tokenizer_the_shards_were_not_made_with_is_refused(layout):
    other = CharTokenizer(list("xyz"))
    model = build_model(ModelConfig(vocab_size=other.vocab_size, d_model=32, n_heads=4, n_layers=2,
                                    context_length=32))
    with pytest.raises(ValueError, match="not by the tokenizer"):
        harness.run_eval(model, other, harness.EvalConfig(families=["perplexity"]),
                         personal=layout.root / "personal", log=lambda _: None)


def test_evaluate_script_writes_eval_and_gitignored_samples(layout, tmp_path):
    from minime.training.checkpoint import save_checkpoint

    model = _model(layout.tok)
    cfg = {"name": "tiny-eval", "model": {**vars(model.cfg)}}
    ckpt = save_checkpoint(tmp_path / "run" / "last.pt", model=model, step=3, config=cfg)
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "evaluate.py"), "--checkpoint", str(ckpt),
         "--tokenizer", str(layout.root / "tokenizer.json"), "--personal", str(layout.root / "personal"),
         "--general", str(layout.root / "general"), "--families", "perplexity,voice",
         "--device", "cpu", "--set", *[f"voice.{k}={v}" for k, v in SMALL["voice"].items()]],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert "impersonation AUC" in result.stdout and "NOTE: dev tokenizer" in result.stdout
    record = json.loads((tmp_path / "run" / "eval-last.json").read_text())
    assert record["checkpoint"]["step"] == 3 and record["tokenizer"]["dev"] is True
    assert (tmp_path / "run" / "samples" / "eval-last.jsonl").exists()

    refused = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "evaluate.py"), "--checkpoint", str(ckpt),
         "--tokenizer", str(layout.root / "tokenizer.json"), "--personal", str(layout.root / "personal"),
         "--split", "test"],
        capture_output=True, text=True, timeout=600,
    )
    assert refused.returncode != 0 and "--final-report" in refused.stderr

    # Another scoring context writes beside the native evaluation, never over it.
    native = (tmp_path / "run" / "eval-last.json").read_bytes()
    common = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "evaluate.py"), "--checkpoint", str(ckpt),
         "--tokenizer", str(layout.root / "tokenizer.json"), "--personal", str(layout.root / "personal"),
         "--families", "perplexity", "--device", "cpu", "--set", "perplexity.context_length=48"],
        capture_output=True, text=True, timeout=600,
    )
    assert common.returncode == 0, common.stderr
    assert "CE scored on 48-token windows (the model's own context is 96)" in common.stdout
    assert (tmp_path / "run" / "eval-last.json").read_bytes() == native
    record = json.loads((tmp_path / "run" / "eval-last-ctx48.json").read_text())
    assert record["perplexity"]["context_length"] == 48
    assert not (tmp_path / "run" / "samples" / "eval-last-ctx48.jsonl").exists()


def test_generate_script_is_reproducible(layout, tmp_path):
    from minime.training.checkpoint import save_checkpoint

    model = _model(layout.tok)
    ckpt = save_checkpoint(tmp_path / "last.pt", model=model, step=1,
                           config={"name": "g", "model": {**vars(model.cfg)}})
    cmd = [sys.executable, str(REPO / "scripts" / "generate.py"), "--checkpoint", str(ckpt),
           "--tokenizer", str(layout.root / "tokenizer.json"), "--prompt", "ok so", "--n", "2",
           "--max-new-tokens", "12", "--device", "cpu", "--seed", "5"]
    runs = [subprocess.run(cmd, capture_output=True, text=True, timeout=300) for _ in range(2)]
    assert runs[0].returncode == 0, runs[0].stderr
    assert "--- 0" in runs[0].stdout and "--- 1" in runs[0].stdout
    assert runs[0].stdout == runs[1].stdout


def test_samples_directory_is_gitignored():
    ignored = subprocess.run(["git", "check-ignore", "-q", "experiments/some-run/samples/eval-last.jsonl"],
                             cwd=REPO)
    assert ignored.returncode == 0
