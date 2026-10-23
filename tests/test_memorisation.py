"""The memorisation check: its prompt sets, extraction, the run index, the report and the gate.

The protocol was decided with the owner before the check ran (README,
"Memorisation check -- protocol"): these tests pin what it counts.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from minime.data.binidx import TokenShard, write_shard
from minime.evaluation.memorisation import (
    GROUPS,
    MAX_TOKENS,
    NEGATIVE,
    POSITIVE,
    MemorisationError,
    RunIndex,
    extract,
    extraction,
    gate,
    memorisation_report,
    run_stats,
    voice_texts,
)
from minime.evaluation.sets import PromptSet, document_prompts, message_prompts

EOT = 0


def _shard(path, docs):
    write_shard(path, docs, vocab_size=5000, eot_id=EOT)
    return TokenShard(path)


def _messages(base, n=40):
    """Messages of 8-12 tokens whose every token is unique, so the next token is never ambiguous."""
    return [[base + 20 * i + j for j in range(8 + i % 5)] for i in range(n)]


class Recall(torch.nn.Module):
    """Writes back the stream it was built from: the token that followed the last two."""

    def __init__(self, stream, vocab=5000, context=128):
        super().__init__()
        self.cfg = type("Cfg", (), {"context_length": context})()
        self.vocab = vocab
        s = [int(t) for t in stream]
        self.next = {(a, b): c for a, b, c in zip(s, s[1:], s[2:])}

    def forward(self, idx, targets=None):
        logits = torch.zeros(idx.shape[0], idx.shape[1], self.vocab)
        for row in range(idx.shape[0]):
            logits[row, -1, self.next.get((int(idx[row, -2]), int(idx[row, -1])), EOT)] = 1.0
        return logits, None


# ------------------------------------------------------------ the prompt sets


def test_message_prompts_take_whole_messages_in_the_length_window(tmp_path):
    docs = [[100 + 20 * i + j for j in range(2 + i % 12)] for i in range(60)]
    shard = _shard(tmp_path / "s", docs)
    ps = message_prompts(shard, n=10, prompt_tokens=16, min_tokens=8, max_tokens=11, eot_id=EOT)
    assert ps.prompts.shape == (10, 16)
    for prompt, target, i in zip(ps.prompts, ps.targets, ps.doc_indices):
        assert 8 <= len(docs[i]) <= 11 and list(target) == docs[i]   # whole, never cropped
        assert prompt[-1] == EOT and int(shard.offsets[i]) >= 16
    eligible = [i for i, d in enumerate(docs) if 8 <= len(d) <= 11 and int(shard.offsets[i]) >= 16]
    assert ps.meta["eligible_documents"] == len(eligible) and ps.meta["kind"] == "message"
    assert message_prompts(shard, 10, 16, 8, 11, EOT).digest == ps.digest
    with pytest.raises(ValueError, match="no message"):
        message_prompts(shard, 5, 16, 50, 60, EOT)


# --------------------------------------------------------------- extraction


def test_a_message_counts_as_extracted_only_when_written_exactly_and_ended():
    ps = PromptSet(np.zeros((4, 2), dtype=np.int64), [np.array([5, 6, 7])] * 4, [0, 1, 2, 3], "d")
    result = extraction([[5, 6, 7], [5, 6, 7], [5, 6, 7, 8], [5, 9, 7]], [True, False, False, True], ps)
    assert result["extracted"] == 1 and result["extracted_documents"] == [0]
    # Cut before its EOT, or run on past the message: not extracted, though every token agreed.
    assert result["mean_matched_fraction"] == pytest.approx((1 + 1 + 1 + 1 / 3) / 4)


def test_a_model_that_recalls_its_training_stream_extracts_train_and_not_val(tmp_path):
    train = _shard(tmp_path / "train", _messages(100))
    val = _shard(tmp_path / "val", _messages(3000))
    model = Recall(train.tokens)
    sets = {name: message_prompts(s, 10, 16, 8, MAX_TOKENS, EOT) for name, s in (("train", train), ("val", val))}
    on_train = extract(model, sets["train"], EOT, device="cpu")
    on_val = extract(model, sets["val"], EOT, device="cpu")
    assert on_train["rate"] == 1.0 and on_train["mean_matched_fraction"] == 1.0
    assert on_train["extracted_documents"] == sets["train"].doc_indices
    assert on_val["rate"] == 0.0 and on_val["mean_matched_fraction"] == 0.0


# ----------------------------------------------------------- the run index


def _brute_longest(stream, seq):
    text = list(stream)
    best = 0
    for i in range(len(seq)):
        for length in range(best + 1, len(seq) - i + 1):
            piece = list(seq[i:i + length])
            if any(text[p:p + length] == piece for p in range(len(text) - length + 1)):
                best = length
            else:
                break
    return best


def test_the_run_index_agrees_with_brute_force():
    rng = np.random.default_rng(0)
    stream = rng.integers(1, 5, size=300)   # four symbols: many shared runs
    index = RunIndex(stream, k=4)
    for _ in range(60):
        seq = rng.integers(1, 5, size=int(rng.integers(1, 25)))
        brute = _brute_longest(stream, seq)
        assert index.longest_run(seq) == (brute if brute >= 4 else 0)
    planted = np.concatenate([[9, 9], stream[100:140], [9]])
    assert index.longest_run(planted) >= 40


def test_run_stats():
    stats = run_stats([0, 4, 8, 16, 32])
    assert stats["n"] == 5 and stats["max"] == 32 and stats["mean"] == pytest.approx(12.0)
    assert stats["share_at_least"] == {"8": 0.6, "16": 0.4, "32": 0.2}


# -------------------------------------------------------- the voice samples


def test_voice_samples_must_be_the_evals_own(tmp_path):
    shard = _shard(tmp_path / "val", _messages(100))
    ps = document_prompts(shard, 6, 16, 8, EOT)
    record = {"eval_sets": {"voice/prompts": ps.describe()}}
    rows = [{"family": "voice", "prompt": i, "doc": int(d), "text": f"t{i}"} for i, d in enumerate(ps.doc_indices)]
    rows.append({"family": "degeneration", "prompt": 0, "doc": 0, "text": "x"})
    path = tmp_path / "samples.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in reversed(rows)))
    assert voice_texts(path, record, ps) == [f"t{i}" for i in range(6)]
    with pytest.raises(MemorisationError, match="not the same prompts"):
        voice_texts(path, {"eval_sets": {"voice/prompts": {"digest": "other"}}}, ps)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows[1:]))
    with pytest.raises(MemorisationError, match="not one per prompt"):
        voice_texts(path, record, ps)
    path.write_text("".join(json.dumps({**r, "doc": r["doc"] + 1}) + "\n" for r in rows))
    with pytest.raises(MemorisationError, match="other documents"):
        voice_texts(path, record, ps)


# --------------------------------------------------------------- the report


def _record(train_rate, val_rate=0.01, overlap_mean=3.0, real_p95=9.0):
    stats = {"n": 512, "mean": overlap_mean, "p50": 2.0, "p95": 10.0, "max": 20,
             "share_at_least": {"8": 0.2, "16": 0.05, "32": 0.0}}
    real = {**stats, "mean": 2.5, "p95": real_p95}
    return {"sets": {"train": {"digest": "a"}, "val": {"digest": "b"}, "voice": {"digest": "c"}},
            "extraction": {"train": {"rate": train_rate, "mean_matched_fraction": 0.4},
                           "val": {"rate": val_rate, "mean_matched_fraction": 0.3}},
            "overlap": {"generated": stats, "real": real}}


def _records(positive=(0.30, 0.28, 0.35, 0.31), negative=(0.01, 0.012, 0.009, 0.011)):
    records = {run: _record(0.05) for runs in GROUPS.values() for run in runs}
    for run, rate in zip(GROUPS[POSITIVE], positive):
        records[run] = _record(rate)
    for run, rate in zip(GROUPS[NEGATIVE], negative):
        records[run] = _record(rate)
    return records


def test_the_report_sets_the_groups_side_by_side():
    report = memorisation_report(_records())
    assert list(report["groups"]) == list(GROUPS)
    gap = report["groups"][POSITIVE]["gap"]
    assert gap["values"] == pytest.approx([0.29, 0.27, 0.34, 0.30]) and gap["sd"] is not None
    assert report["groups"]["flagship"]["gap"]["sd"] is None
    assert report["gate"]["threshold"] == 9.0
    assert report["sensitivity"]["sensitive"] is True and report["sensitivity"]["reading"] is None


def test_the_check_is_insensitive_unless_every_overfit_seed_is_above_every_p1_seed():
    report = memorisation_report(_records(positive=(0.30, 0.28, 0.011, 0.31), negative=(0.01, 0.012, 0.0, 0.0)))
    assert report["sensitivity"]["sensitive"] is False
    assert "no claim of low memorisation" in report["sensitivity"]["reading"]


def test_the_report_refuses_missing_or_unlike_records():
    records = _records()
    records.pop(GROUPS["P4"][2])
    with pytest.raises(MemorisationError, match="no memo-last.json"):
        memorisation_report(records)
    records = _records()
    records[GROUPS["P4"][0]]["sets"]["train"]["digest"] = "other"
    with pytest.raises(MemorisationError, match="different prompt sets"):
        memorisation_report(records)
    records = _records()
    records[GROUPS["P4"][0]]["overlap"]["real"]["p95"] = 99.0
    with pytest.raises(MemorisationError, match="real continuations"):
        memorisation_report(records)


def test_the_groups_are_the_protocols(repo_root):
    assert sum(len(runs) for runs in GROUPS.values()) == 30
    adaptation = repo_root / "reports" / "m10" / "adaptation.json"
    if adaptation.exists():
        assert GROUPS["flagship + C"] == (json.loads(adaptation.read_text())["final_run"],)


# ----------------------------------------------------------------- the gate


def test_the_gate_fails_a_long_shared_run_a_name_or_pii(tmp_path):
    chat = "we should totally get dumplings after class on friday"
    encode = lambda text: [ord(c) for c in text]   # noqa: E731
    index = RunIndex(np.array(encode(chat)), k=4)
    names = tmp_path / "names.txt"
    names.write_text("# a header\nZorblat\n")
    texts = ["hmm maybe next week", "get dumplings after class", "ask Zorblat first",
             "mail me at a@b.com", "no idea tbh"]
    results = gate(texts, encode, index, threshold=8, names_file=names)
    assert [r["pass"] for r in results] == [True, False, False, False, True]
    assert results[1]["longest_run"] == len("get dumplings after class")
    assert results[2]["scrubber"] == {"name": 1} and results[3]["scrubber"] == {"email": 1}
    assert all(set(r) == {"sample", "longest_run", "scrubber", "pass"} for r in results)   # never the text
    (tmp_path / "empty.txt").write_text("# nothing\n")
    with pytest.raises(MemorisationError, match="no names"):
        gate(texts, encode, index, 8, tmp_path / "empty.txt")
