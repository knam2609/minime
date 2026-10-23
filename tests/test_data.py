"""Tests for the token storage format and the mixture sampler.

The mixture proportions test is the one that guards experiment E2: if the
sampler quietly draws 60/40 when the config says 80/20, every conclusion drawn
from that experiment is wrong and nothing else would reveal it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from minime.data import MixtureSampler, ShardWriter, TokenShard, make_val_batches, shard_stem
from minime.data.document import Document, make_id, read_jsonl, write_jsonl


@pytest.fixture
def shard_factory(tmp_path):
    def build(name: str, n_docs: int = 20, doc_len: int = 64, vocab: int = 100, base: int = 0):
        rng = np.random.default_rng(abs(hash(name)) % 2**32)
        path = tmp_path / name
        with ShardWriter(path, vocab_size=vocab, eot_id=vocab - 1, tokenizer="test") as writer:
            for _ in range(n_docs):
                writer.add_document(rng.integers(base, base + 50, size=doc_len))
        return TokenShard(path)

    return build


# ------------------------------------------------------------------ storage


def test_shard_roundtrip_preserves_tokens_and_document_boundaries(tmp_path):
    docs = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    with ShardWriter(tmp_path / "s", vocab_size=100) as writer:
        writer.add_documents(docs)

    shard = TokenShard(tmp_path / "s")
    assert shard.n_docs == 3
    assert shard.n_tokens == 9
    for original, stored in zip(docs, shard.documents()):
        assert list(stored) == original


def test_eot_token_is_appended_between_documents(tmp_path):
    with ShardWriter(tmp_path / "s", vocab_size=100, eot_id=99) as writer:
        writer.add_document([1, 2])
        writer.add_document([3])

    shard = TokenShard(tmp_path / "s")
    assert list(np.asarray(shard.tokens)) == [1, 2, 99, 3, 99]
    assert shard.n_tokens == 5


def test_empty_documents_are_skipped(tmp_path):
    with ShardWriter(tmp_path / "s", vocab_size=10) as writer:
        writer.add_document([])
        writer.add_document([1])
    shard = TokenShard(tmp_path / "s")
    assert shard.n_docs == 1


def test_empty_shard_opens_as_zero_tokens(tmp_path):
    with ShardWriter(tmp_path / "s", vocab_size=10) as writer:
        writer.add_document([])
    shard = TokenShard(tmp_path / "s")
    assert shard.n_tokens == 0 and shard.n_docs == 0
    with pytest.raises(ValueError, match="smaller than one context window"):
        make_val_batches(shard, context_length=4, batch_size=1, n_batches=1)


def test_out_of_range_token_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="out of range"):
        with ShardWriter(tmp_path / "s", vocab_size=10) as writer:
            writer.add_document([1, 999])


def test_metadata_is_written(tmp_path):
    with ShardWriter(tmp_path / "s", vocab_size=50, tokenizer="char-v1") as writer:
        writer.add_document([1, 2, 3])
    shard = TokenShard(tmp_path / "s")
    assert shard.meta["n_tokens"] == 3
    assert shard.meta["tokenizer"] == "char-v1"
    assert shard.meta["vocab_size"] == 50
    assert shard.meta["created"]


def test_shard_stem_survives_dots_in_the_name():
    assert shard_stem("data/personal.v2.bin").name == "personal.v2"
    assert shard_stem("data/personal.v2.idx.npy").name == "personal.v2"
    assert shard_stem("data/personal.v2").name == "personal.v2"


def test_missing_shard_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        TokenShard(tmp_path / "nope")


# ------------------------------------------------------------------ mixture


def test_sampler_shapes_and_target_shift(shard_factory):
    shard = shard_factory("a")
    sampler = MixtureSampler({"a": shard}, {"a": 1.0}, context_length=16, batch_size=8, seed=0)
    x, y = sampler.batch()

    assert x.shape == y.shape == (8, 16)
    assert x.dtype == torch.int64
    # y is x advanced by exactly one position.
    torch.testing.assert_close(x[:, 1:], y[:, :-1])


def test_realised_mixture_matches_requested_weights(shard_factory):
    shards = {"prose": shard_factory("prose"), "code": shard_factory("code")}
    sampler = MixtureSampler(shards, {"prose": 8, "code": 2}, context_length=16, batch_size=64, seed=3)
    for _ in range(40):
        sampler.batch()

    realised = sampler.realised_mixture
    assert sampler.requested_mixture == pytest.approx({"prose": 0.8, "code": 0.2})
    assert realised["prose"] == pytest.approx(0.8, abs=0.03)
    assert realised["code"] == pytest.approx(0.2, abs=0.03)
    assert sampler.tokens_drawn == 40 * 64 * 16


def test_zero_weight_source_is_never_drawn(shard_factory):
    shards = {"prose": shard_factory("prose"), "code": shard_factory("code")}
    sampler = MixtureSampler(shards, {"prose": 1.0, "code": 0.0}, context_length=16, batch_size=32, seed=1)
    for _ in range(10):
        sampler.batch()
    assert sampler.realised_mixture["code"] == 0.0


def test_same_seed_gives_identical_batches(shard_factory):
    shards = {"a": shard_factory("a"), "b": shard_factory("b", base=50)}
    weights = {"a": 1.0, "b": 1.0}
    one = MixtureSampler(shards, weights, context_length=16, batch_size=8, seed=42)
    two = MixtureSampler(shards, weights, context_length=16, batch_size=8, seed=42)
    for _ in range(5):
        x1, y1 = one.batch()
        x2, y2 = two.batch()
        assert torch.equal(x1, x2) and torch.equal(y1, y2)


def test_different_seeds_give_different_batches(shard_factory):
    shards = {"a": shard_factory("a")}
    one = MixtureSampler(shards, {"a": 1.0}, context_length=16, batch_size=8, seed=1)
    two = MixtureSampler(shards, {"a": 1.0}, context_length=16, batch_size=8, seed=2)
    assert not torch.equal(one.batch()[0], two.batch()[0])


def test_sampler_state_restores_the_exact_draw_sequence(shard_factory):
    """Resuming a run must not replay or skip data."""
    shards = {"a": shard_factory("a"), "b": shard_factory("b", base=50)}
    weights = {"a": 3.0, "b": 1.0}

    original = MixtureSampler(shards, weights, context_length=16, batch_size=8, seed=11)
    for _ in range(4):
        original.batch()
    state = original.state_dict()
    expected = [original.batch()[0] for _ in range(3)]

    resumed = MixtureSampler(shards, weights, context_length=16, batch_size=8, seed=11)
    resumed.load_state_dict(state)
    actual = [resumed.batch()[0] for _ in range(3)]

    for want, got in zip(expected, actual):
        assert torch.equal(want, got)
    assert resumed.tokens_drawn == original.tokens_drawn


def test_batches_reach_a_device_other_than_cuda_by_a_blocking_copy(shard_factory, monkeypatch):
    """MPS may read a non-blocking copy's source after ``batch`` has freed it (HANDOFF §3 M10g)."""
    copies = []

    def to(self, device, non_blocking=False):
        copies.append((torch.device(device).type, non_blocking))
        return self

    sampler = MixtureSampler({"a": shard_factory("a")}, {"a": 1.0}, context_length=16, batch_size=8, seed=0)
    monkeypatch.setattr(torch.Tensor, "to", to)
    for device in ("mps", "cuda", "cpu", torch.device("mps")):
        sampler.batch(device)
    assert copies == [("mps", False)] * 2 + [("cuda", True)] * 2 + [("cpu", False)] * 2 + [("mps", False)] * 2


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_batches_on_mps_are_the_batches_drawn(shard_factory):
    """Drawn back to back with no sync between them, as a training step's micro-batches are."""
    shards = {"a": shard_factory("a", n_docs=200), "b": shard_factory("b", n_docs=200, base=50)}
    one = MixtureSampler(shards, {"a": 1.0, "b": 1.0}, context_length=512, batch_size=4, seed=5)
    two = MixtureSampler(shards, {"a": 1.0, "b": 1.0}, context_length=512, batch_size=4, seed=5)
    on_mps = [one.batch("mps") for _ in range(64)]
    torch.mps.synchronize()
    for x, y in on_mps:
        want_x, want_y = two.batch()
        assert torch.equal(x.cpu(), want_x) and torch.equal(y.cpu(), want_y)


def test_unknown_source_in_mixture_is_rejected(shard_factory):
    with pytest.raises(KeyError, match="unknown shard"):
        MixtureSampler({"a": shard_factory("a")}, {"missing": 1.0}, context_length=8, batch_size=2)


def test_shard_too_small_for_context_is_rejected(tmp_path):
    with ShardWriter(tmp_path / "small", vocab_size=10) as writer:
        writer.add_document([1, 2, 3])
    with pytest.raises(ValueError, match="fewer than"):
        MixtureSampler({"small": TokenShard(tmp_path / "small")}, {"small": 1.0},
                       context_length=64, batch_size=2)


# --------------------------------------------------------------- validation


def test_validation_batches_are_identical_across_calls(shard_factory):
    shard = shard_factory("val", n_docs=40)
    first = make_val_batches(shard, context_length=16, batch_size=4, n_batches=3)
    second = make_val_batches(shard, context_length=16, batch_size=4, n_batches=3)

    assert len(first) == 3
    for (x1, y1), (x2, y2) in zip(first, second):
        assert torch.equal(x1, x2) and torch.equal(y1, y2)


def test_validation_windows_are_spread_across_the_shard(shard_factory):
    shard = shard_factory("val", n_docs=40)
    batches = make_val_batches(shard, context_length=16, batch_size=4, n_batches=4)
    firsts = torch.cat([x[:, 0] for x, _ in batches])
    # Evenly spaced windows must not collapse onto a single region.
    assert firsts.unique().numel() > 8


# --------------------------------------------------------------- documents


def test_document_jsonl_roundtrip(tmp_path):
    docs = [
        Document(id=make_id("a", 1), source="chat", text="hi there", author="me"),
        Document(id=make_id("a", 2), source="code", text="def f():\n    pass\n", author="control"),
    ]
    path = tmp_path / "docs.jsonl"
    assert write_jsonl(path, docs) == 2

    loaded = list(read_jsonl(path))
    assert [d.to_dict() for d in loaded] == [d.to_dict() for d in docs]
    assert loaded[0].n_words == 2


def test_document_rejects_unknown_source():
    with pytest.raises(ValueError, match="unknown source"):
        Document(id="x", source="tweets", text="hi")


def test_make_id_is_stable_and_distinct():
    assert make_id("thread", 3) == make_id("thread", 3)
    assert make_id("thread", 3) != make_id("thread", 4)


def test_sample_corpus_has_two_distinguishable_authors(repo_root):
    """The shipped fixture must actually contain a style gap to detect."""
    sam_path = repo_root / "data/sample_synthetic/sam.jsonl"
    if not sam_path.exists():
        pytest.skip("run scripts/make_sample_corpus.py first")

    sam = list(read_jsonl(sam_path))
    control = list(read_jsonl(repo_root / "data/sample_synthetic/control.jsonl"))

    def upper_ratio(docs):
        text = " ".join(d.text for d in docs if d.source == "chat")
        letters = [c for c in text if c.isalpha()]
        return sum(c.isupper() for c in letters) / len(letters)

    assert upper_ratio(sam) < 0.01
    assert upper_ratio(control) > 0.02
    assert {d.source for d in sam} == {"chat", "longform", "code"}
