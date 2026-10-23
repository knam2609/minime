"""Tests for the byte-level BPE tokenizer.

The tokenizer is frozen across every experiment, so the properties pinned here
-- exact round-trip, no OOV, specials that cannot be forged or split, exact
vocab size, deterministic training, a stable fingerprint -- are the ones that
would otherwise fail silently and invalidate comparisons between arms.
"""

from __future__ import annotations

import itertools
import json
import random
import subprocess
import sys

import pytest

pytest.importorskip("tokenizers")

from minime.tokenizer import CharTokenizer, is_dev_tokenizer, load_tokenizer  # noqa: E402
from minime.tokenizer.bpe import EOT, ME, BPETokenizer  # noqa: E402

VOCAB = 16_384


def zipf_corpus(seed: int = 0, n_types: int = 30_000, n_lines: int = 20_000) -> list[str]:
    """Pseudo-words with a Zipfian frequency profile: rich enough for 16k merges."""
    rng = random.Random(seed)
    alphabet = "abcdefghijklmnopqrstuvwxyzéàüñçøß"
    types = ["".join(rng.choice(alphabet) for _ in range(rng.randint(2, 9))) for _ in range(n_types)]
    cum = list(itertools.accumulate(1 / (r + 1) for r in range(n_types)))
    return [" ".join(rng.choices(types, cum_weights=cum, k=rng.randint(5, 25))) for _ in range(n_lines)]


MIXED = [
    "plain ascii text, with punctuation!",
    "Unicode: naïve café, Straße, Ελληνικά, 日本語のテキスト, Tiếng Việt có dấu",
    "emoji 😅👀🔥 and a family 👨‍👩‍👧 and flags 🇻🇳🇦🇺",
    "def f(x):\n    return {'a': [1, 2, 3]}\t# tabs\n\n\n    ",
    "mixed: ok so lowkey 😭 the loss went 3.2 -> 0.34 fr\r\nwindows newline",
    "   leading and trailing spaces   ",
    "\x00\x07 control bytes ​ zero width ﻿ bom",
    "",
]


@pytest.fixture(scope="module")
def tok() -> BPETokenizer:
    return BPETokenizer.train(zipf_corpus(), vocab_size=VOCAB, name="unit-dev")


# ------------------------------------------------------------ vocabulary


def test_vocab_size_is_exactly_the_requested_size(tok):
    assert tok.vocab_size == VOCAB


def test_too_small_a_sample_is_an_error_not_a_smaller_vocab():
    with pytest.raises(ValueError, match="too small"):
        BPETokenizer.train(["hello world"] * 20, vocab_size=VOCAB)


def test_specials_have_fixed_ids(tok):
    assert tok.eot_id == 0 and tok.id_to_token(0) == EOT
    assert tok.me_id == 1 and tok.id_to_token(1) == ME


def test_there_is_no_unk_token(tok):
    assert tok._tok.model.unk_token is None
    vocab = tok._tok.get_vocab()
    assert not any(t in vocab for t in ("<unk>", "[UNK]", "<|unk|>"))


# ------------------------------------------------------------ round trip


@pytest.mark.parametrize("text", MIXED)
def test_roundtrip_is_exact(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_every_byte_is_encodable(tok):
    # The training corpus contains none of these characters.  With byte
    # fallback they still encode, into in-vocab ids, and decode back exactly.
    rare = "".join(chr(c) for c in (0x1F9EC, 0x10348, 0x2603, 0x0E01, 0x05D0, 0x7F, 0x1B))
    ids = tok.encode(rare)
    assert ids and max(ids) < tok.vocab_size
    assert tok.decode(ids) == rare


def test_batch_encoding_matches_single_encoding(tok):
    assert tok.encode_batch(MIXED) == [tok.encode(t) for t in MIXED]


@pytest.mark.parametrize("text", MIXED)
def test_byte_lengths_add_up_to_the_utf8_length(tok, text):
    # bits-per-byte divides by these; one wrong entry skews every source differently.
    lengths = tok.byte_lengths()
    assert len(lengths) == tok.vocab_size
    assert sum(lengths[i] for i in tok.encode(text)) == len(text.encode("utf-8"))


def test_byte_lengths_cover_rare_bytes_and_count_specials_once(tok):
    lengths = tok.byte_lengths()
    rare = "".join(chr(c) for c in (0x1F9EC, 0x10348, 0x7F, 0x1B, 0x00))
    assert sum(lengths[i] for i in tok.encode(rare)) == len(rare.encode("utf-8"))
    assert lengths[tok.eot_id] == lengths[tok.me_id] == 1


def test_char_tokenizer_byte_lengths():
    from minime.tokenizer import CharTokenizer

    char = CharTokenizer(list("aé😅"))
    assert char.byte_lengths() == [1, 1, 2, 4]


# ---------------------------------------------------------------- specials


def test_specials_in_corpus_text_are_plain_text(tok):
    # A web page quoting the delimiter must not forge a document boundary,
    # and nothing in the general corpus may emit the reserved <|me|>.
    text = f"the gpt-2 delimiter is {EOT} and phase 2 uses {ME}"
    ids = tok.encode(text)
    assert tok.eot_id not in ids and tok.me_id not in ids
    assert tok.decode(ids) == text
    for batch_ids in tok.encode_batch([text]):
        assert tok.eot_id not in batch_ids and tok.me_id not in batch_ids


def test_specials_are_never_split_when_allowed(tok):
    ids = tok.encode(f"hello{EOT}world{ME}!", allow_special=True)
    assert ids.count(tok.eot_id) == 1 and ids.count(tok.me_id) == 1
    assert tok.encode(EOT, allow_special=True) == [tok.eot_id]
    assert tok.encode(ME, allow_special=True) == [tok.me_id]


def test_decode_drops_specials_by_default(tok):
    ids = tok.encode("a") + [tok.eot_id] + tok.encode("b")
    assert tok.decode(ids) == "ab"
    assert tok.decode(ids, skip_special=False) == f"a{EOT}b"


# ------------------------------------------------------------- determinism


def test_training_is_deterministic(tok):
    again = BPETokenizer.train(zipf_corpus(), vocab_size=VOCAB, name="unit-dev")
    assert again.fingerprint() == tok.fingerprint()


def test_training_is_deterministic_across_processes(tok, tmp_path):
    # Hash-map iteration order is randomised per process, so an in-process
    # comparison alone could miss order-dependent tie breaking.
    script = (
        "import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'tests')\n"
        "from test_tokenizer_bpe import zipf_corpus, VOCAB\n"
        "from minime.tokenizer.bpe import BPETokenizer\n"
        "print(BPETokenizer.train(zipf_corpus(), vocab_size=VOCAB, name='unit-dev').fingerprint())\n"
    )
    from conftest import REPO_ROOT

    out = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == tok.fingerprint()


def test_different_training_text_gives_a_different_fingerprint(tok):
    other = BPETokenizer.train(zipf_corpus(seed=1), vocab_size=VOCAB)
    assert other.fingerprint() != tok.fingerprint()


# ------------------------------------------------------------- persistence


def test_save_load_preserves_encoding_and_identity(tok, tmp_path):
    path = tmp_path / "unit-dev.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.fingerprint() == tok.fingerprint()
    assert loaded.id == tok.id == f"unit-dev@{tok.fingerprint()}"
    assert loaded.dev is True
    for text in MIXED:
        assert loaded.encode(text) == tok.encode(text)
    assert loaded.encode(EOT) == tok.encode(EOT)  # plain-text mode survives the reload


def test_fingerprint_ignores_metadata(tok):
    relabelled = BPETokenizer(tok._tok, name="other", dev=False, provenance={"x": 1})
    assert relabelled.fingerprint() == tok.fingerprint()


def test_modified_file_is_rejected(tok, tmp_path):
    path = tmp_path / "t.json"
    tok.save(path)
    payload = json.loads(path.read_text())
    payload["tokenizer"]["model"]["merges"] = payload["tokenizer"]["model"]["merges"][:-1]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="modified"):
        BPETokenizer.load(path)


def test_load_tokenizer_dispatches_on_format(tok, tmp_path):
    tok.save(tmp_path / "bpe.json")
    CharTokenizer.fit(["abc"]).save(tmp_path / "char.json")
    assert isinstance(load_tokenizer(tmp_path / "bpe.json"), BPETokenizer)
    assert isinstance(load_tokenizer(tmp_path / "char.json"), CharTokenizer)


def test_missing_file_names_the_training_script(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_tokenizer.py"):
        BPETokenizer.load(tmp_path / "nope.json")


def test_dev_tokenizer_ids_are_recognised(tok):
    assert is_dev_tokenizer(tok.id)
    assert is_dev_tokenizer(CharTokenizer.fit(["ab"]).id)
    assert not is_dev_tokenizer("minime-16k@0123456789abcdef")
