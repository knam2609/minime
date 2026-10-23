"""The text metrics: stylometry, authorship, degeneration, code syntax.

Each metric is checked on inputs whose answer is known by hand, and the
authorship-style ones on the synthetic fixture's two authors, whose styles
differ by construction (``scripts/make_sample_corpus.py``).  A metric that
cannot separate ``sam`` from ``control`` cannot separate the owner from a
model either.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from pathlib import Path

import pytest

from minime.data.document import read_jsonl
from minime.evaluation import authorship, codeeval, degeneration, stylometry


FIXTURE = Path(__file__).resolve().parents[1] / "data" / "sample_synthetic"


@pytest.fixture(scope="module")
def authors():
    """Chat texts per synthetic author, split into disjoint halves."""
    out = {}
    for name in ("sam", "control"):
        texts = [d.text for d in read_jsonl(FIXTURE / f"{name}.jsonl")
                 if d.source == "chat"]
        out[name] = (texts[0::2], texts[1::2])
    return out


# ----------------------------------------------------------------- stylometry


def test_words_are_lowercase_letter_runs_with_apostrophes():
    assert stylometry.words("Don’t PANIC: it's 42 lol 😅") == ["don't", "panic", "it's", "lol"]


def test_mattr_is_plain_ttr_for_short_text_and_window_ttr_for_long():
    assert stylometry.mattr(["a", "b", "a"]) == pytest.approx(2 / 3)
    assert stylometry.mattr(["a"] * 120) == pytest.approx(1 / 50)
    assert stylometry.mattr([str(i) for i in range(120)]) == pytest.approx(1.0)
    assert stylometry.mattr([]) is None


def test_mattr_does_not_depend_on_length_for_a_stationary_text():
    rng = random.Random(0)
    vocab = [f"w{i}" for i in range(40)]
    text = [rng.choice(vocab) for _ in range(4000)]
    assert stylometry.mattr(text[:500]) == pytest.approx(stylometry.mattr(text), abs=0.02)


def test_profile_counts_the_habits_it_claims_to():
    p = stylometry.profile(["Hi there 😅\nok", "yes!!", ""])
    assert p.docs == 3
    assert p.words == 4
    assert p.words_per_doc == pytest.approx(4 / 3)
    assert p.words_per_line == pytest.approx(4 / 3)  # "Hi there 😅", "ok", "yes!!"
    assert p.empty_doc_share == pytest.approx(1 / 3)
    assert p.uppercase_share == pytest.approx(1 / 12)  # H of the 12 letters "Hitherookyes"
    assert p.docs_with_emoji == pytest.approx(1 / 3)
    assert p.emoji_per_100_chars == pytest.approx(100 * 1 / (len("Hi there 😅\nok") + len("yes!!")))


def test_jensen_shannon_is_zero_for_equal_and_one_for_disjoint():
    a, b = Counter({"x": 2, "y": 2}), Counter({"x": 5, "y": 5})
    assert stylometry.jensen_shannon(a, b, ["x", "y"]) == pytest.approx(0.0)
    assert stylometry.jensen_shannon(Counter({"x": 1}), Counter({"y": 1}), ["x", "y"]) == pytest.approx(1.0)
    assert stylometry.jensen_shannon(Counter(), Counter({"y": 1}), ["x", "y"]) is None


def test_ks_statistic_extremes():
    assert stylometry.ks_statistic([1, 2, 3], [1, 2, 3]) == 0.0
    assert stylometry.ks_statistic([1, 2], [5, 6]) == 1.0
    assert stylometry.ks_statistic([], [1]) is None


def test_compare_puts_same_author_closer_than_other_author(authors):
    sam_a, sam_b = authors["sam"]
    ctrl_a, _ = authors["control"]
    same, other = stylometry.compare(sam_a, sam_b), stylometry.compare(ctrl_a, sam_b)
    for key in ("function_word_jsd", "punctuation_jsd", "words_per_doc_ks", "uppercase_share_gap"):
        assert same[key] < other[key], key


def test_chunk_words_drops_the_partial_tail():
    chunks = stylometry.chunk_words(["a b c", "d e f g"], 3)
    assert chunks == [["a", "b", "c"], ["d", "e", "f"]]


def test_burrows_delta_attributes_held_out_text_to_its_author(authors):
    sam_a, sam_b = authors["sam"]
    ctrl_a, ctrl_b = authors["control"]
    ref = {"sam": stylometry.take_words(sam_a, 3000), "control": stylometry.take_words(ctrl_a, 3000)}
    result = stylometry.burrows_delta(ref, {"sam": sam_b, "control": ctrl_b}, target="sam",
                                      chunk_size=100, mfw=50)
    q = result["questioned"]
    assert q["sam"]["nearest_is_target"] >= 0.9
    assert q["control"]["nearest_is_target"] <= 0.1
    assert q["sam"]["delta"]["sam"] < q["sam"]["delta"]["control"]


def test_burrows_delta_vocabulary_ignores_the_questioned_text(authors):
    sam_a, sam_b = authors["sam"]
    ctrl_a, _ = authors["control"]
    ref = {"sam": stylometry.take_words(sam_a, 2000), "control": stylometry.take_words(ctrl_a, 2000)}
    alone = stylometry.burrows_delta(ref, {"q": sam_b}, target="sam", chunk_size=100)
    noisy = stylometry.burrows_delta(ref, {"q": sam_b, "junk": ["zzz qqq " * 400]}, target="sam", chunk_size=100)
    assert alone["questioned"]["q"] == noisy["questioned"]["q"]


def test_burrows_delta_refuses_an_author_without_enough_text():
    with pytest.raises(ValueError, match="at least two"):
        stylometry.burrows_delta({"a": ["one two three"], "b": ["x " * 50]}, {}, target="a", chunk_size=10)


# ----------------------------------------------------------------- authorship


def test_discriminator_separates_two_authors_but_not_one_from_itself(authors):
    sam_a, sam_b = authors["sam"]
    ctrl_a, _ = authors["control"]
    n = 200
    different = authorship.discriminator_auc(sam_a[:n], ctrl_a[:n], folds=4, seed=0)
    same = authorship.discriminator_auc(sam_a[:n], sam_b[:n], folds=4, seed=0)
    assert different["auc"] > 0.95
    assert 0.3 < same["auc"] < 0.7


def test_discriminator_groups_keep_pairs_on_one_side_of_every_split(authors):
    sam_a, sam_b = authors["sam"]
    result = authorship.discriminator_auc(sam_a[:60], sam_b[:60], groups=list(range(60)) * 2, folds=3)
    assert result["folds"] == 3 and result["real"] == result["generated"] == 60


def test_attribution_rates_real_text_by_author(authors):
    sam_a, sam_b = authors["sam"]
    ctrl_a, ctrl_b = authors["control"]
    result = authorship.attribution(sam_a[:300], ctrl_a[:300], {"sam": sam_b[:100], "control": ctrl_b[:100]})
    assert result["sam"]["attributed_to_target"] > 0.9
    assert result["control"]["attributed_to_target"] < 0.1


# -------------------------------------------------------------- degeneration


def test_rep_n_counts_repeated_ngrams_within_a_text():
    # bigrams: (a b) (b a) (a b) (b a) (a b) -> 2 unique of 5
    assert degeneration.rep_n(["a b a b a b"], 2) == pytest.approx(0.6)
    assert degeneration.rep_n(["a b c d"], 2) == 0.0
    assert degeneration.rep_n(["a"], 2) is None


def test_distinct_n_pools_across_texts():
    assert degeneration.distinct_n(["a b", "a b"], 1) == pytest.approx(0.5)
    assert degeneration.distinct_n(["a b", "c d"], 2) == pytest.approx(1.0)


def test_self_bleu_extremes():
    assert degeneration.self_bleu(["the cat sat on the mat"] * 3) == pytest.approx(1.0)
    disjoint = degeneration.self_bleu(["a b c d e", "f g h i j", "k l m n o"])
    assert disjoint < 0.15
    assert degeneration.self_bleu(["only one"]) is None


def _naive_self_bleu(texts, max_n=4, epsilon=0.1):
    tokens = [degeneration.tokenize(t) for t in texts]
    scores = []
    for i, hyp in enumerate(tokens):
        if len(hyp) < max_n:
            continue
        refs = [t for j, t in enumerate(tokens) if j != i]
        log_p = 0.0
        for n in range(1, max_n + 1):
            counts = Counter(degeneration.ngrams(hyp, n))
            ref_max = Counter()
            for ref in refs:
                for g, c in Counter(degeneration.ngrams(ref, n)).items():
                    ref_max[g] = max(ref_max[g], c)
            clipped = sum(min(c, ref_max[g]) for g, c in counts.items())
            log_p += math.log((clipped or epsilon) / sum(counts.values()))
        ref_len = min((abs(len(r) - len(hyp)), len(r)) for r in refs)[1]
        bp = 1.0 if len(hyp) > ref_len else math.exp(1 - ref_len / len(hyp))
        scores.append(bp * math.exp(log_p / max_n))
    return sum(scores) / len(scores)


def test_self_bleu_matches_the_quadratic_definition():
    rng = random.Random(3)
    vocab = "a b c d e f".split()
    texts = [" ".join(rng.choice(vocab) for _ in range(rng.randint(4, 12))) for _ in range(25)]
    assert degeneration.self_bleu(texts) == pytest.approx(_naive_self_bleu(texts), rel=1e-12)


# ------------------------------------------------------------------------ code

HEADER = "def add(a, b):"


def test_a_complete_body_is_valid_and_terminated():
    ok, closed = codeeval.function_parses(HEADER, "\n    return a + b\n\n\nx = 1\n", complete=False)
    assert ok and closed


def test_a_body_on_the_header_line_is_not_valid():
    # `def add(a, b):abc` parses, but it is not the function the prompt asked for.
    assert codeeval.function_parses(HEADER, "abc", complete=True) == (False, False)
    assert codeeval.function_parses(HEADER, " return a\n", complete=True)[0] is False


def test_an_empty_or_broken_body_is_not_valid():
    assert codeeval.function_parses(HEADER, "\nx = 1\n", complete=True)[0] is False
    assert codeeval.function_parses(HEADER, "\n    return (a +\n", complete=True)[0] is False


def test_a_body_cut_by_the_token_limit_loses_its_partial_line():
    ok, closed = codeeval.function_parses(HEADER, "\n    c = a + b\n    return (c", complete=False)
    assert ok and not closed


def test_a_column_zero_line_inside_a_string_does_not_end_the_function():
    body = '\n    """Doc.\nstill the docstring\n    """\n    return a\n'
    assert codeeval.function_parses(HEADER, body, complete=True)[0] is True


def test_function_prompts_end_at_the_header_and_reference_bodies_parse(repo_root, tmp_path):
    from minime.data.binidx import TokenShard, write_shard
    from minime.tokenizer import CharTokenizer

    docs = [d.text for d in read_jsonl(repo_root / "data" / "sample_synthetic" / "control.jsonl")
            if d.source == "code"][:20]
    tok = CharTokenizer.fit(docs)
    write_shard(tmp_path / "code.val", tok.encode_batch(docs), tok.vocab_size, tok.id, tok.eot_id)
    shard = TokenShard(tmp_path / "code.val")

    prompts, digest = codeeval.function_prompts(
        shard, tok.byte_lengths(), tok.decode, n=10, prompt_tokens=64, per_file=2, eot_id=tok.eot_id)
    assert len(prompts) == 10 and len(digest) == 16
    for p in prompts:
        assert p.header.startswith("def ") and len(p.prompt) == 64
        assert tok.decode(p.prompt.tolist()).endswith(p.header[-20:])
        assert codeeval.function_parses(p.header, p.reference_body, complete=True)[0]
    result = codeeval.syntax_validity(prompts, [p.reference_body for p in prompts], [True] * len(prompts))
    assert result["valid"] == result["reference_valid"] == 1.0


def test_prompts_do_not_depend_on_iteration_luck(repo_root, tmp_path):
    from minime.data.binidx import TokenShard, write_shard
    from minime.tokenizer import CharTokenizer

    docs = [d.text for d in read_jsonl(repo_root / "data" / "sample_synthetic" / "control.jsonl")
            if d.source == "code"][:30]
    tok = CharTokenizer.fit(docs)
    write_shard(tmp_path / "c", tok.encode_batch(docs), tok.vocab_size, tok.id, tok.eot_id)
    runs = [codeeval.function_prompts(TokenShard(tmp_path / "c"), tok.byte_lengths(), tok.decode,
                                      n=8, prompt_tokens=32, eot_id=tok.eot_id) for _ in range(2)]
    assert runs[0][1] == runs[1][1]
    # Spread across the shard, not the first files in it.
    assert max(p.doc_index for p in runs[0][0]) >= 15

