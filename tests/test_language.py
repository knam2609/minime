"""Tests for the code-switch language filter.

Synthetic strings only. Both directions are pinned: Vietnamese is caught, and
English -- including accented loanwords and words that happen to look like
unaccented Vietnamese -- is not, since a false positive silently deletes the
owner's writing.
"""

from __future__ import annotations

from collections import Counter

import pytest

from minime.data.personal.language import (
    UNACCENTED, admit, describe, is_vietnamese_word, vietnamese_share,
)


@pytest.mark.parametrize("text", [
    "yeah i think so lol",
    "Is the café open? My résumé is on the Pokémon desk.",
    "ok ok ok",
])
def test_english_has_no_vietnamese(text):
    assert vietnamese_share(text) == 0.0


@pytest.mark.parametrize("text", [
    "hôm nay trời đẹp quá",
    "Khánh ơi, mình đi ăn không",
    "ko biet nua",                         # unaccented chat spelling
    "dc roi nha",
])
def test_vietnamese_is_mostly_vietnamese(text):
    assert vietnamese_share(text) > 0.5


def test_code_switched_text_is_in_between():
    share = vietnamese_share("omg that exam was so hard luon, mình dead")
    assert 0 < share <= 0.5


def test_share_counts_words_not_characters():
    assert vietnamese_share("được yes") == pytest.approx(0.5)
    assert vietnamese_share("được yes no") == pytest.approx(1 / 3)


def test_decomposed_input_is_normalised():
    import unicodedata

    assert vietnamese_share(unicodedata.normalize("NFD", "trời đẹp")) == 1.0


@pytest.mark.parametrize("word,expected", [
    ("được", True), ("mình", True), ("à", True), ("quá", True), ("người", True),
    ("café", False), ("Pokémon", False), ("résumé", False), ("naïve", False),
    ("KO", True), ("Dc", True),
])
def test_word_classification(word, expected):
    assert is_vietnamese_word(word) is expected


def test_unaccented_list_avoids_english_collisions():
    """Each of these is a real Vietnamese chat spelling *and* an English word
    (or an abbreviation a developer writes); including one would drop English."""
    english = {
        "the", "a", "an", "ban", "met", "hay", "la", "ma", "di", "vs", "k", "r", "j",
        "em", "co", "con", "may", "hon", "ha", "ai", "it", "me", "ui", "day", "den", "sang",
        "dung", "com", "dang", "bi", "vi", "lm", "ns", "uk", "hong", "hem", "hoc", "that",
        "tao", "chi", "ok", "hi", "so", "go", "no", "to", "do", "on", "in", "is", "as",
    }
    assert not UNACCENTED & english
    assert all(w == w.lower() and w.isascii() for w in UNACCENTED)


@pytest.mark.parametrize("text", ["😂😂", "!!!", "123 456", ""])
def test_text_without_words_is_kept(text):
    counts = Counter()
    assert vietnamese_share(text) == 0.0
    assert admit(text, 0.5, counts)
    assert counts == Counter(kept_no_words=1)


def test_admit_tallies_each_outcome():
    counts = Counter()
    outcomes = [admit(t, 0.5, counts) for t in
                ["hello there", "hôm nay trời đẹp", "so tired luon", "đẹp nice"]]
    assert outcomes == [True, False, True, True]  # exactly 0.5 is kept
    assert counts == Counter(kept_english=1, dropped_vietnamese=1, kept_code_switched=2)
    assert "dropped 1 mainly-Vietnamese" in describe(counts)


def test_admit_with_no_threshold_keeps_everything_and_counts_nothing():
    counts = Counter()
    assert admit("hôm nay trời đẹp", None, counts)
    assert not counts
