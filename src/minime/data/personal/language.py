"""How Vietnamese a message is, so mainly-Vietnamese messages can be dropped.

The owner writes English with Vietnamese mixed in.  The decision on record:
keep code-switched messages -- switching *is* the voice -- and drop messages
that are mainly Vietnamese, which the English-dominant general corpus and
tokenizer cannot model, and which would otherwise be a small, badly-compressed
second language inside a corpus that is already small.

The measure is per word, and deliberately simple enough to audit:

* a **letter-word** is a run of Unicode letters (digits, emoji and punctuation
  are not words, so an emoji-only message has no words and is always kept);
* a letter-word is **Vietnamese** if it carries a character only Vietnamese
  uses (đ ă ơ ư, hook-above, dot-below, stacked diacritics), or a vowel accent
  Vietnamese shares with French/Spanish *and* the shape of a single Vietnamese
  syllable -- so "được" and "mình" count and "café" and "Pokémon" do not -- or
  if it is a common *unaccented* chat spelling ("ko", "dc", "roi").

The unaccented list is curated, not generated: every entry must be a word an
English message would essentially never contain.  Spellings that collide with
English ("the", "hay", "ban", "met", "em", "con", "ai", "lm") are left out on
purpose; a missed Vietnamese word only moves a share slightly, while a false
one can drop an English message.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

# Characters that only Vietnamese orthography uses (lowercase, NFC).
_STRONG = frozenset(
    "đăằắẳẵặơờớởỡợưừứửữự"   # đ, ă, ơ, ư and their tones
    "ầấẩẫậềếểễệồốổỗộ"       # â ê ô with a tone stacked on top
    "ảẻỉỏủỷạẹịọụỵẽĩũỹỳ"      # hook above, dot below, tilde on e/i/u/y, grave y
)
# Accented vowels Vietnamese shares with Western European languages.
_SHARED = frozenset("àáâãèéêìíòóôõùúý")

_VOWELS = frozenset("aeiouy")

UNACCENTED = frozenset({
    # negation and questions
    "khong", "ko", "hok", "khum", "khg", "chua", "chx", "gi", "nao", "sao", "dau", "bao", "ntn",
    # sentence particles
    "nha", "nhe", "nhen", "nhi", "oi", "roi", "rui", "vay", "zay", "dzay", "thoi", "thui",
    "luon", "lun", "ne", "nek", "ukm",
    # people
    "minh", "mik", "mk", "tui", "toi", "anh", "nguoi", "ng", "mng", "ae", "nhau", "ong",
    # verbs and adjectives
    "biet", "bik", "bjk", "duoc", "dc", "dk", "lam", "thay", "muon", "noi", "nghe", "xem", "coi",
    "choi", "hieu", "thich", "thik", "yeu", "iu", "nho", "quen", "phai", "nen", "chac", "vui",
    "buon", "xinh", "ngu", "uong", "nuoc", "tien", "mua", "sai", "vao", "xuong", "giup", "gium",
    # connectives
    "cua", "voi", "nhung", "nma", "nhma", "cung", "cx", "cug", "thi", "neu", "khi", "luc", "nhu",
    "lai", "nua", "qua", "trc", "truoc", "sau", "cai", "cho", "ve", "vo",
    # time and degree
    "nhieu", "hom", "mai", "ngay", "chieu", "nhanh", "cham", "lau", "som", "tre",
    # exclamations
    "troi", "tr", "vcl", "vl", "vkl", "dmm", "clgt", "cmn",
})

_WORD = re.compile(r"[^\W\d_]+")


def letter_words(text: str) -> list[str]:
    return _WORD.findall(unicodedata.normalize("NFC", text))


def _base(char: str) -> str:
    return "d" if char == "đ" else unicodedata.normalize("NFD", char)[0]


def _one_syllable(word: str) -> bool:
    """Vietnamese writes one syllable per word: a single run of vowels."""
    runs, in_vowel = 0, False
    for char in word:
        vowel = _base(char) in _VOWELS
        runs += vowel and not in_vowel
        in_vowel = vowel
    return runs == 1


def is_vietnamese_word(word: str) -> bool:
    lower = word.casefold()
    if any(char in _STRONG for char in lower):
        return True
    if any(char in _SHARED for char in lower):
        return _one_syllable(lower)
    return lower in UNACCENTED


def vietnamese_share(text: str) -> float:
    """Fraction of the letter-words in ``text`` that are Vietnamese (0.0 if it has none)."""
    words = letter_words(text)
    if not words:
        return 0.0
    return sum(map(is_vietnamese_word, words)) / len(words)


def admit(text: str, max_share: float | None, counts: Counter) -> bool:
    """Keep ``text`` unless it is mainly Vietnamese, tallying which way it went.

    ``max_share=None`` disables the filter (and the tally).
    """
    if max_share is None:
        return True
    if not letter_words(text):
        counts["kept_no_words"] += 1
        return True
    share = vietnamese_share(text)
    if share > max_share:
        counts["dropped_vietnamese"] += 1
        return False
    counts["kept_code_switched" if share > 0 else "kept_english"] += 1
    return True


def describe(counts: Counter) -> str:
    """One line for the census, e.g. ``kept 9,812 English, ... dropped 1,024 mainly-Vietnamese``."""
    return (f"kept {counts['kept_english']:,} English, {counts['kept_code_switched']:,} "
            f"code-switched, {counts['kept_no_words']:,} with no words; dropped "
            f"{counts['dropped_vietnamese']:,} mainly-Vietnamese")
