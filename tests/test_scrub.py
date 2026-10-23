"""Tests for PII redaction.

Both directions matter.  A scrubber that redacts everything is perfectly
private and destroys the corpus, so every "removes X" test is paired with a
"leaves ordinary text alone" test.
"""

from __future__ import annotations

import pytest

from minime.data.document import Document
from minime.data.scrub import ScrubConfig, contains_obvious_pii, scrub, scrub_documents


def redact(text, **kwargs):
    return scrub(text, ScrubConfig(**kwargs))[0]


# ------------------------------------------------------------- removes PII


@pytest.mark.parametrize("text", [
    "email me at jo.bloggs@student.example.edu.au ok",
    "sam+work@gmail.com sent it",
    "contact: a.b-c@sub.domain.co.uk",
])
def test_emails_are_redacted(text):
    out = redact(text)
    assert "@" not in out.replace("<EMAIL>", "")
    assert "<EMAIL>" in out


@pytest.mark.parametrize("text", [
    "call me on +61 412 345 678",
    "my number is 0412 345 678",
    "ring 415-555-0134 tomorrow",
    "+1 (415) 555 0134",
])
def test_phone_numbers_are_redacted(text):
    assert "<PHONE>" in redact(text)


def test_handles_are_redacted():
    out = redact("dm @someuser_123 about it")
    assert "<HANDLE>" in out and "@someuser_123" not in out


def test_urls_are_redacted():
    out = redact("see https://example.com/private/doc?id=99 and www.foo.org")
    assert out.count("<URL>") == 2


def test_names_are_redacted_case_insensitively():
    out = redact("told Anna and anna marie about it", names=["Anna Marie", "Anna"])
    assert "Anna" not in out and "anna" not in out
    assert out.count("<NAME>") == 2


def test_longer_names_win_over_shorter_prefixes():
    out = redact("Anna Marie said hi", names=["Anna", "Anna Marie"])
    assert out == "<NAME> said hi"


def test_counts_are_reported_by_category():
    _, counts = scrub("a@b.com and c@d.com and @handle", ScrubConfig())
    assert counts["email"] == 2
    assert counts["handle"] == 1


# ------------------------------------------------- leaves ordinary text alone


@pytest.mark.parametrize("text", [
    "the loss dropped to 0.3384 after 200 steps",
    "i submitted it in 2024 and again in 2025",
    "version 3.11.2 with torch 2.11.0",
    "we processed 301000000 tokens",
    "batch 32 x 8 accumulation = 4096 tokens",
    "scored 95 out of 100 on the exam",
])
def test_ordinary_numbers_survive(text):
    assert redact(text) == text, "an over-eager regex would corrupt the corpus"


def test_email_at_sign_is_not_double_redacted():
    out = redact("ping me at sam@example.com")
    assert out == "ping me at <EMAIL>"
    assert "<HANDLE>" not in out


def test_casual_text_is_untouched():
    text = "ngl that assignment was cooked... i spent like 3 hours on it :(("
    assert redact(text) == text


def test_long_numbers_are_opt_in():
    text = "the id is 123456789012"
    assert redact(text) == text
    assert "<NUMBER>" in redact(text, long_numbers=True)


def test_short_names_are_ignored():
    """Two-letter 'names' would shred ordinary words."""
    assert redact("it is on the table", names=["is", "on"]) == "it is on the table"


# ----------------------------------------------------------------- helpers


def test_scrub_documents_updates_text_and_totals():
    docs = [
        Document(id="1", source="chat", text="mail me a@b.com"),
        Document(id="2", source="chat", text="call 0412 345 678"),
    ]
    out, totals = scrub_documents(docs, ScrubConfig())
    assert "<EMAIL>" in out[0].text and "<PHONE>" in out[1].text
    assert totals["email"] == 1 and totals["phone"] == 1


def test_contains_obvious_pii():
    assert contains_obvious_pii("reach me at a@b.com")
    assert contains_obvious_pii("0412 345 678")
    assert not contains_obvious_pii("just a normal sentence with 42 things")


def test_names_file_is_optional(tmp_path):
    cfg = ScrubConfig.from_names_file(tmp_path / "missing.txt")
    assert cfg.names == []

    path = tmp_path / "names.txt"
    path.write_text("# other people\nAnna Marie\nBen\n\n")
    cfg = ScrubConfig.from_names_file(path)
    assert cfg.names == ["Anna Marie", "Ben"]


# ------------------------------------------------------------- names file


def test_names_file_lists_full_names_and_parts():
    from minime.data.scrub import NAMES_FILE_HEADER, names_file_lines

    lines = names_file_lines(["Lê Văn Minh", "Jo Li", "Grace Park"], common_words={"grace", "minh"})
    assert lines[:len(NAMES_FILE_HEADER)] == NAMES_FILE_HEADER
    body = lines[len(NAMES_FILE_HEADER) + 1:]
    assert "Lê Văn Minh" in body and "Văn" in body and "Park" in body and "Grace Park" in body
    assert "# common word: Grace" in body and "# common word: Minh" in body
    assert "Jo" not in body and "Li" not in body, "the scrubber ignores names under 3 chars"
    assert "Jo Li" in body


def test_names_file_round_trips_through_the_scrubber(tmp_path):
    from minime.data.scrub import names_file_lines

    path = tmp_path / "names.txt"
    path.write_text("\n".join(names_file_lines(["Grace Park"], {"grace"})), encoding="utf-8")
    cfg = ScrubConfig.from_names_file(path)
    assert sorted(cfg.names) == ["Grace Park", "Park"]
    text, _ = scrub("saying grace with Park", cfg)
    assert text == "saying grace with <NAME>"
