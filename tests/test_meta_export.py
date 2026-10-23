"""Tests for the Meta (Messenger/Instagram) export parser.

Built against a synthetic export rather than real data, so the suite runs on
any machine and no personal message ever enters the repository.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

from minime.data.personal.meta_export import (
    MetaExportConfig,
    detect_platform,
    discover_me,
    find_thread_files,
    fix_mojibake,
    is_system_message,
    parse_export,
    parse_thread_file,
    participant_names,
)
from minime.data.splits import check_no_leakage, split_documents, split_key

ME = "Thang Nguyen"
OTHER = "Someone Else"
MINUTE = 60_000


def mojibake(text: str) -> str:
    """How Meta writes UTF-8: as latin-1 code points."""
    return text.encode("utf-8").decode("latin-1")


def write_thread(root, thread: str, messages: list[dict], platform: str = "messenger",
                 participants: list[str] | None = None, box: str = "inbox"):
    base = root / ("your_instagram_activity" if platform == "instagram" else "your_facebook_activity")
    folder = base / "messages" / box / thread
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "participants": [{"name": n} for n in (participants or [ME, OTHER])],
        "title": OTHER,
        "thread_path": f"inbox/{thread}",
        # Meta exports newest-first; the fixture reproduces that.
        "messages": sorted(messages, key=lambda m: -m["timestamp_ms"]),
    }
    (folder / "message_1.json").write_text(json.dumps(payload), encoding="utf-8")
    return folder


def msg(sender: str, content: str | None, ts_minutes: float, **extra) -> dict:
    out = {"sender_name": sender, "timestamp_ms": int(ts_minutes * MINUTE), "type": "Generic"}
    if content is not None:
        out["content"] = content
    out.update(extra)
    return out


@pytest.fixture
def cfg():
    return MetaExportConfig(me=[ME], gap_seconds=300)


# ------------------------------------------------------------------ basics


def test_only_my_messages_are_kept(tmp_path, cfg):
    write_thread(tmp_path, "t1", [
        msg(ME, "my first message", 0),
        msg(OTHER, "their reply which must never be trained on", 1),
        msg(ME, "my second message", 2),
    ])
    docs = parse_export(tmp_path, cfg)
    text = " ".join(d.text for d in docs)

    assert "my first message" in text and "my second message" in text
    assert "must never be trained on" not in text
    assert all(d.author == "me" and d.source == "chat" for d in docs)


def test_messages_are_ordered_forwards_in_time(tmp_path, cfg):
    write_thread(tmp_path, "t1", [
        msg(ME, "first", 0), msg(ME, "second", 1), msg(ME, "third", 2),
    ])
    docs = parse_export(tmp_path, cfg)
    assert len(docs) == 1
    assert docs[0].text == "first\nsecond\nthird"


def test_burst_of_messages_becomes_one_document(tmp_path, cfg):
    write_thread(tmp_path, "t1", [msg(ME, f"line {i}", i * 0.5) for i in range(4)])
    docs = parse_export(tmp_path, cfg)
    assert len(docs) == 1 and docs[0].meta["messages"] == 4


def test_a_long_gap_starts_a_new_document(tmp_path, cfg):
    write_thread(tmp_path, "t1", [
        msg(ME, "before the gap", 0),
        msg(ME, "hours later", 600),       # 10 hours
    ])
    docs = parse_export(tmp_path, cfg)
    assert len(docs) == 2


def test_someone_elses_message_ends_my_utterance(tmp_path, cfg):
    write_thread(tmp_path, "t1", [
        msg(ME, "my turn", 0),
        msg(OTHER, "their turn", 1),
        msg(ME, "my turn again", 2),
    ])
    docs = parse_export(tmp_path, cfg)
    assert len(docs) == 2, "a reply in between must split the utterance"


# ---------------------------------------------------------------- mojibake


def test_fix_mojibake_repairs_meta_encoding():
    original = "don't 😅 — really"
    mangled = original.encode("utf-8").decode("latin-1")
    assert mangled != original
    assert fix_mojibake(mangled) == original


def test_fix_mojibake_leaves_clean_text_alone():
    for text in ["plain ascii", "already correct don't", "emoji 😅 intact"]:
        assert fix_mojibake(text) == text


def test_parser_repairs_encoding_end_to_end(tmp_path, cfg):
    write_thread(tmp_path, "t1", [msg(ME, "it's fine 😅".encode("utf-8").decode("latin-1"), 0)])
    docs = parse_export(tmp_path, cfg)
    assert docs[0].text == "it's fine 😅"


# ---------------------------------------------------------- system content


@pytest.mark.parametrize("text", [
    "You sent an attachment.",
    "Thang Nguyen sent an attachment.",
    "You sent a link.",
    "Liked a message",
    "You are now connected on Messenger",
    "Missed video call",
    "You called Someone Else.",
])
def test_system_messages_are_recognised(text):
    assert is_system_message(text)


@pytest.mark.parametrize("text", [
    "i sent you the attachment btw",
    "you called it",                      # lowercase subject, no full stop
    "You called it",                      # no full stop
    "sarah called me yesterday.",         # lowercase subject
    "liked that idea",
    "i liked a message you sent",
    "we removed a message from the doc",
])
def test_ordinary_messages_are_not_mistaken_for_system_text(text):
    """A loose subject pattern here silently deletes real writing."""
    assert not is_system_message(text)


def test_system_messages_are_dropped(tmp_path, cfg):
    write_thread(tmp_path, "t1", [
        msg(ME, "You sent an attachment.", 0),
        msg(ME, "real message", 1),
        msg(ME, "Liked a message", 2),
    ])
    docs = parse_export(tmp_path, cfg)
    assert " ".join(d.text for d in docs) == "real message"


def test_unsent_and_empty_messages_are_dropped(tmp_path, cfg):
    write_thread(tmp_path, "t1", [
        msg(ME, "retracted", 0, is_unsent=True),
        msg(ME, None, 1),
        msg(ME, "", 2),
        msg(ME, "kept", 3),
    ])
    docs = parse_export(tmp_path, cfg)
    assert " ".join(d.text for d in docs) == "kept"


def test_share_only_messages_are_dropped(tmp_path, cfg):
    write_thread(tmp_path, "t1", [
        msg(ME, "https://example.com/thing", 0, share={"link": "https://example.com/thing"}),
        msg(ME, "look at this", 1, photos=[{"uri": "photos/1.jpg"}]),
    ])
    docs = parse_export(tmp_path, cfg)
    text = " ".join(d.text for d in docs)
    assert "https://" not in text
    assert "look at this" in text, "a real caption on a photo is still my writing"


def test_very_short_messages_are_dropped_by_min_chars(tmp_path):
    write_thread(tmp_path, "t1", [msg(ME, "k", 0), msg(ME, "okay then", 1)])
    docs = parse_export(tmp_path, MetaExportConfig(me=[ME], min_chars=2))
    assert " ".join(d.text for d in docs) == "okay then"


# ------------------------------------------------------------- privacy/meta


def test_thread_identity_is_hashed_not_named(tmp_path, cfg):
    write_thread(tmp_path, "t1", [msg(ME, "hello", 0)])
    doc = parse_export(tmp_path, cfg)[0]
    assert OTHER not in json.dumps(doc.meta)
    assert len(doc.meta["thread"]) == 16
    assert doc.meta["platform"] == "messenger"


def test_document_ids_are_stable_across_runs(tmp_path, cfg):
    write_thread(tmp_path, "t1", [msg(ME, "hello", 0), msg(ME, "again", 500)])
    first = [d.id for d in parse_export(tmp_path, cfg)]
    second = [d.id for d in parse_export(tmp_path, cfg)]
    assert first == second and len(set(first)) == 2


def test_excluded_threads_are_skipped(tmp_path):
    write_thread(tmp_path, "groupchat_xyz", [msg(ME, "in the group", 0)])
    write_thread(tmp_path, "private_abc", [msg(ME, "one to one", 0)])
    docs = parse_export(tmp_path, MetaExportConfig(me=[ME], exclude_threads=["groupchat"]))
    assert " ".join(d.text for d in docs) == "one to one"


# --------------------------------------------------------------- discovery


def test_instagram_and_messenger_exports_are_both_found(tmp_path, cfg):
    write_thread(tmp_path, "t1", [msg(ME, "from messenger", 0)], platform="messenger")
    write_thread(tmp_path, "t2", [msg(ME, "from instagram", 0)], platform="instagram")

    assert len(find_thread_files(tmp_path)) == 2
    text = " ".join(d.text for d in parse_export(tmp_path, cfg))
    assert "from messenger" in text and "from instagram" in text


def test_malformed_json_does_not_abort_the_export(tmp_path, cfg):
    folder = write_thread(tmp_path, "good", [msg(ME, "survives", 0)])
    broken = folder.parent / "broken"
    broken.mkdir()
    (broken / "message_1.json").write_text("{not json", encoding="utf-8")

    docs = parse_export(tmp_path, cfg)
    assert " ".join(d.text for d in docs) == "survives"
    assert parse_thread_file(broken / "message_1.json", cfg) == []


def test_display_name_matching_is_case_insensitive(tmp_path):
    write_thread(tmp_path, "t1", [msg("thang nguyen", "lowercase sender", 0)])
    docs = parse_export(tmp_path, MetaExportConfig(me=["Thang Nguyen"]))
    assert len(docs) == 1


def test_config_requires_a_name():
    with pytest.raises(ValueError, match="must name at least one"):
        MetaExportConfig(me=[])


# ------------------------------------------------- real-export regressions


def test_mojibake_sender_matches_a_diacritic_name(tmp_path):
    """Meta encodes the sender name too: "Khánh" arrives as "KhÃ¡nh".

    Found on the real export: comparing the raw sender matched nothing, so the
    owner's corpus came out empty.
    """
    owner = "Khánh Nam Nguyễn"
    write_thread(tmp_path, "t1", [msg(mojibake(owner), "hello there", 0)])
    cfg = MetaExportConfig(me=[owner])
    assert cfg.is_me(mojibake(owner))
    assert [d.text for d in parse_export(tmp_path, cfg)] == ["hello there"]


def test_name_matching_ignores_unicode_composition():
    import unicodedata

    decomposed = unicodedata.normalize("NFD", "Khánh Nam Nguyễn")
    assert MetaExportConfig(me=[decomposed]).is_me(mojibake("Khánh Nam Nguyễn"))
    assert not MetaExportConfig(me=["Khánh Nam Nguyen"]).is_me(mojibake("Khánh Nam Nguyễn"))


def test_platform_is_detected_from_the_layout_not_the_folder_name(tmp_path):
    insta = tmp_path / "minimellm_data"
    write_thread(insta, "t1", [msg(ME, "hi", 0)], platform="instagram")
    fb = tmp_path / "instagram-looking-name"
    write_thread(fb, "t1", [msg(ME, "hi", 0)], platform="messenger")
    assert detect_platform(insta) == "instagram"
    assert detect_platform(fb) == "messenger"


def test_a_root_holding_both_platforms_is_refused(tmp_path):
    write_thread(tmp_path, "t1", [msg(ME, "hi", 0)], platform="instagram")
    write_thread(tmp_path, "t2", [msg(ME, "hi", 0)], platform="messenger")
    with pytest.raises(ValueError, match="both"):
        detect_platform(tmp_path)


def test_appledouble_files_are_skipped(tmp_path, cfg):
    """exFAT makes macOS write a binary ``._name`` beside every file."""
    folder = write_thread(tmp_path, "t1", [msg(ME, "real", 0)])
    (folder / "._message_1.json").write_bytes(b"\x00\x05\x16\x07\xff\xfe binary")
    assert [p.name for p in find_thread_files(tmp_path)] == ["message_1.json"]


def test_message_requests_are_not_read(tmp_path, cfg):
    write_thread(tmp_path, "t1", [msg(ME, "accepted thread", 0)])
    write_thread(tmp_path, "t2", [msg(ME, "never accepted", 0)], box="message_requests")
    assert " ".join(d.text for d in parse_export(tmp_path, cfg)) == "accepted thread"


def test_discover_me_finds_the_sender_in_the_most_threads(tmp_path):
    owner = "Khánh Nam Nguyễn"
    for i in range(3):
        write_thread(tmp_path, f"t{i}", [msg(mojibake(owner), "hi", 0), msg(f"Friend {i}", "yo", 1)])
    write_thread(tmp_path, "t9", [msg("Friend 0", "only them", 0)])
    guess = discover_me(tmp_path)
    assert guess.name == owner          # mojibake repaired for display
    assert (guess.threads, guess.runner_up, guess.total) == (3, 2, 4)


def test_discover_me_on_an_empty_export(tmp_path):
    assert discover_me(tmp_path).name is None


def test_participant_names_exclude_the_owner_and_are_repaired(tmp_path):
    owner = "Khánh Nam Nguyễn"
    write_thread(tmp_path, "t1", [msg(mojibake(owner), "hi", 0), msg(mojibake("Lê Văn A"), "yo", 1)],
                 participants=[mojibake(owner), mojibake("Lê Văn A")])
    write_thread(tmp_path, "t2", [msg("Left Group", "bye", 0)], participants=[mojibake(owner)])
    names = participant_names(tmp_path, MetaExportConfig(me=[owner]))
    assert names == ["Left Group", "Lê Văn A"]


# ------------------------------------------------------- language filter


def test_mainly_vietnamese_messages_are_dropped_and_counted(tmp_path, cfg):
    write_thread(tmp_path, "t1", [
        msg(ME, "that was so funny", 0),
        msg(ME, mojibake("hôm nay trời đẹp quá"), 1),
        msg(ME, "omg ok luon", 2),
        msg(ME, "😂😂", 3),
    ])
    docs = parse_export(tmp_path, cfg)
    assert docs[0].text == "that was so funny\nomg ok luon\n😂😂"
    assert cfg.language_counts == Counter(
        kept_english=1, dropped_vietnamese=1, kept_code_switched=1, kept_no_words=1)


def test_language_filter_can_be_disabled(tmp_path):
    write_thread(tmp_path, "t1", [msg(ME, "hôm nay trời đẹp quá", 0)])
    cfg = MetaExportConfig(me=[ME], max_vietnamese_share=None)
    assert [d.text for d in parse_export(tmp_path, cfg)] == ["hôm nay trời đẹp quá"]
    assert not cfg.language_counts


# ------------------------------------------------ conversation-level splits


def test_documents_carry_a_thread_month_split_group(tmp_path, cfg):
    day = 24 * 60
    write_thread(tmp_path, "t1", [msg(ME, "january", 0), msg(ME, "still january", 20 * day),
                                  msg(ME, "february", 40 * day)])
    docs = parse_export(tmp_path, cfg)
    groups = [d.meta["split_group"] for d in docs]
    thread = docs[0].meta["thread"]
    assert groups == [f"{thread}:1970-01", f"{thread}:1970-01", f"{thread}:1970-02"]
    assert OTHER not in "".join(groups)


def test_every_document_of_a_thread_month_lands_in_one_split(tmp_path, cfg):
    """Utterances were split by their own ids, scattering one conversation
    across train and test -- the leak document-level splits exist to prevent."""
    for t in range(40):
        # 30 separate utterances per thread (each an hour apart), within one month
        write_thread(tmp_path, f"t{t}", [msg(ME, f"message {i}", i * 60) for i in range(30)])
    docs = parse_export(tmp_path, cfg)
    assert len(docs) == 40 * 30
    splits = split_documents(docs)
    check_no_leakage(splits)
    where: dict[str, set[str]] = {}
    for name, members in splits.items():
        for doc in members:
            where.setdefault(split_key(doc), set()).add(name)
    assert len(where) == 40 and all(len(names) == 1 for names in where.values())
    assert sum(1 for members in splits.values() if members) >= 2
