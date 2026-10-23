"""Tests for the Meta comments parser.

The parser accepts several schemas because Meta's export format has changed
repeatedly and these tests are written before a real export exists to check
against. Each test therefore pins one *known* shape, so when the real export
lands the failing ones point straight at what moved.
"""

from __future__ import annotations

import json

import pytest

from minime.data.personal.meta_comments import (
    CommentsConfig, find_comment_files, parse_comment_file, parse_comments,
)

ME = "Thang Nguyen"
OTHER = "Someone Else"


def write(tmp_path, name: str, payload, subdir: str = "comments_and_reactions"):
    folder = tmp_path / subdir
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def fb_record(text: str, author: str = ME, timestamp: int = 1_614_556_800) -> dict:
    return {
        "timestamp": timestamp,
        "data": [{"comment": {"timestamp": timestamp, "comment": text, "author": author}}],
        "title": f"{author} commented on a post.",
    }


def ig_record(text: str, timestamp: int = 1_614_556_800) -> dict:
    return {
        "string_map_data": {
            "Comment": {"value": text},
            "Time": {"timestamp": timestamp},
            "Media Owner": {"value": "someone_elses_handle"},
        }
    }


# ------------------------------------------------------------ known schemas


def test_facebook_comments_v2_shape(tmp_path):
    write(tmp_path, "comments.json", {"comments_v2": [fb_record("a real comment here")]})
    docs = parse_comments(tmp_path, CommentsConfig(me=[ME]))
    assert [d.text for d in docs] == ["a real comment here"]
    assert docs[0].source == "comment" and docs[0].author == "me"


def test_facebook_older_comments_key(tmp_path):
    write(tmp_path, "comments.json", {"comments": [fb_record("older export shape")]})
    assert [d.text for d in parse_comments(tmp_path, CommentsConfig(me=[ME]))] == ["older export shape"]


def test_instagram_string_map_shape(tmp_path):
    write(tmp_path, "post_comments_1.json",
          {"comments_media_comments": [ig_record("an instagram comment")]}, subdir="comments")
    assert [d.text for d in parse_comments(tmp_path)] == ["an instagram comment"]


def test_bare_list_root(tmp_path):
    write(tmp_path, "post_comments.json", [ig_record("bare list root")], subdir="comments")
    assert [d.text for d in parse_comments(tmp_path)] == ["bare list root"]


def test_flat_comment_shape(tmp_path):
    write(tmp_path, "comments.json", [{"comment": "flat shape", "author": ME, "timestamp": 1}])
    assert [d.text for d in parse_comments(tmp_path, CommentsConfig(me=[ME]))] == ["flat shape"]


def test_renamed_wrapper_key_still_parses(tmp_path):
    """A single unknown key wrapping the array is the most likely future drift."""
    write(tmp_path, "comments.json", {"some_new_key_v3": [fb_record("survives a rename")]})
    assert [d.text for d in parse_comments(tmp_path, CommentsConfig(me=[ME]))] == ["survives a rename"]


# ----------------------------------------------------------------- filtering


def test_reactions_are_dropped(tmp_path):
    """Facebook bundles reactions with comments; a reaction has no comment body."""
    write(tmp_path, "comments.json", {"comments_v2": [
        {"timestamp": 1, "title": "Thang Nguyen likes a post."},
        fb_record("but this one is a real comment"),
    ]})
    docs = parse_comments(tmp_path, CommentsConfig(me=[ME]))
    assert [d.text for d in docs] == ["but this one is a real comment"]


def test_other_peoples_comments_are_dropped(tmp_path):
    write(tmp_path, "comments.json", {"comments_v2": [
        fb_record("mine to train on", author=ME),
        fb_record("not mine at all", author=OTHER),
    ]})
    docs = parse_comments(tmp_path, CommentsConfig(me=[ME]))
    assert [d.text for d in docs] == ["mine to train on"]


def test_missing_author_is_assumed_mine(tmp_path):
    """Instagram omits the author because it only exports the owner's activity."""
    write(tmp_path, "post_comments.json", [ig_record("no author field")], subdir="comments")
    assert len(parse_comments(tmp_path, CommentsConfig(me=[ME]))) == 1


def test_missing_author_can_be_treated_as_unknown(tmp_path):
    write(tmp_path, "post_comments.json", [ig_record("no author field")], subdir="comments")
    cfg = CommentsConfig(me=[ME], assume_mine_when_author_missing=False)
    assert parse_comments(tmp_path, cfg) == []


def test_media_owner_is_not_used_as_the_author(tmp_path):
    """"Media Owner" is whose post it was; using it would drop every comment."""
    write(tmp_path, "post_comments.json", [ig_record("kept")], subdir="comments")
    docs = parse_comments(tmp_path, CommentsConfig(me=[ME]))
    assert len(docs) == 1
    assert "someone_elses_handle" not in json.dumps(docs[0].meta)


def test_short_comments_are_dropped(tmp_path):
    write(tmp_path, "comments.json", {"comments_v2": [fb_record("x"), fb_record("lol")]})
    docs = parse_comments(tmp_path, CommentsConfig(me=[ME], min_chars=2))
    assert [d.text for d in docs] == ["lol"]


def test_mojibake_is_repaired(tmp_path):
    mangled = "it's great 😅".encode("utf-8").decode("latin-1")
    write(tmp_path, "comments.json", {"comments_v2": [fb_record(mangled)]})
    assert parse_comments(tmp_path, CommentsConfig(me=[ME]))[0].text == "it's great 😅"


# ----------------------------------------------------------------- discovery


def test_only_comment_files_are_picked_up(tmp_path):
    write(tmp_path, "comments.json", {"comments_v2": [fb_record("mine")]})
    write(tmp_path, "reactions.json", {"reactions_v2": [{"title": "liked"}]})
    write(tmp_path, "comments_on_your_posts.json", {"comments_v2": [fb_record("theirs", author=OTHER)]})
    write(tmp_path, "likes_and_reactions_1.json", [{"title": "liked"}])

    names = {p.name for p in find_comment_files(tmp_path)}
    assert names == {"comments.json"}, "reaction files and other people's comments must be excluded"


def test_malformed_json_is_skipped(tmp_path):
    good = write(tmp_path, "comments.json", {"comments_v2": [fb_record("survives")]})
    bad = tmp_path / "comments_and_reactions" / "broken_comments.json"
    bad.write_text("{not json", encoding="utf-8")

    assert parse_comment_file(bad, CommentsConfig(me=[ME])) == []
    assert [d.text for d in parse_comments(tmp_path, CommentsConfig(me=[ME]))] == ["survives"]
    assert good.exists()


def test_schema_drift_is_reported_loudly(tmp_path, capsys):
    """A file full of records that parse to nothing must not look like an empty file."""
    write(tmp_path, "comments.json", {"comments_v2": [{"unexpected": {"body": "text"}}]})
    docs = parse_comments(tmp_path, CommentsConfig(me=[ME]))
    out = capsys.readouterr().out

    assert docs == []
    assert "WARNING" in out and "no known comment schema matched" in out
    assert "inspect_export.py" in out, "the warning should say how to diagnose it"


def test_genuinely_empty_file_is_quiet(tmp_path, capsys):
    write(tmp_path, "comments.json", {"comments_v2": []})
    assert parse_comments(tmp_path, CommentsConfig(me=[ME])) == []
    assert "WARNING" not in capsys.readouterr().out


def test_a_file_of_only_reactions_is_quiet(tmp_path, capsys):
    """Reactions are a correct skip, not a parse failure.

    Without this distinction every Facebook "comments and reactions" file
    reports a false schema drift, which is worse than no diagnostic.
    """
    write(tmp_path, "comments.json", {"comments_v2": [
        {"timestamp": 1, "title": "Thang Nguyen likes a post."},
        {"timestamp": 2, "title": "Thang Nguyen loves a photo."},
    ]})
    assert parse_comments(tmp_path, CommentsConfig(me=[ME])) == []
    assert "WARNING" not in capsys.readouterr().out


def test_a_reaction_and_a_real_comment_together_are_quiet(tmp_path, capsys):
    write(tmp_path, "comments.json", {"comments_v2": [
        {"timestamp": 1, "title": "Thang Nguyen likes a post."},
        fb_record("a real comment"),
    ]})
    docs = parse_comments(tmp_path, CommentsConfig(me=[ME]))
    assert [d.text for d in docs] == ["a real comment"]
    assert "WARNING" not in capsys.readouterr().out


@pytest.mark.parametrize("record,expected", [
    ({"timestamp": 1, "title": "Thang Nguyen likes a post."}, False),
    ({"timestamp": 1, "author": "Thang Nguyen"}, False),
    ({"string_map_data": {"Media Owner": {"value": "some_handle"}}}, False),
    ({"unexpected": {"body": "some actual text"}}, True),
    ({"data": [{"comment": {"comment": "real text"}}]}, True),
])
def test_record_classification(record, expected):
    from minime.data.personal.meta_comments import looks_like_comment_record

    assert looks_like_comment_record(record) is expected


def test_ids_are_stable_and_distinct(tmp_path):
    write(tmp_path, "comments.json",
          {"comments_v2": [fb_record("one", timestamp=1), fb_record("two", timestamp=2)]})
    first = [d.id for d in parse_comments(tmp_path, CommentsConfig(me=[ME]))]
    second = [d.id for d in parse_comments(tmp_path, CommentsConfig(me=[ME]))]
    assert first == second and len(set(first)) == 2


def test_comment_is_its_own_mixture_source(tmp_path):
    """Comments must not be folded into 'chat': different register, separate weight."""
    from minime.data.document import SOURCES

    assert "comment" in SOURCES
    write(tmp_path, "comments.json", {"comments_v2": [fb_record("public writing")]})
    assert parse_comments(tmp_path, CommentsConfig(me=[ME]))[0].source == "comment"


# ------------------------------------------------- the schema inspector itself


def test_inspector_never_prints_string_contents():
    """The whole point of inspect_export.py is that it is safe to paste."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "inspect_export", Path(__file__).resolve().parents[1] / "scripts" / "inspect_export.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    secret = "my private message to a friend"
    described = module.shape({
        "content": secret,
        "sender_name": "A Real Name",
        "nested": [{"deep": {"deeper": {"deepest": secret}}}],
    })
    rendered = json.dumps(described)

    assert secret not in rendered
    assert "A Real Name" not in rendered
    assert f"str(len={len(secret)})" in rendered


# ------------------------------------------------- real-export regressions


def test_appledouble_comment_files_are_skipped(tmp_path):
    """exFAT: macOS writes ``._post_comments_1.json`` beside the real file.
    It is binary, and it has "comment" in its name."""
    real = write(tmp_path, "post_comments_1.json", [ig_record("real comment")], subdir="comments")
    (real.parent / "._post_comments_1.json").write_bytes(b"\x00\x05\x16\x07\xff\xfe")
    assert [p.name for p in find_comment_files(tmp_path)] == ["post_comments_1.json"]


def test_mojibake_author_matches_a_diacritic_name(tmp_path):
    owner = "Khánh Nam Nguyễn"
    encoded = owner.encode("utf-8").decode("latin-1")
    write(tmp_path, "comments.json", {"comments_v2": [fb_record("mine", author=encoded)]})
    assert [d.text for d in parse_comments(tmp_path, CommentsConfig(me=[owner]))] == ["mine"]


def test_mainly_vietnamese_comments_are_dropped(tmp_path):
    write(tmp_path, "post_comments_1.json",
          [ig_record("so cute luon"), ig_record("đẹp quá trời")], subdir="comments")
    cfg = CommentsConfig()
    assert [d.text for d in parse_comments(tmp_path, cfg)] == ["so cute luon"]
    assert cfg.language_counts["dropped_vietnamese"] == 1
