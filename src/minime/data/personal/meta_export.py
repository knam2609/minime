"""Parser for Meta "Download Your Information" exports (Messenger and Instagram).

Both products export the same JSON shape, so one parser covers them.

Four details make or break this parser:

1. **Only my messages.**  Everyone else's text is dropped entirely rather than
   redacted.  It is not mine to train on, and a model that learned both sides
   of a conversation would not be a model of me.
2. **Meta's mojibake.**  The exporter writes UTF-8 bytes as latin-1 code
   points, so "don't" arrives as "donâ€™t".  Left unfixed, every apostrophe and
   emoji in the corpus becomes noise -- and apostrophes and emoji are exactly
   the habits a voice model is supposed to learn.
3. **Reverse chronological order.**  Meta lists newest first.  Grouping
   messages into utterances without sorting produces backwards conversations.
4. **System messages.**  "You sent an attachment." is Facebook's text, not
   mine, and there are thousands of them.

Two parts of an export are left out on purpose: ``message_requests/`` (threads
the owner never accepted) and ``ai_conversations.json`` (writing to an AI, not
to people).  ``find_inbox_dirs`` never looks at either.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from ..document import Document, make_id
from .language import admit

# Facebook/Instagram boilerplate that is not written by any human.
#
# The subject must look like a name ("You", or capitalised words), and the
# punctuation must match Facebook's exactly.  A loose `.+?` subject here makes
# "you called it" a system message and silently deletes real writing -- which
# is the failure mode these patterns are written to avoid, since nothing
# downstream would ever reveal it.
_NAME = r"(?:You|[A-Z][\w'\u2019.-]*(?:\s[A-Z][\w'\u2019.-]*){0,4})"

SYSTEM_PATTERNS = [
    re.compile(rf"^{_NAME} sent an attachment\.$"),
    re.compile(rf"^{_NAME} sent a link\.$"),
    re.compile(rf"^{_NAME} called .+\.$"),
    re.compile(rf"^{_NAME} missed your call\.$"),
    re.compile(r"^You missed a call from .+\.$"),
    re.compile(rf"^{_NAME} started a video chat\.?$"),
    re.compile(r"^(The video chat ended|Video chat ended|Missed video call|Missed call)\.?$"),
    re.compile(rf"^{_NAME} (?:added|removed|named|changed) .*(?:the group|to the group|conversation)\.?$"),
    re.compile(r"^Liked a message$"),
    re.compile(r"^Reacted .+ to your message$"),
    re.compile(r"^You are now connected on Messenger"),
    re.compile(rf"^{_NAME} (?:unsent|removed) a message\.?$"),
    re.compile(r"^This poll is no longer available\.?$"),
    re.compile(r"^Say hi to your new Facebook friend"),
]

ATTACHMENT_KEYS = ("photos", "videos", "audio_files", "gifs", "sticker", "files", "share")

INSTAGRAM_DIR = "your_instagram_activity"
FACEBOOK_DIR = "your_facebook_activity"


@dataclass
class MetaExportConfig:
    me: list[str]
    gap_seconds: int = 300
    min_chars: int = 2
    platform: str = "messenger"
    drop_system: bool = True
    exclude_threads: list[str] = field(default_factory=list)
    # Drop a message when more than this share of its words is Vietnamese
    # (see ``language``); None keeps every message.
    max_vietnamese_share: float | None = 0.5
    # Filled in while parsing: which way the language filter went, per message.
    language_counts: Counter = field(default_factory=Counter, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.me, str):
            self.me = [self.me]
        if not self.me:
            raise ValueError("MetaExportConfig.me must name at least one display name")
        self._me_norm = {_norm_name(n) for n in self.me}

    def is_me(self, sender: str | None) -> bool:
        return bool(sender) and _norm_name(sender) in self._me_norm


def _norm_name(name: str) -> str:
    """The form in which two display names are compared.

    Meta mojibake-encodes sender names like everything else, so "Khánh" arrives
    as "KhÃ¡nh" and never equals the name the owner types.  Without the repair,
    any name with a diacritic matches nothing and the corpus comes out empty.
    NFC because the typed name and the export may compose accents differently.
    """
    return unicodedata.normalize("NFC", fix_mojibake(name)).strip().casefold()


def display_name(name: str) -> str:
    """A sender name as a person would write it: mojibake repaired, NFC."""
    return unicodedata.normalize("NFC", fix_mojibake(name)).strip()


def is_appledouble(path: Path) -> bool:
    """macOS ``._name`` metadata files, which it writes beside every file on exFAT.

    They share the real file's name after the prefix, so a glob for
    ``*comments*.json`` finds them too -- and they are binary.
    """
    return path.name.startswith("._")


def detect_platform(root: str | Path) -> str:
    """Which product an export came from, read from its layout.

    Never from the folder name: that is whatever the owner called it (this
    project's export lives in ``minimellm_data``).
    """
    root = Path(root)
    found = {
        name for name in (INSTAGRAM_DIR, FACEBOOK_DIR)
        if root.name == name or any(p.is_dir() for p in root.glob(f"**/{name}"))
    }
    if len(found) > 1:
        raise ValueError(f"{root} holds both an Instagram and a Facebook export; "
                         "pass each as its own --meta-export root")
    return "instagram" if INSTAGRAM_DIR in found else "messenger"


@dataclass(frozen=True)
class OwnerGuess:
    name: str | None      # display name of the sender present in the most threads
    threads: int          # threads that sender writes in
    runner_up: int        # threads the next most common sender writes in
    total: int            # threads scanned


def discover_me(root: str | Path) -> OwnerGuess:
    """The sender present in the most threads -- in an owner's export, the owner.

    The inventory prints this instead of failing when ``--me`` is not given, in
    the same spirit as ``github_repos.discover_authors``: the name to pass is
    whatever the export actually says, which is not always what the owner
    would guess.
    """
    senders: dict[Path, set[str]] = {}
    shown: dict[str, str] = {}
    for path in find_thread_files(root):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue
        thread = senders.setdefault(path.parent, set())
        for message in payload.get("messages") or []:
            sender = message.get("sender_name")
            if sender:
                key = _norm_name(sender)
                thread.add(key)
                shown.setdefault(key, display_name(sender))
    tally = Counter(key for keys in senders.values() for key in keys)
    ranked = tally.most_common(2)
    if not ranked:
        return OwnerGuess(None, 0, 0, len(senders))
    return OwnerGuess(
        name=shown[ranked[0][0]],
        threads=ranked[0][1],
        runner_up=ranked[1][1] if len(ranked) > 1 else 0,
        total=len(senders),
    )


def participant_names(root: str | Path, cfg: MetaExportConfig) -> list[str]:
    """Everyone else's display names, for the scrubber's names file.

    Taken from both the participants list and message senders, since someone
    who left a group can still have written in it.
    """
    names: dict[str, str] = {}
    for path in find_thread_files(root):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue
        raw = [p.get("name") for p in payload.get("participants") or [] if isinstance(p, dict)]
        raw += [m.get("sender_name") for m in payload.get("messages") or []]
        for name in raw:
            if name and not cfg.is_me(name):
                names.setdefault(_norm_name(name), display_name(name))
    return sorted(names.values(), key=str.casefold)


def fix_mojibake(text: str) -> str:
    """Undo Meta's latin-1-encoded UTF-8.

    Applied only when it actually decodes, so text that is already correct is
    returned unchanged rather than mangled.
    """
    if not text:
        return text
    try:
        repaired = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
    return repaired


def is_system_message(text: str) -> bool:
    return any(pattern.match(text.strip()) for pattern in SYSTEM_PATTERNS)


def _usable_content(message: dict, cfg: MetaExportConfig) -> str | None:
    if message.get("is_unsent"):
        return None
    content = message.get("content")
    if not content:
        return None
    # A share/photo message carries a caption-shaped `content` that is really
    # a URL or filename; drop it rather than teach the model to emit links.
    if any(key in message for key in ATTACHMENT_KEYS) and len(content) < 200:
        if content.startswith("http") or is_system_message(content):
            return None
    text = fix_mojibake(content).strip()
    if cfg.drop_system and is_system_message(text):
        return None
    if len(text) < cfg.min_chars:
        return None
    if not admit(text, cfg.max_vietnamese_share, cfg.language_counts):
        return None
    return text


def find_inbox_dirs(root: str | Path) -> list[Path]:
    """Locate every thread folder in an export, whatever its layout version.

    ``message_requests`` is not among them: those are threads never accepted.
    """
    root = Path(root)
    seen: list[Path] = []
    for pattern in ("**/messages/inbox", "**/messages/archived_threads",
                    "**/messages/filtered_threads", "**/messages/e2ee_cached_threads"):
        for path in root.glob(pattern):
            if path.is_dir():
                seen.append(path)
    return sorted(set(seen))


def find_thread_files(root: str | Path) -> list[Path]:
    files: list[Path] = []
    for inbox in find_inbox_dirs(root):
        files.extend(p for p in sorted(inbox.glob("*/message_*.json")) if not is_appledouble(p))
    return files


def parse_thread_file(path: str | Path, cfg: MetaExportConfig) -> list[Document]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []

    thread_key = payload.get("thread_path") or path.parent.name
    if any(excluded.lower() in str(thread_key).lower() for excluded in cfg.exclude_threads):
        return []

    messages = payload.get("messages") or []
    # Meta exports newest-first; sort so utterances read forwards in time.
    messages.sort(key=lambda m: m.get("timestamp_ms", 0))

    return _group_into_utterances(messages, thread_key, path, cfg)


def _group_into_utterances(
    messages: Iterable[dict], thread_key: str, path: Path, cfg: MetaExportConfig
) -> list[Document]:
    """Merge a burst of consecutive messages from me into one document.

    People write one thought across several messages.  Treating each as its own
    document would teach the model that everything is one line long.
    """
    docs: list[Document] = []
    buffer: list[str] = []
    started_at: int | None = None
    last_at: int | None = None
    thread_id = make_id(cfg.platform, thread_key)

    def flush() -> None:
        nonlocal buffer, started_at, last_at
        if buffer:
            month = datetime.fromtimestamp(started_at / 1000, tz=timezone.utc).strftime("%Y-%m")
            docs.append(Document(
                id=make_id(thread_id, started_at, len(docs)),
                source="chat",
                text="\n".join(buffer),
                author="me",
                meta={
                    "platform": cfg.platform,
                    # The thread is identified by a hash: thread titles are
                    # other people's names.
                    "thread": thread_id,
                    "messages": len(buffer),
                    "start_ms": started_at,
                    "end_ms": last_at,
                    # Utterances from one conversation must not straddle
                    # train and test: a month of a thread is split as a unit.
                    "split_group": f"{thread_id}:{month}",
                },
            ))
        buffer = []
        started_at = None
        last_at = None

    for message in messages:
        timestamp = int(message.get("timestamp_ms", 0))
        if not cfg.is_me(message.get("sender_name")):
            # Somebody else spoke: my next message starts a new utterance.
            flush()
            continue
        text = _usable_content(message, cfg)
        if text is None:
            continue
        if last_at is not None and (timestamp - last_at) > cfg.gap_seconds * 1000:
            flush()
        if started_at is None:
            started_at = timestamp
        buffer.append(text)
        last_at = timestamp

    flush()
    return docs


def iter_export(root: str | Path, cfg: MetaExportConfig) -> Iterator[Document]:
    for path in find_thread_files(root):
        yield from parse_thread_file(path, cfg)


def parse_export(root: str | Path, cfg: MetaExportConfig) -> list[Document]:
    return list(iter_export(root, cfg))
