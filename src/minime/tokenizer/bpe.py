"""Byte-level BPE: the tokenizer every real experiment uses.

The tokenizer is a *controlled variable*.  It is trained once, frozen, and
shared by every experiment arm; two arms tokenised differently are not
comparable, because their losses are measured in different units.  Three
things follow from that, and this module enforces each of them:

* **Identity.**  ``fingerprint()`` hashes the canonical tokenizer definition
  (vocab, merges, pre-tokenizer, decoder).  It is written into every shard, so
  a shard can always be traced to the tokenizer that produced it, and
  ``train.py`` refuses a mixture whose shards disagree.
* **Dev vs. frozen.**  The owner's personal corpus does not exist yet, so a
  *dev* tokenizer trained on general text unblocks the pipeline now and is
  replaced -- once, before any experiment -- by one that includes the personal
  corpus.  The dev flag is stored in the file and its name must end in
  ``-dev``, so it cannot be mistaken for the frozen one.
* **Nothing is out of vocabulary.**  Byte-level BPE starts from all 256 bytes,
  so any string encodes and round-trips exactly.  There is no ``UNK`` token.

Specials are ``<|endoftext|>`` (id 0) and ``<|me|>`` (id 1).  ``<|me|>`` is
reserved and unused in Phase 1; it exists now so that Phase 2 does not have to
retokenise the whole corpus to add it.

Corpus text is encoded with specials treated as *ordinary text*: a web page
that quotes ``<|endoftext|>`` must not forge a document boundary, and nothing
in the general corpus may ever emit ``<|me|>``.  Specials are recognised only
when a caller asks for it with ``allow_special=True``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

from .char import EOT

try:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "the BPE tokenizer needs the `tokenizers` package: pip install -e '.[data]'"
    ) from exc

ME = "<|me|>"
SPECIALS = (EOT, ME)
DEFAULT_VOCAB_SIZE = 16_384

FORMAT = "minime-bpe"
FORMAT_VERSION = 1


def _canonical(tokenizer: Tokenizer) -> str:
    # to_str() serialises the vocab from a hash map; sorting keys makes the
    # same tokenizer produce the same bytes, and so the same fingerprint.
    return json.dumps(json.loads(tokenizer.to_str()), sort_keys=True, ensure_ascii=False)


def _byte_alphabet() -> dict[str, int]:
    """The byte-level alphabet inverted: printable stand-in character -> byte.

    Byte-level BPE spells every byte as a printable character (GPT-2's
    ``bytes_to_unicode``): printable Latin-1 bytes stand for themselves, the
    other 68 are shifted to U+0100 upwards.  A vocabulary entry is a string in
    that alphabet, so its length in bytes is its length in characters.
    """
    printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1),
                 *range(ord("®"), ord("ÿ") + 1)]
    chars = {b: b for b in printable}
    shifted = 0
    for b in range(256):
        if b not in chars:
            chars[b] = 256 + shifted
            shifted += 1
    return {chr(c): b for b, c in chars.items()}


class BPETokenizer:
    """Mirrors ``CharTokenizer``: ``encode``/``decode``/``vocab_size``/``eot_id``/``save``/``load``."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        name: str = "bpe",
        dev: bool = True,
        provenance: dict | None = None,
    ) -> None:
        self._tok = tokenizer
        self.name = name
        self.dev = dev
        self.provenance = provenance or {}
        self._check()

        # A second instance that treats special strings as plain text.  Two
        # instances rather than toggling one flag, so neither mode can leak
        # into the other.
        self._plain = Tokenizer.from_str(tokenizer.to_str())
        self._plain.encode_special_tokens = True

    def _check(self) -> None:
        for expected_id, token in enumerate(SPECIALS):
            actual = self._tok.token_to_id(token)
            if actual != expected_id:
                raise ValueError(f"special {token!r} has id {actual}, expected {expected_id}")
        if self._tok.model.unk_token is not None:
            raise ValueError("a byte-level tokenizer must not have an UNK token")
        if self.vocab_size > 65_535:
            raise ValueError("vocab_size must fit in uint16 for the .bin storage format")

    # ------------------------------------------------------------ training

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        name: str = "bpe",
        dev: bool = True,
        provenance: dict | None = None,
        min_frequency: int = 2,
    ) -> "BPETokenizer":
        """Train on ``texts``.  Deterministic: the same texts give the same tokenizer.

        There is no seed here because BPE training has no randomness; the seed
        in ``scripts/train_tokenizer.py`` chooses the *sample* of texts.
        """
        tok = Tokenizer(models.BPE())  # no unk_token: every byte is in the alphabet
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(SPECIALS),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        tok.train_from_iterator(texts, trainer=trainer)

        # The trainer quietly stops early when the sample cannot support
        # enough merges.  A 15k vocab where 16k was asked for would change the
        # model's parameter count, so it is an error, not a detail.
        if tok.get_vocab_size() != vocab_size:
            raise ValueError(
                f"trained vocab has {tok.get_vocab_size():,} entries, not {vocab_size:,}: "
                "the training sample is too small to support that many merges"
            )
        return cls(tok, name=name, dev=dev, provenance=provenance)

    # ------------------------------------------------------------ encoding

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def eot_id(self) -> int:
        return SPECIALS.index(EOT)

    @property
    def me_id(self) -> int:
        return SPECIALS.index(ME)

    def encode(self, text: str, allow_special: bool = False) -> list[int]:
        tok = self._tok if allow_special else self._plain
        return tok.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        """Corpus encoding: parallel, and specials are always plain text."""
        encodings = self._plain.encode_batch_fast(list(texts), add_special_tokens=False)
        return [e.ids for e in encodings]

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special)

    def id_to_token(self, i: int) -> str | None:
        return self._tok.id_to_token(i)

    def byte_lengths(self) -> list[int]:
        """UTF-8 bytes each token id stands for, for bits-per-byte.

        A special counts as one byte: ``<|endoftext|>`` is a separator the
        model has to predict, like a newline, not thirteen characters of text.
        """
        alphabet = _byte_alphabet()
        lengths = []
        for i in range(self.vocab_size):
            token = self._tok.id_to_token(i)
            if token in SPECIALS:
                lengths.append(1)
            else:
                if any(c not in alphabet for c in token):
                    raise ValueError(f"token {i} is not spelled in the byte-level alphabet")
                lengths.append(len(token))
        return lengths

    # ------------------------------------------------------------ identity

    def fingerprint(self) -> str:
        """Short hash of the encoding function itself; metadata does not change it."""
        return hashlib.sha256(_canonical(self._tok).encode("utf-8")).hexdigest()[:16]

    @property
    def id(self) -> str:
        """``name@fingerprint``, the string stored in ``ShardMeta.tokenizer``."""
        return f"{self.name}@{self.fingerprint()}"

    # --------------------------------------------------------- persistence

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": FORMAT,
            "version": FORMAT_VERSION,
            "name": self.name,
            "dev": self.dev,
            "fingerprint": self.fingerprint(),
            "provenance": self.provenance,
            "tokenizer": json.loads(self._tok.to_str()),
        }
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"no tokenizer at {path}; train one with scripts/train_tokenizer.py"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != FORMAT:
            raise ValueError(f"{path} is not a {FORMAT} file")
        if payload.get("version") != FORMAT_VERSION:
            raise ValueError(f"{path} has format version {payload.get('version')}, "
                             f"expected {FORMAT_VERSION}")
        tok = cls(
            Tokenizer.from_str(json.dumps(payload["tokenizer"])),
            name=payload["name"],
            dev=bool(payload["dev"]),
            provenance=payload.get("provenance") or {},
        )
        # A hand-edited or truncated file would otherwise produce shards that
        # claim a tokenizer they were not made with.
        if tok.fingerprint() != payload["fingerprint"]:
            raise ValueError(
                f"{path}: stored fingerprint {payload['fingerprint']} does not match "
                f"its contents ({tok.fingerprint()}); the file has been modified"
            )
        return tok

    def __repr__(self) -> str:
        kind = "dev" if self.dev else "frozen"
        return f"BPETokenizer({self.id}, {self.vocab_size:,} tokens, {kind})"
