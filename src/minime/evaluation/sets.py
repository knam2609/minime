"""Fixed prompt sets drawn from token shards.

A prompt set is a pure function of a shard and a few integers: the same shard
gives the same prompts on every machine, and ``digest`` proves it.  Prompts in
a set share one length, so ``generation.sample`` can batch them without
padding.

Two shapes, for two kinds of text:

* ``document_prompts`` -- the ``prompt_tokens`` tokens that *precede* a
  document, ending with the previous document's EOT; the target is the
  document itself.  This is the personal-chat set: the context is the owner's
  preceding messages in the same conversation (the corpus is written in
  conversation order), and the model writes the next one.
* ``prefix_prompts`` -- the opening ``prompt_tokens`` tokens *of* a document;
  the target is what follows.  This is the long-form set used for the
  degeneration metrics, where a continuation must be long enough to repeat
  itself.

Selection is evenly spaced over the eligible documents, never "the first n",
because a shard's order is its source's order.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from ..data.binidx import TOKEN_DTYPE, TokenShard


@dataclass
class PromptSet:
    prompts: np.ndarray  # [n, prompt_tokens] int64
    targets: list[np.ndarray]  # the real continuation per prompt, EOT excluded
    doc_indices: list[int]
    digest: str
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.targets)

    def describe(self) -> dict:
        return {"digest": self.digest, "prompts": len(self), **self.meta}


def evenly_spaced(candidates: np.ndarray | list[int], n: int) -> list[int]:
    """``n`` items spread from the first candidate to the last (all if fewer)."""
    candidates = list(candidates)
    if n <= 0 or not candidates:
        return []
    if n >= len(candidates):
        return candidates
    return [candidates[(i * len(candidates)) // n] for i in range(n)]


def _digest(kind: str, prompts: np.ndarray, targets: list[np.ndarray], params: dict) -> str:
    h = hashlib.sha256(f"{kind}:{sorted(params.items())}".encode())
    for prompt, target in zip(prompts, targets):
        h.update(prompt.astype(TOKEN_DTYPE).tobytes())
        h.update(b"|")
        h.update(np.asarray(target, dtype=TOKEN_DTYPE).tobytes())
        h.update(b";")
    return h.hexdigest()[:16]


def _strip_eot(doc: np.ndarray, eot_id: int | None) -> np.ndarray:
    return doc[:-1] if eot_id is not None and doc.size and doc[-1] == eot_id else doc


def _require_index(shard: TokenShard) -> np.ndarray:
    if shard.offsets is None:
        raise ValueError(f"{shard!r} has no document index; prompt sets need document boundaries")
    return shard.offsets


def document_prompts(
    shard: TokenShard,
    n: int,
    prompt_tokens: int,
    max_target_tokens: int,
    eot_id: int | None,
) -> PromptSet:
    """Prompts that end where a document starts; targets are those documents."""
    offsets = _require_index(shard)
    starts = offsets[:-1]
    eligible = [int(i) for i in np.nonzero(starts >= prompt_tokens)[0]
                if _strip_eot(shard.document(int(i)), eot_id).size > 0]
    picks = evenly_spaced(eligible, n)
    if not picks:
        raise ValueError(f"{shard!r} has no document preceded by {prompt_tokens} tokens")

    prompts = np.stack([
        np.asarray(shard.tokens[int(starts[i]) - prompt_tokens : int(starts[i])], dtype=np.int64)
        for i in picks
    ])
    targets = [_strip_eot(shard.document(i), eot_id)[:max_target_tokens].astype(np.int64) for i in picks]
    params = {"prompt_tokens": prompt_tokens, "max_target_tokens": max_target_tokens, "n": n}
    return PromptSet(
        prompts, targets, picks, _digest("document", prompts, targets, params),
        meta={"kind": "document", "eligible_documents": len(eligible), **params},
    )


def message_prompts(
    shard: TokenShard,
    n: int,
    prompt_tokens: int,
    min_tokens: int,
    max_tokens: int,
    eot_id: int,
) -> PromptSet:
    """``document_prompts`` over the documents of ``min_tokens``-``max_tokens`` tokens, uncropped.

    The memorisation check's set: each target is a whole message, so a model
    that writes it and then the EOT has reproduced it exactly.
    """
    offsets = _require_index(shard)
    starts = offsets[:-1]
    lengths = np.diff(offsets) - 1   # the EOT each document ends with
    eligible = [int(i) for i in np.nonzero((starts >= prompt_tokens) & (lengths >= min_tokens)
                                           & (lengths <= max_tokens))[0]]
    picks = evenly_spaced(eligible, n)
    if not picks:
        raise ValueError(f"{shard!r} has no message of {min_tokens}-{max_tokens} tokens after {prompt_tokens}")
    prompts = np.stack([
        np.asarray(shard.tokens[int(starts[i]) - prompt_tokens : int(starts[i])], dtype=np.int64)
        for i in picks
    ])
    targets = [_strip_eot(shard.document(i), eot_id).astype(np.int64) for i in picks]
    params = {"prompt_tokens": prompt_tokens, "min_tokens": min_tokens, "max_tokens": max_tokens, "n": n}
    return PromptSet(
        prompts, targets, picks, _digest("message", prompts, targets, params),
        meta={"kind": "message", "eligible_documents": len(eligible), **params},
    )


def prefix_prompts(
    shard: TokenShard,
    n: int,
    prompt_tokens: int,
    target_tokens: int,
    eot_id: int | None,
) -> PromptSet:
    """Opening ``prompt_tokens`` of documents at least ``prompt + target`` long."""
    offsets = _require_index(shard)
    need = prompt_tokens + target_tokens
    lengths = np.diff(offsets) - (1 if eot_id is not None else 0)
    eligible = [int(i) for i in np.nonzero(lengths >= need)[0]]
    picks = evenly_spaced(eligible, n)
    if not picks:
        raise ValueError(f"{shard!r} has no document of {need} or more tokens")

    docs = [_strip_eot(shard.document(i), eot_id).astype(np.int64) for i in picks]
    prompts = np.stack([d[:prompt_tokens] for d in docs])
    targets = [d[prompt_tokens:need] for d in docs]
    params = {"prompt_tokens": prompt_tokens, "target_tokens": target_tokens, "n": n}
    return PromptSet(
        prompts, targets, picks, _digest("prefix", prompts, targets, params),
        meta={"kind": "prefix", "eligible_documents": len(eligible), **params},
    )
