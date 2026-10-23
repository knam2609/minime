"""Code: does the model write syntactically valid Python function bodies?

HumanEval is out of reach at 17M parameters, but syntax is not.  The task:

1. From held-out code files that parse as Python 3, take module-level
   functions whose body starts on a line after the ``def`` header.
2. Prompt with the ``prompt_tokens`` shard tokens that end at that header --
   the file so far, including the header, and the end of the previous file if
   the prefix is short -- exactly what the model saw in training.
3. Sample a continuation (stopping at ``<|endoftext|>``).
4. ``ast.parse`` the header plus the body, cut where the function ends:
   before a non-blank line at column 0 (the first such cut that parses, since
   a multi-line string may hold a column-0 line), or at the end of the
   continuation.

``valid`` is the share of prompts whose function parses.  A continuation that
runs out of tokens before the function ends loses its partial last line and is
parsed anyway (a body cut between statements is still valid code), and
``terminated`` records how many functions actually ended.  The real bodies go
through the same rule, so ``reference_valid`` should be 1.0; if it is not,
the procedure, not the model, is wrong.

Code perplexity is the ``code`` slice of the perplexity family.
"""

from __future__ import annotations

import ast
import hashlib
import warnings
from dataclasses import dataclass

import numpy as np

from ..data.binidx import TOKEN_DTYPE, TokenShard


@dataclass
class CodePrompt:
    doc_index: int
    prompt: np.ndarray  # [prompt_tokens] shard tokens ending at the cut
    header: str  # source text from the ``def`` line up to the cut
    reference_body: str  # the real text from the cut to the end of the function


def _parse(source: str) -> ast.Module | None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # invalid escape sequences etc. in old code
        try:
            return ast.parse(source)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            return None


def _is_function(source: str, header_lines: int) -> bool:
    """``source`` parses to one function whose body starts after the header.

    The second condition matters: ``def f(x):abc`` is valid Python (a one-line
    body), so without it an untrained model scores by emitting any identifier
    on the header line.  Prompts are chosen from functions with the body on
    its own lines, and the generated function must match that shape.
    """
    tree = _parse(source)
    if tree is None or len(tree.body) != 1:
        return False
    node = tree.body[0]
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.body[0].lineno > header_lines


def _functions(text: str) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    tree = _parse(text)
    if tree is None:
        return []
    return [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.col_offset == 0
        and node.body[0].lineno > node.lineno  # body on its own line(s)
    ]


def _line_starts(data: bytes) -> list[int]:
    starts = [0]
    for i, b in enumerate(data):
        if b == 0x0A:
            starts.append(i + 1)
    return starts


def function_prompts(
    shard: TokenShard,
    byte_lengths,
    decode,
    n: int,
    prompt_tokens: int,
    per_file: int = 2,
    eot_id: int | None = None,
) -> tuple[list[CodePrompt], str]:
    """Up to ``n`` prompts, at most ``per_file`` per file, files evenly spaced.

    ``decode`` turns token ids into text; ``byte_lengths`` maps ids to UTF-8
    lengths so a character position in the file can be found in token space.
    The prompt stops at the last token boundary at or before the end of the
    header line, so a token that merges the newline with the body's
    indentation is left for the model to write.
    """
    if shard.offsets is None:
        raise ValueError(f"{shard!r} has no document index")
    offsets = shard.offsets
    byte_lengths = np.asarray(byte_lengths, dtype=np.int64)
    candidates: list[CodePrompt] = []
    # Visit files spread across the shard first (every stride-th), then the
    # ones in between, and stop once n prompts are found: decoding and parsing
    # every held-out file would be wasted work, and the front of the shard is
    # one stretch of the source's stream order.
    files_needed = max(1, -(-n // max(1, per_file)))
    stride = max(1, shard.n_docs // files_needed)
    order = [i for r in range(stride) for i in range(r, shard.n_docs, stride)]
    for doc_index in order:
        if len(candidates) >= n:
            break
        start, end = int(offsets[doc_index]), int(offsets[doc_index + 1])
        ids = np.asarray(shard.tokens[start:end], dtype=np.int64)
        if eot_id is not None and ids.size and ids[-1] == eot_id:
            ids = ids[:-1]
        text = decode(ids.tolist())
        data = text.encode("utf-8")
        if len(data) != int(byte_lengths[ids].sum()):
            continue  # not clean UTF-8 at token level; positions would not line up
        functions = _functions(text)
        if not functions:
            continue
        line_starts = _line_starts(data)
        cum = np.concatenate([[0], np.cumsum(byte_lengths[ids])])
        for node in functions[:per_file]:
            header_end = line_starts[node.body[0].lineno - 1] - 1  # the newline ending the header
            k = int(np.searchsorted(cum, header_end, side="right")) - 1  # tokens fully before it
            if start + k < prompt_tokens or k <= 0:
                continue
            def_start = line_starts[node.lineno - 1]
            if cum[k] <= def_start:
                continue
            body_end = line_starts[node.end_lineno] if node.end_lineno < len(line_starts) else len(data)
            candidates.append(CodePrompt(
                doc_index=doc_index,
                prompt=np.asarray(shard.tokens[start + k - prompt_tokens : start + k], dtype=np.int64),
                header=data[def_start : cum[k]].decode("utf-8", errors="replace"),
                reference_body=data[cum[k] : body_end].decode("utf-8", errors="replace"),
            ))
            if len(candidates) >= n:
                break

    h = hashlib.sha256(f"code:{prompt_tokens}:{per_file}:{n}".encode())
    for c in candidates:
        h.update(c.prompt.astype(TOKEN_DTYPE).tobytes())
        h.update(b";")
    return candidates, h.hexdigest()[:16]


def function_cuts(header: str, body: str, complete: bool) -> list[str]:
    """Where the function written as ``header + body`` could end, in order.

    A function ends before a non-blank line at column 0.  Usually the first
    such line is the end, but a multi-line string may legally put a line at
    column 0 inside the body, so every column-0 line is a candidate.  The last
    candidate is the whole text -- minus a trailing partial line unless
    ``complete`` (the text stopped at EOT, not mid-line at the token limit).
    """
    lines = (header + body).split("\n")
    header_lines = header.count("\n") + 1
    cuts = ["\n".join(lines[:i]) + "\n"
            for i in range(header_lines, len(lines))
            if lines[i].strip() and lines[i][0] not in " \t"]
    kept = lines if complete or len(lines) <= header_lines else lines[:-1]
    cuts.append("\n".join(kept) + "\n")
    return cuts


def function_parses(header: str, body: str, complete: bool) -> tuple[bool, bool]:
    """``(valid, terminated)``: is there an end at which the function parses?

    Valid if any candidate end (``function_cuts``) yields exactly one function
    with its body on the lines after the header; terminated if the text
    reached a column-0 line after the header, i.e. the function closed before
    the continuation ran out.  Real bodies go through the same rule, so the
    rule's generosity is measured, not assumed.
    """
    cuts = function_cuts(header, body, complete)
    header_lines = header.count("\n") + 1
    return any(_is_function(c, header_lines) for c in cuts), len(cuts) > 1


def syntax_validity(prompts: list[CodePrompt], bodies: list[str], ended: list[bool]) -> dict:
    """Share of generated functions that parse, beside the real bodies' share."""
    valid = terminated = reference_valid = 0
    for prompt, body, hit_eot in zip(prompts, bodies, ended):
        ok, closed = function_parses(prompt.header, body, complete=hit_eot)
        valid += ok
        terminated += closed or hit_eot
        reference_valid += function_parses(prompt.header, prompt.reference_body, complete=True)[0]
    n = len(prompts)
    return {
        "prompts": n,
        "valid": valid / n if n else None,
        "terminated": terminated / n if n else None,
        "reference_valid": reference_valid / n if n else None,
    }
