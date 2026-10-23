"""Streaming access to Hugging Face datasets, with nothing kept on disk.

The general corpus is several gigabytes of text.  Downloading it first would
put it on the internal disk and leave it there; streaming reads each file over
HTTP, yields rows and keeps nothing.  The only cache is dataset metadata, and
that goes under ``data_root()`` so it follows ``MINIME_DATA_ROOT`` to the
external drive.

Every source is pinned to a commit.  A stream is deterministic for a fixed
revision (files in order, rows in order), so pinning is what makes "the first
630M tokens of FineWeb-Edu" the same tokens on every machine and every re-run.

That determinism is also what makes a dropped connection recoverable: reopen
the split, skip the rows already yielded, carry on.  The consumer sees every
row exactly once, in order, as if nothing had happened.  Only network errors
are retried; anything else -- a gated dataset, a schema problem -- raises.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

from ...config import data_root


class SchemaDriftError(KeyError):
    """A row is missing a field the converter depends on."""

    def __str__(self) -> str:  # KeyError would otherwise repr() the message
        return str(self.args[0])


@dataclass(frozen=True)
class HFSource:
    dataset: str
    config: str | None
    splits: tuple[str, ...]
    revision: str
    license: str

    def describe(self) -> dict:
        return {
            "dataset": self.dataset,
            "config": self.config,
            "splits": list(self.splits),
            "revision": self.revision,
            "license": self.license,
        }


def field(row: dict, key: str, source: HFSource):
    try:
        return row[key]
    except KeyError:
        raise SchemaDriftError(
            f"{source.dataset}@{source.revision[:10]}: row has no {key!r} field "
            f"(fields: {sorted(row)}); the dataset schema differs from the one this "
            "converter was written against"
        ) from None


# The library defaults (10 s read timeout, 20 read retries 5 s apart) were
# observed to fail on a loaded laptop: a process starved of CPU misses its own
# read deadline, and five of those in a row end a multi-hour ingestion.  These
# only make the stream more patient; they cannot change what it yields.
READ_TIMEOUT_S = 60
READ_RETRIES = 120
RETRY_INTERVAL_S = 10


def _patient_streaming() -> None:
    import huggingface_hub.constants as hub
    from datasets import config

    hub.HF_HUB_DOWNLOAD_TIMEOUT = max(hub.HF_HUB_DOWNLOAD_TIMEOUT, READ_TIMEOUT_S)
    hub.HF_HUB_ETAG_TIMEOUT = max(hub.HF_HUB_ETAG_TIMEOUT, READ_TIMEOUT_S)
    config.STREAMING_READ_MAX_RETRIES = max(config.STREAMING_READ_MAX_RETRIES, READ_RETRIES)
    config.STREAMING_READ_RETRY_INTERVAL = max(config.STREAMING_READ_RETRY_INTERVAL, RETRY_INTERVAL_S)


def _transient_errors() -> tuple[type[BaseException], ...]:
    errors: list[type[BaseException]] = [ConnectionError, TimeoutError]
    try:
        import requests

        # Not RequestException: an HTTPError (404 on a bad revision) is permanent,
        # and huggingface_hub already retries 5xx responses itself.
        errors += [
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ]
    except ImportError:  # pragma: no cover
        pass
    try:
        import aiohttp

        errors.append(aiohttp.ClientError)
    except ImportError:  # pragma: no cover
        pass
    return tuple(errors)


def _hf_opener(source: HFSource) -> Callable[[str], Iterable[dict]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "streaming general sources needs `datasets`: pip install -e '.[data]'"
        ) from exc
    _patient_streaming()
    cache_dir = data_root() / "cache" / "huggingface"

    def open_split(split: str) -> Iterable[dict]:
        try:
            return load_dataset(
                source.dataset,
                source.config,
                split=split,
                streaming=True,
                revision=source.revision,
                cache_dir=str(cache_dir),
            )
        except Exception as exc:
            text = str(exc).lower()
            if "gated" in text or "401" in text or "403" in text:
                raise PermissionError(
                    f"{source.dataset} refused access ({exc.__class__.__name__}). It may be "
                    "gated: accept its terms on the Hub and run `huggingface-cli login`."
                ) from exc
            raise

    return open_split


def stream_rows(
    source: HFSource,
    open_split: Callable[[str], Iterable[dict]] | None = None,
    attempts: int = 8,
    wait: float = 30.0,
    log: Callable[[str], None] = lambda message: print(message, flush=True),
) -> Iterator[dict]:
    """Yield raw rows from every split of ``source``, in order, surviving dropped connections.

    ``attempts`` bounds *consecutive* failures without progress; a stream that
    keeps moving between failures is never abandoned.  ``open_split`` exists
    for tests.
    """
    open_split = open_split or _hf_opener(source)
    transient = _transient_errors()
    for split in source.splits:
        done = failures = 0
        progress_at_failure = -1
        while True:
            try:
                for row in itertools.islice(open_split(split), done, None):
                    yield row
                    done += 1
                break
            except transient as exc:
                failures = 1 if done > progress_at_failure else failures + 1
                progress_at_failure = done
                if failures > attempts:
                    raise
                log(f"  {source.dataset}:{split}: {exc.__class__.__name__} after {done:,} rows; "
                    f"reopening and skipping ahead in {wait:.0f}s (attempt {failures}/{attempts})")
                time.sleep(wait)
