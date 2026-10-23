"""What ``scripts/train.py`` reads: the mixture's shards and the validation sets.

A run can draw from more than one shard directory -- an adaptation arm mixes
the general corpus (``web``, ``dialogue``, ``code``) with the personal one
(``chat``, ``comment``) -- so each source must be found in exactly one of
them.  A name in two directories would make the run's data depend on the
order of its ``--data`` arguments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..data.binidx import TokenShard


def resolve_shard(data_dirs: Sequence[str | Path] | str | Path, name: str) -> TokenShard:
    """Source ``name`` from the one directory that has it.

    Accepts ``<name>.train`` (a per-source shard) or a plain ``<name>`` shard.
    """
    return TokenShard(find_shard(data_dirs, name))


def find_shard(data_dirs: Sequence[str | Path] | str | Path, name: str) -> Path:
    """Where ``resolve_shard`` would read source ``name``, without opening it."""
    if isinstance(data_dirs, (str, Path)):
        data_dirs = [data_dirs]
    found = []
    for data_dir in map(Path, data_dirs):
        for candidate in (data_dir / f"{name}.train", data_dir / name):
            # Append rather than with_suffix: "chat.train".with_suffix(".bin")
            # would look for "chat.bin".
            if Path(f"{candidate}.bin").exists():
                found.append(candidate)
                break
    where = ", ".join(map(str, data_dirs))
    if not found:
        raise FileNotFoundError(
            f"no shard for source '{name}' in {where} (looked for {name}.train.bin and {name}.bin)"
        )
    if len(found) > 1:
        raise ValueError(
            f"source '{name}' is in more than one data directory ({', '.join(map(str, found))}): "
            "which one the run reads would depend on the order of --data"
        )
    return found[0]


def parse_val_shards(specs: Sequence[str], data_dirs: Sequence[str | Path]) -> dict[str, Path]:
    """``--val-shard`` arguments -> ``{name: shard path}``, in order.

    ``SHARD`` is a shard in the first data directory, named after itself
    (``val`` -> ``{"val": <dir>/val}``: a pretrain's single set, as before).
    ``NAME=PATH`` names a shard anywhere, so a run can validate on several:
    ``general=/data/shards/general/val personal=/data/shards/personal/val``.
    The first set is the run's ``val_loss``.
    """
    if not specs:
        raise ValueError("at least one validation set is needed")
    out: dict[str, Path] = {}
    for spec in specs:
        name, sep, path = spec.partition("=")
        if not sep:
            name, path = spec, str(Path(data_dirs[0]) / spec)
        if not name or not path:
            raise ValueError(f"--val-shard wants SHARD or NAME=PATH, got {spec!r}")
        if name in out:
            raise ValueError(f"two validation sets named {name!r}")
        out[name] = Path(path)
    return out


__all__ = ["find_shard", "parse_val_shards", "resolve_shard"]
