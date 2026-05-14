"""Small filesystem helpers used across the package.

The functions here exist to remove three classes of bugs:

1. Partial writes — a writer crashing mid-write leaves a corrupt file behind.
   :func:`atomic_write_yaml` writes to a temp file then renames, so readers
   only see complete content.
2. Silent missing directories — ``Path.mkdir`` with ``parents=True`` is the
   right default but the call site clutter is repetitive. :func:`ensure_dir`
   centralizes it.
3. Drifted data — when the project claims a file's contents are pinned, we
   need a cheap integrity check. :func:`compute_sha256` is the chosen one.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: Path | str) -> Path:
    """Create ``path`` (and parents) if missing. Returns the resolved Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def compute_sha256(path: Path | str, *, chunk_size: int = 1 << 20) -> str:
    """Stream the file at ``path`` through SHA-256 and return the hex digest.

    Streaming (rather than reading the whole file into memory) lets us hash
    the ~130 MB raw UCI dataset cheaply.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Cannot hash non-file path: {p}")
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write_yaml(path: Path | str, data: Any) -> None:
    """Write ``data`` to ``path`` as YAML, atomically.

    Writes to a sibling tempfile then renames over the target — a partial write
    can never appear at the final path. Caller-visible failure modes:
    file-system errors propagate; serialization errors propagate.
    """
    target = Path(path)
    ensure_dir(target.parent)
    fd, tmp_name = tempfile.mkstemp(
        prefix=target.name + ".",
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=False)
        os.replace(tmp_name, target)
    except Exception:
        # Clean up the orphaned tempfile so we don't leak it.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
