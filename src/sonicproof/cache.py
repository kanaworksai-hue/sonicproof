"""On-disk embedding cache so re-checking a file is instant.

Lives in ``SONICPROOF_CACHE``, else ``%LOCALAPPDATA%\\sonicproof\\Cache`` on Windows, else
``$XDG_CACHE_HOME/sonicproof`` / ``~/.cache/sonicproof``. Writes are atomic (unique temp file + rename) so
threads and processes can share it; an unreadable entry is simply a miss.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from .model import MODEL_ID, WINDOW_SECONDS, embed_tag

CACHE_VERSION = 2  # bump when preprocessing changes (resampling, windowing, frame sampling …)
EMBED_KINDS = {"audio", "image", "video", "vocab"}  # depend on the model's size, dtype and snapshot


def cache_dir() -> Path:
    root = os.environ.get("SONICPROOF_CACHE") or _default_root()
    p = Path(root).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _default_root() -> str:
    if sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        return os.path.join(os.environ["LOCALAPPDATA"], "sonicproof", "Cache")
    return os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "sonicproof")


def _key(kind: str, *parts: str) -> str:
    h = hashlib.sha1()
    base = [f"v{CACHE_VERSION}", MODEL_ID, f"w{WINDOW_SECONDS}"]
    if kind in EMBED_KINDS:
        base.append(embed_tag())
    for part in (*base, *parts):
        h.update(part.encode())
        h.update(b"\0")
    return h.hexdigest()


def file_digest(path: str | Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_array(kind: str, *parts: str) -> np.ndarray | None:
    p = cache_dir() / kind / f"{_key(kind, *parts)}.npy"
    try:
        a = np.load(p, allow_pickle=False)
    except Exception:  # missing, half-written by an older version, corrupt: treat as a miss
        return None
    if a.dtype.kind == "f" and not np.isfinite(a).all() and kind != "level":
        return None  # an embedding with NaN in it (written before inputs were sanitised): recompute
    return a


def save_array(arr: np.ndarray, kind: str, *parts: str) -> None:
    if arr.dtype.kind == "f" and not np.isfinite(arr).all() and kind != "level":
        return  # never cache a broken embedding
    d = cache_dir() / kind
    d.mkdir(exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".", suffix=".tmp")
    dest = d / f"{_key(kind, *parts)}.npy"
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, arr)
        for attempt in range(4):
            try:
                os.replace(tmp, dest)
                return
            except PermissionError:  # Windows: another process has the old file open for a moment
                if attempt == 3:
                    return  # best effort: the next run computes it again
                time.sleep(0.05 * (attempt + 1))
    finally:
        Path(tmp).unlink(missing_ok=True)
