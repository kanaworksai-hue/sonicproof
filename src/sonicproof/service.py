"""One engine and one JSON shape for every front end: the CLI's ``--json``, the MCP server and ``sp serve``.

``Engine`` loads EmbeddingGemma 2 once per process (text + audio first; the vision encoder only when a picture
is involved) and keeps opened libraries, so long-lived servers never load the model twice.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from pathlib import Path

from . import audio as audio_io
from . import qc as qc_mod
from .errors import InputError, SetupError, SonicproofError

Progress = Callable[[int, int, str], None]

INSTALL_HINT = "pip install 'sonicproof[model]' (or: uv sync --extra model)"


# -- errors -------------------------------------------------------------------------------------
def _one_line(text: str, limit: int = 300) -> str:
    line = next((x.strip() for x in str(text).splitlines() if x.strip()), "")
    return line if len(line) <= limit else line[: limit - 1] + "…"


def as_error(exc: BaseException, file: str | Path | None = None) -> SonicproofError:
    """Any exception -> a SonicproofError with a stable code and a one-line message (``__cause__`` = exc)."""
    if isinstance(exc, SonicproofError):
        if file is not None and exc.file is None:
            exc.file = str(file)
        return exc
    f = str(file) if file is not None else (getattr(exc, "filename", None) and str(exc.filename))
    name = Path(f).name if f else None
    if isinstance(exc, ImportError):
        mod = getattr(exc, "name", None) or "a model dependency"
        out: SonicproofError = SetupError(
            f"This needs the model extra ({mod} is not installed): {INSTALL_HINT}", code="missing_extra"
        )
    elif isinstance(exc, FileNotFoundError):
        out = InputError(f"No such file: {f or exc}", code="not_found", file=f)
    elif isinstance(exc, IsADirectoryError | NotADirectoryError):
        what = "is a folder, not a file" if isinstance(exc, IsADirectoryError) else "is not a folder"
        out = InputError(
            f"{f} {what}", code="is_dir" if isinstance(exc, IsADirectoryError) else "bad_value", file=f
        )
    elif isinstance(exc, PermissionError):
        out = InputError(f"Permission denied: {f or exc}", code="unreadable", file=f)
    elif isinstance(exc, OSError) and ("huggingface" in str(exc).lower() or "couldn't connect" in str(exc)):
        out = SetupError(
            "EmbeddingGemma 2 is not downloaded and the Hugging Face Hub is unreachable. Run "
            "`sonicproof doctor --load` once with network, or point HF_HOME at an existing cache.",
            code="model_unavailable",
        )
    elif isinstance(exc, ValueError):
        out = InputError(_one_line(exc), code="bad_value", file=f)
    elif isinstance(exc, OSError | RuntimeError):  # decoders: libsndfile, PIL, ffmpeg
        msg = _one_line(exc)
        out = InputError(
            f"Cannot read {name}: {msg}" if name and name not in msg else msg, code="unreadable", file=f
        )
    else:
        out = SonicproofError(f"Unexpected {type(exc).__name__}: {_one_line(exc)}", code="internal", file=f)
    out.__cause__ = exc
    return out


def error_dict(exc: BaseException, file: str | Path | None = None) -> dict:
    return as_error(exc, file).to_dict()


def probe_audio(path: str | Path) -> Path:
    """Cheap up-front check of an audio input (before any model is loaded). Raises InputError / SetupError."""
    p = Path(path).expanduser()
    shown = str(path)
    if not p.exists():
        raise InputError(f"No such file: {shown}", code="not_found", file=shown)
    if p.is_dir():
        raise InputError(f"{shown} is a folder, not a file", code="is_dir", file=shown)
    if p.stat().st_size == 0:
        raise InputError(f"{shown} is empty (0 bytes)", code="empty", file=shown)
    import soundfile as sf

    try:
        sf.info(str(p))  # header only
    except Exception:
        try:
            audio_io.load(p)  # ffmpeg formats; raises the precise reason (no audio track, needs ffmpeg …)
        except Exception as e:
            raise as_error(e, shown) from e
    return p


def need_prompt(file: str | Path, prompt: str | None) -> None:
    """A prompt is given, or the file has a sidecar with one (checked before the model loads)."""
    if prompt is not None and not prompt.strip():
        raise InputError("The prompt is empty.", code="no_prompt", file=str(file))
    if prompt is None and not audio_io.sidecar_prompt(Path(file).expanduser()):
        raise InputError(
            f"No prompt given and no .json sidecar with a prompt next to {Path(file).name}.",
            code="no_prompt",
            file=str(file),
        )


def require_model(allow_download: bool = True) -> None:
    """Fail fast, in one line, when the model can't be loaded: missing extra, offline without a cache, or (for
    ``allow_download=False``) simply not downloaded yet."""
    from .model import (
        DOWNLOAD_SIZE,
        hub_cache_dir,
        model_cached,
        require_model_available,
        require_model_extra,
    )

    require_model_extra()
    require_model_available()
    if not allow_download and not model_cached() and os.environ.get("SONICPROOF_ALLOW_DOWNLOAD") != "1":
        raise SetupError(
            f"EmbeddingGemma 2 is not downloaded yet (looked in {hub_cache_dir()}). "
            f"Run `sonicproof doctor --load` once in a terminal "
            f"({DOWNLOAD_SIZE}, first run only), or give this server the HF_HOME you use.",
            code="model_unavailable",
        )


# -- result shapes ------------------------------------------------------------------------------
def qc_record(path: str | Path, loop: bool = False, target_lufs: float | None = None) -> dict:
    a = audio_io.load(Path(path).expanduser())
    checks = qc_mod.run(a, loop=loop, target_lufs=target_lufs)
    return {
        "file": str(path),
        "status": qc_mod.summary(checks),
        "duration": a.duration,
        "sample_rate": a.sr,
        "channels": a.channels,
        "checks": [c.to_dict() for c in checks],
    }


def qc_record_safe(path: str | Path, loop: bool = False, target_lufs: float | None = None) -> dict:
    """``qc_record``, or ``{"file", "status": "error", "error", "checks": []}`` for an unreadable file."""
    try:
        probe_audio(path)
        return qc_record(path, loop, target_lufs)
    except Exception as e:
        return {"file": str(path), "status": "error", "error": error_dict(e, path), "checks": []}


def qc_records(files, loop: bool = False, target_lufs: float | None = None) -> list[dict]:
    if not files:
        raise InputError("No files given.", code="bad_value")
    return [qc_record_safe(f, loop, target_lufs) for f in files]


def worst_status(records: list[dict]) -> str:
    """error > fail > warn > info > pass over qc records."""
    order = ["pass", "info", "warn", "fail", "error"]
    return max((r.get("status", "pass") for r in records), key=order.index, default="pass")


def rank_records(judge, prompt: str, files, n_alt: int = 3) -> list[dict]:
    """Takes best first; unreadable takes go last with ``verdict: "error"`` (they never sink the batch)."""
    if not files:
        raise InputError("No files given.", code="bad_value")
    good, bad = [], []
    for f in files:
        try:
            probe_audio(f)
            good.append(f)
        except Exception as e:
            bad.append({"file": str(f), "verdict": "error", "error": error_dict(e, f)})
    results = []
    if good:
        t = judge.text([prompt])[0]
        for f in good:
            try:
                results.append(judge.check(Path(f).expanduser(), prompt, n_alt=n_alt, _t=t).to_dict())
            except Exception as e:
                bad.append({"file": str(f), "verdict": "error", "error": error_dict(e, f)})
    results.sort(key=_rank_key, reverse=True)
    return results + bad


def _rank_key(r: dict) -> tuple:
    # silent takes (note "silent") always rank last among the readable ones
    return (r.get("note") != "silent", r["score"])


def overall(verdict: str, tech: str) -> tuple[str, str]:
    """SHIP / REVIEW / REDO from the adherence verdict and the QC status."""
    if verdict in ("strong", "plausible") and tech == "pass":
        return "ship", "Sounds like the prompt and is technically clean."
    if verdict == "miss":
        return "redo", "Does not sound like the prompt — regenerate."
    if tech == "fail":
        return "redo", "Technical problems — fix before using."
    return "review", "Usable, but listen first: see the flagged items."


def report_record(check: dict, qc: dict) -> dict:
    status, note = overall(check["verdict"], qc["status"])
    return {
        "file": qc["file"],
        "overall": status,
        "note": note,
        "adherence": check,
        "qc": {"status": qc["status"], "checks": qc["checks"]},
        "duration": qc["duration"],
        "sample_rate": qc["sample_rate"],
        "channels": qc["channels"],
    }


def index_record(lib, st) -> dict:
    return {
        "library": str(lib.root),
        "index": str(lib.location),
        "files": len(lib),
        "duration": lib.total_duration(),
        "added": st.added,
        "changed": st.changed,
        "removed": st.removed,
        "unchanged": st.unchanged,
        "embedded": st.embedded,
        "failed": st.failed,
    }


def dedupe_record(lib, groups, failed: list[str] | None = None) -> dict:
    from .library import REDUNDANT

    return {
        "library": str(lib.root),
        "files": len(lib),
        "failed": failed or [],
        "groups": [g.to_dict() for g in groups],
        "redundant_files": sum(m.relation in REDUNDANT for g in groups for m in g.members),
        "reclaimable_bytes": sum(g.reclaimable_bytes for g in groups),
    }


def search_record(
    lib, mode: str, query, hits, like=None, picture=None, failed: list[str] | None = None
) -> dict:
    out = {"library": str(lib.root), "mode": mode, "query": query, "results": [h.to_dict() for h in hits]}
    if like is not None:
        out["like"] = str(like)
    if picture is not None:
        out["picture"] = str(picture)
    out["failed"] = failed or []
    return out


def picture_kind(path: str | Path) -> str:
    from .visual import VIDEO_EXTS

    return "video" if Path(path).suffix.lower() in VIDEO_EXTS else "image"


# -- engine -------------------------------------------------------------------------------------
class Engine:
    """Process-wide model + libraries. Thread-safe; model calls are serialised by the embedder.

    ``allow_download=False`` (the MCP server) refuses to fetch the model inside a call: a 1.5 GB download
    would outlast any tool timeout. ``SONICPROOF_ALLOW_DOWNLOAD=1`` lifts it.
    """

    def __init__(self, device: str | None = None, allow_download: bool = True):
        self._device = device
        self.allow_download = allow_download
        self._judge = None
        self._lock = threading.RLock()
        self._libs: dict[tuple[str, bool], object] = {}
        self._lib_locks: dict[tuple[str, bool], threading.Lock] = {}

    @property
    def loaded(self) -> str | None:
        """None, "audio" (text + audio) or "vision" (full model)."""
        j = self._judge
        if j is None:
            return None
        return "vision" if j.embedder.vision else "audio"

    def judge(self, vision: bool = False):
        with self._lock:
            if self._judge is not None and (self._judge.embedder.vision or not vision):
                return self._judge
            require_model(self.allow_download)
            from .judge import Judge
            from .model import Embedder, prefer_offline

            prefer_offline()
            old = self._judge
            j = Judge(embedder=Embedder(self._device, vision=vision))
            _ = j.V
            if old is not None:
                # Swap in the full model. Text/audio embeddings are identical, so a call still holding the
                # old Judge carries on with the new embedder; clearing it instead would make that Judge fall
                # back to get_embedder() and load a third copy of the model.
                old._embedder = j.embedder
            self._judge = j
            if old is not None:
                del old
                _free_accelerator_memory()
            return j

    # -- single files ----------------------------------------------------------------------
    def check(self, file: str | Path, prompt: str | None = None, top: int = 5) -> dict:
        p = probe_audio(file)
        need_prompt(p, prompt)
        return self.judge().check(p, prompt, n_alt=top).to_dict()

    def rank(self, prompt: str, files: list[str | Path], top: int = 3) -> list[dict]:
        if not files:
            raise InputError("No files given.", code="bad_value")
        return rank_records(self.judge(), prompt, files, n_alt=top)

    def qc(self, files: list[str | Path], loop: bool = False, target_lufs: float | None = None) -> list[dict]:
        return qc_records(files, loop, target_lufs)

    def report(
        self,
        file: str | Path,
        prompt: str | None = None,
        loop: bool = False,
        target_lufs: float | None = None,
    ) -> dict:
        p = probe_audio(file)
        need_prompt(p, prompt)
        q = qc_record(p, loop, target_lufs)
        return report_record(self.check(p, prompt, top=3), q)

    def fit(
        self,
        file: str | Path,
        picture: str | Path | None = None,
        scene: str | None = None,
        top: int = 5,
        kind: str | None = None,
    ) -> dict:
        """``picture`` defaults to the file itself when it is a video (its own soundtrack vs its picture)."""
        p = probe_audio(file)
        pic, kind = resolve_picture(p, picture, kind)
        return self.judge(vision=True).fit(p, pic, kind, scene, top).to_dict()

    def describe_picture(self, picture: str | Path, top: int = 8) -> dict:
        """What the picture shows, in vocabulary words ("pictured as")."""
        from .vocab import category_of

        pic, kind = resolve_picture(None, picture)
        j = self.judge(vision=True)
        v, frames = j.video(pic) if kind == "video" else (j.image(pic), 1)
        idx, _, z = j.describe_picture(v, top)
        items = [
            {"text": j.vocab[i], "score": float(s), "category": category_of(j.vocab[i])}
            for i, s in zip(idx[:top], z[:top], strict=True)
        ]
        return {"picture": str(picture), "kind": kind, "frames": frames, "pictured_as": items}

    # -- libraries -------------------------------------------------------------------------
    def library(self, directory: str | Path, in_cache: bool = False):
        from .library import Library

        root = str(library_root(directory))
        key = (root, in_cache)
        with self._lock:
            if key not in self._libs:
                self._libs[key] = Library(root, in_cache=in_cache)
                self._lib_locks[key] = threading.Lock()
            return self._libs[key], self._lib_locks[key]

    def index(self, directory: str | Path, in_cache: bool = False, progress: Progress | None = None) -> dict:
        lib, lock = self.library(directory, in_cache)
        with lock:
            st = lib.update(lambda w: self.judge().embedder.embed_audio([w])[0], progress=progress)
        return index_record(lib, st)

    def search(
        self,
        directory: str | Path,
        query: str | None = None,
        like: str | Path | None = None,
        picture: str | Path | None = None,
        top: int = 10,
        in_cache: bool = False,
        refresh: bool = True,
        progress: Progress | None = None,
    ) -> dict:
        """Words (any language), a sound (``like``) or a picture (+ optional words about the scene)."""
        from . import library as libmod

        if like is not None and (query or picture is not None):
            raise InputError(
                "Search by words, by sound (like) or by picture — one at a time.", code="bad_value"
            )
        if like is None and picture is None and not query:
            raise InputError(
                "Nothing to search for: give words, a sound (like) or a picture.", code="bad_value"
            )
        like_p = probe_audio(like) if like is not None else None
        pic, kind = resolve_picture(None, picture) if picture is not None else (None, None)
        library_root(directory)
        failed = self.index(directory, in_cache, progress)["failed"] if refresh else []
        lib, lock = self.library(directory, in_cache)
        j = self.judge(vision=picture is not None)
        with lock:
            if like_p is not None:
                hits, mode = libmod.search_like(lib, j, like_p, top), "like"
            elif pic is not None:
                v = j.video(pic)[0] if kind == "video" else j.image(pic)
                hits, mode = libmod.search_picture(lib, j, v, query, top), "picture"
            else:
                hits, mode = libmod.search_text(lib, j, query, top), "words"
        return search_record(lib, mode, query, hits, like, picture, failed)

    def dedupe(
        self,
        directory: str | Path,
        match: float | None = None,
        in_cache: bool = False,
        refresh: bool = True,
        progress: Progress | None = None,
    ) -> dict:
        from .library import MATCH, find_duplicates

        if match is not None and not 0.5 <= match <= 1.0:
            raise InputError(f"match must be between 0.5 and 1.0 (got {match}).", code="bad_value")
        library_root(directory)
        failed = self.index(directory, in_cache)["failed"] if refresh else []
        lib, lock = self.library(directory, in_cache)
        with lock:
            groups = find_duplicates(lib, match_min=match if match is not None else MATCH, progress=progress)
        return dedupe_record(lib, groups, failed)


def library_root(directory: str | Path) -> Path:
    """A folder that is safe to scan as a library (not /, not your home folder itself, not a system root)."""
    from .library import check_root

    return check_root(Path(directory).expanduser())


def resolve_picture(
    file: str | Path | None, picture: str | Path | None, kind: str | None = None
) -> tuple[Path, str]:
    """The picture to judge against: ``picture`` (image or video), else ``file`` itself when it is a video."""
    import shutil

    from .visual import VIDEO_EXTS, has_video_stream

    if picture is None:
        if file is None or Path(file).suffix.lower() not in VIDEO_EXTS:
            raise InputError(
                "Give a picture (image or video), or pass a video file to judge its own sound.",
                code="bad_value",
            )
        picture, kind = file, "video"
    p = Path(picture).expanduser()
    shown = str(picture)
    if not p.exists():
        raise InputError(f"No such file: {shown}", code="not_found", file=shown)
    if p.is_dir():
        raise InputError(f"{shown} is a folder, not a file", code="is_dir", file=shown)
    if p.stat().st_size == 0:
        raise InputError(f"{shown} is empty (0 bytes)", code="empty", file=shown)
    kind = kind or picture_kind(p)
    if kind == "video":
        if shutil.which("ffprobe") is None:
            raise SetupError(
                "Reading video needs ffmpeg (ffprobe was not found on PATH).", code="needs_ffmpeg"
            )
        if not has_video_stream(p):
            raise InputError(
                f"{p.name} has no video track (or cannot be read)", code="unreadable", file=shown
            )
    else:
        try:
            from PIL import Image

            with Image.open(p) as im:
                im.verify()
        except Exception as e:
            raise InputError(
                f"{p.name} is not an image Sonicproof can read", code="unreadable", file=shown
            ) from e
    return p, kind


def _free_accelerator_memory() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
