"""``sp serve``: the local workbench — a FastAPI app on 127.0.0.1 plus the built web UI.

Nothing leaves the machine. The model is loaded once (lazily, on the first job) and shared by every request;
anything slow runs as a job on one worker thread and reports progress, which the UI polls.

Security model: the server binds to 127.0.0.1 only and answers only to a localhost ``Host`` (DNS rebinding).
Writes must come from the workbench's own origin, and ``/api/*`` refuses anything a browser marks as
cross-site or same-site (another site, or another local web app, linking to or embedding the API). Only media
files the user brought in can be read back, always with ``nosniff`` and a sandboxing CSP, so a file can
never run as a page of this origin. No CORS headers are ever sent, so other origins can't read responses.

A per-session secret token was considered and left out: with the checks above the only browser requests
that reach ``/api/*`` come from the workbench page itself; a token would have to ride in every
``<audio src>`` / ``<img src>`` URL and would break the Vite dev proxy, for no protection beyond very old
browsers that don't send ``Sec-Fetch-Site``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import mimetypes
import os
import queue
import re
import shutil
import socket
import sys
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__, cache, library, service
from .audio import read_sidecar, sidecar_prompt
from .errors import InputError, SetupError, SonicproofError
from .library import AUDIO_EXTS, INDEX_DIR, index_location
from .visual import IMAGE_EXTS, VIDEO_EXTS

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}
PICK_VERDICTS = ("strong", "plausible")
# The workbench keeps library indexes in the cache: opening a folder never writes into it.
IN_CACHE = True
# Cache folders that only make things faster: safe to delete at any time ("Clear cache").
CACHE_KINDS = {
    "embeddings": ("audio", "image", "video", "vocab", "fp"),
    "thumbnails": ("thumb_meta", "thumb_peaks", "thumb_spec"),
}
# Limits on what one request may bring in.
MAX_UPLOAD_FILES = 1000  # Starlette's own multipart limit
MAX_UPLOAD_BYTES = {"audio": 1 << 29, "video": 1 << 31, "image": 1 << 27, "sidecar": 1 << 20}
MAX_SCAN_FILES = 5000
MAX_SCAN_DIRS = 20_000
MAX_SCAN_DEPTH = 16
MAX_TOP = 500
LUFS_RANGE = (-70.0, 0.0)
SKIP_DIRS = library.SKIP_DIRS
WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
# Media responses: never sniffed into HTML, never run as a page.
MEDIA_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": "sandbox; default-src 'none'",
}
APP_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
    "media-src 'self' blob:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
)
STATUS_CODES = {400: "bad_request", 403: "forbidden", 404: "not_available", 405: "bad_request", 409: "busy"}


def _now() -> float:
    return time.time()


class ApiError(HTTPException):
    """An HTTP error with a stable ``code`` the UI translates; ``detail`` is the English fallback."""

    def __init__(self, status: int, code: str, detail: str):
        super().__init__(status, detail)
        self.code = code


def _finite(x: Any) -> Any:
    """Non-finite floats (NaN, ±inf) become None, so a response is always valid JSON."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    return x


# -- jobs -----------------------------------------------------------------------------------------
class JobCancelled(Exception):
    pass


@dataclass
class Job:
    id: str
    kind: str
    label: str = ""
    status: str = "queued"  # queued | running | done | error | cancelled
    done: int = 0
    total: int = 0
    result: Any = None
    error: str | None = None
    error_code: str | None = None
    cancel_requested: bool = False
    created: float = field(default_factory=_now)
    finished: float | None = None

    def view(self, with_result: bool = True) -> dict:
        d = asdict(self)
        if not with_result:
            d.pop("result")
        return _finite(d)

    def check(self):
        """Called from progress callbacks: stop here if the user cancelled."""
        if self.cancel_requested:
            raise JobCancelled()

    def progress(self, i: int, n: int, label: str):
        self.check()
        self.done, self.total, self.label = i, n, label


class Jobs:
    """One worker thread: model work is serial anyway, and order is predictable for the UI."""

    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._q: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        threading.Thread(target=self._work, daemon=True, name="sonicproof-jobs").start()

    def submit(self, kind: str, fn, label: str = "") -> Job:
        job = Job(id=uuid.uuid4().hex[:10], kind=kind, label=label)
        with self._lock:
            self._jobs[job.id] = job
            # keep the table small: forget finished jobs older than ten minutes
            old = [k for k, j in self._jobs.items() if j.finished and _now() - j.finished > 600]
            for k in old:
                del self._jobs[k]
        self._q.put((job, fn))
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def active(self) -> list[Job]:
        with self._lock:
            return [j for j in self._jobs.values() if j.status in ("queued", "running")]

    def cancel(self, job_id: str) -> Job | None:
        j = self.get(job_id)
        if j is not None and j.status in ("queued", "running"):
            j.cancel_requested = True
        return j

    def _work(self):
        while True:
            job, fn = self._q.get()
            if job.cancel_requested:
                job.status, job.error, job.error_code = "cancelled", "Cancelled.", "cancelled"
                job.finished = _now()
                continue
            job.status = "running"
            try:
                job.result = fn(job)
                job.status = "done"
            except JobCancelled:
                job.status, job.error, job.error_code = "cancelled", "Cancelled.", "cancelled"
            except SonicproofError as e:
                job.status, job.error, job.error_code = "error", e.message, e.code
            except Exception as e:  # report to the UI, keep the worker alive
                job.error = str(e) or e.__class__.__name__
                job.error_code = "internal"
                job.status = "error"
                traceback.print_exc()
            job.finished = _now()


# -- inbox ----------------------------------------------------------------------------------------
@dataclass
class Item:
    id: str
    name: str
    path: str
    digest: str
    prompt: str | None = None
    prompt_source: str | None = None  # "sidecar" | "user"
    seed: int | None = None
    loop: bool = False
    status: str = "queued"  # queued | working | done | error
    check: dict | None = None
    qc: dict | None = None
    heard_as: list | None = None
    suggest: list | None = None  # plain wordings of the prompt, offered when no take matches
    overall: str | None = None
    note: str | None = None
    error: str | None = None
    error_code: str | None = None  # errors.py code (empty, unreadable, …) the UI translates
    added: float = field(default_factory=_now)


def _group_key(prompt: str | None) -> str | None:
    return re.sub(r"\s+", " ", prompt.strip().lower()) if prompt and prompt.strip() else None


def _lufs(value: Any) -> float | None:
    """A loudness target: None, or a finite number of LUFS in range. Raises ValueError otherwise."""
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(value)
    t = float(value)
    if not math.isfinite(t) or not LUFS_RANGE[0] <= t <= LUFS_RANGE[1]:
        raise ValueError(value)
    return t


def _sidecar(f: Path) -> tuple[dict, str | None]:
    """The sidecar's metadata and prompt; a sidecar of the wrong shape counts as no sidecar."""
    try:
        meta = read_sidecar(f)
        meta = meta if isinstance(meta, dict) else {}
        return meta, (sidecar_prompt(f) if meta else None)
    except Exception:
        return {}, None


def _replace(src: str | Path, dst: Path):
    # Windows: an antivirus or indexer may hold the target for a moment
    for attempt in range(6):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * (attempt + 1))


class Workbench:
    def __init__(self, engine: service.Engine, home: Path | None = None):
        self.engine = engine
        self.jobs = Jobs()
        self.home = home or (cache.cache_dir() / "workbench")
        self.uploads = self.home / "uploads"
        self.uploads.mkdir(parents=True, exist_ok=True)
        self.state_file = self.home / "state.json"
        self.items: dict[str, Item] = {}
        self.libraries: list[str] = []  # recently opened, newest first
        self.pictures: set[str] = set()
        self.settings: dict = {"target_lufs": None}
        self.version = 0
        self._lock = threading.RLock()
        self._load()

    # persistence -------------------------------------------------------------------------
    def _load(self):
        try:
            st = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(st, dict):
            return
        for d in st.get("items", []) if isinstance(st.get("items"), list) else []:
            try:
                it = Item(**d)
            except TypeError:
                continue
            if Path(it.path).exists():
                if it.status in ("queued", "working"):
                    it.status = "queued"
                self.items[it.id] = it
        libs = st.get("libraries", [])
        self.libraries = (
            [p for p in libs if isinstance(p, str) and Path(p).is_dir()] if isinstance(libs, list) else []
        )
        settings = st.get("settings")
        if isinstance(settings, dict):
            try:
                self.settings["target_lufs"] = _lufs(settings.get("target_lufs"))
            except ValueError:
                pass  # a bad value from an older version: drop it
        if any(it.status == "queued" for it in self.items.values()):
            self.analyze([i for i, it in self.items.items() if it.status == "queued"])

    def save(self):
        # The whole write happens under the lock, through a uniquely named temp file: the analysis worker
        # and request threads save concurrently, and a reader must never see a half-written state.json.
        with self._lock:
            self.version += 1
            data = json.dumps(
                {
                    "items": [asdict(it) for it in self.items.values()],
                    "libraries": self.libraries,
                    "settings": self.settings,
                },
                ensure_ascii=False,
            )
            fd, tmp = tempfile.mkstemp(dir=self.home, prefix=".state-", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(data)
                _replace(tmp, self.state_file)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
                raise

    # access control: only media files the user brought in can be served ---------------------
    def allowed(self, path: Path) -> bool:
        try:
            p = path.resolve()
        except (OSError, ValueError):
            return False
        if _kind(p) == "other":
            return False  # never text, HTML, SVG, keys… even inside an opened library
        if p.is_relative_to(self.uploads.resolve()):
            return True
        with self._lock:
            if str(p) in self.pictures or any(Path(it.path).resolve() == p for it in self.items.values()):
                return True
            return any(p.is_relative_to(Path(root)) for root in self.libraries)

    # items -------------------------------------------------------------------------------
    def add_files(self, files: list[Path]) -> dict:
        added, skipped = [], 0
        with self._lock:
            known = {it.digest for it in self.items.values()}
            for f in files:
                if f.suffix.lower() not in AUDIO_EXTS or f.name.startswith("."):
                    continue
                try:
                    digest = cache.file_digest(f)
                except OSError:
                    continue
                if digest in known:
                    skipped += 1
                    continue
                known.add(digest)
                meta, prompt = _sidecar(f)
                seed = meta.get("seed") if isinstance(meta.get("seed"), int) else None
                it = Item(
                    id=uuid.uuid4().hex[:10],
                    name=f.name,
                    path=str(f.resolve()),
                    digest=digest,
                    prompt=prompt,
                    prompt_source="sidecar" if prompt else None,
                    seed=seed,
                    loop=bool(re.search(r"loop", f.stem, re.I)),
                )
                self.items[it.id] = it
                added.append(it.id)
        self.save()
        job = self.analyze(added) if added else None
        return {"added": len(added), "skipped": skipped, "job": job.id if job else None}

    def analyze(self, ids: list[str]) -> Job:
        for i in ids:
            if i in self.items:
                self.items[i].status = "queued"

        def run(job: Job):
            job.total = len(ids)
            for n, i in enumerate(ids):
                it = self.items.get(i)
                job.done = n
                if it is None:
                    continue
                if job.cancel_requested:  # leave the rest marked, so they don't wait forever
                    for rest in ids[n:]:
                        if (r := self.items.get(rest)) and r.status == "queued":
                            r.status, r.error, r.error_code = (
                                "error",
                                "Cancelled before it was checked.",
                                "cancelled",
                            )
                    with contextlib.suppress(Exception):
                        self.save()
                    raise JobCancelled()
                job.label = it.name
                it.status = "working"
                self.version += 1
                try:
                    self._analyze_one(it, job)
                    it.status, it.error, it.error_code = "done", None, None
                except SonicproofError as e:
                    it.status, it.error, it.error_code = "error", e.message, e.code
                except Exception as e:
                    it.status, it.error, it.error_code = "error", str(e) or e.__class__.__name__, None
                try:
                    self.save()
                except Exception:  # a failed save must not strand the rest of the queue
                    traceback.print_exc()
            job.done = job.total
            return {"items": ids}

        return self.jobs.submit("analyze", run, label="")

    def _analyze_one(self, it: Item, job: Job):
        it.qc = service.qc_record(it.path, it.loop, self.settings.get("target_lufs"))
        if self.engine.loaded is None:
            job.label = "@model"  # the UI shows "loading the model…"
        j = self.engine.judge()
        job.label = it.name
        if it.prompt:
            it.check = j.check(it.path, it.prompt, n_alt=5).to_dict()
            it.check["prompt_source"] = it.prompt_source or "argument"
            it.heard_as = it.check["heard_as"]
            it.suggest = j.phrasings(it.prompt, 2)
            it.overall, it.note = service.overall(it.check["verdict"], it.qc["status"])
        else:
            it.check = None
            it.heard_as = [asdict(a) for a in j.listen(it.path, 5)]
            it.suggest = None
            it.overall, it.note = None, None

    def update_item(self, item_id: str, body: dict) -> tuple[Item, Job]:
        it = self.items.get(item_id)
        if it is None:
            raise ApiError(404, "no_item", "No such item.")
        if "prompt" in body:
            raw = body["prompt"]
            if raw is not None and not isinstance(raw, str):
                raise ApiError(400, "bad_value", "prompt must be text.")
            p = (raw or "").strip()
            it.prompt, it.prompt_source = (p, "user") if p else (_sidecar(Path(it.path))[1], "sidecar")
            if not it.prompt:
                it.prompt_source = None
        if "loop" in body:
            it.loop = bool(body["loop"])
        self.save()
        return it, self.analyze([item_id])

    def remove_items(self, ids: list[str]) -> int:
        """Forget items; copies we made of them in uploads/ go too, unless another item still uses them."""
        with self._lock:
            gone = [self.items.pop(i) for i in ids if i in self.items]
        # state first: if a copy can't be deleted (Windows: open in another handle), the Inbox is still
        # consistent and "Clear cache" picks the leftover up later
        self.save()
        keep = self._uploads_in_use()
        uploads = self.uploads.resolve()
        for it in gone:
            p = Path(it.path).resolve()
            if not p.is_relative_to(uploads):
                continue  # the user's own file: never touched
            for f in (p, p.with_name(p.name + ".json"), p.with_suffix(".json")):
                if f not in keep and f.is_file():
                    try:
                        f.unlink()
                    except OSError as e:
                        print(
                            f"sonicproof: could not delete {f.name} now ({e.__class__.__name__})",
                            file=sys.stderr,
                        )
            self._prune(p.parent)
        return len(gone)

    def _prune(self, d: Path):
        """Remove empty folders from ``d`` up to (not including) uploads/."""
        root = self.uploads.resolve()
        d = d.resolve()
        while d != root and d.is_relative_to(root):
            try:
                d.rmdir()
            except OSError:
                return
            d = d.parent

    # cache --------------------------------------------------------------------------------
    def _uploads_in_use(self) -> set[Path]:
        keep = set()
        with self._lock:
            paths = [it.path for it in self.items.values()] + list(self.pictures)
        for p in paths:
            q = Path(p).resolve()
            keep |= {q, q.with_name(q.name + ".json"), q.with_suffix(".json")}
        return keep

    def cache_usage(self) -> dict:
        root = cache.cache_dir()
        kinds = {k: sum(_size(root / d) for d in dirs) for k, dirs in CACHE_KINDS.items()}
        keep_idx = {index_location(Path(r), True).resolve() for r in self.libraries}
        idx = [f for f in (root / "indexes").glob("*.npz")] if (root / "indexes").is_dir() else []
        keep = self._uploads_in_use()
        up = [f for f in self.uploads.rglob("*") if f.is_file()]
        return {
            "path": str(root),
            "embeddings": kinds["embeddings"],
            "thumbnails": kinds["thumbnails"],
            "indexes": sum(_size(f) for f in idx),
            "indexes_unused": sum(_size(f) for f in idx if f.resolve() not in keep_idx),
            "uploads": sum(_size(f) for f in up),
            "uploads_unused": sum(_size(f) for f in up if f.resolve() not in keep),
            "total": _size(root),
        }

    def clear_cache(self) -> dict:
        """Delete what only speeds things up: embeddings, thumbnails, indexes of libraries no longer in the
        recent list and uploaded copies nothing uses. Inbox items and recent libraries keep working."""
        before = _size(cache.cache_dir())
        root = cache.cache_dir()
        for dirs in CACHE_KINDS.values():
            for d in dirs:
                shutil.rmtree(root / d, ignore_errors=True)
        keep_idx = {index_location(Path(r), True).resolve() for r in self.libraries}
        for f in (root / "indexes").glob("*.npz") if (root / "indexes").is_dir() else []:
            if f.resolve() not in keep_idx:
                with contextlib.suppress(OSError):
                    f.unlink()
        keep = self._uploads_in_use()
        for f in sorted(self.uploads.rglob("*"), key=lambda x: len(x.parts), reverse=True):
            if f.is_file() and f.resolve() not in keep:
                with contextlib.suppress(OSError):  # still open somewhere (Windows): next time
                    f.unlink()
            elif f.is_dir():
                self._prune(f)
        return {"freed": max(0, before - _size(cache.cache_dir())), "usage": self.cache_usage()}

    def inbox(self) -> dict:
        with self._lock:
            items = sorted(list(self.items.values()), key=lambda it: it.added)
            groups: dict[str, list[Item]] = {}
            order: list[str] = []
            for it in items:
                key = _group_key(it.prompt) or f"#{it.id}"
                if key not in groups:
                    groups[key] = []
                    order.append(key)
                groups[key].append(it)
            out = []
            for key in order:
                members = groups[key]
                scored = [m for m in members if m.check]
                scored.sort(key=lambda m: _finite(m.check.get("score")) or -1e9, reverse=True)
                pick = None
                if len(members) > 1 and scored:
                    # Only a take whose current analysis finished, matches well and has no failing qc
                    # check; no fallback, so a group of broken takes never gets "Use this one".
                    pick = next(
                        (
                            m
                            for m in scored
                            if m.status == "done"
                            and m.check.get("verdict") in PICK_VERDICTS
                            and m.qc
                            and m.qc.get("status") != "fail"
                        ),
                        None,
                    )
                ordered = scored + [m for m in members if not m.check]
                out.append(
                    {
                        "key": key,
                        "prompt": members[0].prompt,
                        "pick": pick.id if pick else None,
                        "settled": all(m.status in ("done", "error") for m in members),
                        "suggest": next((m.suggest for m in members if m.suggest), None),
                        "items": [asdict(m) for m in ordered],
                    }
                )
            return _finite(
                {"version": self.version, "groups": out, "count": len(items), "settings": self.settings}
            )

    # libraries -------------------------------------------------------------------------
    def remember_library(self, root: str):
        with self._lock:
            if root in self.libraries:
                self.libraries.remove(root)
            self.libraries.insert(0, root)
            del self.libraries[8:]
        self.save()

    def library_view(self, root: str) -> dict:
        lib, _ = self.engine.library(root, IN_CACHE)
        j = self.engine.judge()
        heard = j.heard_as_many(lib.emb)
        entries = []
        for e, h in zip(lib.entries, heard, strict=True):
            d = e.to_dict()
            for k in ("mtime_ns", "digest"):
                d.pop(k)
            d["heard_as"] = h
            entries.append(d)
        return {
            "library": str(lib.root),
            "files": len(lib),
            "duration": lib.total_duration(),
            "entries": entries,
        }


# -- helpers --------------------------------------------------------------------------------------
def _dist() -> Path | None:
    d = Path(str(resources.files("sonicproof") / "web"))
    return d if (d / "index.html").exists() else None


def _kind(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in IMAGE_EXTS:
        return "image"
    return "other"


def _safe_part(part: str) -> str | None:
    """One path component of an uploaded name, made safe on every OS (None = drop it)."""
    part = re.sub(r"[\x00-\x1f\x7f]", "", part).strip()
    if part in ("", ".", "..") or re.fullmatch(r"[A-Za-z]:", part):
        return None  # a drive ("C:")
    part = part.replace(":", "_")  # "D:x.wav" is drive-relative on Windows, "a:b" an NTFS stream
    part = part.rstrip(" .")  # Windows silently strips these
    if not part:
        return None
    if part.split(".")[0].upper() in WINDOWS_RESERVED:
        part = "_" + part
    raw = part.encode("utf-8")
    if len(raw) > 200:  # most file systems stop at 255 bytes
        stem, dot, ext = part.rpartition(".")
        ext = dot + ext if dot and len(ext) <= 10 else ""
        part = (stem if ext else part).encode("utf-8")[: 200 - len(ext.encode())].decode(
            "utf-8", "ignore"
        ) + ext
    return part


def _safe_rel(name: str) -> Path:
    parts = [q for p in re.split(r"[\\/]+", name or "") if (q := _safe_part(p))]
    return Path(*parts) if parts else Path("file")


def _upload_kind(name: str) -> str | None:
    if name.lower().endswith(".json"):
        return "sidecar"
    k = _kind(Path(name))
    return None if k == "other" else k


def _check_root(raw: Any) -> str:
    """A folder the user may open as a library (not ``/``, not the home folder itself, not a system root)."""
    if not isinstance(raw, str) or not raw.strip():
        raise ApiError(400, "not_folder", "Choose a folder.")
    try:
        p = Path(raw).expanduser()
        if not p.is_dir():
            raise ApiError(400, "not_folder", f"Not a folder: {raw}")
        root = library.check_root(p)
    except InputError as e:
        raise ApiError(400, "library_root", e.message) from e
    except (OSError, ValueError) as e:
        raise ApiError(400, "not_folder", f"Not a folder: {raw}") from e
    return str(Path(root).resolve())


def _scan_inbox(root: Path) -> list[Path]:
    """Audio files under a folder dropped on the Inbox, with limits: hidden and system folders are skipped,
    and a folder with too many files is refused instead of hashing the whole disk."""
    home_lib = (Path.home() / "Library").resolve()
    out: list[Path] = []
    n_dirs = 0
    base = len(root.parts)
    for d, dirs, files in os.walk(root):
        n_dirs += 1
        here = Path(d)
        keep = []
        for x in sorted(dirs):
            sub = here / x
            if (
                x.startswith(".")
                or x in SKIP_DIRS
                or sub == home_lib
                or len(sub.parts) - base > MAX_SCAN_DEPTH
            ):
                continue
            keep.append(x)
        dirs[:] = keep
        out += [here / f for f in sorted(files) if Path(f).suffix.lower() in AUDIO_EXTS and f[0] != "."]
        if len(out) > MAX_SCAN_FILES or n_dirs > MAX_SCAN_DIRS:
            raise ApiError(
                413,
                "too_many_files",
                f"{root.name or root} holds too many files to add at once (limit {MAX_SCAN_FILES}); "
                "add a smaller folder, or open it as a library.",
            )
    return out


def _int(value: Any, default: int, lo: int = 1, hi: int = MAX_TOP, name: str = "top") -> int:
    if value is None or value == "":
        return default
    try:
        if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
            raise ValueError(value)
        n = int(value)
    except (TypeError, ValueError):
        raise ApiError(400, "bad_value", f"{name} must be a whole number from {lo} to {hi}.") from None
    if not lo <= n <= hi:
        raise ApiError(400, "bad_value", f"{name} must be a whole number from {lo} to {hi}.")
    return n


def _str_list(value: Any, name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ApiError(400, "bad_value", f"{name} must be a list of paths.")
    return value


def _parse_host(host: str) -> tuple[str, str] | None:
    """``localhost:7788`` → ("localhost", "7788"); anything that isn't a plain host[:port] → None."""
    m = re.fullmatch(r"(\[[0-9a-fA-F:.]+\]|[A-Za-z0-9.\-]+)(?::(\d{1,5}))?", host or "")
    return (m.group(1).lower(), m.group(2) or "") if m else None


def _same_origin(origin: str, host: str) -> bool:
    try:
        u = urlsplit(origin)
    except ValueError:
        return False
    return u.scheme == "http" and not u.path.strip("/") and u.netloc.lower() == host.lower()


# -- app ----------------------------------------------------------------------------------------
def create_app(engine: service.Engine | None = None, home: Path | None = None) -> FastAPI:
    wb = Workbench(engine or service.Engine(), home)
    app = FastAPI(title="Sonicproof", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.wb = wb

    def refuse(detail: str, code: str = "forbidden") -> JSONResponse:
        return JSONResponse({"detail": detail, "code": code}, status_code=403)

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        # The server binds to 127.0.0.1; also refuse other Host names (DNS rebinding), writes from any other
        # origin (other sites and other local web apps), and API requests a browser marks as coming from
        # another site (links, <img>/<audio> embeds): a web page can neither drive nor probe the workbench.
        host = request.headers.get("host") or ""
        parsed = _parse_host(host)
        if parsed is None or parsed[0] not in LOCAL_HOSTS:
            return refuse("Sonicproof only answers on localhost.")
        origin = request.headers.get("origin")
        if request.method not in ("GET", "HEAD") and origin and not _same_origin(origin, host):
            return refuse("Cross-site request refused.", "cross_site")
        if request.url.path.startswith("/api/"):
            if request.headers.get("sec-fetch-site", "").lower() in ("cross-site", "same-site"):
                return refuse("Cross-site request refused.", "cross_site")
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        return response

    # errors: always JSON {"detail", "code"} ---------------------------------------------------------
    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        code = getattr(exc, "code", None) or STATUS_CODES.get(exc.status_code, "error")
        detail = exc.detail if isinstance(exc.detail, str) else "Request failed."
        return JSONResponse(
            {"detail": detail, "code": code}, status_code=exc.status_code, headers=exc.headers
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(x) for x in first.get("loc", ()) if x != "body")
        msg = f"Invalid request: {where + ' — ' if where else ''}{first.get('msg', 'bad value')}"
        return JSONResponse({"detail": msg, "code": "bad_request"}, status_code=422)

    @app.exception_handler(SonicproofError)
    async def sonicproof_error(request: Request, exc: SonicproofError):
        status = 503 if isinstance(exc, SetupError) else 400
        return JSONResponse({"detail": exc.message, "code": exc.code}, status_code=status)

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception):
        return JSONResponse(
            {"detail": f"Internal error ({exc.__class__.__name__}); see the terminal.", "code": "internal"},
            status_code=500,
        )

    def job_response(job: Job) -> dict:
        return {"job": job.id}

    # status -------------------------------------------------------------------------------
    @app.get("/api/status")
    def status():
        from .model import MODEL_ID, model_cached

        return {
            "version": __version__,
            "model": MODEL_ID,
            "model_cached": model_cached(),
            "loaded": wb.engine.loaded,
            "jobs": [j.view(False) for j in wb.jobs.active()],
            "state": wb.version,
            "libraries": list(wb.libraries),
        }

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        j = wb.jobs.get(job_id)
        if j is None:
            raise ApiError(404, "no_job", "No such job.")
        return j.view()

    @app.post("/api/jobs/{job_id}/cancel")
    def job_cancel(job_id: str):
        j = wb.jobs.cancel(job_id)
        if j is None:
            raise ApiError(404, "no_job", "No such job.")
        return j.view(False)

    # inbox --------------------------------------------------------------------------------
    @app.get("/api/inbox")
    def inbox():
        return wb.inbox()

    @app.post("/api/inbox/upload")
    async def inbox_upload(files: list[UploadFile] = File(...), paths: list[str] = Form(default=[])):
        saved = await _save_uploads(wb, files, paths)
        return wb.add_files([p for p in saved if _kind(p) == "audio"])

    @app.post("/api/inbox/paths")
    def inbox_paths(body: dict):
        files: list[Path] = []
        for raw in _str_list(body.get("paths"), "paths"):
            try:
                p = Path(raw).expanduser()
                is_dir, is_file = p.is_dir(), p.is_file()
            except (OSError, ValueError):
                is_dir = is_file = False
            if is_dir:
                files += _scan_inbox(Path(_check_root(raw)))
            elif is_file:
                files.append(p)
            else:
                raise ApiError(400, "not_found", f"Not found: {raw}")
            if len(files) > MAX_SCAN_FILES:
                raise ApiError(
                    413, "too_many_files", f"Too many files to add at once (limit {MAX_SCAN_FILES})."
                )
        return wb.add_files(files)

    @app.patch("/api/inbox/{item_id}")
    def inbox_update(item_id: str, body: dict):
        it, job = wb.update_item(item_id, body)
        return {"item": _finite(asdict(it)), "job": job.id}

    @app.post("/api/inbox/reanalyze")
    def inbox_reanalyze(body: dict | None = None):
        ids = _str_list((body or {}).get("ids"), "ids") or list(wb.items)
        return job_response(wb.analyze(ids))

    @app.delete("/api/inbox/{item_id}")
    def inbox_remove(item_id: str):
        wb.remove_items([item_id])
        return {"ok": True}

    @app.delete("/api/inbox")
    def inbox_clear():
        wb.remove_items(list(wb.items))
        return {"ok": True}

    @app.get("/api/cache")
    def cache_usage():
        return wb.cache_usage()

    @app.post("/api/cache/clear")
    def cache_clear():
        if wb.jobs.active():
            raise ApiError(409, "busy", "Busy — try again when the current task has finished.")
        return wb.clear_cache()

    @app.post("/api/settings")
    def settings(body: dict):
        if "target_lufs" in body:
            try:
                t = _lufs(body["target_lufs"])
            except ValueError:
                raise ApiError(
                    400,
                    "bad_value",
                    f"target_lufs must be a number from {LUFS_RANGE[0]:g} to {LUFS_RANGE[1]:g}.",
                ) from None
            wb.settings["target_lufs"] = t
        wb.save()
        return wb.settings

    # files ----------------------------------------------------------------------------------
    @app.post("/api/upload")
    async def upload(files: list[UploadFile] = File(...), paths: list[str] = Form(default=[])):
        saved = await _save_uploads(wb, files, paths)
        out = []
        for p in saved:
            kind = _kind(p)
            if kind in ("image", "video"):
                if not _picture_ok(p, kind):
                    with contextlib.suppress(OSError):
                        p.unlink()
                    out.append({"path": None, "name": p.name, "kind": "broken", "error_code": "unreadable"})
                    continue
                with wb._lock:
                    wb.pictures.add(str(p.resolve()))
            out.append({"path": str(p), "name": p.name, "kind": kind})
        return {"files": out}

    @app.get("/api/media")
    def media(path: str):
        p = Path(path)
        if not p.is_file() or not wb.allowed(p):
            raise ApiError(404, "not_available", "Not available.")
        mt = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        if not mt.startswith(("audio/", "video/", "image/")) or mt == "image/svg+xml":
            mt = "application/octet-stream"
        return FileResponse(p, media_type=mt, headers=MEDIA_HEADERS)

    @app.get("/api/thumb")
    def thumb(path: str):
        from .thumbs import thumbnail

        p = Path(path)
        if not p.is_file() or not wb.allowed(p):
            raise ApiError(404, "not_available", "Not available.")
        try:
            data = thumbnail(p)
        except SonicproofError as e:
            raise ApiError(422, e.code, e.message) from e
        except Exception as e:
            raise ApiError(422, "unreadable", f"Could not read {p.name}.") from e
        return JSONResponse(_finite(data), headers=MEDIA_HEADERS)

    @app.get("/api/fs")
    def fs(path: str | None = None):
        try:
            base = Path(path).expanduser() if path else Path.home()
            ok = base.is_dir()
        except (OSError, ValueError):
            ok = False
        if not ok:
            raise ApiError(400, "not_folder", f"Not a folder: {path}")
        base = base.resolve()
        dirs, audio = [], 0
        try:
            for c in sorted(base.iterdir(), key=lambda x: x.name.lower()):
                if c.name.startswith("."):
                    continue
                if c.is_dir():
                    dirs.append({"name": c.name, "path": str(c)})
                elif c.suffix.lower() in AUDIO_EXTS:
                    audio += 1
        except OSError:
            pass
        places = []
        for name in ("Desktop", "Downloads", "Music", "Documents"):
            p = Path.home() / name
            if p.is_dir():
                places.append({"name": name, "path": str(p)})
        return {
            "path": str(base),
            "parent": str(base.parent) if base.parent != base else None,
            "dirs": dirs,
            "audio_files": audio,
            "indexed": (base / INDEX_DIR).is_dir() or index_location(base, IN_CACHE).exists(),
            "home": str(Path.home()),
            "places": places,
            "recent": list(wb.libraries),
        }

    # library --------------------------------------------------------------------------------
    def _available(raw: Any, what: str = "That file") -> Path:
        if not isinstance(raw, str) or not raw:
            raise ApiError(400, "not_available", f"{what} is not available.")
        p = Path(raw)
        if not (p.is_file() and wb.allowed(p)):
            raise ApiError(400, "not_available", f"{what} is not available.")
        return p

    @app.post("/api/library/open")
    def library_open(body: dict):
        root = _check_root(body.get("path"))
        wb.remember_library(root)

        def run(job: Job):
            job.label = "@scan"
            st = wb.engine.index(root, IN_CACHE, progress=job.progress)
            job.check()
            job.label = "@model" if wb.engine.loaded is None else ""
            view = wb.library_view(root)
            view["update"] = st
            return view

        return job_response(wb.jobs.submit("index", run))

    @app.post("/api/library/search")
    def library_search(body: dict):
        root = _check_root(body.get("path"))
        raw_q = body.get("query")
        if raw_q is not None and not isinstance(raw_q, str):
            raise ApiError(400, "bad_value", "query must be text.")
        query = (raw_q or "").strip() or None
        like = str(_available(body["like"])) if body.get("like") else None
        picture = str(_available(body["picture"], "That picture")) if body.get("picture") else None
        top = _int(body.get("top"), 30)

        def run(job: Job):
            if picture and wb.engine.loaded != "vision":
                job.label = "@vision"
            return wb.engine.search(
                root,
                query=query,
                like=like,
                picture=picture,
                top=top,
                in_cache=IN_CACHE,
                progress=job.progress,
            )

        return job_response(wb.jobs.submit("search", run))

    @app.post("/api/library/dedupe")
    def library_dedupe(body: dict):
        root = _check_root(body.get("path"))

        def run(job: Job):
            return wb.engine.dedupe(root, in_cache=IN_CACHE, progress=job.progress)

        return job_response(wb.jobs.submit("dedupe", run))

    # picture --------------------------------------------------------------------------------
    @app.post("/api/picture/describe")
    def picture_describe(body: dict):
        pic = _available(body.get("picture"), "That picture")
        top = _int(body.get("top"), 8)

        def run(job: Job):
            if wb.engine.loaded != "vision":
                job.label = "@vision"
            return wb.engine.describe_picture(pic, top=top)

        return job_response(wb.jobs.submit("describe", run))

    @app.post("/api/picture/fit")
    def picture_fit(body: dict):
        pic = _available(body.get("picture"), "That picture")
        raw_scene = body.get("scene")
        scene = (raw_scene.strip() if isinstance(raw_scene, str) else "") or None
        own = bool(body.get("own"))
        sounds = [_available(p, Path(p).name) for p in _str_list(body.get("files"), "files")]
        sounds += [Path(wb.items[i].path) for i in _str_list(body.get("items"), "items") if i in wb.items]
        if own:
            sounds = [pic]
        if not sounds:
            raise ApiError(400, "no_sounds", "Add some sounds to the Inbox first, or pick a library.")

        def run(job: Job):
            if wb.engine.loaded != "vision":
                job.label = "@vision"
            job.total = len(sounds)
            out = []
            for n, s in enumerate(sounds):
                job.check()
                job.done = n
                r = wb.engine.fit(s, pic, scene=scene, top=5)
                r["qc"] = service.qc_record(s)["status"] if _kind(s) == "audio" else None
                out.append(r)
                job.label = s.name
            job.done = job.total
            out.sort(key=lambda r: r["score"], reverse=True)
            return {"picture": str(pic), "scene": scene, "results": out}

        return job_response(wb.jobs.submit("fit", run))

    # web UI -----------------------------------------------------------------------------------
    @app.get("/{rest:path}")
    def spa(rest: str):
        if rest.startswith("api/"):
            raise ApiError(404, "not_available", "Not found.")
        dist = _dist()
        if dist is None:
            return HTMLResponse(
                "<h1>Sonicproof</h1><p>The web UI is not built. "
                "Run <code>npm install && npm run build</code> in <code>web/</code>.</p>",
                status_code=503,
            )
        f = (dist / rest).resolve() if rest else dist / "index.html"
        if rest and f.is_file() and f.is_relative_to(dist.resolve()):
            return FileResponse(f, headers={"Content-Security-Policy": APP_CSP})
        return Response(
            (dist / "index.html").read_bytes(),
            media_type="text/html",
            headers={"Content-Security-Policy": APP_CSP, "Cache-Control": "no-cache"},
        )

    return app


def _size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    total = 0
    for d, _, files in os.walk(p):
        for f in files:
            try:
                total += os.stat(os.path.join(d, f)).st_size
            except OSError:
                pass
    return total


def _picture_ok(p: Path, kind: str) -> bool:
    """Can we read this picture? Checked on upload, so the UI can say so before anything runs."""
    if kind == "image":
        try:
            from PIL import Image

            with Image.open(p) as im:
                im.verify()
            return True
        except Exception:
            return False
    if shutil.which("ffprobe") is None:
        return True  # can't tell; the job reports it if it really is broken
    from .visual import has_video_stream

    return has_video_stream(p)


async def _save_uploads(wb: Workbench, files: list[UploadFile], paths: list[str]) -> list[Path]:
    """Keep the dropped folder structure (sidecars must sit next to their audio). Only media and .json
    sidecars are kept; each stays inside its batch folder and under a size limit."""
    if len(files) > MAX_UPLOAD_FILES:
        raise ApiError(413, "too_many_files", f"Too many files at once (limit {MAX_UPLOAD_FILES}).")
    batch = wb.uploads / hashlib.sha1(f"{_now()}{uuid.uuid4()}".encode()).hexdigest()[:12]
    base = batch.resolve()
    saved: list[Path] = []
    try:
        for i, f in enumerate(files):
            rel = _safe_rel(paths[i] if i < len(paths) and paths[i] else (f.filename or "file"))
            kind = _upload_kind(rel.name)
            if kind is None:
                continue  # notes, images of other kinds, …: not ours to keep
            limit = MAX_UPLOAD_BYTES[kind]
            if f.size is not None and f.size > limit:
                raise ApiError(413, "too_large", f"{rel.name} is too large (limit {limit >> 20} MB).")
            dest = batch / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.resolve().is_relative_to(base):
                raise ApiError(400, "bad_value", f"Unsafe file name: {rel.name}")
            n = 0
            with open(dest, "wb") as out:
                saved.append(dest)
                while chunk := await f.read(1 << 20):
                    n += len(chunk)
                    if n > limit:
                        raise ApiError(413, "too_large", f"{rel.name} is too large (limit {limit >> 20} MB).")
                    out.write(chunk)
    except BaseException:
        for p in saved:
            with contextlib.suppress(OSError):
                p.unlink()
        shutil.rmtree(batch, ignore_errors=True)
        raise
    return saved


def _bind(host: str, port: int) -> socket.socket:
    """Bind the port before anything is printed, so a busy port is one clear line, not a uvicorn trace."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    if os.name != "nt":  # on Windows SO_REUSEADDR would let us share a port someone else is using
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError as e:
        sock.close()
        raise SetupError(
            f"Port {port} is busy — is Sonicproof already running? Try --port {port + 1}.", code="port_busy"
        ) from e
    sock.set_inheritable(True)
    return sock


def serve(host: str = "127.0.0.1", port: int = 7788, open_browser: bool = True) -> None:
    import uvicorn

    sock = _bind(host, port)
    url = f"http://{'localhost' if host in ('127.0.0.1', '::1') else host}:{port}/"
    try:
        app = create_app()
        server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
        if open_browser:
            import webbrowser

            threading.Timer(1.0, lambda: webbrowser.open(url)).start()
        print(f"Sonicproof workbench on {url}  (Ctrl+C to stop)", flush=True)
        server.run(sockets=[sock])
    finally:
        sock.close()
