"""Local sound library: an incremental vector index, search by words / sound / picture, and dedupe.

The index lives in ``<library>/.sonicproof/index.npz`` (or in the cache when the folder is read-only or
``in_cache=True``). Rows are reused while a file's size and modification time are unchanged; changed or
new files are looked up in the embedding cache by content hash before anything is re-embedded.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import stft

from . import __version__, cache
from .audio import MODEL_SR, load, load_for_model, resample
from .errors import InputError, SetupError
from .judge import SILENT_NOTE, is_silent_level, silence_level
from .model import MODEL_ID

AUDIO_EXTS = {
    ".wav",
    ".wave",
    ".flac",
    ".aif",
    ".aiff",
    ".caf",
    ".w64",
    ".mp3",
    ".ogg",
    ".oga",
    ".opus",
    ".m4a",
    ".aac",
    ".wma",
}
LOSSLESS_EXTS = {".wav", ".wave", ".flac", ".aif", ".aiff", ".caf", ".w64"}
INDEX_DIR = ".sonicproof"
INDEX_NAME = "index.npz"
INDEX_SCHEMA = 2  # bump when Entry or the npz layout changes in a way old readers can't handle
DIM = 768

# Folders that are never a sound library: scanning them would walk the whole disk or the OS.
_UNIX_ROOTS = (
    "/", "/System", "/usr", "/bin", "/sbin", "/etc", "/var", "/private", "/private/var", "/private/etc",
    "/Library", "/Users", "/home", "/root", "/dev", "/proc", "/sys", "/boot", "/opt", "/tmp", "/private/tmp",
    "/Volumes", "/mnt", "/media",
)  # fmt: skip
_WINDOWS_ROOTS = ("Windows", "Program Files", "Program Files (x86)", "ProgramData", "Users")

Progress = Callable[[int, int, str], None]


# -- index ------------------------------------------------------------------------------------
@dataclass
class Entry:
    path: str  # relative to the library root, POSIX separators
    size: int
    mtime_ns: int
    digest: str
    duration: float
    sample_rate: int
    channels: int
    lossless: bool
    lufs: float | None
    peak_db: float
    active: float | None = None  # seconds above -60 dBFS (None in indexes written before it existed)

    @property
    def silent(self) -> bool:
        """Silent or nearly empty — same rule as qc's silence check."""
        return is_silent_level(self.peak_db, self.active)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class UpdateStats:
    files: int = 0
    added: int = 0
    changed: int = 0
    removed: int = 0
    unchanged: int = 0
    embedded: int = 0  # had to run the model (the rest came from the embedding cache)
    failed: list[str] = field(default_factory=list)


# Folder names never walked (system data, cloud placeholders, package trees).
SKIP_DIRS = {"Library", "Mobile Documents", "node_modules", "$RECYCLE.BIN", "System Volume Information"}


def scan(root: Path) -> list[Path]:
    """Audio files under ``root``, skipping hidden folders (including our own index folder), system folders,
    and symlinks that point outside ``root``."""
    real = root.resolve()
    out = []
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if not x.startswith(".") and x not in SKIP_DIRS)
        for f in sorted(files):
            p = Path(d) / f
            if p.suffix.lower() not in AUDIO_EXTS or f[0] == ".":
                continue
            if p.is_symlink() and not p.resolve().is_relative_to(real):
                continue
            out.append(p)
    return out


def check_root(path: str | Path) -> Path:
    """Validate a library folder: expanded, resolved, an existing directory and not a system root.

    Raises ``InputError`` (``not_found`` / ``bad_value``) with a one-line message naming ``path``.
    """
    shown = str(path)
    raw = Path(path).expanduser()
    root = raw.resolve()
    if not root.exists():
        raise InputError(f"Folder not found: {shown}", code="not_found", file=shown)
    if not root.is_dir():
        raise InputError(
            f"{shown} is a file — give the folder that holds your sounds.", code="bad_value", file=shown
        )
    if _is_system_root(raw.absolute()) or _is_system_root(root):
        raise InputError(
            f"{shown} is too broad to use as a sound library (a disk, system or home folder) — "
            "pick the folder that holds your sounds.",
            code="bad_value",
            file=shown,
        )
    return root


def _is_system_root(p: Path) -> bool:
    if p.parent == p:  # "/" or a Windows drive / share root ("C:\\")
        return True
    try:
        if os.path.ismount(p):  # a whole disk: /Volumes/<disk>, /mnt/usb, /media/<user>/<disk>
            return True
    except OSError:
        pass
    try:
        if p == Path.home().resolve() or p == Path.home():
            return True
    except RuntimeError:  # no home directory
        pass
    if sys.platform == "win32":
        anchor = Path(p.anchor)
        if any(p == anchor / name for name in _WINDOWS_ROOTS):
            return True
        windir = os.environ.get("SystemRoot") or os.environ.get("WINDIR")
        return bool(windir) and p == Path(windir)
    return any(p == Path(r) for r in _UNIX_ROOTS)


def _cache_index_path(root: Path) -> Path:
    key = hashlib.sha1(str(root.resolve()).encode()).hexdigest()[:16]
    return cache.cache_dir() / "indexes" / f"{key}.npz"


def index_location(root: Path, in_cache: bool = False) -> Path:
    if not in_cache and os.access(root, os.W_OK):
        return root / INDEX_DIR / INDEX_NAME
    return _cache_index_path(root)


class Library:
    def __init__(self, root: str | Path, in_cache: bool = False):
        self.root = check_root(root)
        self.location = index_location(self.root, in_cache)
        self.note: str | None = None  # set when the index had to move to the cache
        self.entries: list[Entry] = []
        self.emb = np.zeros((0, DIM), dtype=np.float32)
        # Prefer the folder's own index, else the cache copy. With in_cache the index is always saved to
        # the cache, but an existing folder index still seeds it (nothing is written into the folder).
        cands = [self.root / INDEX_DIR / INDEX_NAME, _cache_index_path(self.root)]
        if in_cache:
            cands.reverse()
        for cand in cands:
            if cand.exists() and self._read(cand):
                if not in_cache:
                    self.location = cand
                break

    def __len__(self) -> int:
        return len(self.entries)

    def _read(self, p: Path) -> bool:
        """Load an index if it is intact and was built with this model; anything else means rebuild."""
        try:
            with np.load(p, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"]))
                emb = z["emb"].astype(np.float32)
            if meta.get("model") != MODEL_ID or int(meta.get("schema", 1)) > INDEX_SCHEMA:
                return False  # another model, or written by a newer Sonicproof: start over
            entries = [_entry(e) for e in meta["entries"]]
        except Exception:  # BadZipFile (truncated), EOFError, TypeError / KeyError (fields), garbage …
            return False
        if emb.ndim != 2 or emb.shape[1] != DIM or len(entries) != len(emb):
            return False
        self.entries, self.emb = entries, emb
        return True

    def save(self) -> None:
        """Write the index; a folder that turns out to be read-only falls back to the cache."""
        try:
            self._write(self.location)
        except OSError:
            fallback = _cache_index_path(self.root)
            if self.location == fallback:
                raise
            self._write(fallback)
            self.location = fallback
            self.note = f"{self.root.name} is read-only; the index is kept in the Sonicproof cache."

    def _write(self, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        _hide(dest.parent)
        meta = {
            "model": MODEL_ID,
            "schema": INDEX_SCHEMA,
            "sonicproof": __version__,
            "root": str(self.root),
            "entries": [e.to_dict() for e in self.entries],
        }
        tmp = dest.with_name(f"{dest.stem}.{os.getpid()}.{threading.get_ident()}.tmp.npz")
        try:
            np.savez(
                tmp, emb=self.emb.astype(np.float32), meta=np.array(json.dumps(meta, ensure_ascii=False))
            )
            tmp.replace(dest)
        finally:
            tmp.unlink(missing_ok=True)

    def update(
        self, embed: Callable[[np.ndarray], np.ndarray], progress: Progress | None = None
    ) -> UpdateStats:
        """Bring the index in line with the folder. ``embed`` maps a mono 16 kHz wave to a vector."""
        files = scan(self.root)
        old = {e.path: (e, self.emb[i]) for i, e in enumerate(self.entries)}
        st = UpdateStats(files=len(files))
        entries, vecs = [], []
        for n, f in enumerate(files):
            rel = f.relative_to(self.root).as_posix()
            if progress:
                progress(n, len(files), rel)
            try:
                s = f.stat()
            except OSError:
                continue
            prev = old.pop(rel, None)
            if prev and prev[0].size == s.st_size and prev[0].mtime_ns == s.st_mtime_ns:
                entries.append(prev[0])
                vecs.append(prev[1])
                st.unchanged += 1
                continue
            try:
                e, v, fresh = _describe(f, rel, s, embed)
            except (ImportError, SetupError):
                raise  # the machine can't embed at all: not this file's fault, and don't save an empty index
            except Exception as ex:  # unreadable / corrupt file: report and keep going
                st.failed.append(f"{rel}: {getattr(ex, 'message', None) or ex}")
                continue
            entries.append(e)
            vecs.append(v)
            st.embedded += fresh
            if prev:
                st.changed += 1
            else:
                st.added += 1
        st.removed = len(old)
        if progress:
            progress(len(files), len(files), "")
        self.entries = entries
        self.emb = np.stack(vecs).astype(np.float32) if vecs else np.zeros((0, DIM), dtype=np.float32)
        self.save()
        return st

    def abspath(self, e: Entry) -> Path:
        return self.root / e.path

    def total_duration(self) -> float:
        return float(sum(e.duration for e in self.entries))


def _entry(d: dict) -> Entry:
    """Entry from index JSON: unknown keys (a newer index) are ignored; a missing required key raises."""
    return Entry(**{k: d[k] for k in Entry.__dataclass_fields__ if k in d})


def _hide(d: Path) -> None:
    """Mark the ``.sonicproof`` folder hidden on Windows (dot-folders already are elsewhere)."""
    if sys.platform != "win32" or d.name != INDEX_DIR:
        return
    try:
        import ctypes

        ctypes.windll.kernel32.SetFileAttributesW(str(d), 0x02)  # FILE_ATTRIBUTE_HIDDEN
    except Exception:
        pass


def _lufs(a) -> float | None:
    """Integrated loudness, or None when it can't be measured (shorter than 400 ms, silent, odd data)."""
    from .qc import integrated_loudness

    try:
        val = integrated_loudness(a.data, a.sr)
    except Exception:
        return None
    return round(float(val), 2) if np.isfinite(val) else None


def _describe(f: Path, rel: str, s: os.stat_result, embed) -> tuple[Entry, np.ndarray, bool]:
    digest = cache.file_digest(f)
    a = load(f)
    mono = a.mono()
    lufs = _lufs(a)
    peak = float(np.abs(a.data).max()) if a.data.size else 0.0
    _, active = silence_level(a.data, a.sr)
    v = cache.load_array("audio", digest)
    fresh = v is None
    if fresh:
        v = np.asarray(embed(resample(mono, a.sr, MODEL_SR)), dtype=np.float32)
        if v.shape != (DIM,):
            raise SetupError(f"The embedder returned shape {v.shape}, expected ({DIM},).")
        cache.save_array(v, "audio", digest)
    e = Entry(
        path=rel,
        size=s.st_size,
        mtime_ns=s.st_mtime_ns,
        digest=digest,
        duration=round(a.duration, 4),
        sample_rate=a.sr,
        channels=a.channels,
        lossless=f.suffix.lower() in LOSSLESS_EXTS,
        lufs=lufs,
        peak_db=round(20 * np.log10(max(peak, 1e-12)), 2),
        active=round(active, 3),
    )
    return e, v, fresh


# -- duplicates -------------------------------------------------------------------------------
# Embeddings say "these sound alike"; they cannot tell a re-encoded copy from two takes of the same
# idea. Candidate pairs (close embeddings) are therefore confirmed on the signal itself: log-mel
# fingerprints, aligned in time, must correlate almost perfectly. Gain, codec, sample rate, channel
# count and trimmed ends do not change that; a different take or a different recording does.
CANDIDATE_COS = 0.94
MATCH = 0.98
FULL_OVERLAP = 0.98  # matched span covers this much of the longer file -> same audio, else trimmed
FP_RATE = 40  # fingerprint frames per second (hop 400 at 16 kHz)
FP_SECONDS = 120.0  # only the first two minutes are compared
FP_FLOOR_DB = 70.0
_N_FFT = 512
_N_MELS = 32


def _mel_filterbank() -> np.ndarray:
    def mel(f):
        return 2595 * np.log10(1 + f / 700)

    pts = 700 * (10 ** (np.linspace(mel(40), mel(MODEL_SR / 2), _N_MELS + 2) / 2595) - 1)
    bins = np.fft.rfftfreq(_N_FFT, 1 / MODEL_SR)
    fb = np.zeros((_N_MELS, len(bins)))
    for i in range(_N_MELS):
        lo, c, hi = pts[i : i + 3]
        fb[i] = np.clip(np.minimum((bins - lo) / (c - lo), (hi - bins) / (hi - c)), 0, None)
    return fb


_FB = _mel_filterbank()


# Not a copy: the fingerprinted first two minutes match but the rest of a longer file did not (or only one
# of the two is long). Reported for a listen, never counted as redundant or reclaimable.
SAME_START = "same start"
REDUNDANT = ("identical", "same audio", "trimmed")


def fingerprint(wave: np.ndarray, tail: bool = False) -> np.ndarray:
    """(frames, 32) log-mel energies of a mono 16 kHz wave (its first two minutes, or with ``tail`` its
    last two), floored 70 dB under its peak."""
    n = int(FP_SECONDS * MODEL_SR)
    w = np.asarray(wave[-n:] if tail else wave[:n], dtype=np.float32)
    hop = MODEL_SR // FP_RATE
    if len(w) < _N_FFT:
        w = np.pad(w, (0, _N_FFT - len(w)))
    _, _, Z = stft(w, MODEL_SR, nperseg=_N_FFT, noverlap=_N_FFT - hop, boundary=None, padded=False)
    L = 10 * np.log10(_FB @ (np.abs(Z) ** 2) + 1e-10)
    return np.maximum(L, L.max() - FP_FLOOR_DB).T.astype(np.float32)


def _corr_at(A: np.ndarray, B: np.ndarray, lag: int) -> tuple[float, int]:
    a0, b0 = max(0, lag), max(0, -lag)
    n = min(len(A) - a0, len(B) - b0)
    if n <= 0:
        return -1.0, 0
    x, y = A[a0 : a0 + n], B[b0 : b0 + n]
    x = (x - x.mean(0)).ravel()
    y = (y - y.mean(0)).ravel()
    den = float(np.sqrt((x @ x) * (y @ y)))
    return (float(x @ y) / den if den > 0 else 0.0), n


def match_fingerprints(A: np.ndarray, B: np.ndarray, min_overlap: float = 0.5) -> tuple[float, float, float]:
    """Best aligned correlation of two fingerprints: (correlation, offset in s, coverage of the longer).

    ``A[t]`` lines up with ``B[t - lag]``; offsets are searched coarse-to-fine over every alignment where
    the overlap covers at least ``min_overlap`` of the shorter file.
    """
    pool = 5

    def coarse(X):
        n = len(X) // pool
        return X[: n * pool].reshape(n, pool, -1).mean(1) if n >= 2 else X

    if min(len(A), len(B)) < 2 * pool:  # too short to pool: search every offset at full resolution
        ca, cb, scale = A, B, 1
    else:
        ca, cb, scale = coarse(A), coarse(B), pool
    minov = max(2, int(min(len(ca), len(cb)) * min_overlap))
    lags = range(-(len(cb) - minov), len(ca) - minov + 1)
    rough = sorted(((_corr_at(ca, cb, g)[0], g) for g in lags), reverse=True)[:3]
    best = (-1.0, 0, 0)
    for _, g in rough:
        for lag in range(g * scale - scale, g * scale + scale + 1):
            c, n = _corr_at(A, B, lag)
            if c > best[0] and n >= min(len(A), len(B)) * min_overlap:
                best = (c, lag, n)
    c, lag, n = best
    return c, lag / FP_RATE, n / max(len(A), len(B), 1)


@dataclass
class Member:
    path: str
    relation: str  # "keep" | "identical" | "same audio" | "trimmed" | "same start"
    match: float  # fingerprint correlation with the kept file (1.0 for identical bytes)
    similarity: float  # embedding cosine with the kept file
    duration: float
    sample_rate: int
    channels: int
    lossless: bool
    size: int


@dataclass
class DupGroup:
    keep: str
    reason: str
    members: list[Member]
    reclaimable_bytes: int

    def to_dict(self) -> dict:
        return asdict(self)


def _quality_key(e: Entry) -> tuple:
    clipped = e.peak_db >= -0.01
    return (e.lossless, not clipped, e.sample_rate, e.channels, round(e.duration, 1), -len(e.path))


def _keep_reason(keep: Entry, others: list[Entry]) -> str:
    bits = []
    if keep.lossless and not all(o.lossless for o in others):
        bits.append("lossless")
    if any(o.sample_rate < keep.sample_rate for o in others):
        bits.append(f"{keep.sample_rate / 1000:g} kHz")
    if any(o.channels < keep.channels for o in others):
        bits.append("stereo" if keep.channels == 2 else f"{keep.channels} ch")
    if any(o.duration < keep.duration - 0.05 for o in others):
        bits.append("longest (not trimmed)")
    if any(o.peak_db >= -0.01 for o in others) and keep.peak_db < -0.01:
        bits.append("not clipped")
    return ", ".join(bits) or "equal quality — shortest path"


def find_duplicates(
    lib: Library,
    candidate_cos: float = CANDIDATE_COS,
    match_min: float = MATCH,
    progress: Progress | None = None,
) -> list[DupGroup]:
    E, entries = lib.emb, lib.entries
    n = len(entries)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    edges: dict[tuple[int, int], float] = {}
    by_digest: dict[str, int] = {}
    for i, e in enumerate(entries):
        if e.digest in by_digest:
            j = by_digest[e.digest]
            edges[(j, i)] = 1.0
            parent[find(i)] = find(j)
        else:
            by_digest[e.digest] = i

    pairs = []
    block = 2048
    for s in range(0, n, block):
        S = E[s : s + block] @ E.T
        for r, c in zip(*np.nonzero(S >= candidate_cos), strict=True):
            i = s + int(r)
            if int(c) > i and entries[i].digest != entries[int(c)].digest:
                pairs.append((i, int(c)))

    fps: dict[int, np.ndarray] = {}

    def fp(i: int, tail: bool = False) -> np.ndarray:
        if (i, tail) not in fps:
            d = entries[i].digest
            tag = f"r{FP_RATE}s{FP_SECONDS}" + ("t" if tail else "")
            x = cache.load_array("fp", d, tag)
            if x is None:
                x = fingerprint(load_for_model(lib.abspath(entries[i])), tail=tail)
                cache.save_array(x.astype(np.float16), "fp", d, tag)
            fps[i, tail] = x.astype(np.float32)
        return fps[i, tail]

    def whole_file_matches(i: int, j: int) -> bool:
        """Past the fingerprinted first two minutes: do the last two minutes match as well?"""
        if max(entries[i].duration, entries[j].duration) <= FP_SECONDS + 0.5:
            return True  # the fingerprint already covered all of both files
        try:
            return match_fingerprints(fp(i, True), fp(j, True))[0] >= match_min
        except Exception:
            return False

    for k, (i, j) in enumerate(pairs):
        if progress:
            progress(k, len(pairs), entries[j].path)
        try:
            c, _, _ = match_fingerprints(fp(i), fp(j))
        except Exception:
            continue
        if c >= match_min:
            edges[(i, j)] = c
            parent[find(j)] = find(i)
    if progress:
        progress(len(pairs), len(pairs), "")

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = []
    for idx in groups.values():
        if len(idx) < 2:
            continue
        keep = max(idx, key=lambda i: _quality_key(entries[i]))
        others = [i for i in idx if i != keep]
        members = [_member(entries[keep], "keep", 1.0, 1.0)]
        for i in sorted(others, key=lambda i: entries[i].path):
            e = entries[i]
            if e.digest == entries[keep].digest:
                members.append(_member(e, "identical", 1.0, 1.0))
                continue
            sim = float(E[i] @ E[keep])
            c, _, cov = match_fingerprints(fp(keep), fp(i))
            rel = "same audio" if cov >= FULL_OVERLAP else "trimmed"
            members.append(_member(e, rel if whole_file_matches(keep, i) else SAME_START, c, sim))
        out.append(
            DupGroup(
                keep=entries[keep].path,
                reason=_keep_reason(entries[keep], [entries[i] for i in others]),
                members=members,
                reclaimable_bytes=sum(m.size for m in members if m.relation in REDUNDANT),
            )
        )
    return sorted(out, key=lambda g: (-len(g.members), g.keep))


def _member(e: Entry, relation: str, match: float, sim: float) -> Member:
    return Member(
        path=e.path,
        relation=relation,
        match=round(match, 4),
        similarity=round(sim, 4),
        duration=e.duration,
        sample_rate=e.sample_rate,
        channels=e.channels,
        lossless=e.lossless,
        size=e.size,
    )


# -- search -----------------------------------------------------------------------------------
@dataclass
class Hit:
    path: str
    score: float  # calibrated σ for words / pictures; cosine for --like
    cosine: float
    verdict: str | None
    heard_as: str | None
    duration: float
    note: str | None = None  # "silent": no usable signal, verdict forced to "miss" and ranked last

    def to_dict(self) -> dict:
        return asdict(self)


def _order(lib: Library, score: np.ndarray, keep: np.ndarray | None = None) -> list[int]:
    """Best first; silent entries go after every audible one."""
    idx = [i for i in np.argsort(-score) if keep is None or keep[i]]
    return sorted(idx, key=lambda i: lib.entries[i].silent)  # stable: keeps score order within each group


def _hits(lib: Library, judge, order, score: np.ndarray, cos: np.ndarray, verdicts: bool):
    from .judge import verdict

    out = []
    for i in order:
        e = lib.entries[i]
        vs = judge._calibrate(judge.V, lib.emb[i], judge._vr_mu, judge._vr_sd)[3]
        silent = e.silent
        out.append(
            Hit(
                path=e.path,
                score=float(score[i]),
                cosine=float(cos[i]),
                verdict=("miss" if silent else verdict(float(score[i]))) if verdicts else None,
                heard_as=judge.vocab[int(np.argmax(vs))],
                duration=e.duration,
                note=SILENT_NOTE if silent else None,
            )
        )
    return out


def search_text(lib: Library, judge, query: str, n: int = 10) -> list[Hit]:
    """Clips ranked by the same calibrated score ``check`` uses — any language."""
    if not len(lib):
        return []
    t = judge.text([query])[0]
    score, cos = judge.prompt_scores(t, lib.emb)
    return _hits(lib, judge, _order(lib, score)[:n], score, cos, True)


def search_like(lib: Library, judge, path: str | Path, n: int = 10) -> list[Hit]:
    """Clips that sound most like ``path`` (the file itself, if it is in the library, is skipped)."""
    if not len(lib):
        return []
    a = judge.audio(path)
    digest = cache.file_digest(path)
    cos = lib.emb @ a
    keep = np.array([e.digest != digest for e in lib.entries])
    return _hits(lib, judge, _order(lib, cos, keep)[:n], cos, cos, False)


def search_picture(lib: Library, judge, v: np.ndarray, scene: str | None = None, n: int = 10) -> list[Hit]:
    """Clips that best suit a picture (image or video embedding), scored like ``fit``."""
    if not len(lib):
        return []
    t = judge.text([scene])[0] if scene else None
    r = judge.fit_scores(v, lib.emb, t)
    return _hits(lib, judge, _order(lib, r["score"])[:n], r["score"], r["cosine"], True)
