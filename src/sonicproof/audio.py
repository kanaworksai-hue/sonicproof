"""Audio I/O: decode anything (soundfile, falling back to ffmpeg) into float32 arrays."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .errors import InputError, SetupError

MODEL_SR = 16_000
SILENCE_DB = -60.0  # frames (10 ms RMS) at or below this are silence
MIN_SIGNAL_S = (
    0.01  # less sound than one 10 ms frame above SILENCE_DB counts as silent (a 50 ms click is not)
)
_CHUNK = 1 << 20  # samples per block when scanning long files, to keep memory flat


@dataclass
class Audio:
    """Decoded audio. ``data`` is (frames, channels) float32 in [-1, 1]."""

    data: np.ndarray
    sr: int
    path: Path | None = None
    subtype: str | None = None

    @property
    def channels(self) -> int:
        return self.data.shape[1]

    @property
    def duration(self) -> float:
        return self.data.shape[0] / self.sr

    def mono(self) -> np.ndarray:
        """Channel average for the model. NaN / inf samples (float files) become 0 / ±1 so they cannot
        poison the embedding; qc still sees and reports them in ``data``."""
        m = self.data.mean(axis=1).astype(np.float32)
        return m if np.isfinite(m).all() else np.nan_to_num(m, nan=0.0, posinf=1.0, neginf=-1.0)


def frame_rms_db(x: np.ndarray, sr: int, ms: float = 10.0) -> np.ndarray:
    """RMS level (dBFS) of consecutive ``ms`` frames. ``x`` is (frames,) or (frames, channels); for several
    channels the frame RMS is taken over all of them, so out-of-phase stereo does not cancel. A file shorter
    than one frame gives a single value; an empty one gives ``[-240]``."""
    x = x if x.ndim == 2 else x[:, None]
    n = max(1, int(sr * ms / 1000))
    frames = len(x) // n
    if frames == 0:
        ms_all = float(np.mean(x.astype(np.float64) ** 2)) if x.size else 0.0
        return np.array([20 * np.log10(max(np.sqrt(ms_all), 1e-12))])
    out = np.empty(frames)
    step = max(1, _CHUNK // n)  # frames per block
    for f0 in range(0, frames, step):
        f1 = min(frames, f0 + step)
        blk = x[f0 * n : f1 * n].reshape(f1 - f0, n, x.shape[1])
        out[f0:f1] = np.square(blk, dtype=np.float32).mean(axis=(1, 2))
    return 20 * np.log10(np.maximum(np.sqrt(out), 1e-12))


def is_silent(a: Audio | np.ndarray, sr: int | None = None) -> bool:
    """True for audio with (almost) nothing in it: no 10 ms frame above -60 dBFS (the same rule as qc's
    silence check; short one-shots such as UI clicks are not silent). Takes an :class:`Audio` or a raw
    array (pass ``sr``; defaults to the model's 16 kHz)."""
    if isinstance(a, Audio):
        x, sr = a.data, a.sr
    else:
        x, sr = np.asarray(a), sr or MODEL_SR
    if x.size == 0:
        return True
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0) if not np.isfinite(x).all() else x
    env = frame_rms_db(x, sr, 10)
    frame_s = max(1, int(sr * 0.01)) / sr
    return float(np.count_nonzero(env > SILENCE_DB)) * min(frame_s, len(x) / sr) < MIN_SIGNAL_S - 1e-9


_NO_AUDIO = ("does not contain any stream", "matches no streams", "no audio stream", "Output file is empty")


def _ffmpeg_decode(path: Path, sr: int | None = None, mono: bool = False) -> tuple[np.ndarray, int]:
    if shutil.which("ffmpeg") is None:
        raise SetupError(
            f"Cannot read {path.name}: this format needs ffmpeg, which is not installed "
            "(brew install ffmpeg / apt install ffmpeg).",
            code="needs_ffmpeg",
            file=str(path),
        )
    probe_sr = sr or _ffprobe_sr(path) or 48_000
    channels = 1 if mono else (_ffprobe_channels(path) or 2)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(path),
        "-vn",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ar",
        str(probe_sr),
        "-ac",
        str(channels),
        "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=False)
    stderr = out.stderr.decode(errors="ignore").strip()
    if out.returncode != 0:
        # keep ffmpeg's own (long) output as the cause, not in the one-line message
        cause = RuntimeError(f"ffmpeg: {stderr}")
        if any(s in stderr for s in _NO_AUDIO):
            raise InputError(f"{path.name} has no audio track.", code="no_audio", file=str(path)) from cause
        raise InputError(
            f"Cannot read {path.name}: not a supported audio/video file, or it is damaged.",
            code="unreadable",
            file=str(path),
        ) from cause
    usable = len(out.stdout) // (4 * channels) * 4 * channels
    data = np.frombuffer(out.stdout[:usable], dtype=np.float32).reshape(-1, channels)
    return data.copy(), probe_sr


def _ffprobe(path: Path, entry: str) -> int | None:
    if shutil.which("ffprobe") is None:
        return None
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            f"stream={entry}",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return int(out.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def _ffprobe_sr(path: Path) -> int | None:
    return _ffprobe(path, "sample_rate")


def _ffprobe_channels(path: Path) -> int | None:
    return _ffprobe(path, "channels")


# Formats libsndfile decodes without ffmpeg.
_SNDFILE_NATIVE = {
    ".wav",
    ".wave",
    ".flac",
    ".ogg",
    ".oga",
    ".mp3",
    ".aif",
    ".aiff",
    ".aifc",
    ".caf",
    ".w64",
    ".rf64",
}


def load(path: str | Path) -> Audio:
    """Load a file at its native rate and channel count.

    Raises :class:`InputError` (not_found / is_dir / empty / unreadable / no_audio) for files it can't use and
    :class:`SetupError` (needs_ffmpeg) when the format needs ffmpeg and it is missing (a broken WAV / FLAC /
    OGG / MP3 is ``unreadable`` either way). A valid file with zero
    frames loads fine (``qc`` reports it as ``file.empty``)."""
    path = Path(path)
    if not path.exists():
        raise InputError(f"No such file: {path}", code="not_found", file=str(path))
    if path.is_dir():
        raise InputError(f"{path} is a folder, not an audio file.", code="is_dir", file=str(path))
    try:
        size = path.stat().st_size
    except OSError as e:
        raise InputError(f"Cannot read {path.name}: {e.strerror}.", code="unreadable", file=str(path)) from e
    if size == 0:
        raise InputError(f"{path.name} is empty (0 bytes).", code="empty", file=str(path))
    try:
        info = sf.info(str(path))
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return Audio(data=data, sr=sr, path=path, subtype=info.subtype)
    except (sf.SoundFileError, RuntimeError, TypeError, ValueError) as e:
        if path.suffix.lower() in _SNDFILE_NATIVE and shutil.which("ffmpeg") is None:
            # a format libsndfile reads itself: it is the file that is broken, not a missing decoder
            raise InputError(
                f"Cannot read {path.name}: not a valid {path.suffix.lstrip('.').upper()} file.",
                code="unreadable",
                file=str(path),
            ) from e
        data, sr = _ffmpeg_decode(path)
        return Audio(data=data, sr=sr, path=path)


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x.astype(np.float32)
    g = np.gcd(sr_from, sr_to)
    return resample_poly(x, sr_to // g, sr_from // g, axis=0).astype(np.float32)


def load_for_model(path: str | Path) -> np.ndarray:
    """Mono 16 kHz float32 — the input EmbeddingGemma 2 expects."""
    a = load(path)
    if len(a.data) == 0:
        return np.zeros(0, np.float32)
    return resample(a.mono(), a.sr, MODEL_SR)


def _looks_like_sidecar(meta: dict) -> bool:
    return any(k in meta for k in ("prompt", "text", "description", "prompt_used")) or any(
        str(k).startswith("sound_") for k in meta
    )


def read_sidecar(path: str | Path) -> dict | None:
    """Stable Audio (and similar) tools write ``<file>.json`` next to the audio; some write ``<stem>.json``.

    Always returns a dict or None. ``<stem>.json`` is only taken when it carries a prompt-like field
    (prompt / text / description / prompt_used / sound_*), so an unrelated JSON with the same name is
    ignored."""
    path = Path(path)
    exact, stem = path.with_name(path.name + ".json"), path.with_suffix(".json")
    for cand in (exact, stem):
        if cand == path or not cand.is_file():
            continue
        try:
            meta = json.loads(cand.read_text(encoding="utf-8"))
        except (ValueError, OSError):  # JSONDecodeError and UnicodeDecodeError are ValueErrors
            continue
        if isinstance(meta, dict) and (cand == exact or _looks_like_sidecar(meta)):
            return meta
    return None


def sidecar_prompt(path: str | Path) -> str | None:
    meta = read_sidecar(path)
    if not meta:
        return None
    parts = []
    for key in ("prompt", "text", "description"):
        val = meta.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
            break
    # Stable Audio SFX: the structured sound_* fields carry most of the description. Only "what" and
    # "how" are used — recording-space words ("outdoor, close-up, dry") blur the match.
    fields = meta.get("sound_fields")
    if isinstance(fields, dict):
        parts += [
            fields[k].strip()
            for k in ("source", "action")
            if isinstance(fields.get(k), str) and fields[k].strip()
        ]
    if parts:
        return ", ".join(parts)
    used = meta.get("prompt_used")
    if isinstance(used, str) and used.strip():
        # Drop the trained-in "TrackType: X, VocalType: Y," prefix.
        return re.sub(r"^(\w+Type:\s*[^,]+,\s*)+", "", used.strip())
    return None
