"""Pictures for ``fit``: still images (Pillow) and video frames (ffmpeg)."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import InputError, SetupError

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".ogv", ".mpg", ".mpeg", ".wmv"}

# EmbeddingGemma 2 samples video at 1 fps and its processor keeps at most 32 frames; we do the sampling
# ourselves with ffmpeg so any container works and long clips are spread evenly instead of cut off.
VIDEO_FPS = 1.0
MAX_FRAMES = 32
FRAME_WIDTH = 640  # the processor resizes again; this only keeps decoding light


@dataclass
class Frames:
    frames: np.ndarray  # (T, H, W, 3) uint8
    times: list[float]  # seconds, one per frame
    duration: float


def _check_file(path: Path) -> None:
    if not path.exists():
        raise InputError(f"No such file: {path}", code="not_found", file=str(path))
    if path.is_dir():
        raise InputError(f"{path} is a folder, not a picture.", code="is_dir", file=str(path))


def load_image(path: str | Path):
    path = Path(path)
    _check_file(path)
    try:
        from PIL import Image, ImageOps
    except ImportError:
        raise SetupError(
            "Reading images needs Pillow: pip install 'sonicproof[model]'", code="missing_extra"
        ) from None
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            return im.convert("RGB")
    except OSError as e:  # UnidentifiedImageError, truncated files
        raise InputError(f"Cannot read {path.name} as an image.", code="unreadable", file=str(path)) from e


def _need_ffmpeg(tool: str = "ffprobe") -> None:
    if shutil.which(tool) is None:
        raise SetupError(f"Reading video needs ffmpeg ({tool} was not found on PATH).", code="needs_ffmpeg")


def _probe(path: Path) -> dict:
    _need_ffmpeg("ffprobe")
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,width,height:stream_side_data=rotation:stream_tags=rotate:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        err = InputError(f"Cannot read {path.name} as a video.", code="unreadable", file=str(path))
        raise err from RuntimeError(out.stderr.strip())
    try:
        return json.loads(out.stdout or "{}")
    except json.JSONDecodeError as e:
        raise InputError(f"Cannot read {path.name} as a video.", code="unreadable", file=str(path)) from e


def has_video_stream(path: str | Path) -> bool:
    """Does ``path`` (a video by its extension) hold a picture? Raises ``SetupError(needs_ffmpeg)``
    without ffprobe, so callers don't mistake "can't look" for "no video"."""
    path = Path(path)
    if path.suffix.lower() not in VIDEO_EXTS or not path.is_file():
        return False
    _need_ffmpeg("ffprobe")
    try:
        info = _probe(path)
    except InputError:
        return False
    return any(s.get("codec_type") == "video" for s in info.get("streams", []))


def _rotation(stream: dict) -> int:
    """Display rotation in degrees (phone videos store portrait as rotated landscape)."""
    rot = stream.get("tags", {}).get("rotate")
    for sd in stream.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = sd["rotation"]
    try:
        return round(float(rot or 0)) % 360
    except (TypeError, ValueError):
        return 0


def video_frame_times(duration: float, fps: float = VIDEO_FPS, max_frames: int = MAX_FRAMES) -> list[float]:
    """Frame centres at ``fps``; clips longer than ``max_frames`` seconds are sampled evenly instead."""
    n = max(1, min(max_frames, int(np.floor(duration * fps)) or 1))
    step = duration / n
    return [round((i + 0.5) * step, 3) for i in range(n)]


def load_video(path: str | Path, fps: float = VIDEO_FPS, max_frames: int = MAX_FRAMES) -> Frames:
    path = Path(path)
    _check_file(path)
    info = _probe(path)
    _need_ffmpeg("ffmpeg")
    vs = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    if not vs:
        raise InputError(f"{path.name} has no video stream.", code="bad_value", file=str(path))
    w, h = int(vs[0]["width"]), int(vs[0]["height"])
    if _rotation(vs[0]) in (90, 270):  # ffmpeg auto-rotates, so the frames come out portrait
        w, h = h, w
    try:
        duration = float(info.get("format", {}).get("duration") or 0.0)
    except ValueError:
        duration = 0.0
    out_w = max(2, min(FRAME_WIDTH, w) // 2 * 2)
    out_h = max(2, round(h * out_w / w / 2) * 2)
    times = video_frame_times(duration) if duration > 0 else [0.0]
    rate = len(times) / duration if duration > 0 else 1.0
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(path),
        "-an",
        "-vf",
        f"fps={rate:.6f}:start_time=0,scale={out_w}:{out_h}",
        "-frames:v",
        str(len(times)),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=False)
    if out.returncode != 0:
        err = InputError(f"Cannot read {path.name} as a video.", code="unreadable", file=str(path))
        raise err from RuntimeError(out.stderr.decode(errors="ignore").strip())
    frame_bytes = out_w * out_h * 3
    n = len(out.stdout) // frame_bytes
    if n == 0:
        raise InputError(f"Could not read any frames from {path.name}.", code="unreadable", file=str(path))
    frames = np.frombuffer(out.stdout[: n * frame_bytes], dtype=np.uint8).reshape(n, out_h, out_w, 3)
    return Frames(frames=frames.copy(), times=times[:n], duration=duration)
