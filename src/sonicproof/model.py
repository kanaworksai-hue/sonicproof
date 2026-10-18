"""EmbeddingGemma 2 wrapper: one shared 768-d space for text, audio, images and video.

Rules from the model card that this module enforces:
- audio is mono 16 kHz and gets no prefix; text always goes through a task prompt;
- fp32 or bf16 only — never fp16 (silent NaNs);
- Matryoshka truncation is followed by re-normalisation.

The vision encoder (170M) is only loaded when a picture is involved (``vision=True``): text and audio
embeddings are identical either way, so the lighter text + audio model serves check / rank / search.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

from .audio import MODEL_SR
from .errors import SetupError

MODEL_ID = os.environ.get("SONICPROOF_MODEL", "google/embeddinggemma-2")
DEFAULT_TEXT_PROMPT = "SearchQuery"
WINDOW_SECONDS = 30.0  # long files are embedded in windows and mean-pooled
MIN_WINDOW_SECONDS = 1.0
DIM = 768  # the full embedding size; calibration data (reference bank, indexes) is stored at this size
DOWNLOAD_SIZE = "~1.5 GB"
INSTALL_HINT = "pip install 'sonicproof[model]' (or: uv sync --extra model)"

_lock = threading.Lock()


def _env(name: str) -> str | None:
    """An environment variable, with an empty string treated as unset (``-e HF_HOME="$HF_HOME"``)."""
    return os.environ.get(name) or None


def hub_cache_dir() -> Path:
    """Where Hugging Face keeps (or will download) models, resolved the way huggingface_hub does it:
    ``HF_HUB_CACHE`` > ``HF_HOME/hub`` > ``XDG_CACHE_HOME/huggingface/hub`` > ``~/.cache/huggingface/hub``.
    (No hub import: it freezes HF_HUB_OFFLINE on import.)"""
    if hub := _env("HF_HUB_CACHE"):
        return Path(hub).expanduser()
    home = _env("HF_HOME") or os.path.join(_env("XDG_CACHE_HOME") or "~/.cache", "huggingface")
    return Path(home).expanduser() / "hub"


def _snapshots() -> list[Path]:
    snaps = hub_cache_dir() / f"models--{MODEL_ID.replace('/', '--')}" / "snapshots"
    try:
        return sorted(s for s in snaps.iterdir() if (s / "config.json").exists())
    except OSError:
        return []


def model_cached() -> bool:
    """Is the model in the Hugging Face cache?"""
    return bool(_snapshots())


def model_revision() -> str:
    """Short id of the cached snapshot ("none" before the first download), part of the embedding cache key."""
    snaps = _snapshots()
    if not snaps:
        return "none"
    ref = hub_cache_dir() / f"models--{MODEL_ID.replace('/', '--')}" / "refs" / "main"
    try:
        main = ref.read_text().strip()
        if any(s.name == main for s in snaps):
            return main[:12]
    except OSError:
        pass
    return max(snaps, key=lambda s: s.stat().st_mtime).name[:12]


def prefer_offline() -> None:
    """Skip the Hugging Face network round-trip when the model is already cached."""
    if "HF_HUB_OFFLINE" not in os.environ and model_cached():
        os.environ["HF_HUB_OFFLINE"] = "1"


def _offline() -> bool:
    return (os.environ.get("HF_HUB_OFFLINE") or "").strip().lower() in ("1", "true", "yes", "on")


def require_model_extra() -> None:
    """Raise ``SetupError(missing_extra)`` unless torch and sentence-transformers are installed."""
    missing = [m for m in ("torch", "sentence_transformers") if importlib.util.find_spec(m) is None]
    if missing:
        raise SetupError(
            f"This needs the model extra ({', '.join(missing)} not installed): {INSTALL_HINT}",
            code="missing_extra",
        )


def require_model_available() -> None:
    """Raise ``SetupError(model_unavailable)`` when the model is not cached and downloading is off."""
    if not model_cached() and _offline():
        raise SetupError(
            f"EmbeddingGemma 2 is not downloaded and Hugging Face is offline (HF_HUB_OFFLINE). "
            f"Run `sonicproof doctor --load` once with network to download it ({DOWNLOAD_SIZE}), "
            "or set HF_HOME to a folder that already has it.",
            code="model_unavailable",
        )


_DEVICE_HINT = "use cpu, mps or cuda[:N], or unset it"


def pick_device() -> str:
    """CUDA > MPS > CPU (or ``SONICPROOF_DEVICE``, validated). On Apple Silicon, MPS fp32 gives the same
    embeddings as CPU, faster."""
    require_model_extra()
    import torch

    env = (_env("SONICPROOF_DEVICE") or "").strip().lower()
    if env:
        try:
            dev = torch.device(env)
        except (RuntimeError, ValueError, TypeError):
            raise SetupError(
                f"SONICPROOF_DEVICE={env} is not a device; {_DEVICE_HINT}.", code="bad_device"
            ) from None
        ok = {
            "cpu": lambda: True,
            "cuda": lambda: torch.cuda.is_available() and (dev.index or 0) < torch.cuda.device_count(),
            "mps": lambda: torch.backends.mps.is_available(),
        }.get(dev.type, lambda: False)
        if not ok():
            raise SetupError(
                f"SONICPROOF_DEVICE={env} but it is not available here; {_DEVICE_HINT}.", code="bad_device"
            )
        return env
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def _dtype_name() -> str:
    """The dtype ``Embedder`` will use: bf16 on CUDA with bf16 support, fp32 everywhere else.
    Works out the answer without importing torch where CUDA can't exist (macOS)."""
    env = (_env("SONICPROOF_DEVICE") or "").strip().lower()
    if env and not env.startswith("cuda"):
        return "float32"
    if sys.platform == "darwin" or importlib.util.find_spec("torch") is None:
        return "float32"
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return "bfloat16"
    except Exception:
        pass
    return "float32"


def embed_tag() -> str:
    """What else, besides the input, an embedding depends on: size, dtype and model snapshot."""
    return f"d{DIM}-{_dtype_name()}-{model_revision()}"


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


class Embedder:
    def __init__(self, device: str | None = None, dim: int = DIM, vision: bool = False):
        import logging

        require_model_extra()
        require_model_available()
        # torch prints a harmless "Redirects are currently not supported in MacOs" note on import.
        logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
        import torch
        from sentence_transformers import SentenceTransformer

        self.device = device or pick_device()
        self.dim = dim
        self.vision = vision
        cuda = self.device.startswith("cuda")
        dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float32
        try:
            self.model = SentenceTransformer(
                MODEL_ID,
                device=self.device,
                model_kwargs={"torch_dtype": dtype},
                config_kwargs={} if vision else {"vision_config": None},  # 740M vs 570M (text + audio)
            )
        except OSError as ex:  # offline / no network / missing files; anything else is a real bug
            if model_cached():
                raise
            raise SetupError(
                f"Could not download EmbeddingGemma 2 ({DOWNLOAD_SIZE}). Check the network and run "
                "`sonicproof doctor --load`, or set HF_HOME to a folder that already has it.",
                code="model_unavailable",
            ) from ex

    def _truncate(self, x: np.ndarray) -> np.ndarray:
        return _normalize(x[..., : self.dim]) if self.dim < x.shape[-1] else _normalize(x)

    def embed_text(
        self, texts: list[str], prompt_name: str | None = DEFAULT_TEXT_PROMPT, batch_size: int = 32
    ) -> np.ndarray:
        with _lock:
            emb = self.model.encode(
                texts,
                prompt_name=prompt_name,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return self._truncate(emb)

    def embed_audio(self, waves: list[np.ndarray]) -> np.ndarray:
        """Embed mono 16 kHz arrays. Long inputs are windowed and mean-pooled."""
        out = []
        for w in waves:
            wins = split_windows(w)
            with _lock:
                emb = self.model.encode(
                    [{"array": x, "sampling_rate": MODEL_SR} for x in wins],
                    batch_size=1,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            out.append(_normalize(np.asarray(emb).mean(axis=0)))
        return self._truncate(np.stack(out))

    def embed_images(self, images: list) -> np.ndarray:
        """Embed PIL images (one vector each)."""
        self._need_vision()
        with _lock:
            emb = self.model.encode(
                images,
                batch_size=4,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return self._truncate(np.asarray(emb))

    def embed_video(self, frames: np.ndarray) -> np.ndarray:
        """Embed one clip given as sampled RGB frames, shape (T, H, W, 3) uint8."""
        self._need_vision()
        with _lock:
            emb = self.model.encode(
                [{"video": frames}], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
            )
        return self._truncate(np.asarray(emb))[0]

    def _need_vision(self) -> None:
        if not self.vision:
            raise RuntimeError("This embedder was loaded without the vision encoder (use vision=True).")


def split_windows(w: np.ndarray, seconds: float = WINDOW_SECONDS) -> list[np.ndarray]:
    w = np.asarray(w, dtype=np.float32)
    min_len = int(MIN_WINDOW_SECONDS * MODEL_SR)
    if len(w) < min_len:  # very short one-shots: pad with silence so the encoder has enough frames
        w = np.pad(w, (0, min_len - len(w)))
    size = int(seconds * MODEL_SR)
    if len(w) <= size:
        return [w]
    wins = [w[i : i + size] for i in range(0, len(w), size)]
    if len(wins[-1]) < min_len:
        wins = wins[:-1]
    return wins


@lru_cache(maxsize=2)
def get_embedder(device: str | None = None, dim: int = DIM, vision: bool = False) -> Embedder:
    """Process-wide cached model (loading takes a few seconds)."""
    return Embedder(device=device, dim=dim, vision=vision)
