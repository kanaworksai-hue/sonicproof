"""Prompt adherence: does this clip sound like what was asked for?

Raw cosine similarity in EmbeddingGemma 2's shared space is a poor verdict on its own: audio embeddings
sit close together, so every caption scores ~0.6-0.7 against every clip. Sonicproof turns it into a
calibrated score by asking two relative questions and averaging the answers (both in standard deviations):

- clip side: how does the prompt compare with ~440 everyday sound descriptions (``vocab.py``) on this clip?
- prompt side: how much closer is the prompt to this clip than to a bank of 110 reference clips
  (``data/refbank.npz``, embeddings only)?

``fit`` (does this sound suit this picture?) follows the same idea with two pieces of evidence:

- direct: how much closer is the picture to this clip than to the reference clips?
- described: the picture is first described with the vocabulary ("pictured as …" — image-text matching is
  the model's strong side), and the clip is checked against those descriptions like a prompt.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np

from . import cache
from .audio import MIN_SIGNAL_S, MODEL_SR, SILENCE_DB, frame_rms_db, load, resample, sidecar_prompt
from .errors import InputError, SetupError
from .model import DEFAULT_TEXT_PROMPT, DIM, MODEL_ID, Embedder, get_embedder
from .visual import MAX_FRAMES, VIDEO_FPS, load_image, load_video
from .vocab import category_of, entries

# Verdict cut-offs on the calibrated score (σ). Tuned on an internal labelled set so that most
# mismatched prompt/clip pairs fall under "weak" and most correct ones reach "plausible" or better.
THRESHOLDS = (("strong", 2.0), ("plausible", 1.1), ("weak", 0.3))
VERDICT_HINT = {
    "strong": "clearly sounds like the prompt",
    "plausible": "sounds like the prompt",
    "weak": "only loosely matches the prompt",
    "miss": "does not sound like the prompt",
}
AGREES = 0.85  # text-text cosine above which a heard description agrees with the prompt (any language)
FIT_HINT = {
    "strong": "the sound clearly suits the picture",
    "plausible": "the sound suits the picture",
    "weak": "only loosely suits the picture",
    "miss": "the sound does not suit the picture",
}
DESCRIBE_TOP = 5  # how many "pictured as" descriptions feed the described-side evidence
DESCRIBE_TEMP = 0.02  # softmax temperature over their image-text cosines


# Silent or nearly empty clips: the model still places them somewhere in the space (often near short
# percussive descriptions), so their scores mean nothing. Same rule as qc's silence check
# (``audio.is_silent``); the level is cached so a cache hit needs no decode.
SILENT_NOTE = "silent"


def silence_level(data: np.ndarray, sr: int) -> tuple[float, float]:
    """(peak dBFS, seconds of 10 ms frames above -60 dBFS) of (frames,) or (frames, channels) audio."""
    x = np.asarray(data, dtype=np.float32)
    if not x.size:
        return float("-inf"), 0.0
    if not np.isfinite(x).all():  # NaN / inf samples: measure the rest (same as qc and the model input)
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    env = frame_rms_db(x, sr)
    frame = max(1, int(sr * 0.01))
    active = int((env > SILENCE_DB).sum()) * min(frame, len(x)) / sr
    return 20 * np.log10(max(float(np.abs(x).max()), 1e-12)), float(active)


def is_silent_level(peak_db: float, active: float | None) -> bool:
    """``audio.is_silent`` from stored numbers (``active`` is None in indexes that predate it)."""
    return peak_db <= SILENCE_DB or (active is not None and active < MIN_SIGNAL_S)


def verdict(score: float) -> str:
    for name, cut in THRESHOLDS:
        if score >= cut:
            return name
    return "miss"


@dataclass
class Alternative:
    text: str
    score: float
    category: str | None
    agrees: bool  # the description says roughly what the prompt says


@dataclass
class CheckResult:
    file: str
    prompt: str
    prompt_source: str  # "argument" | "sidecar"
    score: float  # calibrated, in σ
    verdict: str
    cosine: float
    z_clip: float
    z_prompt: float
    vocab_rank: int  # where the prompt lands among all vocabulary descriptions for this clip
    vocab_size: int
    heard_as: list[Alternative] = field(default_factory=list)
    note: str | None = None  # "silent": no usable signal, verdict forced to "miss" (ranked last)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FitResult:
    file: str
    picture: str
    picture_kind: str  # "image" | "video"
    frames: int
    scene: str | None
    score: float  # calibrated, in σ
    verdict: str
    cosine: float
    z_direct: float
    z_described: float
    z_scene: float | None
    pictured_as: list[Alternative] = field(default_factory=list)  # agrees = the clip also matches it
    heard_as: list[Alternative] = field(default_factory=list)
    note: str | None = None  # "silent", as in CheckResult

    def to_dict(self) -> dict:
        return asdict(self)


class Judge:
    def __init__(self, embedder: Embedder | None = None, device: str | None = None, vision: bool = False):
        self._embedder = embedder
        if embedder is not None:
            _check_dim(embedder)
        self._device = device
        self._vision = vision
        self.vocab = entries()
        self._V: np.ndarray | None = None
        bank = np.load(resources.files("sonicproof") / "data" / "refbank.npz")
        if str(bank["model"]) != MODEL_ID:
            raise SetupError(
                f"SONICPROOF_MODEL={MODEL_ID}, but the reference bank was built for {bank['model']}; "
                "unset SONICPROOF_MODEL or rebuild the bank.",
                code="model_unavailable",
            )
        self.R = bank["emb"].astype(np.float32)

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = _check_dim(get_embedder(self._device, vision=self._vision))
        return self._embedder

    # -- embeddings -------------------------------------------------------------------------
    def text(self, texts: list[str]) -> np.ndarray:
        return self.embedder.embed_text(texts, prompt_name=DEFAULT_TEXT_PROMPT)

    @property
    def V(self) -> np.ndarray:
        if self._V is None:
            tag = hashlib.sha1("\n".join(self.vocab).encode()).hexdigest()
            V = cache.load_array("vocab", DEFAULT_TEXT_PROMPT, tag)
            if V is None:
                V = self.text(self.vocab)
                cache.save_array(V, "vocab", DEFAULT_TEXT_PROMPT, tag)
            self._V = V
            VR = V @ self.R.T
            self._vr_mu, self._vr_sd = VR.mean(1), VR.std(1)
        return self._V

    def audio(self, path: str | Path) -> np.ndarray:
        digest = _digest(path)
        a = cache.load_array("audio", digest)
        if a is None:
            x = load(path)
            self._save_level(x.data, x.sr, digest)
            a = self.embedder.embed_audio([resample(x.mono(), x.sr, MODEL_SR)])[0]
            cache.save_array(a, "audio", digest)
        return a

    def level(self, path: str | Path) -> tuple[float, float]:
        """(peak dBFS, active seconds) — cached next to the embedding, so a cache hit needs no decode."""
        digest = _digest(path)
        lv = cache.load_array("level", digest)
        if lv is None or lv.shape != (2,):
            x = load(path)
            return self._save_level(x.data, x.sr, digest)
        return float(lv[0]), float(lv[1])

    def _save_level(self, data: np.ndarray, sr: int, digest: str) -> tuple[float, float]:
        peak, active = silence_level(data, sr)
        cache.save_array(np.array([max(peak, -400.0), active], np.float64), "level", digest)
        return peak, active

    def silent(self, path: str | Path) -> bool:
        return is_silent_level(*self.level(path))

    def image(self, path: str | Path) -> np.ndarray:
        digest = _digest(path)
        v = cache.load_array("image", digest)
        if v is None:
            v = self.embedder.embed_images([load_image(path)])[0]
            cache.save_array(v, "image", digest)
        return v

    def video(self, path: str | Path) -> tuple[np.ndarray, int]:
        """Clip embedding from frames sampled at 1 fps (≤ 32, spread evenly). Returns (embedding, frames)."""
        digest = _digest(path)
        key = f"fps{VIDEO_FPS}x{MAX_FRAMES}"
        v = cache.load_array("video", digest, key)
        if v is None:
            fr = load_video(path)
            v = np.concatenate([self.embedder.embed_video(fr.frames), [len(fr.frames)]]).astype(np.float32)
            cache.save_array(v, "video", digest, key)
        return v[:-1], int(v[-1])

    # -- scoring ----------------------------------------------------------------------------
    def _calibrate(self, T: np.ndarray, a: np.ndarray, t_mu: np.ndarray, t_sd: np.ndarray):
        """Return (cosine, z_clip, z_prompt, score) for each row of T against clip a."""
        va = self.V @ a
        cos = T @ a
        z_clip = (cos - va.mean()) / va.std()
        z_prompt = (cos - t_mu) / t_sd
        return cos, z_clip, z_prompt, (z_clip + z_prompt) / 2

    def check(
        self, path: str | Path, prompt: str | None = None, n_alt: int = 5, _t: np.ndarray | None = None
    ) -> CheckResult:
        source = "argument"
        if not prompt:
            prompt = sidecar_prompt(path)
            source = "sidecar"
            if not prompt:
                name = Path(path).name
                raise InputError(
                    f"No prompt given and no sidecar prompt found for {name}.", code="no_prompt", file=name
                )
        t = self.text([prompt])[0] if _t is None else _t
        V = self.V
        a = self.audio(path)
        tR = self.R @ t
        cos, zc, zp, score = (x.item() for x in self._calibrate(t[None], a, tR.mean(), tR.std()))

        # the same calibrated score for every vocabulary entry -> rank + "sounds more like"
        _, _, _, vs = self._calibrate(V, a, self._vr_mu, self._vr_sd)
        rank = int((vs > score).sum()) + 1
        agrees = (V @ t) >= AGREES
        alts = [
            Alternative(self.vocab[i], float(vs[i]), category_of(self.vocab[i]), bool(agrees[i]))
            for i in np.argsort(-vs)[:n_alt]
        ]
        silent = self.silent(path)
        return CheckResult(
            file=str(path),
            prompt=prompt,
            prompt_source=source,
            score=score,
            verdict="miss" if silent else verdict(score),
            cosine=cos,
            z_clip=zc,
            z_prompt=zp,
            vocab_rank=rank,
            vocab_size=len(self.vocab),
            heard_as=alts,
            note=SILENT_NOTE if silent else None,
        )

    def listen(self, path: str | Path, n_alt: int = 5) -> list[Alternative]:
        """What the clip sounds like, without a prompt (the "heard as" list on its own)."""
        a = self.audio(path)
        _, _, _, vs = self._calibrate(self.V, a, self._vr_mu, self._vr_sd)
        return [
            Alternative(self.vocab[i], float(vs[i]), category_of(self.vocab[i]), False)
            for i in np.argsort(-vs)[:n_alt]
        ]

    def phrasings(self, prompt: str, n: int = 3) -> list[str]:
        """Vocabulary descriptions closest in meaning to ``prompt``: plain wordings the model knows well,
        offered as a rewrite when no take sounds like the prompt."""
        s = self.V @ self.text([prompt])[0]
        own = prompt.strip().lower()
        return [self.vocab[i] for i in np.argsort(-s)[: n + 1] if self.vocab[i].lower() != own][:n]

    def heard_as_many(self, A: np.ndarray) -> list[str]:
        """Top "heard as" description for every clip embedding in ``A``."""
        if not len(A):
            return []
        VA = self.V @ A.T
        vs = (VA - VA.mean(0)) / VA.std(0) + (VA - self._vr_mu[:, None]) / self._vr_sd[:, None]
        return [self.vocab[i] for i in np.argmax(vs, axis=0)]

    def rank(self, prompt: str, paths: list[str | Path], n_alt: int = 3) -> list[CheckResult]:
        t = self.text([prompt])[0]
        results = [self.check(p, prompt, n_alt=n_alt, _t=t) for p in paths]
        return sorted(results, key=lambda r: (r.note == SILENT_NOTE, -r.score))  # silent takes last

    # -- many clips at once (library search) -------------------------------------------------
    def clip_stats(self, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per clip: mean and std of its similarity to every vocabulary description."""
        VA = self.V @ A.T
        return VA.mean(0), VA.std(0)

    def prompt_scores(self, t: np.ndarray, A: np.ndarray, stats=None) -> tuple[np.ndarray, np.ndarray]:
        """Calibrated prompt score of text embedding ``t`` against every clip in ``A``: (score, cosine)."""
        mu, sd = stats if stats is not None else self.clip_stats(A)
        cos = A @ t
        tR = self.R @ t
        return ((cos - mu) / sd + (cos - tR.mean()) / tR.std()) / 2, cos

    def describe_picture(self, v: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vocabulary descriptions closest to a picture: (indices, softmax weights, z within vocabulary)."""
        s = self.V @ v
        top = np.argsort(-s)[: max(n, DESCRIBE_TOP)]
        w = np.exp((s[top[:DESCRIBE_TOP]] - s[top[0]]) / DESCRIBE_TEMP)
        return top, w / w.sum(), (s[top] - s.mean()) / s.std()

    def fit_scores(
        self, v: np.ndarray, A: np.ndarray, scene: np.ndarray | None = None, stats=None
    ) -> dict[str, np.ndarray]:
        """Calibrated fit of picture embedding ``v`` to every clip in ``A`` (all values per clip)."""
        _ = self.V  # vocabulary statistics
        stats = stats if stats is not None else self.clip_stats(A)
        cos = A @ v
        vR = self.R @ v
        z_direct = (cos - vR.mean()) / vR.std()
        top, w, _ = self.describe_picture(v, DESCRIBE_TOP)
        mu, sd = stats
        d = top[:DESCRIBE_TOP]
        VA = self.V[d] @ A.T
        vs = ((VA - mu) / sd + (VA - self._vr_mu[d, None]) / self._vr_sd[d, None]) / 2
        z_described = w @ vs
        parts = [z_direct, z_described]
        out = {"cosine": cos, "z_direct": z_direct, "z_described": z_described}
        if scene is not None:
            out["z_scene"], _ = self.prompt_scores(scene, A, stats)
            parts.append(out["z_scene"])
        out["score"] = np.mean(parts, axis=0)
        return out

    def fit(
        self,
        path: str | Path,
        picture: str | Path,
        kind: str,
        scene: str | None = None,
        n_alt: int = 5,
    ) -> FitResult:
        """Does the sound in ``path`` suit the image or video ``picture``?"""
        if kind == "video":
            v, frames = self.video(picture)
        else:
            v, frames = self.image(picture), 1
        a = self.audio(path)
        t = self.text([scene])[0] if scene else None
        V = self.V
        r = self.fit_scores(v, a[None], t)
        score = float(r["score"][0])
        silent = self.silent(path)

        _, _, _, vs = self._calibrate(V, a, self._vr_mu, self._vr_sd)
        top, _, zv = self.describe_picture(v, n_alt)
        pictured = [
            Alternative(self.vocab[i], float(z), category_of(self.vocab[i]), bool(vs[i] >= THRESHOLDS[1][1]))
            for i, z in zip(top[:n_alt], zv[:n_alt], strict=True)
        ]
        pic_set = set(top[:DESCRIBE_TOP].tolist())
        heard = [
            Alternative(self.vocab[i], float(vs[i]), category_of(self.vocab[i]), i in pic_set)
            for i in np.argsort(-vs)[:n_alt]
        ]
        return FitResult(
            file=str(path),
            picture=str(picture),
            picture_kind=kind,
            frames=frames,
            scene=scene,
            score=score,
            verdict="miss" if silent else verdict(score),
            cosine=float(r["cosine"][0]),
            z_direct=float(r["z_direct"][0]),
            z_described=float(r["z_described"][0]),
            z_scene=float(r["z_scene"][0]) if t is not None else None,
            pictured_as=pictured,
            heard_as=heard,
            note=SILENT_NOTE if silent else None,
        )


def _digest(path: str | Path) -> str:
    """Content hash for the cache, with the same errors ``audio.load`` gives for a bad path."""
    p = Path(path)
    if not p.exists():
        raise InputError(f"No such file: {path}", code="not_found", file=str(path))
    if p.is_dir():
        raise InputError(f"{path} is a folder, not a file.", code="is_dir", file=str(path))
    try:
        return cache.file_digest(p)
    except OSError as e:
        raise InputError(f"Cannot read {p.name}: {e.strerror}.", code="unreadable", file=str(path)) from e


def _check_dim(embedder):
    dim = getattr(embedder, "dim", DIM)
    if dim != DIM:
        raise ValueError(
            f"Judge needs {DIM}-d embeddings (the reference bank and indexes are {DIM}-d), got dim={dim}."
        )
    return embedder
