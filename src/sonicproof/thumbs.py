"""Waveform and spectrogram thumbnails for the workbench (computed once per file content, then cached)."""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
from scipy.signal import stft

from . import cache
from .audio import load

COLUMNS = 240  # time columns for both the waveform and the spectrogram
ROWS = 64  # log-frequency rows
F_LO = 40.0
FLOOR_DB = 80.0


def peaks(mono: np.ndarray, columns: int = COLUMNS) -> np.ndarray:
    """(columns, 2) min / max per column, in [-1, 1]."""
    if len(mono) == 0:
        return np.zeros((columns, 2), np.float32)
    edges = np.linspace(0, len(mono), columns + 1).astype(int)
    out = np.zeros((columns, 2), np.float32)
    for i in range(columns):
        seg = mono[edges[i] : max(edges[i + 1], edges[i] + 1)]
        out[i] = seg.min(), seg.max()
    return out


def spectrogram(mono: np.ndarray, sr: int, columns: int = COLUMNS, rows: int = ROWS) -> np.ndarray:
    """(rows, columns) uint8 log-frequency spectrogram, 0 = silence floor, 255 = loudest; row 0 is lowest."""
    n_fft = 1024 if sr >= 32_000 else 512
    mono = np.nan_to_num(np.asarray(mono, np.float32))
    if not np.any(mono):  # empty or digital silence: all dark (the scaling below would light it up)
        return np.zeros((rows, columns), np.uint8)
    if len(mono) < n_fft:
        mono = np.pad(mono, (0, n_fft - len(mono)))
    hop = max(1, (len(mono) - n_fft) // max(columns - 1, 1))
    hop = min(hop, n_fft)  # short files: overlap, then resample columns below
    _, _, Z = stft(mono, sr, nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)
    P = np.abs(Z) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    edges = np.geomspace(min(F_LO, sr / 4), sr / 2, rows + 1)  # very low rates: start below Nyquist
    S = np.zeros((rows, P.shape[1]), np.float64)
    for r in range(rows):
        sel = (freqs >= edges[r]) & (freqs < edges[r + 1])
        if not sel.any():  # low rows narrower than one bin: take the nearest bin
            sel = np.zeros_like(freqs, bool)
            sel[np.argmin(np.abs(freqs - np.sqrt(edges[r] * edges[r + 1])))] = True
        S[r] = P[sel].mean(0)
    L = 10 * np.log10(S + 1e-12)
    L = np.clip((L - (L.max() - FLOOR_DB)) / FLOOR_DB, 0, 1)
    idx = np.linspace(0, L.shape[1] - 1, columns).round().astype(int)
    return (L[:, idx] * 255).astype(np.uint8)


def thumbnail(path: str | Path) -> dict:
    digest = cache.file_digest(path)
    key = f"c{COLUMNS}r{ROWS}"
    pk = cache.load_array("thumb_peaks", digest, key)
    sp = cache.load_array("thumb_spec", digest, key)
    meta = cache.load_array("thumb_meta", digest, key)
    if pk is None or sp is None or meta is None:
        a = load(path)
        mono = np.nan_to_num(a.data.mean(axis=1))
        pk = peaks(mono)
        sp = spectrogram(mono, a.sr)
        meta = np.array([a.duration, a.sr, a.channels], np.float64)
        cache.save_array(pk, "thumb_peaks", digest, key)
        cache.save_array(sp, "thumb_spec", digest, key)
        cache.save_array(meta, "thumb_meta", digest, key)
    return {
        "duration": float(meta[0]),
        "sample_rate": int(meta[1]),
        "channels": int(meta[2]),
        "peaks": np.round(pk, 3).tolist(),
        "spec": base64.b64encode(sp.tobytes()).decode(),
        "spec_rows": int(sp.shape[0]),
        "spec_cols": int(sp.shape[1]),
    }
