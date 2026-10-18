import numpy as np
import pytest
import soundfile as sf

from sonicproof import thumbs


@pytest.mark.parametrize("n", [0, 1, 100, 16_000])
@pytest.mark.parametrize("sr", [8_000, 44_100])
def test_peaks_and_spectrogram_shapes(n, sr):
    x = (0.3 * np.sin(2 * np.pi * 300 * np.arange(n) / sr)).astype(np.float32)
    pk = thumbs.peaks(x)
    sp = thumbs.spectrogram(x, sr)
    assert pk.shape == (thumbs.COLUMNS, 2) and np.isfinite(pk).all()
    assert sp.shape == (thumbs.ROWS, thumbs.COLUMNS) and sp.dtype == np.uint8


def test_silence_spectrogram_is_dark():
    assert not thumbs.spectrogram(np.zeros(44_100, np.float32), 44_100).any()
    assert not thumbs.spectrogram(np.zeros(0, np.float32), 44_100).any()


def test_tone_lights_its_row():
    sr = 44_100
    sp = thumbs.spectrogram(np.sin(2 * np.pi * 1000 * np.arange(sr) / sr).astype(np.float32), sr)
    row = int(np.argmax(sp.mean(axis=1)))
    edges = np.geomspace(thumbs.F_LO, sr / 2, thumbs.ROWS + 1)
    assert edges[row] <= 1000 < edges[row + 1] * 1.2


def test_thumbnail_is_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("SONICPROOF_CACHE", str(tmp_path / "cache"))
    p = tmp_path / "t.wav"
    sf.write(p, 0.2 * np.random.default_rng(0).standard_normal((8_000, 2)), 8_000)
    calls = []
    real = thumbs.load
    monkeypatch.setattr(thumbs, "load", lambda path: calls.append(path) or real(path))
    first = thumbs.thumbnail(p)
    second = thumbs.thumbnail(p)
    assert len(calls) == 1 and first == second
    assert first["sample_rate"] == 8_000 and first["channels"] == 2 and abs(first["duration"] - 1.0) < 1e-6
    assert len(first["peaks"]) == thumbs.COLUMNS and first["spec_rows"] == thumbs.ROWS


def test_thumbnail_of_empty_file(tmp_path, monkeypatch):
    monkeypatch.setenv("SONICPROOF_CACHE", str(tmp_path / "cache"))
    p = tmp_path / "e.wav"
    sf.write(p, np.zeros((0, 1), np.float32), 44_100)
    t = thumbs.thumbnail(p)
    assert t["duration"] == 0 and len(t["peaks"]) == thumbs.COLUMNS
