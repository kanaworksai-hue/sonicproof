"""Judge logic with a stand-in embedder, plus end-to-end tests with the real model (skipped when
torch or EmbeddingGemma 2 is missing)."""

import hashlib

import numpy as np
import pytest
import soundfile as sf

from sonicproof.judge import THRESHOLDS, Judge, silence_level, verdict
from sonicproof.model import model_cached

SR = 16_000


class FakeEmbedder:
    """Deterministic vectors: text by hash; audio from ``audio_vec(wave)`` (default: by dominant band)."""

    vision = False
    dim = 768

    def __init__(self, audio_vec=None, dim=768):
        self.audio_vec = audio_vec
        self.dim = dim

    def embed_text(self, texts, prompt_name=None):
        seeds = [int(hashlib.sha1(t.encode()).hexdigest()[:8], 16) for t in texts]
        X = np.array([np.random.default_rng(s).standard_normal(768) for s in seeds], np.float32)
        return X / np.linalg.norm(X, axis=1, keepdims=True)

    def embed_audio(self, waves):
        out = []
        for w in waves:
            if self.audio_vec is not None:
                out.append(np.asarray(self.audio_vec(w), np.float64))
            else:
                band = int(np.argmax(np.abs(np.fft.rfft(w[:SR]))) // 50)
                out.append(np.random.default_rng(band).standard_normal(768))
        X = np.array(out, np.float32)
        return X / np.linalg.norm(X, axis=1, keepdims=True)


def test_verdict_thresholds_are_ordered():
    cuts = [c for _, c in THRESHOLDS]
    assert cuts == sorted(cuts, reverse=True)
    assert verdict(9) == "strong" and verdict(-9) == "miss"


def _wav(path, x):
    sf.write(path, np.asarray(x, np.float32), SR)
    return path


def test_silent_clips_are_a_miss_and_rank_last_but_short_ones_are_judged(tmp_path, monkeypatch):
    monkeypatch.setenv("SONICPROOF_CACHE", str(tmp_path / "cache"))
    prompt = "a wooden door creaks open"
    j = Judge(embedder=FakeEmbedder())
    t = j.text([prompt])[0]
    j = Judge(embedder=FakeEmbedder(audio_vec=lambda w: t))  # every clip is a perfect match on paper
    tone = 0.3 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR)
    loud = _wav(tmp_path / "loud.wav", tone)
    silence = _wav(tmp_path / "silence.wav", np.zeros(3 * SR))
    tiny = _wav(tmp_path / "tiny.wav", 0.5 * np.ones(SR // 20))  # 50 ms
    quiet = _wav(tmp_path / "quiet.wav", 0.0005 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR))  # -66 dBFS

    ok = j.check(loud, prompt)
    assert ok.note is None and ok.verdict == verdict(ok.score) != "miss"
    assert j.check(tiny, prompt).note is None  # short one-shots are judged, not written off as silent
    for p in (silence, quiet):
        r = j.check(p, prompt)
        assert r.verdict == "miss" and r.note == "silent", p.name
        assert r.to_dict()["note"] == "silent"
    ranked = j.rank(prompt, [silence, loud, quiet])
    assert ranked[0].file == str(loud) and {r.note for r in ranked[1:]} == {"silent"}
    again = Judge(embedder=FakeEmbedder(audio_vec=lambda w: t)).check(silence, prompt)  # from the cache
    assert again.note == "silent"


def test_out_of_phase_stereo_is_not_silent():
    x = 0.3 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR)
    peak, active = silence_level(np.stack([x, -x], axis=1), SR)
    assert peak > -20 and active > 0.9
    assert silence_level(np.zeros((0, 2)), SR) == (float("-inf"), 0.0)


def test_missing_prompt_is_an_input_error(tmp_path, monkeypatch):
    from sonicproof.errors import InputError

    monkeypatch.setenv("SONICPROOF_CACHE", str(tmp_path / "cache"))
    p = _wav(tmp_path / "x.wav", np.zeros(SR))
    with pytest.raises(InputError) as e:
        Judge(embedder=FakeEmbedder()).check(p)
    assert e.value.code == "no_prompt" and e.value.file == "x.wav"


def test_judge_refuses_truncated_embeddings():
    with pytest.raises(ValueError, match="768"):
        Judge(embedder=FakeEmbedder(dim=256))


@pytest.mark.skipif(not model_cached(), reason="EmbeddingGemma 2 not downloaded")
def test_check_prefers_matching_prompt(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("SONICPROOF_CACHE", str(tmp_path / "cache"))
    from sonicproof.model import prefer_offline

    prefer_offline()
    sr = 16_000
    t = np.arange(sr * 3) / sr
    beep = 0.3 * np.sin(2 * np.pi * 1000 * t) * (np.sin(2 * np.pi * 2 * t) > 0)  # on/off 1 kHz beeps
    p = tmp_path / "beeps.wav"
    sf.write(p, beep, sr)
    j = Judge()
    good = j.check(p, "a high pitched electronic beeping tone")
    bad = j.check(p, "a large orchestra playing a symphony")
    assert good.score > bad.score
    assert np.isfinite(good.score) and good.heard_as
    again = j.check(p, "a high pitched electronic beeping tone")  # served from the embedding cache
    assert again.score == pytest.approx(good.score, abs=1e-5)


@pytest.mark.skipif(not model_cached(), reason="EmbeddingGemma 2 not downloaded")
def test_fit_scores_an_image_and_matches_the_vectorised_path(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("SONICPROOF_CACHE", str(tmp_path / "cache"))
    from PIL import Image

    from sonicproof.model import prefer_offline

    prefer_offline()
    sr = 16_000
    p = tmp_path / "hum.wav"
    sf.write(p, 0.2 * np.sin(2 * np.pi * 120 * np.arange(sr * 2) / sr), sr)
    img = tmp_path / "frame.png"
    Image.new("RGB", (64, 64), (20, 40, 200)).save(img)
    j = Judge(vision=True)
    r = j.fit(p, img, "image", scene="a calm blue screen", n_alt=3)
    assert np.isfinite(r.score) and r.verdict in ("strong", "plausible", "weak", "miss")
    assert len(r.pictured_as) == 3 and len(r.heard_as) == 3 and r.z_scene is not None
    t = j.text(["a calm blue screen"])[0]
    many = j.fit_scores(j.image(img), np.stack([j.audio(p)]), t)
    assert many["score"][0] == pytest.approx(r.score, abs=1e-5)


def test_nan_samples_do_not_make_a_clip_silent():
    from sonicproof.judge import is_silent_level, silence_level

    x = (0.3 * np.sin(2 * np.pi * 880 * np.arange(SR) / SR)).astype(np.float32)
    x[::50] = np.nan
    assert not is_silent_level(*silence_level(x, SR))
