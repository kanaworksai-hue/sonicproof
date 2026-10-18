"""service.Engine and the shared error / record helpers, with a stand-in embedder (no model needed)."""

import hashlib
import threading
import time

import numpy as np
import pytest
import soundfile as sf

from sonicproof import service
from sonicproof.errors import InputError, SetupError
from sonicproof.judge import Judge

SR = 16_000


class FakeEmbedder:
    """Deterministic vectors: text by hash, audio by its dominant frequency band (so tones differ)."""

    dim = 768

    def __init__(self, device=None, vision=False):
        self.vision = vision

    def embed_text(self, texts, prompt_name=None):
        out = []
        for t in texts:
            seed = int(hashlib.sha1(t.encode()).hexdigest()[:8], 16)
            out.append(np.random.default_rng(seed).standard_normal(768))
        X = np.array(out, np.float32)
        return X / np.linalg.norm(X, axis=1, keepdims=True)

    def embed_audio(self, waves):
        out = []
        for w in waves:
            spec = np.abs(np.fft.rfft(w[:SR]))
            band = int(np.argmax(spec) // 50)
            out.append(np.random.default_rng(band).standard_normal(768))
        X = np.array(out, np.float32)
        return X / np.linalg.norm(X, axis=1, keepdims=True)


class FakeEngine(service.Engine):
    def judge(self, vision=False):
        if self._judge is None:
            self._judge = Judge(embedder=FakeEmbedder())
        return self._judge


def tone(path, f=500, seconds=1.5, amp=0.3):
    t = np.arange(int(SR * seconds)) / SR
    sf.write(path, (amp * np.sin(2 * np.pi * f * t) * np.exp(-2 * t)).astype(np.float32), SR)
    return path


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SONICPROOF_CACHE", str(tmp_path / "cache"))


# -- errors ---------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "exc, code, exit_code",
    [
        (ModuleNotFoundError("No module named 'torch'", name="torch"), "missing_extra", 3),
        (FileNotFoundError(2, "No such file", "x.wav"), "not_found", 2),
        (IsADirectoryError(21, "Is a directory", "d"), "is_dir", 2),
        (ValueError("bad\nsecond line"), "bad_value", 2),
        (RuntimeError("ffmpeg failed on x.m4a: long\nnoise"), "unreadable", 2),
        (KeyError("boom"), "internal", 2),
    ],
)
def test_as_error_gives_a_code_and_one_line(exc, code, exit_code):
    e = service.as_error(exc)
    assert (e.code, e.exit_code) == (code, exit_code)
    assert "\n" not in e.message
    assert set(e.to_dict()) == {"code", "message", "file"}


def test_probe_audio_rejects_folders_empty_and_garbage(tmp_path):
    (tmp_path / "empty.wav").write_bytes(b"")
    (tmp_path / "junk.wav").write_bytes(b"not audio at all" * 10)
    for name, code in (
        ("nope.wav", "not_found"),
        (".", "is_dir"),
        ("empty.wav", "empty"),
        ("junk.wav", "unreadable"),
    ):
        with pytest.raises(InputError) as ei:
            service.probe_audio(tmp_path / name)
        assert ei.value.code == code
    assert service.probe_audio(tone(tmp_path / "ok.wav")).name == "ok.wav"


# -- batches ----------------------------------------------------------------------------------------
def test_qc_and_rank_keep_the_batch_when_one_file_is_bad(tmp_path):
    e = FakeEngine()
    good = tone(tmp_path / "a.wav")
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"\0" * 100)
    recs = e.qc([good, bad])
    assert [r["status"] != "error" for r in recs] == [True, False]
    assert recs[1]["checks"] == [] and recs[1]["error"]["code"] in ("unreadable", "empty")
    assert service.worst_status(recs) == "error"
    ranked = e.rank("a beep", [bad, good])
    assert ranked[0]["file"] == str(good) and ranked[-1]["verdict"] == "error"
    with pytest.raises(InputError):
        e.qc([])
    with pytest.raises(InputError):
        e.rank("a beep", [])


def test_engine_expands_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    tone(tmp_path / "x.wav")
    assert FakeEngine().qc(["~/x.wav"])[0]["status"] != "error"
    assert FakeEngine().check("~/x.wav", "a beep")["verdict"]


def test_search_and_fit_argument_errors(tmp_path):
    e = FakeEngine()
    a = tone(tmp_path / "a.wav")
    with pytest.raises(InputError):
        e.search(tmp_path, query="rain", like=a)
    with pytest.raises(InputError):
        e.search(tmp_path)
    with pytest.raises(InputError) as ei:
        e.fit(a)  # no picture, and a .wav has none of its own
    assert ei.value.code == "bad_value"
    with pytest.raises(InputError):
        e.check(a)  # no prompt, no sidecar
    with pytest.raises(InputError):
        e.dedupe(tmp_path, match=0.1)


# -- model loading ----------------------------------------------------------------------------------
def _counting_embedder(monkeypatch, delay=0.0):
    made = []

    class Slow(FakeEmbedder):
        def __init__(self, device=None, vision=False):
            time.sleep(delay)
            super().__init__(device, vision)
            made.append(vision)

    import sonicproof.judge as judge_mod
    import sonicproof.model as model_mod

    monkeypatch.setattr(model_mod, "Embedder", Slow)
    monkeypatch.setattr(service, "require_model", lambda allow_download=True: None)

    def no_fallback(*a, **k):
        raise AssertionError("a Judge fell back to get_embedder(): a third model would be loaded")

    monkeypatch.setattr(judge_mod, "get_embedder", no_fallback)
    return made


def test_model_swap_reuses_the_new_embedder(monkeypatch):
    made = _counting_embedder(monkeypatch)
    e = service.Engine()
    old = e.judge()
    new = e.judge(vision=True)
    assert old.embedder is new.embedder  # a call still holding the old Judge keeps working, no reload
    assert e.judge() is new and e.loaded == "vision"
    assert made == [False, True]


def test_concurrent_first_calls_load_the_model_once(monkeypatch):
    made = _counting_embedder(monkeypatch, delay=0.2)
    e = service.Engine()
    got = []
    ts = [threading.Thread(target=lambda: got.append(e.judge())) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert made == [False] and len({id(j) for j in got}) == 1


def test_no_download_inside_a_call(tmp_path, monkeypatch):
    import sonicproof.model as model_mod

    monkeypatch.setattr(model_mod, "require_model_extra", lambda: None)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("SONICPROOF_ALLOW_DOWNLOAD", raising=False)
    e = service.Engine(allow_download=False)
    with pytest.raises(SetupError) as ei:
        e.judge()
    assert ei.value.code == "model_unavailable" and "doctor --load" in ei.value.message
