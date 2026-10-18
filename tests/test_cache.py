"""Embedding cache: location, keys and concurrent use — no model needed."""

import threading
from pathlib import Path

import numpy as np

from sonicproof import cache


def test_cache_dir_follows_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SONICPROOF_CACHE", str(tmp_path / "sp"))
    assert cache.cache_dir() == tmp_path / "sp"
    monkeypatch.delenv("SONICPROOF_CACHE")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(cache.sys, "platform", "linux")
    assert cache.cache_dir() == tmp_path / "xdg" / "sonicproof"


def test_windows_default_is_localappdata(tmp_path, monkeypatch):
    monkeypatch.delenv("SONICPROOF_CACHE", raising=False)
    monkeypatch.setattr(cache.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    assert Path(cache._default_root()) == tmp_path / "Local" / "sonicproof" / "Cache"


def test_key_depends_on_model_version_and_embedding_tag(monkeypatch):
    base = cache._key("audio", "abc")
    assert base == cache._key("audio", "abc")
    assert base != cache._key("audio", "abd")
    monkeypatch.setattr(cache, "MODEL_ID", "someone/other-model")
    assert cache._key("audio", "abc") != base
    monkeypatch.undo()
    monkeypatch.setattr(cache, "CACHE_VERSION", cache.CACHE_VERSION + 1)
    assert cache._key("audio", "abc") != base
    monkeypatch.undo()
    monkeypatch.setattr(cache, "embed_tag", lambda: "d768-bfloat16-deadbeef")
    assert cache._key("audio", "abc") != base  # dtype / snapshot change -> embeddings recomputed
    thumb = cache._key("thumb_peaks", "abc")
    monkeypatch.undo()
    assert cache._key("thumb_peaks", "abc") == thumb  # non-embedding kinds don't care


def test_save_load_is_safe_across_threads(tmp_path, monkeypatch):
    monkeypatch.setenv("SONICPROOF_CACHE", str(tmp_path / "c"))
    arr = np.arange(4096, dtype=np.float32)
    errors = []

    def work():
        try:
            for _ in range(150):
                cache.save_array(arr, "audio", "same")
                got = cache.load_array("audio", "same")
                assert got is None or np.array_equal(got, arr)
        except Exception as e:  # pragma: no cover - the failure we are testing for
            errors.append(e)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert np.array_equal(cache.load_array("audio", "same"), arr)
    assert not [p for p in (tmp_path / "c" / "audio").iterdir() if p.suffix == ".tmp"]


def test_corrupt_entry_is_a_miss(tmp_path, monkeypatch):
    monkeypatch.setenv("SONICPROOF_CACHE", str(tmp_path / "c"))
    cache.save_array(np.ones(3), "audio", "x")
    p = tmp_path / "c" / "audio" / f"{cache._key('audio', 'x')}.npy"
    p.write_bytes(p.read_bytes()[:20])  # truncated
    assert cache.load_array("audio", "x") is None
    p.write_bytes(b"")
    assert cache.load_array("audio", "x") is None
    assert cache.load_array("audio", "never-saved") is None


def test_non_finite_embeddings_are_never_cached_or_served(tmp_path, monkeypatch):
    import numpy as np

    from sonicproof import cache

    monkeypatch.setenv("SONICPROOF_CACHE", str(tmp_path))
    bad = np.full(4, np.nan, np.float32)
    cache.save_array(bad, "audio", "x")
    assert cache.load_array("audio", "x") is None
    d = cache.cache_dir() / "audio"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / (cache._key("audio", "y") + ".npy"), bad)  # written by an older version
    assert cache.load_array("audio", "y") is None
