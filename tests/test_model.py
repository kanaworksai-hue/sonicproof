"""Model setup checks (extras, cache lookup, device) — no model or torch needed."""

import importlib.util
import sys
import types

import pytest

from sonicproof import model
from sonicproof.errors import SetupError


def _snapshot(hub, model_id=model.MODEL_ID, rev="abc123def4567890"):
    snap = hub / f"models--{model_id.replace('/', '--')}" / "snapshots" / rev
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    return snap


@pytest.fixture
def clean_env(monkeypatch):
    for k in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(k, raising=False)
    for k in ("HF_HUB_OFFLINE", "SONICPROOF_DEVICE"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_hub_cache_dir_resolution_order(tmp_path, clean_env):
    clean_env.setenv("HOME", str(tmp_path / "home"))
    clean_env.setenv("USERPROFILE", str(tmp_path / "home"))  # Path.home() on Windows
    assert model.hub_cache_dir() == tmp_path / "home" / ".cache" / "huggingface" / "hub"
    clean_env.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert model.hub_cache_dir() == tmp_path / "xdg" / "huggingface" / "hub"
    clean_env.setenv("HF_HOME", "")  # copied from `-e HF_HOME="$HF_HOME"` with HF_HOME unset
    assert model.hub_cache_dir() == tmp_path / "xdg" / "huggingface" / "hub"
    clean_env.setenv("HF_HOME", str(tmp_path / "hf"))
    assert model.hub_cache_dir() == tmp_path / "hf" / "hub"
    clean_env.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "legacy"))  # older name, still read by the hub
    assert model.hub_cache_dir() == tmp_path / "legacy"
    clean_env.setenv("HF_HUB_CACHE", str(tmp_path / "hubcache"))
    assert model.hub_cache_dir() == tmp_path / "hubcache"


def test_model_cached_and_revision(tmp_path, clean_env):
    clean_env.setenv("HF_HOME", str(tmp_path / "hf"))
    assert not model.model_cached() and model.model_revision() == "none"
    _snapshot(tmp_path / "hf" / "hub")
    assert model.model_cached()
    assert model.model_revision() == "abc123def456"


def test_missing_extra(monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(model.importlib.util, "find_spec", lambda n: None if n == "torch" else real(n))
    with pytest.raises(SetupError) as e:
        model.require_model_extra()
    assert e.value.code == "missing_extra" and "sonicproof[model]" in e.value.message
    with pytest.raises(SetupError, match="model extra"):
        model.Embedder()  # fails before importing torch
    with pytest.raises(SetupError):
        model.pick_device()


def test_offline_without_cache_is_model_unavailable(tmp_path, clean_env):
    clean_env.setenv("HF_HOME", str(tmp_path / "empty"))
    clean_env.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(SetupError) as e:
        model.require_model_available()
    assert e.value.code == "model_unavailable"
    assert "doctor --load" in e.value.message and "1.5 GB" in e.value.message
    _snapshot(tmp_path / "empty" / "hub")
    model.require_model_available()  # cached: fine offline


def _fake_torch(cuda=False, mps=False, devices=1):
    t = types.ModuleType("torch")

    class Device:
        def __init__(self, s):
            kind, _, idx = s.partition(":")
            if kind not in ("cpu", "cuda", "mps") or (idx and not idx.isdigit()):
                raise RuntimeError(f"Expected one of cpu, cuda, mps device type: {s}")
            self.type, self.index = kind, int(idx) if idx else None

    t.device = Device
    t.cuda = types.SimpleNamespace(is_available=lambda: cuda, device_count=lambda: devices if cuda else 0)
    t.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))
    return t


@pytest.mark.parametrize(
    ("cuda", "mps", "want"), [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")]
)
def test_pick_device_fallback_order(clean_env, cuda, mps, want):
    clean_env.setitem(sys.modules, "torch", _fake_torch(cuda, mps))
    clean_env.setattr(model, "require_model_extra", lambda: None)
    assert model.pick_device() == want


@pytest.mark.parametrize(("env", "ok"), [("cpu", True), ("mps", True), ("cuda", False), ("gpu", False)])
def test_pick_device_validates_env(clean_env, env, ok):
    clean_env.setitem(sys.modules, "torch", _fake_torch(cuda=False, mps=True))
    clean_env.setattr(model, "require_model_extra", lambda: None)
    clean_env.setenv("SONICPROOF_DEVICE", env)
    if ok:
        assert model.pick_device() == env
    else:
        with pytest.raises(SetupError) as e:
            model.pick_device()
        assert e.value.code == "bad_device" and env in e.value.message


def test_embed_tag_mentions_size_dtype_and_snapshot(tmp_path, clean_env):
    clean_env.setenv("HF_HOME", str(tmp_path / "hf"))
    clean_env.setenv("SONICPROOF_DEVICE", "cpu")
    model._dtype_name.cache_clear()
    try:
        assert model.embed_tag() == "d768-float32-none"
    finally:
        model._dtype_name.cache_clear()
