import json
import shutil

import numpy as np
import pytest
import soundfile as sf

from sonicproof import audio
from sonicproof.audio import MODEL_SR, Audio, is_silent, load, load_for_model, read_sidecar, sidecar_prompt
from sonicproof.errors import InputError, SetupError
from sonicproof.model import split_windows


def _tone(path, sr=44_100, seconds=1.0, channels=2):
    t = np.arange(int(sr * seconds)) / sr
    x = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(path, np.stack([x] * channels, axis=1), sr, subtype="PCM_24")


def test_load_native_and_model_rate(tmp_path):
    p = tmp_path / "tone.wav"
    _tone(p)
    a = load(p)
    assert a.sr == 44_100 and a.channels == 2 and abs(a.duration - 1.0) < 1e-3
    m = load_for_model(p)
    assert m.ndim == 1 and abs(len(m) - MODEL_SR) <= 1
    assert m.dtype == np.float32


def test_sidecar_prompt_stable_audio_sfx(tmp_path):
    p = tmp_path / "x.wav"
    _tone(p)
    meta = {
        "prompt": "",
        "original_idea": "my notes",
        "sound_fields": {"source": "glass bottle", "action": "shattering", "material": "", "space": "dry"},
        "prompt_used": "TrackType: SFX, glass bottle, shattering, dry.",
    }
    (tmp_path / "x.wav.json").write_text(json.dumps(meta))
    assert sidecar_prompt(p) == "glass bottle, shattering"


def test_sidecar_prompt_strips_tracktype(tmp_path):
    p = tmp_path / "m.wav"
    _tone(p)
    (tmp_path / "m.wav.json").write_text(
        json.dumps({"prompt_used": "TrackType: Music, VocalType: Instrumental, Lo-fi piano. BPM: 80."})
    )
    assert sidecar_prompt(p) == "Lo-fi piano. BPM: 80."


def test_no_sidecar(tmp_path):
    p = tmp_path / "n.wav"
    _tone(p)
    assert sidecar_prompt(p) is None


def test_split_windows_pads_and_splits():
    assert len(split_windows(np.zeros(100, np.float32))[0]) == MODEL_SR  # padded to 1 s
    w = np.zeros(int(65 * MODEL_SR), np.float32)
    assert [len(x) // MODEL_SR for x in split_windows(w)] == [30, 30, 5]
    w = np.zeros(int(60.5 * MODEL_SR), np.float32)
    assert len(split_windows(w)) == 2  # 0.5 s tail dropped


# --- errors, sidecars, silence ----------------------------------------------------------------------------


def _code(path):
    with pytest.raises(InputError) as e:
        load(path)
    assert e.value.file == str(path)
    return e.value.code, str(e.value)


def test_load_errors_are_classified(tmp_path):
    assert _code(tmp_path / "missing.wav")[0] == "not_found"
    assert _code(tmp_path)[0] == "is_dir"
    (tmp_path / "zero.wav").write_bytes(b"")
    code, msg = _code(tmp_path / "zero.wav")
    assert code == "empty" and "zero.wav" in msg


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_corrupt_file_is_unreadable_with_short_message(tmp_path):
    p = tmp_path / "junk.wav"
    p.write_bytes(np.random.default_rng(0).bytes(4000))
    code, msg = _code(p)
    assert code == "unreadable" and "junk.wav" in msg and "\n" not in msg and "ffmpeg:" not in msg


def test_missing_ffmpeg_is_a_setup_error(tmp_path, monkeypatch):
    p = tmp_path / "junk.m4a"
    p.write_bytes(b"not audio at all" * 50)
    monkeypatch.setattr(audio.shutil, "which", lambda name: None)
    with pytest.raises(SetupError) as e:
        load(p)
    assert e.value.code == "needs_ffmpeg" and "junk.m4a" in str(e.value)


def test_ffmpeg_no_stream_is_no_audio(tmp_path, monkeypatch):
    import subprocess

    p = tmp_path / "clip.mp4"
    p.write_bytes(b"\0" * 64)

    def fake_run(cmd, **kw):
        err = b"[out#0/f32le] Output file does not contain any stream\n"
        return subprocess.CompletedProcess(cmd, 1, b"", err)

    monkeypatch.setattr(audio.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(audio, "_ffprobe", lambda path, entry: None)
    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    code, msg = _code(p)
    assert code == "no_audio" and msg == "clip.mp4 has no audio track."


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_ffmpeg_fallback_keeps_rate_and_channels(tmp_path):
    import subprocess

    src, m4a = tmp_path / "t.wav", tmp_path / "t.m4a"
    _tone(src, sr=22_050, seconds=0.5, channels=2)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), str(m4a)], check=True)
    a = load(m4a)
    assert a.sr == 22_050 and a.channels == 2 and abs(a.duration - 0.5) < 0.1


def test_zero_frame_wav_loads_and_model_input_is_empty(tmp_path):
    p = tmp_path / "empty.wav"
    sf.write(p, np.zeros((0, 2), np.float32), 48_000)
    assert len(load(p).data) == 0
    assert load_for_model(p).shape == (0,)


def test_sidecar_shapes(tmp_path):
    p = tmp_path / "a.wav"
    _tone(p)
    side = tmp_path / "a.wav.json"
    for bad in ("[1, 2]", "3", '"text"', "null", "{not json"):
        side.write_text(bad)
        assert read_sidecar(p) is None and sidecar_prompt(p) is None
    side.write_text(json.dumps({"prompt": None, "text": "rain on a tin roof", "sound_fields": "oops"}))
    assert sidecar_prompt(p) == "rain on a tin roof"


def test_stem_json_needs_prompt_fields(tmp_path):
    p = tmp_path / "b.wav"
    _tone(p)
    stem = tmp_path / "b.json"
    stem.write_text(json.dumps({"version": 3, "items": []}))  # some unrelated JSON with the same name
    assert read_sidecar(p) is None
    stem.write_text(json.dumps({"seed": 1, "sound_source": "dog"}))
    assert read_sidecar(p) == {"seed": 1, "sound_source": "dog"}
    stem.write_text(json.dumps({"prompt": "dog barking"}))
    assert sidecar_prompt(p) == "dog barking"
    (tmp_path / "b.wav.json").write_text(json.dumps({"prompt": "exact wins"}))
    assert sidecar_prompt(p) == "exact wins"


def test_is_silent():
    sr = 16_000
    assert is_silent(np.zeros(0, np.float32))
    assert is_silent(np.zeros(sr, np.float32))
    assert is_silent(np.full(sr, 1e-4, np.float32))  # -80 dBFS
    blip = np.zeros(sr, np.float32)
    blip[: int(sr * 0.05)] = 0.5  # a 50 ms click is short, not silent
    assert not is_silent(blip)
    tone = 0.1 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
    assert not is_silent(tone)
    assert not is_silent(Audio(np.stack([tone, -tone], axis=1), sr))  # out of phase does not cancel
    assert is_silent(Audio(np.zeros((44_100, 2), np.float32), 44_100))
    nan = tone.copy()
    nan[::2] = np.nan
    assert not is_silent(nan)
