"""End to end through the CLI with the real model and ffmpeg: compressed audio, video soundtracks and frames,
library search. Skipped unless torch, the downloaded model and ffmpeg are all present (CI runs it on CPU)."""

import json
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from sonicproof import cli
from sonicproof.model import model_cached

pytestmark = [
    pytest.mark.skipif(not model_cached(), reason="EmbeddingGemma 2 not downloaded"),
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg"),
]

SR = 16_000
VERDICTS = {"strong", "plausible", "weak", "miss"}


@pytest.fixture(scope="module")
def judges():
    pytest.importorskip("torch")
    return {}


@pytest.fixture
def run(tmp_path, monkeypatch, judges):
    monkeypatch.setenv("SONICPROOF_CACHE", str(tmp_path / "cache"))
    real = cli._judge

    def cached(vision=False):  # load each model once for the whole module
        if vision not in judges:
            judges[vision] = real(vision=vision)
        return judges[vision]

    monkeypatch.setattr(cli, "_judge", cached)
    runner = CliRunner()

    def invoke(*args):
        r = runner.invoke(cli.app, [str(a) for a in args])
        assert r.exit_code == 0, r.output
        return json.loads(r.stdout)

    return invoke


def _ffmpeg(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *map(str, args)], check=True)


def _beeps(path, seconds=3.0):
    t = np.arange(int(SR * seconds)) / SR
    sf.write(path, 0.3 * np.sin(2 * np.pi * 1000 * t) * (np.sin(2 * np.pi * 2 * t) > 0), SR)
    return path


def _noise(path, seconds=3.0):
    sf.write(path, 0.2 * np.random.default_rng(0).standard_normal(int(SR * seconds)), SR)
    return path


def test_check_and_rank_compressed_audio(run, tmp_path):
    mp3, m4a = tmp_path / "beeps.mp3", tmp_path / "noise.m4a"
    _ffmpeg("-i", _beeps(tmp_path / "beeps.wav"), mp3)
    _ffmpeg("-i", _noise(tmp_path / "noise.wav"), m4a)
    prompt = "a high pitched electronic beeping tone"
    wav = run("check", tmp_path / "beeps.wav", prompt, "--json")
    r = run("check", mp3, prompt, "--json")
    assert r["verdict"] in VERDICTS and r["heard_as"]
    assert r["score"] == pytest.approx(wav["score"], abs=0.5)  # lossy copy, same sound
    ranked = run("rank", prompt, m4a, mp3, "--json")
    assert [x["file"] for x in ranked] == [str(mp3), str(m4a)]


def test_fit_a_video_with_its_own_soundtrack(run, tmp_path):
    clip = tmp_path / "clip.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=10:duration=3",
        "-i", _beeps(tmp_path / "beeps.wav"),
        "-shortest", "-pix_fmt", "yuv420p", "-c:a", "aac", clip,
    )  # fmt: skip
    r = run("fit", clip, "a test pattern on a screen", "--json")
    assert r["picture_kind"] == "video" and r["frames"] >= 2
    assert r["verdict"] in VERDICTS and np.isfinite(r["score"]) and r["pictured_as"]


def test_search_a_library_by_words_and_by_example(run, tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    _ffmpeg("-i", _beeps(tmp_path / "beeps.wav"), lib / "beeps.mp3")
    _noise(lib / "noise.wav")
    idx = run("index", lib, "--json")
    assert idx["files"] == 2 and not idx["failed"]
    hits = run("search", "electronic beeping", "--in", lib, "--json")
    assert hits["results"][0]["path"] == "beeps.mp3"
    like = run("search", "--like", tmp_path / "beeps.wav", "--in", lib, "--json")
    assert like["results"][0]["path"] == "beeps.mp3"
