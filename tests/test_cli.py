"""CLI: --json shapes, error objects and exit codes, with a stand-in embedder (no model needed)."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf
from test_service import SR, FakeEmbedder, tone
from typer.testing import CliRunner

from sonicproof import cli, service
from sonicproof.judge import Judge


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setenv("SONICPROOF_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(cli, "_judge", lambda vision=False: Judge(embedder=FakeEmbedder(vision=vision)))
    runner = CliRunner()

    def invoke(*args, **kw):
        return runner.invoke(cli.app, [str(a) for a in args], **kw)

    return invoke


def _err(r) -> dict:
    return json.loads(r.stdout)["error"]


def test_check_json_and_errors(run, tmp_path):
    a = tone(tmp_path / "a.wav")
    r = run("check", a, "a beep", "--json")
    assert r.exit_code == 0
    assert set(json.loads(r.stdout)) >= {"file", "prompt", "score", "verdict", "heard_as", "cosine"}
    r = run("check", a, "--json")  # no prompt, no sidecar
    assert r.exit_code == 2 and _err(r)["code"] == "no_prompt"
    r = run("check", tmp_path / "nope.wav", "x", "--json")
    assert r.exit_code == 2 and _err(r)["code"] == "not_found" and "nope.wav" in _err(r)["message"]
    r = run("check", tmp_path, "x", "--json")
    assert r.exit_code == 2 and _err(r)["code"] == "is_dir"
    assert run("check", a, "x", "--top", "-1").exit_code == 2


def test_rank_puts_unreadable_takes_last(run, tmp_path):
    a, b = tone(tmp_path / "a.wav", 500), tone(tmp_path / "b.wav", 3000)
    (tmp_path / "bad.wav").write_bytes(b"x" * 64)
    r = run("rank", "a beep", a, b, "--json")
    assert r.exit_code == 0 and len(json.loads(r.stdout)) == 2
    r = run("rank", "a beep", tmp_path / "bad.wav", a, "--json")
    out = json.loads(r.stdout)
    assert r.exit_code == 2 and out[-1]["verdict"] == "error" and out[0]["file"] == str(a)


def test_qc_json_is_always_a_list_and_exit_codes(run, tmp_path):
    a = tone(tmp_path / "a.wav")
    r = run("qc", a, "--json")
    assert r.exit_code == 0 and json.loads(r.stdout) == [service.qc_record(a)]
    sf.write(tmp_path / "quiet.wav", np.zeros(SR * 2, np.float32), SR)
    r = run("qc", a, tmp_path / "quiet.wav", "--json")
    assert r.exit_code == 1 and [x["status"] for x in json.loads(r.stdout)][1] == "fail"
    (tmp_path / "bad.wav").write_bytes(b"")
    r = run("qc", a, tmp_path / "bad.wav", tmp_path / "quiet.wav", "--json")
    out = json.loads(r.stdout)
    assert r.exit_code == 2 and out[1]["status"] == "error" and out[1]["checks"] == [] and len(out) == 3
    for bad in ("nan", "0", "-100"):
        assert run("qc", a, "--target-lufs", bad).exit_code == 2


def test_report_strict(run, tmp_path):
    a = tone(tmp_path / "a.wav")
    r = run("report", a, "a beep", "--json")
    rec = json.loads(r.stdout)
    assert r.exit_code == 0
    assert set(rec) == {"file", "overall", "note", "adherence", "qc", "duration", "sample_rate", "channels"}
    r = run("report", a, "a beep", "--json", "--strict")
    assert r.exit_code == (0 if rec["overall"] == "ship" else 1)


def test_library_commands(run, tmp_path, monkeypatch):
    # index embeds through the patched _judge only on the "not downloaded yet" path; force it so the test
    # never loads the real model (and behaves the same with or without the model extra)
    monkeypatch.setattr("sonicproof.model.model_cached", lambda *a, **k: False)
    lib = tmp_path / "lib"
    lib.mkdir()
    tone(lib / "a.wav", 500)
    tone(lib / "b.wav", 3000)
    (lib / "bad.wav").write_bytes(b"x" * 64)
    r = run("index", lib, "--json")
    rec = json.loads(r.stdout)
    assert r.exit_code == 0 and rec["files"] == 2 and len(rec["failed"]) == 1
    r = run("search", "a beep", "--in", lib, "--json")
    out = json.loads(r.stdout)
    assert r.exit_code == 0 and out["mode"] == "words" and len(out["results"]) == 2
    r = run("search", "--like", lib / "a.wav", "--in", lib, "--json")
    assert r.exit_code == 0 and json.loads(r.stdout)["mode"] == "like"
    r = run("dedupe", lib, "--json")
    assert r.exit_code == 0 and set(json.loads(r.stdout)) >= {"library", "files", "groups", "failed"}
    assert run("search", "rain", "--json").exit_code == 2  # --in is required
    assert run("search", "rain", "--like", lib / "a.wav", "--in", lib).exit_code == 2  # two modes
    assert run("index").exit_code == 2 and run("dedupe").exit_code == 2  # DIR is required
    assert run("dedupe", lib, "--match", "0").exit_code == 2
    r = run("search", "rain", "--in", "/", "--json")
    assert r.exit_code == 2 and _err(r)["code"] == "bad_value"


def test_unexpected_errors_are_one_line_unless_debug(run, tmp_path, monkeypatch):
    a = tone(tmp_path / "a.wav")

    def boom(vision=False):
        raise KeyError("surprise")

    monkeypatch.setattr(cli, "_judge", boom)
    r = run("check", a, "x", "--json")
    assert r.exit_code == 2 and _err(r)["code"] == "internal" and "Traceback" not in r.output
    monkeypatch.setattr(cli, "_debug", False)
    r = run("--debug", "check", a, "x")
    assert isinstance(r.exception, KeyError)


def test_missing_model_extra_is_a_setup_error(run, tmp_path, monkeypatch):
    a = tone(tmp_path / "a.wav")

    def no_torch(vision=False):
        raise ModuleNotFoundError("No module named 'torch'", name="torch")

    monkeypatch.setattr(cli, "_judge", no_torch)
    r = run("check", a, "x", "--json")
    assert (
        r.exit_code == 3 and _err(r)["code"] == "missing_extra" and "sonicproof[model]" in _err(r)["message"]
    )


def test_doctor_load_reports_the_model_it_just_downloaded(run, monkeypatch):
    import types

    import sonicproof.model as model

    downloaded = {"done": False}

    def fake_judge(vision=False):
        downloaded["done"] = True  # what loading the model for the first time does
        return Judge(embedder=FakeEmbedder())

    monkeypatch.setattr(cli, "_judge", fake_judge)
    monkeypatch.setattr(model, "require_model_extra", lambda: None)
    monkeypatch.setattr(model, "pick_device", lambda: "cpu")
    monkeypatch.setattr(model, "model_cached", lambda: downloaded["done"])
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(__version__="0.0"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    r = run("doctor", env={"COLUMNS": "200"})
    assert r.exit_code == 0 and "(not downloaded)" in r.stdout and "Inference" not in r.stdout
    downloaded["done"] = False
    r = run("doctor", "--load", env={"COLUMNS": "200"})
    assert r.exit_code == 0 and "Inference" in r.stdout
    assert "(cached)" in r.stdout and "not downloaded" not in r.stdout


@pytest.mark.parametrize("cuda", [None, "12.8"])
def test_doctor_warns_when_a_cpu_only_pytorch_leaves_the_nvidia_gpu_idle(run, monkeypatch, cuda):
    import shutil
    import types

    import sonicproof.model as model

    monkeypatch.setattr(model, "require_model_extra", lambda: None)
    monkeypatch.setattr(model, "pick_device", lambda: "cpu")
    monkeypatch.setattr(model, "model_cached", lambda: True)
    fake = types.SimpleNamespace(__version__="2.14.1", version=types.SimpleNamespace(cuda=cuda))
    monkeypatch.setitem(sys.modules, "torch", fake)
    real_which = shutil.which
    monkeypatch.setattr(shutil, "which", lambda n: "nvidia-smi" if n == "nvidia-smi" else real_which(n))
    r = run("doctor", env={"COLUMNS": "200"})
    assert r.exit_code == 0
    assert ("CPU-only" in r.stdout) == (cuda is None and sys.platform != "darwin")  # Macs have no NVIDIA GPU


def test_qc_table_fits_80_columns(run, tmp_path):
    a = tone(tmp_path / "a.wav", seconds=0.8, amp=0.99)  # cut-off tail etc: long hints
    r = run("qc", a, env={"COLUMNS": "80"})
    lines = r.stdout.splitlines()
    assert lines and max(len(x) for x in lines) <= 80
    assert all("…" not in x for x in lines)
    assert any("fade" in x for x in lines)  # the hint is there, readable


def test_skill_install_and_show(run, tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))  # Path.home() on Windows
    dest = home / ".claude" / "skills" / "sonicproof" / "SKILL.md"
    assert run("skill", "install").exit_code == 0 and dest.read_text(encoding="utf-8") == cli._skill_text()
    assert run("skill", "install").exit_code == 0  # already up to date
    dest.write_text("edited", encoding="utf-8")
    assert run("skill", "install").exit_code == 2 and dest.read_text(encoding="utf-8") == "edited"
    assert run("skill", "install", "--force").exit_code == 0 and dest.read_text(encoding="utf-8") != "edited"
    monkeypatch.chdir(tmp_path)
    assert run("skill", "install", "--project").exit_code == 0
    assert (tmp_path / ".claude" / "skills" / "sonicproof" / "SKILL.md").exists()
    r = run("skill", "show")
    assert r.exit_code == 0 and r.stdout.startswith("---") and "name: sonicproof" in r.stdout


def _py(code_or_args, env_extra, tmp_path):
    env = {**os.environ, **env_extra, "SONICPROOF_CACHE": str(tmp_path / "cache")}
    return subprocess.run([sys.executable, *code_or_args], capture_output=True, env=env, cwd=tmp_path)


def test_non_utf8_stdout_does_not_crash(tmp_path):
    a = tone(tmp_path / "木の扉.wav")
    for extra in ([], ["--json"]):
        p = _py(["-m", "sonicproof.cli", "qc", str(a), *extra], {"PYTHONIOENCODING": "cp1252"}, tmp_path)
        assert p.returncode == 0, p.stderr.decode(errors="replace")
        assert "木の扉" in p.stdout.decode("utf-8")


def test_empty_hf_home_counts_as_unset(tmp_path):
    names = ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE")
    code = f"import os, sonicproof.cli; print(*(v in os.environ for v in {names!r}))"
    p = _py(["-c", code], dict.fromkeys(names, ""), tmp_path)
    assert p.stdout.decode().split() == ["False", "False", "False"]


def test_check_json_is_strict_for_audio_with_nan_samples(tmp_path, monkeypatch):
    import math

    import numpy as np
    import soundfile as sf

    from sonicproof import cli

    x = (0.3 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).astype(np.float32)
    x[::16] = np.nan
    p = tmp_path / "nan.wav"
    sf.write(p, x, 16000, subtype="FLOAT")
    assert (
        cli._json({"a": math.nan, "b": [math.inf, 1.0]})
        == '{\n  "a": null,\n  "b": [\n    null,\n    1.0\n  ]\n}'
    )
    from sonicproof.audio import load

    m = load(p).mono()
    assert np.isfinite(m).all()  # the model never sees NaN
