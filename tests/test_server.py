"""Workbench API and the shared service layer, with a stand-in embedder (no model needed)."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from sonicproof import service
from sonicproof.judge import Judge

SR = 16_000


class FakeEmbedder:
    """Deterministic vectors: text by hash, audio by its dominant frequency band (so tones differ)."""

    vision = False

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


def _tone(path, f, seconds=1.5, sidecar=None):
    t = np.arange(int(SR * seconds)) / SR
    x = 0.3 * np.sin(2 * np.pi * f * t) * np.exp(-2 * t)
    sf.write(path, x.astype(np.float32), SR)
    if sidecar:
        (path.parent / (path.name + ".json")).write_text(json.dumps(sidecar))


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SONICPROOF_CACHE", str(tmp_path / "cache"))
    from sonicproof.server import create_app

    app = create_app(FakeEngine(), home=tmp_path / "wb")
    with TestClient(app, base_url="http://localhost") as c:
        yield c


def _wait(c, job_id, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "error", "cancelled"):
            return j
        time.sleep(0.05)
    raise TimeoutError(job_id)


def test_inbox_groups_takes_by_sidecar_prompt_and_picks_one(client, tmp_path):
    d = tmp_path / "takes"
    d.mkdir()
    for i, f in enumerate((300, 900, 2500)):
        _tone(d / f"beep_v{i + 1}.wav", f, sidecar={"prompt": "a short beep", "seed": 10 + i})
    _tone(d / "loose.wav", 600)
    r = client.post("/api/inbox/paths", json={"paths": [str(d)]}).json()
    assert r["added"] == 4
    assert _wait(client, r["job"])["status"] == "done"
    st = client.get("/api/inbox").json()
    groups = {g["prompt"]: g for g in st["groups"]}
    beep = groups["a short beep"]
    assert len(beep["items"]) == 3 and beep["settled"]
    scores = [it["check"]["score"] for it in beep["items"]]
    assert scores == sorted(scores, reverse=True)  # best first
    assert {it["seed"] for it in beep["items"]} == {10, 11, 12}
    assert all(it["qc"]["checks"] and it["overall"] for it in beep["items"])
    assert beep["pick"] in (None, beep["items"][0]["id"])
    assert beep["suggest"] and all(isinstance(x, str) for x in beep["suggest"])
    loose = groups[None]["items"][0]
    assert loose["check"] is None and loose["heard_as"]  # no prompt: still "heard as"

    # adding the same files again is a no-op
    again = client.post("/api/inbox/paths", json={"paths": [str(d)]}).json()
    assert again == {"added": 0, "skipped": 4, "job": None}

    # a typed prompt replaces the sidecar's and re-checks
    r = client.patch(f"/api/inbox/{loose['id']}", json={"prompt": "ピッという音"}).json()
    _wait(client, r["job"])
    it = next(
        i for g in client.get("/api/inbox").json()["groups"] for i in g["items"] if i["id"] == loose["id"]
    )
    assert it["prompt_source"] == "user" and it["check"]["prompt"] == "ピッという音"


def test_upload_keeps_sidecars_and_media_is_served_only_for_known_files(client, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _tone(src / "a.wav", 440, sidecar={"prompt": "a bell"})
    files = [
        ("files", ("a.wav", (src / "a.wav").read_bytes(), "audio/wav")),
        ("files", ("a.wav.json", (src / "a.wav.json").read_bytes(), "application/json")),
    ]
    r = client.post(
        "/api/inbox/upload", files=files, data={"paths": ["batch/a.wav", "batch/a.wav.json"]}
    ).json()
    assert r["added"] == 1
    _wait(client, r["job"])
    it = client.get("/api/inbox").json()["groups"][0]["items"][0]
    assert it["prompt"] == "a bell" and it["prompt_source"] == "sidecar"
    assert client.get("/api/media", params={"path": it["path"]}).status_code == 200
    th = client.get("/api/thumb", params={"path": it["path"]}).json()
    assert len(th["peaks"]) == th["spec_cols"] and th["duration"] == pytest.approx(1.5, abs=0.01)
    secret = tmp_path / "secret.wav"
    _tone(secret, 100)
    assert client.get("/api/media", params={"path": str(secret)}).status_code == 404

    # removing the item deletes our copy (and its sidecar), never the user's original
    up = tmp_path / "wb" / "uploads"
    assert client.delete(f"/api/inbox/{it['id']}").json()["ok"]
    assert not [f for f in up.rglob("*") if f.is_file()]
    assert (src / "a.wav").exists()


def test_cache_usage_and_clear_keep_what_is_in_use(client, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for name, f in (("keep.wav", 440), ("drop.wav", 900)):
        _tone(src / name, f)
    files = [("files", (n, (src / n).read_bytes(), "audio/wav")) for n in ("keep.wav", "drop.wav")]
    r = client.post("/api/inbox/upload", files=files).json()
    _wait(client, r["job"])
    items = {i["name"]: i for g in client.get("/api/inbox").json()["groups"] for i in g["items"]}
    client.get("/api/thumb", params={"path": items["keep.wav"]["path"]})
    # an orphan left behind by an older version
    stray = tmp_path / "wb" / "uploads" / "old" / "x.wav"
    stray.parent.mkdir(parents=True)
    stray.write_bytes(b"0" * 1000)
    u = client.get("/api/cache").json()
    assert u["embeddings"] > 0 and u["thumbnails"] > 0 and u["uploads_unused"] == 1000
    r = client.post("/api/cache/clear").json()
    assert r["freed"] > 1000
    assert r["usage"]["embeddings"] == r["usage"]["thumbnails"] == r["usage"]["uploads_unused"] == 0
    assert not stray.parent.exists()
    assert Path(items["keep.wav"]["path"]).exists() and Path(items["drop.wav"]["path"]).exists()

    assert client.delete("/api/inbox").json()["ok"]
    assert not [f for f in (tmp_path / "wb" / "uploads").rglob("*") if f.is_file()]


def test_library_index_search_and_dedupe(client, tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    _tone(lib / "low.wav", 200)
    _tone(lib / "high.wav", 3000)
    x, _ = sf.read(lib / "low.wav")
    sf.write(lib / "low_quiet.wav", 0.5 * x, SR)  # a copy at a lower level
    j = _wait(client, client.post("/api/library/open", json={"path": str(lib)}).json()["job"])
    assert j["status"] == "done", j["error"]
    view = j["result"]
    assert view["files"] == 3 and all(e["heard_as"] for e in view["entries"])
    assert view["update"]["added"] == 3

    s = _wait(
        client, client.post("/api/library/search", json={"path": str(lib), "query": "a hum"}).json()["job"]
    )
    assert s["result"]["mode"] == "words" and len(s["result"]["results"]) == 3
    like = _wait(
        client,
        client.post("/api/library/search", json={"path": str(lib), "like": str(lib / "low.wav")}).json()[
            "job"
        ],
    )
    assert like["result"]["results"][0]["path"] == "low_quiet.wav"  # the file itself is skipped

    d = _wait(client, client.post("/api/library/dedupe", json={"path": str(lib)}).json()["job"])
    groups = d["result"]["groups"]
    assert len(groups) == 1 and {m["path"] for m in groups[0]["members"]} == {"low.wav", "low_quiet.wav"}
    assert (lib / "low_quiet.wav").exists()  # nothing deleted

    fs = client.get("/api/fs", params={"path": str(lib)}).json()
    assert fs["audio_files"] == 3 and fs["indexed"]
    assert not (lib / ".sonicproof").exists()  # the workbench never writes into the folder


def test_local_only(client):
    assert client.get("/api/status", headers={"host": "evil.example:7788"}).status_code == 403
    r = client.post("/api/settings", json={"target_lufs": -16}, headers={"origin": "https://evil.example"})
    assert r.status_code == 403
    assert client.post("/api/settings", json={"target_lufs": -16}).json()["target_lufs"] == -16


def test_service_shapes_match_the_cli_json(tmp_path, monkeypatch):
    monkeypatch.setenv("SONICPROOF_CACHE", str(tmp_path / "cache"))
    _tone(tmp_path / "x.wav", 500)
    e = FakeEngine()
    rep = e.report(tmp_path / "x.wav", "a beep")
    assert set(rep) == {"file", "overall", "note", "adherence", "qc", "duration", "sample_rate", "channels"}
    assert rep["overall"] in ("ship", "review", "redo")
    assert all({"code", "data"} <= set(c) for c in rep["qc"]["checks"])
    ranked = e.rank("a beep", [tmp_path / "x.wav"])
    assert ranked[0]["verdict"] in ("strong", "plausible", "weak", "miss")
    from typer.testing import CliRunner

    from sonicproof.cli import app

    out = CliRunner().invoke(app, ["qc", str(tmp_path / "x.wav"), "--json"])
    assert json.loads(out.stdout) == [service.qc_record(tmp_path / "x.wav")]  # qc --json is always a list


# -- state consistency -----------------------------------------------------------------------------
def test_concurrent_saves_never_corrupt_the_state(client, tmp_path):
    import threading

    wb = client.app.state.wb
    for i in range(60):
        _tone(tmp_path / f"t{i}.wav", 200 + 40 * i, seconds=0.3)
    job = wb.add_files([tmp_path / f"t{i}.wav" for i in range(60)])["job"]  # the worker saves too
    errors, bad_reads = [], 0
    state = tmp_path / "wb" / "state.json"

    def saver():
        try:
            for _ in range(100):
                wb.save()
        except Exception as e:  # pragma: no cover - the bug this guards against
            errors.append(e)

    threads = [threading.Thread(target=saver) for _ in range(4)]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        try:
            json.loads(state.read_text(encoding="utf-8"))
        except ValueError:
            bad_reads += 1
        except PermissionError:  # Windows: the file is being replaced right now
            pass
        time.sleep(0.001)
    for t in threads:
        t.join()
    assert not errors and bad_reads == 0
    assert len(json.loads(state.read_text(encoding="utf-8"))["items"]) == 60
    _wait(client, job)
    assert not list((tmp_path / "wb").glob("*.tmp"))  # no temp files left behind


def test_empty_wav_keeps_the_inbox_working(client, tmp_path):
    empty = tmp_path / "empty.wav"
    sf.write(empty, np.zeros(0, np.float32), SR)
    zero = tmp_path / "zero.wav"
    zero.write_bytes(b"")
    r = client.post("/api/inbox/paths", json={"paths": [str(empty), str(zero)]}).json()
    _wait(client, r["job"])
    res = client.get("/api/inbox")
    assert res.status_code == 200
    json.loads(res.text)  # strict JSON: no NaN / Infinity
    items = [i for g in res.json()["groups"] for i in g["items"]]
    assert all(i["status"] in ("done", "error") for i in items)


def test_sidecar_of_the_wrong_shape_is_ignored(client, tmp_path):
    d = tmp_path / "odd"
    d.mkdir()
    _tone(d / "a.wav", 500)
    (d / "a.wav.json").write_text("[1, 2, 3]")
    _tone(d / "b.wav", 900)
    (d / "b.wav.json").write_text('"just a string"')
    r = client.post("/api/inbox/paths", json={"paths": [str(d)]})
    assert r.status_code == 200 and r.json()["added"] == 2
    _wait(client, r.json()["job"])
    assert all(i["prompt"] is None for g in client.get("/api/inbox").json()["groups"] for i in g["items"])


def test_restart_keeps_items_and_resumes_queued_ones(tmp_path, monkeypatch):
    monkeypatch.setenv("SONICPROOF_CACHE", str(tmp_path / "cache"))
    from sonicproof.server import create_app

    _tone(tmp_path / "a.wav", 440)
    _tone(tmp_path / "b.wav", 1200)
    home = tmp_path / "wb"
    with TestClient(create_app(FakeEngine(), home=home), base_url="http://localhost") as c:
        r = c.post(
            "/api/inbox/paths", json={"paths": [str(tmp_path / "a.wav"), str(tmp_path / "b.wav")]}
        ).json()
        _wait(c, r["job"])
    # pretend the server stopped halfway through: one item still queued
    st = json.loads((home / "state.json").read_text())
    st["items"][1].update(status="working", check=None, qc=None)
    st["settings"] = {"target_lufs": float("inf")}  # a bad value from an older version
    (home / "state.json").write_text(json.dumps(st))
    with TestClient(create_app(FakeEngine(), home=home), base_url="http://localhost") as c:
        assert c.get("/api/inbox").json()["settings"]["target_lufs"] is None
        jobs = c.get("/api/status").json()["jobs"]
        for j in jobs:
            _wait(c, j["id"])
        items = [i for g in c.get("/api/inbox").json()["groups"] for i in g["items"]]
        assert len(items) == 2 and all(i["status"] == "done" and i["qc"] for i in items)


@pytest.mark.parametrize("value", [1e999, "NaN", "abc", [1], True, 5, -100, {"x": 1}])
def test_settings_rejects_bad_values(client, value):
    r = client.post(
        "/api/settings",
        content=json.dumps({"target_lufs": value}),
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 400 and r.json()["code"] == "bad_value"
    assert client.get("/api/inbox").status_code == 200
    assert client.post("/api/settings", json={"target_lufs": "-18"}).json()["target_lufs"] == -18
    assert client.post("/api/settings", json={"target_lufs": None}).json()["target_lufs"] is None


def test_bad_parameters_are_400_not_500(client, tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    _tone(lib / "x.wav", 300)
    for top in ("many", 0, 10_000, 2.5):
        r = client.post("/api/library/search", json={"path": str(lib), "query": "x", "top": top})
        assert r.status_code == 400 and r.json()["code"] == "bad_value", top
    r = client.post("/api/inbox/paths", json={"paths": "/tmp"})  # a string, not a list
    assert r.status_code == 400 and r.json()["code"] == "bad_value"
    r = client.post("/api/inbox/paths", json={"paths": [str(tmp_path / "nope.wav")]})
    assert r.status_code == 400 and r.json()["code"] == "not_found"
    assert client.patch("/api/inbox/nope", json={"prompt": "x"}).json()["code"] == "no_item"
    assert client.get("/api/jobs/nope").json()["code"] == "no_job"


def test_unexpected_errors_are_json(tmp_path, monkeypatch):
    monkeypatch.setenv("SONICPROOF_CACHE", str(tmp_path / "cache"))
    from sonicproof.server import create_app

    app = create_app(FakeEngine(), home=tmp_path / "wb")
    monkeypatch.setattr(app.state.wb, "cache_usage", lambda: 1 / 0)
    with TestClient(app, base_url="http://localhost", raise_server_exceptions=False) as c:
        r = c.get("/api/cache")
    assert r.status_code == 500 and r.json()["code"] == "internal"


def test_jobs_can_be_cancelled(client, tmp_path):
    import threading

    wb = client.app.state.wb
    gate = threading.Event()
    blocker = wb.jobs.submit("test", lambda job: gate.wait(10))  # keeps the worker busy
    lib = tmp_path / "lib"
    lib.mkdir()
    _tone(lib / "x.wav", 300)
    queued = client.post("/api/library/open", json={"path": str(lib)}).json()["job"]
    assert client.post(f"/api/jobs/{queued}/cancel").json()["cancel_requested"]
    gate.set()
    _wait(client, blocker.id)
    j = _wait(client, queued)
    assert j["status"] == "cancelled" and j["error_code"] == "cancelled"

    # a running job stops at its next progress tick
    def slow(job):
        for i in range(200):
            job.progress(i, 200, "x")
            time.sleep(0.01)

    running = wb.jobs.submit("slow", slow)
    while wb.jobs.get(running.id).status != "running":
        time.sleep(0.01)
    client.post(f"/api/jobs/{running.id}/cancel")
    assert _wait(client, running.id)["status"] == "cancelled"


def test_remove_keeps_state_consistent_when_a_copy_cannot_be_deleted(client, tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    _tone(src / "a.wav", 440)
    r = client.post(
        "/api/inbox/upload", files=[("files", ("a.wav", (src / "a.wav").read_bytes(), "audio/wav"))]
    ).json()
    _wait(client, r["job"])
    item = client.get("/api/inbox").json()["groups"][0]["items"][0]

    def locked(self, *a, **k):  # Windows: the file is open in another handle
        raise PermissionError(13, "in use")

    monkeypatch.setattr(Path, "unlink", locked)
    assert client.delete(f"/api/inbox/{item['id']}").status_code == 200
    monkeypatch.undo()
    assert client.get("/api/inbox").json()["count"] == 0
    assert json.loads((tmp_path / "wb" / "state.json").read_text())["items"] == []
    assert Path(item["path"]).exists()  # left for "Clear cache"
    client.post("/api/cache/clear")
    assert not Path(item["path"]).exists()


def test_cache_clear_is_refused_while_a_job_runs(client):
    import threading

    gate = threading.Event()
    job = client.app.state.wb.jobs.submit("test", lambda j: gate.wait(10))
    r = client.post("/api/cache/clear")
    assert r.status_code == 409 and r.json()["code"] == "busy"
    gate.set()
    _wait(client, job.id)


def test_busy_port_is_one_clear_error():
    import socket

    from sonicproof.errors import SetupError
    from sonicproof.server import _bind

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen()
    port = s.getsockname()[1]
    try:
        with pytest.raises(SetupError) as e:
            _bind("127.0.0.1", port)
        assert e.value.code == "port_busy" and str(port) in e.value.message
    finally:
        s.close()


def test_pick_never_falls_back_to_a_take_that_fails_qc(tmp_path):
    from sonicproof.server import Item, Workbench

    wb = Workbench(engine=None, home=tmp_path / "wb")  # inbox() never touches the engine

    def item(i, score, qc_status, status="done"):
        check = {"score": score, "verdict": "strong"}
        qc = {"status": qc_status}
        return Item(i, f"{i}.wav", f"/x/{i}.wav", i, "a dog barking", status=status, check=check, qc=qc)

    wb.items = {"a": item("a", 3.0, "fail"), "b": item("b", 2.5, "fail")}
    assert wb.inbox()["groups"][0]["pick"] is None
    wb.items["c"] = item("c", 2.2, "warn")
    assert wb.inbox()["groups"][0]["pick"] == "c"
    wb.items["d"] = item("d", 2.9, "pass", status="error")  # stale check left by a failed re-check
    assert wb.inbox()["groups"][0]["pick"] == "c"
