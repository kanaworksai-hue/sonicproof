"""Workbench security: what the local server will serve, accept and write (no model needed)."""

import json
import os
from pathlib import PureWindowsPath

import pytest
from fastapi.testclient import TestClient
from test_server import FakeEngine, _tone, _wait

from sonicproof import server


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SONICPROOF_CACHE", str(tmp_path / "cache"))
    app = server.create_app(FakeEngine(), home=tmp_path / "wb")
    with TestClient(app, base_url="http://localhost:7788") as c:
        yield c


@pytest.fixture
def lib(tmp_path, client):
    """An opened library holding a sound, an image, and files that must never be served."""
    d = tmp_path / "lib"
    d.mkdir()
    _tone(d / "a.wav", 440)
    from PIL import Image

    Image.new("RGB", (8, 8), (200, 10, 10)).save(d / "cover.png")
    (d / "readme.html").write_text("<script>fetch('/api/fs')</script>")
    (d / "cover.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>')
    (d / "notes.txt").write_text("topsecret")
    (d / "id_rsa.wav.txt").write_text("key")
    outside = tmp_path / "outside.wav"
    _tone(outside, 300)
    os.symlink(outside, d / "link.wav")
    _wait(client, client.post("/api/library/open", json={"path": str(d)}).json()["job"])
    return d


def media(c, path, **headers):
    return c.get("/api/media", params={"path": str(path)}, headers=headers)


# -- what can be read --------------------------------------------------------------------------------
def test_media_serves_only_audio_image_video_with_safe_headers(client, lib):
    r = media(client, lib / "a.wav")
    assert r.status_code == 200 and r.headers["content-type"].startswith("audio/")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert "sandbox" in r.headers["content-security-policy"]
    assert media(client, lib / "cover.png").status_code == 200
    for name in ("readme.html", "cover.svg", "notes.txt", "id_rsa.wav.txt"):
        assert media(client, lib / name).status_code == 404, name
    th = client.get("/api/thumb", params={"path": str(lib / "a.wav")})
    assert th.status_code == 200 and "sandbox" in th.headers["content-security-policy"]
    assert client.get("/api/thumb", params={"path": str(lib / "notes.txt")}).status_code == 404


def test_media_refuses_traversal_absolute_and_symlinks_out(client, lib, tmp_path):
    up = tmp_path / "wb" / "uploads"
    for p in (
        f"{up}/x/../../state.json",
        f"{up}/../state.json",
        "/etc/hosts",
        str(lib / ".." / "outside.wav"),
        str(lib / "link.wav"),  # points outside the library
        str(up) + "\\..\\state.json",
    ):
        assert media(client, p).status_code == 404, p
        assert client.get("/api/thumb", params={"path": p}).status_code == 404, p


def test_spa_never_leaves_the_build(client):
    for p in ("/../server.py", "/assets/../../server.py", "/%2e%2e/server.py"):
        r = client.get(p)
        assert "def create_app" not in r.text
    r = client.get("/")
    if r.status_code == 200:
        assert "frame-ancestors 'none'" in r.headers["content-security-policy"]


@pytest.mark.parametrize("root", ["/", "~", "/System", "/usr"])
def test_too_broad_library_roots_are_refused(client, root):
    r = client.post("/api/library/open", json={"path": root})
    assert r.status_code == 400 and r.json()["code"] in ("library_root", "not_folder")
    r = client.post("/api/inbox/paths", json={"paths": [root]})
    assert r.status_code == 400


def test_inbox_folder_scan_has_limits(client, tmp_path, monkeypatch):
    d = tmp_path / "many"
    d.mkdir()
    for i in range(6):
        (d / f"{i}.wav").write_bytes(b"x")
    (d / "Library").mkdir()
    (d / "Library" / "hidden.wav").write_bytes(b"x")
    monkeypatch.setattr(server, "MAX_SCAN_FILES", 5)
    r = client.post("/api/inbox/paths", json={"paths": [str(d)]})
    assert r.status_code == 413 and r.json()["code"] == "too_many_files"
    monkeypatch.setattr(server, "MAX_SCAN_FILES", 100)
    assert "hidden.wav" not in {p.name for p in server._scan_inbox(d)}


# -- who may ask ------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "host", ["evil.example:7788", "localhost.evil.com", "testserver", "localhost:80@evil", ""]
)
def test_foreign_hosts_are_refused(client, host):
    assert client.get("/api/status", headers={"host": host}).status_code == 403


@pytest.mark.parametrize(
    "origin",
    [
        "https://evil.example",
        "null",
        "http://localhost.evil.com",
        "http://localhost:9999",  # another local web app
        "https://localhost:7788",
        "http://[::1]:7788",  # same machine, but not the origin this page was loaded from
        "file://",
    ],
)
def test_writes_from_other_origins_are_refused(client, origin):
    r = client.post("/api/settings", json={"target_lufs": -16}, headers={"origin": origin})
    assert r.status_code == 403
    assert "access-control-allow-origin" not in r.headers


def test_own_origin_may_write(client):
    r = client.post("/api/settings", json={"target_lufs": -16}, headers={"origin": "http://localhost:7788"})
    assert r.status_code == 200
    r = client.post(
        "/api/settings",
        json={"target_lufs": -16},
        headers={"origin": "http://[::1]:7788", "host": "[::1]:7788"},
    )
    assert r.status_code == 200


@pytest.mark.parametrize("site", ["cross-site", "same-site"])
def test_api_refuses_requests_from_other_sites(client, lib, site):
    # a link or <img>/<audio> embed on another page: neither readable nor an existence oracle
    assert media(client, lib / "a.wav", **{"sec-fetch-site": site}).status_code == 403
    assert client.get("/api/status", headers={"sec-fetch-site": site}).status_code == 403
    assert media(client, lib / "a.wav", **{"sec-fetch-site": "same-origin"}).status_code == 200
    assert (
        media(client, lib / "a.wav", **{"sec-fetch-site": "none"}).status_code == 200
    )  # typed in the URL bar


def test_an_attacker_file_cannot_drive_the_api(client, lib, tmp_path):
    """The P0 chain: a library .html can't be served, so it can't run as this origin."""
    r = media(client, lib / "readme.html")
    assert r.status_code == 404 and "<script>" not in r.text
    private = tmp_path / "private"
    private.mkdir()
    (private / "id_rsa").write_text("MY-PRIVATE-KEY")
    assert media(client, private / "id_rsa").status_code == 404


# -- what can be written -------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name",
    [
        "../../x.wav",
        "/tmp/abs.wav",
        "..\\..\\win.wav",
        "C:\\evil\\c.wav",
        "D:/evil/x.wav",
        "D:x.wav",
        "a/b:stream.wav",
        "CON.wav",
        "a\x00b.wav",
        ("x" * 300) + ".wav",
        "",
    ],
)
def test_upload_names_stay_inside_the_batch(client, tmp_path, name):
    src = tmp_path / "s.wav"
    _tone(src, 440)
    r = client.post(
        "/api/upload",
        files=[("files", ("s.wav", src.read_bytes(), "audio/wav"))],
        data={"paths": [name or "s.wav"]},
    )
    assert r.status_code == 200, r.text
    saved = [p for p in (tmp_path / "wb" / "uploads").rglob("*") if p.is_file()]
    up = (tmp_path / "wb" / "uploads").resolve()
    assert saved and all(p.resolve().is_relative_to(up) for p in saved)
    assert all(len(p.name.encode()) <= 200 for p in saved)
    assert not list(tmp_path.glob("*.wav"))[1:]  # nothing new next to the source


def test_safe_rel_on_windows_paths():
    batch = PureWindowsPath("C:/Users/u/.cache/sonicproof/workbench/uploads/abc123")
    for name in ("D:/evil/x.wav", "a/b:stream.wav", "C:x.wav", "C:\\evil\\c.wav", "\\\\server\\share\\x.wav"):
        rel = server._safe_rel(name)
        joined = batch / PureWindowsPath(*rel.parts)
        assert joined.drive == "C:" and str(joined).startswith(str(batch)), (name, joined)
        assert not any(":" in p for p in rel.parts)
    assert server._safe_rel("CON.wav").name == "_CON.wav"
    assert server._safe_rel("lpt1").name == "_lpt1"


def test_uploads_have_size_and_count_limits(client, tmp_path, monkeypatch):
    monkeypatch.setitem(server.MAX_UPLOAD_BYTES, "audio", 1000)
    r = client.post("/api/upload", files=[("files", ("big.wav", b"0" * 5000, "audio/wav"))])
    assert r.status_code == 413 and r.json()["code"] == "too_large"
    assert not [p for p in (tmp_path / "wb" / "uploads").rglob("*") if p.is_file()]  # partial file removed
    monkeypatch.setattr(server, "MAX_UPLOAD_FILES", 2)
    files = [("files", (f"{i}.wav", b"0", "audio/wav")) for i in range(3)]
    r = client.post("/api/upload", files=files)
    assert r.status_code == 413 and r.json()["code"] == "too_many_files"


def test_upload_keeps_only_media_and_sidecars(client, tmp_path):
    files = [
        ("files", ("a.wav.json", b'{"prompt": "x"}', "application/json")),
        ("files", ("notes.txt", b"hello", "text/plain")),
        ("files", ("page.html", b"<script></script>", "text/html")),
    ]
    assert client.post("/api/upload", files=files).status_code == 200
    names = {p.name for p in (tmp_path / "wb" / "uploads").rglob("*") if p.is_file()}
    assert names == {"a.wav.json"}


def test_broken_picture_is_reported_on_upload(client, tmp_path):
    r = client.post("/api/upload", files=[("files", ("bad.jpg", b"not an image", "image/jpeg"))]).json()
    assert r["files"][0]["kind"] == "broken" and r["files"][0]["path"] is None


def test_remove_never_touches_user_files(client, tmp_path):
    d = tmp_path / "mine"
    d.mkdir()
    _tone(d / "u.wav", 440, sidecar={"prompt": "a bell"})
    r = client.post("/api/inbox/paths", json={"paths": [str(d / "u.wav")]}).json()
    _wait(client, r["job"])
    up = tmp_path / "src.wav"
    _tone(up, 900, sidecar={"prompt": "a gong"})
    r = client.post(
        "/api/inbox/upload",
        files=[
            ("files", ("b/src.wav", up.read_bytes(), "audio/wav")),
            ("files", ("b/src.wav.json", (tmp_path / "src.wav.json").read_bytes(), "application/json")),
        ],
        data={"paths": ["b/src.wav", "b/src.wav.json"]},
    ).json()
    _wait(client, r["job"])
    assert client.delete("/api/inbox").json()["ok"]
    assert (d / "u.wav").exists() and (d / "u.wav.json").exists()
    uploads = tmp_path / "wb" / "uploads"
    assert not [p for p in uploads.rglob("*") if p.is_file()]
    assert not [p for p in uploads.iterdir()]  # empty batch folders pruned


def test_crafted_state_cannot_delete_outside_uploads(tmp_path, monkeypatch):
    monkeypatch.setenv("SONICPROOF_CACHE", str(tmp_path / "cache"))
    victim = tmp_path / "victim"
    victim.mkdir()
    _tone(victim / "v.wav", 440)
    home = tmp_path / "wb"
    (home / "uploads").mkdir(parents=True)
    items = [
        {
            "id": "a",
            "name": "v.wav",
            "path": f"{home}/uploads/../../victim/v.wav",
            "digest": "1",
            "status": "done",
        },
        {"id": "b", "name": "uploads", "path": str(home / "uploads"), "digest": "2", "status": "done"},
    ]
    (home / "state.json").write_text(json.dumps({"items": items}))
    app = server.create_app(FakeEngine(), home=home)
    with TestClient(app, base_url="http://localhost") as c:
        assert c.delete("/api/inbox").status_code == 200
    assert (victim / "v.wav").exists()
