"""Library index, fingerprints and dedupe — no model needed (a stand-in embedder is used)."""

import os

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import resample, resample_poly

from sonicproof.library import (
    MATCH,
    Library,
    find_duplicates,
    fingerprint,
    match_fingerprints,
    scan,
)

SR = 16_000


def _texture(seed: int, seconds: float = 4.0) -> np.ndarray:
    """Noise bursts with a seed-specific rhythm and colour: easy to tell apart, hard to confuse."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    x = rng.standard_normal(n).astype(np.float32)
    env = np.zeros(n, dtype=np.float32)
    for start in rng.integers(0, n - SR // 4, 10):
        env[start : start + SR // 8] = rng.uniform(0.3, 1.0)
    tone = np.sin(2 * np.pi * rng.uniform(200, 2000) * np.arange(n) / SR)
    return (0.2 * x * env + 0.1 * tone * env).astype(np.float32)


def test_fingerprint_survives_gain_trim_and_resampling():
    a = _texture(1)
    fa = fingerprint(a)
    quiet = fingerprint(a * 0.25)
    trimmed = fingerprint(a[int(0.3 * SR) :])
    resampled = fingerprint(resample_poly(resample_poly(a, 441, 160), 160, 441).astype(np.float32))
    for other in (quiet, trimmed, resampled):
        c, _, _ = match_fingerprints(fa, other)
        assert c >= MATCH
    c, offset, cov = match_fingerprints(fa, trimmed)
    assert offset == pytest.approx(0.3, abs=0.05)
    assert cov < 0.98
    c_other, _, _ = match_fingerprints(fa, fingerprint(_texture(2)))
    assert c_other < 0.8


def _write(p, x, sr=SR):
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(p, x, sr)


def _const_embed(w):  # every clip "sounds alike": dedupe must rely on the fingerprints
    v = np.ones(768, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_index_is_incremental_and_dedupe_groups_copies(tmp_path, monkeypatch):
    monkeypatch.setenv("SONICPROOF_CACHE", str(tmp_path / "cache"))
    root = tmp_path / "lib"
    a, b = _texture(1), _texture(2)
    _write(root / "a.wav", resample_poly(a, 441, 160).astype(np.float32), 44100)
    _write(root / "copies" / "a_quiet.wav", a * 0.5)
    _write(root / "copies" / "a_22k.wav", resample_poly(a, 22050, 16000).astype(np.float32), 22050)
    _write(root / "copies" / "a_trim.wav", a[SR // 2 :])
    _write(root / "b.wav", b)
    _write(root / "b_same_bytes.wav", b)
    _write(root / ".hidden" / "ignored.wav", a)
    (root / "notes.txt").write_text("not audio")

    assert len(scan(root)) == 6
    lib = Library(root)
    st = lib.update(_const_embed)
    assert (st.added, st.unchanged, st.embedded) == (6, 0, 5)  # the byte-identical copy hits the cache
    assert lib.location == root / ".sonicproof" / "index.npz"

    lib2 = Library(root)
    assert len(lib2) == 6
    os.remove(root / "copies" / "a_quiet.wav")
    _write(root / "c.wav", _texture(3))
    st2 = lib2.update(_const_embed)
    assert (st2.added, st2.removed, st2.unchanged) == (1, 1, 5)

    groups = {g.keep: g for g in find_duplicates(lib2)}
    # a.wav and a_22k.wav are both upsampled from the same 16 kHz source: same content, smaller file wins
    assert set(groups) == {"copies/a_22k.wav", "b.wav"}
    rel = {m.path: m.relation for m in groups["copies/a_22k.wav"].members}
    assert rel == {"copies/a_22k.wav": "keep", "a.wav": "same audio", "copies/a_trim.wav": "trimmed"}
    assert {m.relation for m in groups["b.wav"].members} == {"keep", "identical"}
    assert "longest" in groups["copies/a_22k.wav"].reason
    assert "same range, lower rate" in groups["copies/a_22k.wav"].reason


def test_in_cache_index_lives_outside_the_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("SONICPROOF_CACHE", str(tmp_path / "cache"))
    root = tmp_path / "lib"
    _write(root / "a.wav", _texture(1, 1.0))
    lib = Library(root, in_cache=True)
    lib.update(_const_embed)
    assert not (root / ".sonicproof").exists()
    assert str(lib.location).startswith(str(tmp_path / "cache"))
    assert len(Library(root, in_cache=True)) == 1


def test_in_cache_reuses_an_existing_folder_index_without_writing_to_it(tmp_path, monkeypatch):
    monkeypatch.setenv("SONICPROOF_CACHE", str(tmp_path / "cache"))
    root = tmp_path / "lib"
    _write(root / "a.wav", _texture(1, 1.0))
    Library(root).update(_const_embed)
    folder_index = root / ".sonicproof" / "index.npz"
    before = folder_index.stat().st_mtime_ns
    lib = Library(root, in_cache=True)
    assert len(lib) == 1  # seeded from the folder's index
    st = lib.update(_const_embed)
    assert st.unchanged == 1 and st.embedded == 0
    assert str(lib.location).startswith(str(tmp_path / "cache"))
    assert folder_index.stat().st_mtime_ns == before


# -- index robustness (B6) ------------------------------------------------------------------------
def _small_lib(tmp_path, monkeypatch, n=2):
    monkeypatch.setenv("SONICPROOF_CACHE", str(tmp_path / "cache"))
    root = tmp_path / "lib"
    for i in range(n):
        _write(root / f"s{i}.wav", _texture(i + 1, 1.0))
    Library(root).update(_const_embed)
    return root, root / ".sonicproof" / "index.npz"


def _rewrite(index, emb=None, entries=None, **meta_extra):
    import json as _json

    with np.load(index) as z:
        meta = _json.loads(str(z["meta"]))
        e = z["emb"]
    if entries is not None:
        meta["entries"] = entries(meta["entries"])
    meta.update(meta_extra)
    np.savez(index, emb=e if emb is None else emb(e), meta=np.array(_json.dumps(meta)))


@pytest.mark.parametrize(
    "damage",
    [
        lambda p: p.write_bytes(p.read_bytes()[: len(p.read_bytes()) // 2]),  # truncated zip
        lambda p: p.write_bytes(b"hello"),
        lambda p: _rewrite(p, emb=lambda e: e[:1]),  # fewer rows than entries
        lambda p: _rewrite(p, entries=lambda es: [{k: v for k, v in x.items() if k != "digest"} for x in es]),
        lambda p: _rewrite(p, schema=99),  # written by a newer Sonicproof
        lambda p: _rewrite(p, model="someone/else"),
    ],
)
def test_damaged_index_is_rebuilt(tmp_path, monkeypatch, damage):
    root, index = _small_lib(tmp_path, monkeypatch)
    damage(index)
    lib = Library(root)
    assert len(lib) == 0  # nothing half-loaded
    st = lib.update(_const_embed)
    assert st.added == 2 and len(Library(root)) == 2


def test_unknown_entry_fields_are_ignored(tmp_path, monkeypatch):
    root, index = _small_lib(tmp_path, monkeypatch)
    _rewrite(index, entries=lambda es: [{**x, "future_field": 1} for x in es])
    lib = Library(root)
    assert len(lib) == 2
    assert lib.update(_const_embed).unchanged == 2


def test_index_records_schema(tmp_path, monkeypatch):
    import json as _json

    from sonicproof.library import INDEX_SCHEMA

    _, index = _small_lib(tmp_path, monkeypatch, 1)
    with np.load(index) as z:
        assert _json.loads(str(z["meta"]))["schema"] == INDEX_SCHEMA


def test_model_change_invalidates_index(tmp_path, monkeypatch):
    import sonicproof.library as library

    root, _ = _small_lib(tmp_path, monkeypatch)
    monkeypatch.setattr(library, "MODEL_ID", "someone/other")
    assert len(Library(root)) == 0


def test_update_stats_and_failed_files(tmp_path, monkeypatch):
    root, _ = _small_lib(tmp_path, monkeypatch)
    calls = []

    def embed(w):
        calls.append(1)
        return _const_embed(w)

    st = os.stat(root / "s0.wav")
    os.utime(root / "s0.wav", ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))  # touched, same bytes
    _write(root / "s1.wav", _texture(9, 1.0))  # new content
    (root / "broken.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVEjunk")
    (root / "s2.wav").write_bytes(b"")  # empty
    lib = Library(root)
    r = lib.update(embed)
    assert (r.changed, r.added, r.unchanged) == (2, 0, 0)
    assert r.embedded == 1 and len(calls) == 1  # the touched file came from the embedding cache
    assert sorted(f.split(":")[0] for f in r.failed) == ["broken.wav", "s2.wav"]
    assert len(lib) == 2


def test_missing_model_extra_is_not_a_bad_file(tmp_path, monkeypatch):
    from sonicproof.errors import SetupError

    monkeypatch.setenv("SONICPROOF_CACHE", str(tmp_path / "cache"))
    root = tmp_path / "lib"
    _write(root / "a.wav", _texture(1, 1.0))

    def no_torch(w):
        raise SetupError("needs the model extra", code="missing_extra")

    with pytest.raises(SetupError):
        Library(root).update(no_torch)
    assert not (root / ".sonicproof").exists()  # no empty index left behind

    def import_error(w):
        raise ModuleNotFoundError("No module named 'torch'")

    with pytest.raises(ImportError):
        Library(root).update(import_error)
    assert not (root / ".sonicproof").exists()


def test_read_only_folder_falls_back_to_cache(tmp_path, monkeypatch):
    import sonicproof.library as library

    monkeypatch.setenv("SONICPROOF_CACHE", str(tmp_path / "cache"))
    root = tmp_path / "lib"
    _write(root / "a.wav", _texture(1, 1.0))
    lib = Library(root)
    assert lib.location == root / ".sonicproof" / "index.npz"  # os.access says yes (as on Windows ACLs)
    real = library.Library._write

    def deny_folder(self, dest):
        if str(dest).startswith(str(root)):
            raise PermissionError(13, "Access is denied")
        return real(self, dest)

    monkeypatch.setattr(library.Library, "_write", deny_folder)
    lib.update(_const_embed)
    assert str(lib.location).startswith(str(tmp_path / "cache")) and "read-only" in lib.note
    monkeypatch.undo()
    monkeypatch.setenv("SONICPROOF_CACHE", str(tmp_path / "cache"))
    assert len(Library(root)) == 1  # found again in the cache


# -- check_root ---------------------------------------------------------------------------------
def test_check_root(tmp_path, monkeypatch):
    from pathlib import Path

    from sonicproof.errors import InputError
    from sonicproof.library import check_root

    (tmp_path / "lib").mkdir()
    assert check_root(tmp_path / "lib") == (tmp_path / "lib").resolve()
    monkeypatch.setenv("HOME", str(tmp_path / "lib"))  # "~" below is this folder: home itself is refused
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "lib"))  # Path.home() on Windows
    cases = {
        tmp_path / "nope": "not_found",
        tmp_path / "lib" / ".." / "lib" / "x.wav": "not_found",
        "/": "bad_value",
        "~": "bad_value",
        "/usr": "bad_value",
        "/etc": "bad_value",
    }
    _write(tmp_path / "f.wav", _texture(1, 1.0))
    cases[tmp_path / "f.wav"] = "bad_value"
    for p, code in cases.items():
        if str(p) in ("/usr", "/etc") and not Path(p).exists():  # Windows
            continue
        with pytest.raises(InputError) as e:
            check_root(p)
        assert e.value.code == code, p
        assert e.value.file == str(p)
    with pytest.raises(InputError):
        Library("/")


# -- fingerprints (B12) and dedupe keep choice ----------------------------------------------------
def _brute(A, B):
    from sonicproof.library import _corr_at

    best = -1.0
    for lag in range(-(len(B) - 1), len(A)):
        c, n = _corr_at(A, B, lag)
        if n >= min(len(A), len(B)) * 0.5:
            best = max(best, c)
    return best


@pytest.mark.parametrize("seconds", [0.1, 0.2, 0.25])
def test_short_fingerprint_matches_brute_force(seconds):
    a = _texture(5, 3.0)
    mid = int(1.3 * SR)
    fa, fb = fingerprint(a), fingerprint(a[mid : mid + int(seconds * SR)])
    assert len(fb) < 10
    c, _, _ = match_fingerprints(fa, fb)
    assert c == pytest.approx(_brute(fa, fb), abs=1e-6)
    assert c >= MATCH


def test_different_takes_do_not_match():
    for s in range(3):
        c, _, _ = match_fingerprints(fingerprint(_texture(10 + s)), fingerprint(_texture(20 + s)))
        assert c < MATCH


def test_keep_prefers_lossless_unclipped_and_not_upsampled(tmp_path, monkeypatch):
    import soundfile as _sf

    monkeypatch.setenv("SONICPROOF_CACHE", str(tmp_path / "cache"))
    root = tmp_path / "lib"
    a = _texture(3)
    _write(root / "a_16k.wav", a)
    _write(root / "a_48k.wav", resample_poly(a, 3, 1).astype(np.float32), 48000)  # upsampled: nothing new
    clipped = np.clip(resample_poly(a, 441, 160) * 8, -1, 1).astype(np.float32)
    _write(root / "a_44k_clipped.wav", clipped, 44100)
    _sf.write(root / "a_lossy.ogg", resample_poly(a, 3, 1).astype(np.float32), 48000)
    lib = Library(root)
    lib.update(_const_embed)
    (g,) = find_duplicates(lib)
    assert g.keep == "a_16k.wav"
    assert g.reason == "lossless, not clipped, same range, lower rate"
    assert {m.path for m in g.members} == {"a_16k.wav", "a_48k.wav", "a_44k_clipped.wav", "a_lossy.ogg"}


def _wideband(seed: int, sr: int = 44_100, seconds: float = 4.0) -> np.ndarray:
    """Like ``_texture`` but with real content up to the Nyquist frequency of ``sr``."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    env = np.zeros(n, dtype=np.float32)
    for start in rng.integers(0, n - sr // 4, 10):
        env[start : start + sr // 8] = rng.uniform(0.3, 1.0)
    return (0.2 * rng.standard_normal(n) * env).astype(np.float32)


def test_keep_prefers_a_higher_rate_only_when_it_holds_more(tmp_path, monkeypatch):
    monkeypatch.setenv("SONICPROOF_CACHE", str(tmp_path / "cache"))
    root = tmp_path / "lib"
    a = _wideband(7)
    _write(root / "orig.wav", a, 44_100)
    _write(root / "small_16k.wav", resample_poly(a, 160, 441).astype(np.float32), 16_000)
    lib = Library(root)
    lib.update(_const_embed)
    (g,) = find_duplicates(lib)
    assert g.keep == "orig.wav" and "wider frequency range" in g.reason


def test_keep_ignores_a_quieter_upsampled_copy(tmp_path, monkeypatch):
    # the original vs a copy turned down 6 dB and upsampled to 48 kHz: the upsampled copy holds nothing more
    monkeypatch.setenv("SONICPROOF_CACHE", str(tmp_path / "cache"))
    root = tmp_path / "lib"
    a = _wideband(8)
    _write(root / "orig.wav", a, 44_100)
    quiet = 0.5 * resample(a, len(a) * 160 // 147)  # sharp, like ffmpeg / sox (resample_poly is soft)
    _write(root / "orig_quieter_48k.flac", quiet.astype(np.float32), 48_000)
    lib = Library(root)
    lib.update(_const_embed)
    (g,) = find_duplicates(lib)
    assert g.keep == "orig.wav"
    assert "higher level" in g.reason and "kHz" not in g.reason


def test_transitive_groups_and_identical(tmp_path, monkeypatch):
    monkeypatch.setenv("SONICPROOF_CACHE", str(tmp_path / "cache"))
    root = tmp_path / "lib"
    a = _texture(4)
    _write(root / "a.wav", a)
    _write(root / "a_copy.wav", a)  # same bytes
    _write(root / "a_trim.wav", a[SR // 4 :])
    _write(root / "a_trim_more.wav", a[SR // 2 :])
    lib = Library(root)
    lib.update(_const_embed)
    (g,) = find_duplicates(lib)
    rel = {m.path: m.relation for m in g.members}
    assert rel["a.wav"] == "keep" and rel["a_copy.wav"] == "identical"
    assert rel["a_trim.wav"] == rel["a_trim_more.wav"] == "trimmed"
    assert all(m.match >= MATCH for m in g.members)


# -- silent entries in search ----------------------------------------------------------------------
def test_search_puts_silent_entries_last(tmp_path, monkeypatch):
    from test_judge import FakeEmbedder

    from sonicproof.judge import Judge
    from sonicproof.library import search_text

    monkeypatch.setenv("SONICPROOF_CACHE", str(tmp_path / "cache"))
    root = tmp_path / "lib"
    _write(root / "loud.wav", _texture(1, 1.0))
    _write(root / "silence.wav", np.zeros(SR, np.float32))
    _write(root / "faint.wav", 0.0005 * np.ones(SR, np.float32))  # -66 dBFS
    j = Judge(embedder=FakeEmbedder(audio_vec=lambda w: np.ones(768)))  # silence "sounds" exactly right
    lib = Library(root)
    lib.update(lambda w: j.embedder.embed_audio([w])[0])
    hits = search_text(lib, j, "door")
    assert hits[0].path == "loud.wav"
    tail = {h.path: h for h in hits[1:]}
    assert set(tail) == {"silence.wav", "faint.wav"}
    assert all(h.verdict == "miss" and h.note == "silent" for h in tail.values())
    assert hits[0].note is None


def test_scan_skips_system_folders_and_symlinks_leaving_the_root(tmp_path):
    from sonicproof.library import scan

    root, outside = tmp_path / "lib", tmp_path / "outside"
    (root / "Library").mkdir(parents=True)
    (root / "node_modules").mkdir()
    outside.mkdir()
    for p in (root / "a.wav", root / "Library" / "b.wav", root / "node_modules" / "c.wav", outside / "o.wav"):
        p.write_bytes(b"x")
    (root / "in_link.wav").symlink_to(root / "a.wav")
    (root / "out_link.wav").symlink_to(outside / "o.wav")
    assert sorted(p.name for p in scan(root)) == ["a.wav", "in_link.wav"]


def test_long_files_that_only_share_the_fingerprinted_start_are_not_redundant(tmp_path, monkeypatch):
    import sonicproof.library as library
    from sonicproof.service import dedupe_record

    monkeypatch.setenv("SONICPROOF_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(library, "FP_SECONDS", 4.0)  # stand-in for the two-minute window
    root = tmp_path / "lib"
    start = _texture(5, 4.0)
    a = np.r_[start, _texture(6, 2.0)]
    _write(root / "a.wav", a)
    _write(root / "b.wav", np.r_[start, _texture(7, 2.0)])  # same first "two minutes", different ending
    _write(root / "a_quiet.wav", 0.5 * a)  # a real copy of the whole file
    lib = Library(root)
    lib.update(_const_embed)
    (g,) = find_duplicates(lib)
    rel = {m.path: m.relation for m in g.members}
    assert rel["b.wav"] == "same start" and rel["a_quiet.wav"] == "same audio"
    sizes = {m.path: m.size for m in g.members}
    assert g.reclaimable_bytes == sum(sizes[p] for p, r in rel.items() if r in ("same audio",))
    rec = dedupe_record(lib, [g])
    assert rec["redundant_files"] == 1


def test_check_root_refuses_whole_disks(monkeypatch, tmp_path):
    import os

    import sonicproof.library as library
    from sonicproof.errors import InputError

    with pytest.raises(InputError):
        library.check_root(
            "/Volumes" if os.path.isdir("/Volumes") else "/mnt" if os.path.isdir("/mnt") else "/"
        )
    monkeypatch.setattr(library.os.path, "ismount", lambda p: str(p) == str(tmp_path.resolve()))
    with pytest.raises(InputError):
        library.check_root(tmp_path)
    (tmp_path / "sfx").mkdir()
    assert library.check_root(tmp_path / "sfx") == (tmp_path / "sfx").resolve()
