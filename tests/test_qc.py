import json

import numpy as np
import soundfile as sf
from typer.testing import CliRunner

from sonicproof import qc
from sonicproof.audio import Audio
from sonicproof.cli import app

SR = 44_100


def _audio(x: np.ndarray) -> Audio:
    x = np.asarray(x, np.float32)
    return Audio(x if x.ndim == 2 else x[:, None], SR)


def _decay(seconds=1.0, amp=0.3, f=440.0):
    t = np.arange(int(SR * seconds)) / SR
    return amp * np.sin(2 * np.pi * f * t) * np.exp(-6 * t)


def _status(checks, key):
    return next(c.status for c in checks if c.key == key)


def test_clean_decay_passes():
    x = np.stack([_decay(), _decay()], axis=1)
    checks = qc.run(_audio(x))
    assert qc.summary(checks) in ("pass", "warn")
    for key in ("true_peak", "clipping", "dc", "lead_silence", "tail_cut", "clicks"):
        assert _status(checks, key) == "pass", key
    assert _status(checks, "stereo") == "info"  # dual mono


def test_clipping_and_true_peak_fail():
    x = np.clip(_decay(amp=4.0), -1, 1)
    checks = qc.run(_audio(x))
    assert _status(checks, "clipping") == "fail"
    assert _status(checks, "true_peak") in ("warn", "fail")


def test_dc_offset():
    assert _status(qc.run(_audio(_decay() + 0.05)), "dc") == "fail"


def test_out_of_phase_is_caught_and_not_mistaken_for_silence():
    x = _decay()
    checks = qc.run(_audio(np.stack([x, -x], axis=1)))
    assert _status(checks, "stereo") == "fail"
    assert all(c.key != "silence" for c in checks)


def test_cut_tail_and_leading_silence():
    t = np.arange(SR) / SR
    x = np.concatenate([np.zeros(SR // 2), 0.3 * np.sin(2 * np.pi * 220 * t)])  # sustained, then stops
    checks = qc.run(_audio(x))
    assert _status(checks, "tail_cut") == "warn"
    assert _status(checks, "lead_silence") == "warn"


def test_click_detected():
    x = 0.01 * np.sin(2 * np.pi * 220 * np.arange(SR) / SR)
    x[SR // 2] += 0.6
    assert _status(qc.run(_audio(x)), "clicks") == "warn"


def test_loop_seam():
    t = np.arange(SR) / SR
    smooth = 0.3 * np.sin(2 * np.pi * 10 * t)  # whole number of cycles: seamless
    assert _status(qc.run(_audio(smooth), loop=True), "loop") == "pass"
    jumpy = 0.3 * np.sin(2 * np.pi * 10.25 * t)
    assert _status(qc.run(_audio(jumpy), loop=True), "loop") == "fail"


def test_loop_seam_sparse_crackle():
    # steady crackle where the clip happens to start on a pop and end in a gap between pops
    rng = np.random.default_rng(0)
    x = 0.01 * rng.standard_normal(SR * 4)
    for i in (*rng.integers(SR // 10, SR * 4 - SR // 5, 40), 200):
        x[i : i + 400] += 0.5 * np.exp(-np.arange(400) / 60) * rng.choice([-1, 1])
    assert _status(qc.run(_audio(x), loop=True), "loop") == "pass"


def test_target_lufs():
    checks = qc.run(_audio(np.stack([_decay()] * 2, axis=1)), target_lufs=-14)
    c = next(c for c in checks if c.key == "loudness")
    assert c.status in ("warn", "fail") and "dB" in c.hint


def test_cli_qc_json_and_exit_code(tmp_path):
    good, bad = tmp_path / "good.wav", tmp_path / "bad.wav"
    sf.write(good, np.stack([_decay()] * 2, axis=1), SR)
    sf.write(bad, np.clip(_decay(amp=4.0), -1, 1), SR)
    runner = CliRunner()
    r = runner.invoke(app, ["qc", str(good), "--json"])
    assert r.exit_code == 0, r.output
    out = json.loads(r.output)
    out = out[0] if isinstance(out, list) else out  # `qc --json` is becoming always-an-array
    assert out["status"] in ("pass", "warn")
    r = runner.invoke(app, ["qc", str(bad), "--json"])
    assert r.exit_code == 1


# --- edge cases (no model needed) -------------------------------------------------------------------------


def _codes(checks):
    return {c.key: c.code for c in checks}


def _strict_json(checks):
    return json.dumps([c.to_dict() for c in checks], allow_nan=False)


def _noise(seconds=1.0, amp=0.1, channels=2, seed=0, sr=SR):
    rng = np.random.default_rng(seed)
    return (amp * rng.standard_normal((int(sr * seconds), channels))).astype(np.float32)


def test_empty_audio_is_one_clean_fail():
    for loop in (False, True):
        checks = qc.run(Audio(np.zeros((0, 2), np.float32), 48_000), loop=loop)
        assert [c.code for c in checks] == ["file.empty"]
        assert qc.summary(checks) == "fail"
        _strict_json(checks)


def test_single_frame_and_non_finite_samples_stay_json_safe():
    for loop in (False, True):
        _strict_json(qc.run(Audio(np.full((1, 1), 0.5, np.float32), SR), loop=loop))
    x = _noise()
    x[100, 0], x[200, 1] = np.nan, np.inf
    checks = qc.run(Audio(x, SR))
    assert checks[0].code == "file.nonfinite" and checks[0].data["count"] == 2
    _strict_json(checks)


def test_surround_loudness_is_measured():
    for ch in (6, 8):
        a = Audio(_noise(channels=ch, sr=48_000), 48_000)
        c = next(c for c in qc.run(a) if c.key == "loudness")
        assert c.code != "loudness.unmeasured" and np.isfinite(c.data["lufs"])


def test_integrated_loudness_matches_pyloudnorm_and_skips_lfe():
    import pyloudnorm as pyln

    x = _noise(seconds=3.0, channels=5)
    x[:, 3:] *= 0.5
    assert abs(qc.integrated_loudness(x, SR) - pyln.Meter(SR).integrated_loudness(x)) < 1e-4
    six = np.concatenate([x[:, :3], np.zeros((len(x), 1), np.float32), x[:, 3:]], axis=1)  # L R C LFE Ls Rs
    base = qc.integrated_loudness(six, SR)
    assert abs(base - qc.integrated_loudness(x, SR)) < 1e-4  # same mix plus an empty LFE
    six[:, 3] = 0.5  # a loud LFE does not count
    assert abs(qc.integrated_loudness(six, SR) - base) < 1e-4
    assert qc.integrated_loudness(np.zeros((SR, 2), np.float32), SR) == float("-inf")


def test_short_file_loudness_is_too_short_not_silent():
    t = np.arange(int(SR * 0.05)) / SR
    checks = qc.run(_audio(0.3 * np.sin(2 * np.pi * 440 * t)))
    c = next(c for c in checks if c.key == "loudness")
    assert c.code == "loudness.too_short" and c.status == "info"
    tiny = qc.run(_audio(np.zeros(int(SR * 0.05))))
    assert _codes(tiny)["loudness"] == "loudness.silent"


def test_silent_file_true_peak_and_ending_are_not_judged():
    checks = qc.run(_audio(np.zeros((SR * 2, 2))))
    codes = _codes(checks)
    assert codes["silence"] == "silence.all"
    assert codes["true_peak"] == "true_peak.silent" and codes["tail_cut"] == "tail_cut.na"
    tp = next(c for c in checks if c.key == "true_peak")
    assert tp.status == "info" and tp.data["dbtp"] is None and "inf" in tp.value
    _strict_json(checks)


def test_mono_is_info():
    c = next(c for c in qc.run(_audio(_decay())) if c.key == "stereo")
    assert c.code == "stereo.mono" and c.status == "info"


def test_click_on_left_channel_only_reports_its_time():
    x = np.zeros((SR * 3, 2), np.float32)
    x[:, 0] = 0.01 * np.sin(2 * np.pi * 220 * np.arange(SR * 3) / SR)
    x[int(SR * 1.5), 0] += 0.6
    c = next(c for c in qc.run(Audio(x, SR)) if c.key == "clicks")
    assert c.code == "clicks.found" and c.data["count"] == 1
    assert abs(c.data["first"] - 1.5) < 0.01


def test_clicks_across_block_boundaries_match_a_single_pass(monkeypatch):
    x = 0.01 * np.sin(2 * np.pi * 220 * np.arange(SR * 2) / SR).astype(np.float32)
    for i in (1000, 30_000, 65_535, 65_536, 70_000):
        x[i] += 0.6
    whole = qc.clicks(_audio(x))
    monkeypatch.setattr(qc, "_CHUNK", 4096)
    monkeypatch.setattr(qc, "_blocks", lambda n, size=4096, pad=0: _small_blocks(n, size, pad))
    assert qc.clicks(_audio(x)).data == whole.data


def _small_blocks(n, size, pad):
    for s0 in range(0, n, size):
        s1 = min(n, s0 + size)
        yield s0, s1, max(0, s0 - pad), min(n, s1 + pad)


def test_true_peak_blocks_match_whole_file(monkeypatch):
    from scipy.signal import resample_poly

    x = _noise(seconds=1.0, amp=0.3)
    whole = float(np.abs(resample_poly(x, 4, 1, axis=0)).max())
    monkeypatch.setattr(qc, "_blocks", lambda n, size=4096, pad=0: _small_blocks(n, 4096, pad))
    assert abs(qc._true_peak_linear(_audio(x)) - whole) < 1e-6


def test_lowpassed_audio_is_dull():
    from scipy.signal import butter, sosfilt

    sos = butter(16, 6000, fs=SR, output="sos")
    x = sosfilt(sos, _noise(seconds=2.0, amp=0.2), axis=0)
    assert _codes(qc.run(_audio(x)))["bandwidth"] == "bandwidth.dull"


def test_dual_mono_and_wide_stereo():
    m = _decay()
    assert _codes(qc.run(_audio(np.stack([m, m], axis=1))))["stereo"] == "stereo.dual_mono"
    rng = np.random.default_rng(1)
    common = 0.1 * rng.standard_normal(SR * 2)
    side = 0.12 * rng.standard_normal(SR * 2)
    c = next(c for c in qc.run(_audio(np.stack([common + side, common - side], axis=1))) if c.key == "stereo")
    assert c.status == "warn"


def test_loudness_tiers():
    def lufs_code(level):
        x = _noise(seconds=2.0, amp=1.0)
        x *= 10 ** ((level - qc.integrated_loudness(x, SR)) / 20)
        return qc.loudness(Audio(x, SR), None).code

    assert lufs_code(-45) == "loudness.inaudible"
    assert lufs_code(-30) == "loudness.quiet"
    assert lufs_code(-6) == "loudness.loud"
    assert lufs_code(-16) == "loudness.ok"


def test_loudness_target_warns_when_gain_would_clip():
    sr = 48000
    t = np.arange(sr * 2) / sr
    x = (0.02 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)  # quiet bed …
    x[sr : sr + 480] *= 45  # … with one near-full-scale transient: a boost to -14 LUFS would clip it
    checks = {c.key: c for c in qc.run(Audio(np.stack([x, x], 1), sr), target_lufs=-14)}
    c = checks["loudness"]
    assert c.code == "loudness.target_limit"
    assert c.data["peak_after"] > -1 and "limiter" in c.hint
    quiet = {c.key: c for c in qc.run(Audio(np.stack([x, x], 1) * 0.01, sr), target_lufs=-40)}
    assert quiet["loudness"].code != "loudness.target_limit"
