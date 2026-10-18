"""Model-free quality control: the things you can measure without listening."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pyloudnorm as pyln
from scipy.signal import lfilter, resample_poly

from .audio import _CHUNK, SILENCE_DB, Audio, frame_rms_db

PASS, WARN, FAIL, INFO = "pass", "warn", "fail", "info"
MIN_LOUDNESS_S = 0.4  # one BS.1770 gating block


@dataclass
class Check:
    key: str
    label: str
    status: str
    value: str
    hint: str = ""
    # For translated UIs: ``code`` names the finding (e.g. "loudness.quiet") and ``data`` holds the numbers
    # that ``value`` and ``hint`` are written from.
    code: str = ""
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def db(x: float) -> float:
    return 20 * np.log10(max(float(x), 1e-12))


def _blocks(n: int, size: int = _CHUNK, pad: int = 0):
    """(start, end, lo, hi): ``[start, end)`` blocks covering ``n`` samples, ``[lo, hi)`` the same plus up to
    ``pad`` samples of context on each side."""
    for s0 in range(0, n, size):
        s1 = min(n, s0 + size)
        yield s0, s1, max(0, s0 - pad), min(n, s1 + pad)


def _channel_weights(channels: int) -> list[float]:
    """ITU-R BS.1770-4 channel weights.

    Up to 5 channels: pyloudnorm's L, R, C, Ls, Rs order (1, 1, 1, 1.41, 1.41). Six or more (5.1, 7.1): WAV /
    ffmpeg order L, R, C, LFE, then surrounds — LFE is left out (weight 0) and every surround channel gets
    1.41, as BS.1770 does for 7.1."""
    if channels <= 5:
        return [1.0, 1.0, 1.0, 1.41, 1.41][:channels]
    return [1.0, 1.0, 1.0, 0.0] + [1.41] * (channels - 4)


def integrated_loudness(data: np.ndarray, sr: int) -> float:
    """BS.1770-4 integrated loudness (LUFS) for any channel count; ``-inf`` for silence.

    Same K-weighting filters and gating as ``pyloudnorm.Meter.integrated_loudness`` (400 ms blocks, 75 %
    overlap, -70 LUFS absolute and -10 LU relative gates), but filtered block by block so a long file costs
    a few MB instead of several copies of itself, and not limited to 5 channels. Raises ``ValueError`` when
    the audio is shorter than one 400 ms block."""
    x = data if data.ndim == 2 else data[:, None]
    n, ch = x.shape
    t_g, step = 0.4, 0.25
    if n < t_g * sr:
        raise ValueError("Audio must have length greater than the block size.")
    meter = pyln.Meter(sr)
    stages = [(f.b, f.a, f.passband_gain) for f in meter._filters.values()]
    n_blocks = int(np.round((n / sr - t_g) / (t_g * step))) + 1
    j = np.arange(n_blocks)
    lo = (t_g * (j * step) * sr).astype(np.int64)
    hi = np.minimum((t_g * (j * step + 1) * sr).astype(np.int64), n)
    pos = np.unique(np.concatenate([lo, hi]))  # cumulative energy is only needed at block edges
    z = np.zeros((ch, n_blocks))
    for c in range(ch):
        cum = np.zeros(len(pos))  # sum of squared filtered samples before each position
        zi = [np.zeros(max(len(a), len(b)) - 1) for b, a, _ in stages]
        total = 0.0
        for s0, s1, _, _ in _blocks(n):
            y = x[s0:s1, c].astype(np.float64)
            for k, (b, a, gain) in enumerate(stages):  # filter state carries across blocks
                y, zi[k] = lfilter(b, a, y, zi=zi[k])
                y = gain * y
            cs = np.cumsum(y * y)
            sel = (pos > s0) & (pos <= s1)
            cum[sel] = total + cs[pos[sel] - s0 - 1]
            total += float(cs[-1])
        idx = {p: i for i, p in enumerate(pos)}
        z[c] = (cum[[idx[v] for v in hi]] - cum[[idx[v] for v in lo]]) / (t_g * sr)
    g = np.array(_channel_weights(ch))[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        block = -0.691 + 10 * np.log10((g * z).sum(axis=0))
        gated = block >= -70.0
        if not gated.any():
            return float("-inf")
        rel = -0.691 + 10 * np.log10((g[:, 0] * z[:, gated].mean(axis=1)).sum()) - 10.0
        gated = (block > rel) & (block > -70.0)
        if not gated.any():
            return float("-inf")
        return float(-0.691 + 10 * np.log10((g[:, 0] * z[:, gated].mean(axis=1)).sum()))


def _has_signal(a: Audio) -> bool:
    return bool((frame_rms_db(a.data, a.sr, 10) > SILENCE_DB).any())


def loudness(a: Audio, target: float | None) -> Check:
    if not _has_signal(a):
        return Check(
            "loudness",
            "Loudness",
            FAIL,
            "silent",
            "The file is silent.",
            "loudness.silent",
        )
    if a.duration < MIN_LOUDNESS_S:
        return Check(
            "loudness",
            "Loudness",
            INFO,
            "n/a",
            "Too short for integrated loudness (needs at least 0.4 s).",
            "loudness.too_short",
            {"seconds": round(a.duration, 3)},
        )
    try:
        lufs = integrated_loudness(a.data, a.sr)
    except (ValueError, FloatingPointError, MemoryError):
        lufs = float("nan")
    if np.isneginf(lufs):  # signal, but all of it below the -70 LUFS gate
        lufs = -70.0
    if not np.isfinite(lufs):
        return Check(
            "loudness", "Loudness", INFO, "n/a", "Loudness could not be measured.", "loudness.unmeasured"
        )
    value = f"{lufs:.1f} LUFS"
    d = {"lufs": round(float(lufs), 2)}
    if target is not None:
        gain = target - lufs
        st = PASS if abs(gain) <= 1 else WARN if abs(gain) <= 3 else FAIL
        hint = "" if st == PASS else f"Apply {gain:+.1f} dB to hit {target:g} LUFS."
        d |= {"gain": round(float(gain), 1), "target": target}
        return Check("loudness", "Loudness", st, value, hint, "loudness.target" if hint else "loudness.ok", d)
    if lufs < -40:
        return Check(
            "loudness",
            "Loudness",
            FAIL,
            value,
            "Almost inaudible — normalize (e.g. to -16 LUFS).",
            "loudness.inaudible",
            d,
        )
    if lufs < -28:
        d["gain"] = round(float(-16 - lufs), 1)
        return Check(
            "loudness",
            "Loudness",
            WARN,
            value,
            f"Quiet. Normalize: {-16 - lufs:+.1f} dB reaches -16 LUFS (games / web).",
            "loudness.quiet",
            d,
        )
    if lufs > -8:
        d["gain"] = round(float(-14 - lufs), 1)
        return Check(
            "loudness",
            "Loudness",
            WARN,
            value,
            f"Very loud. {-14 - lufs:+.1f} dB reaches -14 LUFS (streaming).",
            "loudness.loud",
            d,
        )
    return Check("loudness", "Loudness", PASS, value, "", "loudness.ok", d)


def _true_peak_linear(a: Audio) -> float:
    """Max magnitude after 4x oversampling (BS.1770 true peak), per channel and in blocks so memory stays
    flat; each block carries 64 samples of context so the interpolation filter sees the same neighbours."""
    x = a.data
    if len(x) <= 8:
        return float(np.abs(x).max()) if x.size else 0.0
    pad, peak = 64, 0.0
    for c in range(x.shape[1]):
        for s0, s1, lo, hi in _blocks(len(x), pad=pad):
            up = resample_poly(x[lo:hi, c], 4, 1).astype(np.float32, copy=False)
            peak = max(peak, float(np.abs(up[(s0 - lo) * 4 : (s1 - lo) * 4]).max()))
    return peak


def true_peak(a: Audio, signal: bool = True) -> tuple[Check, float]:
    peak = _true_peak_linear(a)
    tp = db(peak)
    if not signal:  # silent file: a "pass" at -240 dBTP would be noise
        value = f"{tp:+.1f} dBTP" if peak > 0 else "-inf dBTP"
        d = {"dbtp": round(float(tp), 1) if peak > 0 else None}
        return Check("true_peak", "True peak", INFO, value, "", "true_peak.silent", d), tp
    d = {"dbtp": round(float(tp), 1)}
    if tp > 0:
        return Check(
            "true_peak",
            "True peak",
            FAIL,
            f"{tp:+.1f} dBTP",
            "Inter-sample peaks above 0 dBTP distort on playback; use a true-peak limiter at -1 dBTP.",
            "true_peak.over",
            d,
        ), tp
    if tp > -0.95:  # -1 dBTP ceilings land at -0.99x; don't nag about rounding
        return Check(
            "true_peak",
            "True peak",
            WARN,
            f"{tp:+.1f} dBTP",
            "Little headroom; lossy encoding (MP3/OGG/AAC) may clip. Aim for -1 dBTP.",
            "true_peak.hot",
            d,
        ), tp
    return Check("true_peak", "True peak", PASS, f"{tp:+.1f} dBTP", "", "true_peak.ok", d), tp


def clipping(a: Audio) -> Check:
    runs = 0
    for ch in range(a.channels):  # one channel at a time keeps long files cheap
        x = a.data[:, ch]
        h = ((x >= 0.999) | (x <= -0.999)).view(np.int8)
        z = np.zeros(1, np.int8)  # int8 throughout: a list literal here would promote everything to int64
        edges = np.diff(np.concatenate([z, h, z]))
        starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
        runs += int(((ends - starts) >= 3).sum())
    if runs == 0:
        return Check("clipping", "Clipping", PASS, "none", "", "clipping.none", {"runs": 0})
    st = WARN if runs < 5 else FAIL
    return Check(
        "clipping",
        "Clipping",
        st,
        f"{runs} clipped run{'s' * (runs > 1)}",
        "Flat-topped samples at full scale; regenerate or lower the gain before limiting.",
        "clipping.runs",
        {"runs": runs},
    )


def dc_offset(a: Audio) -> Check:
    off = float(np.abs(a.data.mean(axis=0)).max())
    value = f"{off * 100:.2f}%"
    d = {"percent": round(off * 100, 2)}
    if off > 0.03:
        return Check(
            "dc",
            "DC offset",
            FAIL,
            value,
            "Apply a high-pass filter (~20 Hz) to remove the offset.",
            "dc.high",
            d,
        )
    if off > 0.01:
        return Check(
            "dc", "DC offset", WARN, value, "Small DC offset; a 20 Hz high-pass removes it.", "dc.small", d
        )
    return Check("dc", "DC offset", PASS, value, "", "dc.ok", d)


def silence(a: Audio, loop: bool) -> list[Check]:
    env = frame_rms_db(a.data, a.sr, 10)  # RMS over all channels: out-of-phase stereo does not cancel
    loud = np.where(env > SILENCE_DB)[0]
    if len(loud) == 0:
        return [Check("silence", "Silence", FAIL, "entire file", "No signal above -60 dBFS.", "silence.all")]
    lead = loud[0] * 0.01
    tail = (len(env) - 1 - loud[-1]) * 0.01
    out = []
    out.append(
        Check(
            "lead_silence",
            "Leading silence",
            PASS if lead <= 0.1 else WARN,
            f"{lead:.2f} s",
            "" if lead <= 0.1 else f"Trim {lead:.2f} s at the start so the sound fires on time.",
            "lead_silence.ok" if lead <= 0.1 else "lead_silence.long",
            {"seconds": round(float(lead), 2)},
        )
    )
    d = {"seconds": round(float(tail), 2)}
    if tail > 1.0 or (loop and tail > 0.05):
        out.append(
            Check(
                "tail_silence",
                "Trailing silence",
                WARN,
                f"{tail:.2f} s",
                f"Trim {tail:.2f} s of dead air at the end.",
                "tail_silence.long",
                d,
            )
        )
    else:
        out.append(Check("tail_silence", "Trailing silence", PASS, f"{tail:.2f} s", "", "tail_silence.ok", d))
    return out


def truncated_tail(a: Audio, peak_db: float, signal: bool = True) -> Check:
    if not signal:
        return Check("tail_cut", "Ending", INFO, "n/a", "", "tail_cut.na")
    n = max(1, int(a.sr * 0.02))
    end_db = db(np.sqrt(np.mean(np.square(a.data[-n:], dtype=np.float64))))
    if end_db > SILENCE_DB + 15 and end_db > peak_db - 30:
        return Check(
            "tail_cut",
            "Ending",
            WARN,
            f"ends at {end_db:.0f} dBFS",
            "The sound is cut off mid-decay; add a 20-200 ms fade-out (or use --loop if it loops).",
            "tail_cut.cut",
            {"dbfs": round(float(end_db))},
        )
    return Check("tail_cut", "Ending", PASS, "decays naturally", "", "tail_cut.ok")


def clicks(a: Audio) -> Check:
    """Isolated sample jumps far larger than the local signal — digital clicks / pops."""
    if len(a.data) < a.sr // 50:
        return Check("clicks", "Clicks / pops", PASS, "none", "", "clicks.none", {"count": 0})
    w = max(8, int(a.sr * 0.005))
    kernel = np.ones(w) / w
    count, first = 0, None
    for ch in range(a.channels):
        x = a.data[:, ch]
        # a click is a sample far off the line through its neighbours (impulse), much larger than the
        # usual sample-to-sample roughness around it; bright transients (coins, keys) are rough throughout.
        # r[i] is about sample i + 1. Blocks overlap by w so the moving RMS matches a whole-file pass.
        n, ch_count, prev, ch_first = len(x) - 2, 0, 0, None
        for s0, s1, lo, hi in _blocks(n, pad=w):
            seg = x[lo : hi + 2]
            r = np.abs(seg[1:-1] - (seg[:-2] + seg[2:]) / 2)
            local = np.sqrt(np.convolve(r**2, kernel, mode="same"))
            spikes = ((r > 0.15) & (r > 8 * local))[s0 - lo : s1 - lo]
            if spikes.any():
                if ch_first is None:
                    ch_first = s0 + int(np.argmax(spikes))
                ch_count += int(
                    np.count_nonzero(np.diff(np.concatenate([[prev], spikes.astype(np.int8)])) == 1)
                )
            prev = int(spikes[-1]) if len(spikes) else prev
        count = max(count, ch_count)
        if ch_first is not None:
            t = (ch_first + 1) / a.sr
            first = t if first is None else min(first, t)
    if count == 0 or first is None:
        return Check("clicks", "Clicks / pops", PASS, "none", "", "clicks.none", {"count": 0})
    return Check(
        "clicks",
        "Clicks / pops",
        WARN,
        f"{count} (first at {first:.2f} s)",
        "Short discontinuities; de-click or cross-fade around them.",
        "clicks.found",
        {"count": count, "first": round(float(first), 2)},
    )


def mean_spectrum(a: Audio) -> tuple[np.ndarray, np.ndarray]:
    """(frequencies in Hz, mean power) over the whole file, summed over channels."""
    nfft = 4096
    hop = nfft // 2
    max_frames = 512  # evenly spread over long files: the average spectrum barely moves, memory stays small
    spec = 0.0
    for ch in range(a.channels):  # per channel, so out-of-phase content is not lost
        m = a.data[:, ch]
        if len(m) < nfft:
            m = np.pad(m, (0, nfft - len(m)))
        frames = np.lib.stride_tricks.sliding_window_view(m, nfft)[::hop]
        if len(frames) > max_frames:
            frames = frames[np.linspace(0, len(frames) - 1, max_frames).round().astype(int)]
        spec = spec + (np.abs(np.fft.rfft(frames * np.hanning(nfft), axis=1)) ** 2).mean(axis=0)
    return np.fft.rfftfreq(nfft, 1 / a.sr), np.asarray(spec, dtype=np.float64)


def bandwidth(a: Audio) -> Check:
    freqs, spec = mean_spectrum(a)
    spec_db = 10 * np.log10(np.maximum(spec, 1e-20))
    above = np.where(spec_db > spec_db.max() - 70)[0]
    ceiling = float(freqs[above[-1]]) if len(above) else 0.0
    value = f"content up to {ceiling / 1000:.1f} kHz (sample rate {a.sr / 1000:g} kHz)"
    d = {"khz": round(ceiling / 1000, 1), "sr_khz": a.sr / 1000}
    if a.sr >= 44_100 and ceiling < 12_000:
        return Check(
            "bandwidth",
            "Frequency range",
            WARN,
            value,
            "Little energy above 12 kHz — dull or upsampled from a low rate. Fine for lo-fi/muffled sounds.",
            "bandwidth.dull",
            d,
        )
    return Check("bandwidth", "Frequency range", INFO, value, "", "bandwidth.ok", d)


def stereo(a: Audio) -> Check:
    if a.channels < 2:
        return Check("stereo", "Stereo", INFO, "mono", "", "stereo.mono")
    L, R = a.data[:, 0], a.data[:, 1]
    # block-wise sums in float64 instead of whole-file temporaries
    same, ll, rr, lr, side2, mid2 = True, 0.0, 0.0, 0.0, 0.0, 0.0
    for s0, s1, _, _ in _blocks(len(L)):
        lb, rb = L[s0:s1].astype(np.float64), R[s0:s1].astype(np.float64)
        same = same and np.allclose(lb, rb, atol=1e-4)
        ll, rr, lr = ll + float(lb @ lb), rr + float(rb @ rb), lr + float(lb @ rb)
        side2 += float(np.square((lb - rb) / 2).sum())
        mid2 += float(np.square((lb + rb) / 2).sum())
    if same:
        return Check(
            "stereo",
            "Stereo",
            INFO,
            "dual mono (L = R)",
            "Could be stored as mono to halve size.",
            "stereo.dual_mono",
        )
    denom = np.sqrt(ll * rr)
    corr = float(lr / denom) if denom > 0 else 0.0
    side = db(np.sqrt(side2 / len(L)))
    mid = db(np.sqrt(mid2 / len(L)))
    value = f"correlation {corr:+.2f}"
    d = {"correlation": round(corr, 2)}
    if corr < -0.3:
        return Check(
            "stereo",
            "Stereo phase",
            FAIL,
            value,
            "Channels are out of phase; the sound will cancel on mono speakers (phones). Flip one channel.",
            "stereo.out_of_phase",
            d,
        )
    if corr < -0.1 or side > mid + 3:
        return Check(
            "stereo",
            "Stereo phase",
            WARN,
            value,
            "Wide/phasey image; check how it sounds folded to mono.",
            "stereo.wide",
            d,
        )
    return Check("stereo", "Stereo phase", PASS, value, "", "stereo.ok", d)


def loop_seam(a: Audio) -> Check:
    x = a.data
    jump = float(np.abs(x[-1] - x[0]).max())
    # 300 ms windows: 50 ms ones land between crackles of sparse textures (fire, rain) and report fake gaps
    n = max(1, min(int(a.sr * 0.3), len(x) // 4))
    head = db(np.sqrt(np.mean(x[:n] ** 2)))
    tail = db(np.sqrt(np.mean(x[-n:] ** 2)))
    level_gap = abs(head - tail)
    value = f"jump {jump:.3f}, level gap {level_gap:.1f} dB"
    d = {"jump": round(jump, 3), "gap_db": round(float(level_gap), 1)}
    if jump > 0.1:
        return Check(
            "loop",
            "Loop seam",
            FAIL,
            value,
            "Audible click at the loop point; cross-fade end into start.",
            "loop.click",
            d,
        )
    if level_gap > 6 or jump > 0.03:
        return Check(
            "loop",
            "Loop seam",
            WARN,
            value,
            "Start and end don't match; cross-fade 50-200 ms or cut at a zero crossing.",
            "loop.mismatch",
            d,
        )
    return Check("loop", "Loop seam", PASS, value, "", "loop.ok", d)


def _finite(v):
    """JSON-safe: NaN / inf numbers become None (strict JSON has no NaN)."""
    if isinstance(v, float | np.floating) and not np.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: _finite(x) for k, x in v.items()}
    return v


def run(a: Audio, loop: bool = False, target_lufs: float | None = None) -> list[Check]:
    if len(a.data) == 0 or a.data.size == 0:
        return [
            Check("file", "File", FAIL, "empty", "The file has no audio samples (0 frames).", "file.empty")
        ]
    pre = []
    bad = int(a.data.size - np.count_nonzero(np.isfinite(a.data)))
    if bad:  # float files can carry NaN / inf; measure the rest as if those samples were clipped / zero
        pre.append(
            Check(
                "file",
                "File",
                FAIL,
                f"{bad} invalid sample{'s' * (bad > 1)}",
                "The file contains NaN / infinite samples; re-export it.",
                "file.nonfinite",
                {"count": bad},
            )
        )
        a = Audio(np.nan_to_num(a.data, nan=0.0, posinf=1.0, neginf=-1.0), a.sr, a.path, a.subtype)
    signal = _has_signal(a)
    tp_check, tp = true_peak(a, signal)
    loud = loudness(a, target_lufs)
    gain = loud.data.get("gain") if loud.code == "loudness.target" else None
    if gain is not None and tp is not None and np.isfinite(tp) and tp + gain > -1.0:
        # A plain gain change would clip: say so instead of suggesting it blindly.
        loud.code = "loudness.target_limit"
        loud.hint += f" Peaks would reach {tp + gain:+.1f} dBTP — use a limiter (ceiling −1 dBTP)."
        loud.data["peak_after"] = round(float(tp + gain), 1)
    checks = [*pre, loud, tp_check, clipping(a), dc_offset(a), *silence(a, loop)]
    checks.append(loop_seam(a) if loop else truncated_tail(a, tp, signal))
    checks += [clicks(a), bandwidth(a), stereo(a)]
    for c in checks:
        c.data = _finite(c.data)
    return checks


def summary(checks: list[Check]) -> str:
    sts = {c.status for c in checks}
    return FAIL if FAIL in sts else WARN if WARN in sts else PASS
