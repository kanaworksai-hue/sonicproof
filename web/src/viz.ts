import { api, media, type QcCheck, type Thumb, type Verdict } from "./api";
import { h, ICON, svg } from "./dom";
import { sigma, t, tm } from "./i18n";
import { store } from "./store";

// -- thumbnails (waveform over spectrogram) ---------------------------------------------------
const thumbs = new Map<string, Promise<Thumb>>();

function getThumb(path: string): Promise<Thumb> {
  let p = thumbs.get(path);
  if (!p) {
    p = api.thumb(path);
    thumbs.set(path, p);
    p.catch(() => thumbs.delete(path));
  }
  return p;
}

// Spectrogram palette: near-black → deep blue → teal → pale amber, quiet enough to sit under the wave.
const STOPS: [number, [number, number, number]][] = [
  [0, [17, 18, 21]],
  [0.35, [22, 38, 58]],
  [0.6, [24, 92, 104]],
  [0.82, [86, 160, 140]],
  [1, [235, 214, 160]],
];
const LUT = (() => {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const x = i / 255;
    let k = 0;
    while (k < STOPS.length - 2 && x > STOPS[k + 1][0]) k++;
    const [x0, c0] = STOPS[k];
    const [x1, c1] = STOPS[k + 1];
    const f = Math.min(1, Math.max(0, (x - x0) / (x1 - x0)));
    for (let j = 0; j < 3; j++) lut[i * 3 + j] = c0[j] + (c1[j] - c0[j]) * f;
  }
  return lut;
})();

const observer = new IntersectionObserver(
  (entries) => {
    for (const e of entries) {
      if (e.isIntersecting) {
        observer.unobserve(e.target);
        (e.target as HTMLElement & { _load?: () => void })._load?.();
      }
    }
  },
  { rootMargin: "200px" },
);

export interface WaveView {
  el: HTMLElement;
  setProgress(frac: number | null): void;
}

/** A spectrogram strip with the waveform drawn over it. Loads when scrolled into view. */
export function waveView(path: string, opts: { height?: number; color?: string; spectrum?: boolean } = {}): WaveView {
  const height = opts.height ?? 44;
  const canvas = h("canvas", { class: "wave-canvas" });
  const head = h("div", { class: "playhead" });
  const el = h("div", { class: "wave loading", style: `height:${height}px` }, canvas, head) as HTMLElement & {
    _load?: () => void;
  };
  let thumb: Thumb | null = null;

  const draw = () => {
    if (!thumb) return;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(10, el.clientWidth);
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext("2d")!;
    if (opts.spectrum !== false) {
      const { spec_rows: R, spec_cols: C } = thumb;
      const raw = Uint8Array.from(atob(thumb.spec), (c) => c.charCodeAt(0));
      const img = new ImageData(C, R);
      for (let r = 0; r < R; r++) {
        for (let c = 0; c < C; c++) {
          const v = raw[r * C + c];
          const o = ((R - 1 - r) * C + c) * 4;
          img.data[o] = LUT[v * 3];
          img.data[o + 1] = LUT[v * 3 + 1];
          img.data[o + 2] = LUT[v * 3 + 2];
          img.data[o + 3] = 255;
        }
      }
      const off = document.createElement("canvas");
      off.width = C;
      off.height = R;
      off.getContext("2d")!.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.globalAlpha = 0.9;
      ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
      ctx.globalAlpha = 1;
    }
    const peaks = thumb.peaks;
    const mid = canvas.height / 2;
    const amp = canvas.height * 0.46;
    const col = canvas.width / peaks.length;
    ctx.fillStyle = opts.color ?? "rgba(236, 238, 242, 0.82)";
    for (let i = 0; i < peaks.length; i++) {
      const [lo, hi] = peaks[i];
      const y0 = mid - hi * amp;
      const y1 = mid - lo * amp;
      ctx.fillRect(i * col, y0, Math.max(1, col - 0.6 * dpr), Math.max(1 * dpr, y1 - y0));
    }
  };

  el._load = () => {
    getThumb(path)
      .then((tb) => {
        thumb = tb;
        el.classList.remove("loading");
        draw();
      })
      .catch(() => {
        el.classList.remove("loading");
        el.classList.add("failed");
      });
  };
  observer.observe(el);
  new ResizeObserver(() => draw()).observe(el);

  return {
    el,
    setProgress(frac) {
      if (frac === null) {
        head.style.display = "none";
      } else {
        head.style.display = "block";
        head.style.left = `${(frac * 100).toFixed(2)}%`;
      }
    },
  };
}

// -- player: one audio element, only ever started by a click -----------------------------------
type Listener = (path: string | null, frac: number) => void;

class Player {
  private audio = new Audio();
  private current: string | null = null;
  private listeners = new Set<Listener>();

  constructor() {
    this.audio.preload = "none";
    this.audio.autoplay = false;
    const tick = () => this.emit();
    this.audio.addEventListener("timeupdate", tick);
    this.audio.addEventListener("pause", tick);
    this.audio.addEventListener("ended", () => {
      this.current = null;
      this.emit();
    });
  }

  private emit() {
    const d = this.audio.duration || 1;
    const frac = this.audio.currentTime / d;
    this.listeners.forEach((f) => f(this.audio.paused ? (this.current ? this.current : null) : this.current, frac));
  }

  playing(path: string) {
    return this.current === path && !this.audio.paused;
  }

  toggle(path: string) {
    if (this.current === path) {
      if (this.audio.paused) this.start();
      else this.audio.pause();
    } else {
      this.current = path;
      this.audio.src = media(path);
      this.start();
    }
    this.emit();
  }

  private start() {
    // a file the browser can't decode just doesn't play; the card already says what's wrong with it
    this.audio.play().catch(() => {
      this.current = null;
      this.emit();
    });
  }

  seek(path: string, frac: number) {
    if (this.current !== path) this.toggle(path);
    const go = () => (this.audio.currentTime = frac * (this.audio.duration || 0));
    if (this.audio.readyState >= 1) go();
    else this.audio.addEventListener("loadedmetadata", go, { once: true });
  }

  stop() {
    this.audio.pause();
    this.current = null;
    this.emit();
  }

  subscribe(f: Listener) {
    this.listeners.add(f);
    return () => this.listeners.delete(f);
  }
}

export const player = new Player();

/** Play button + waveform that follows the player. */
export function playable(path: string, opts: { height?: number; spectrum?: boolean; silent?: boolean } = {}) {
  const wave = waveView(path, opts);
  const btn = h("button", { class: "play", title: t("common.play"), "aria-label": t("common.play") });
  const sync = () => {
    const on = player.playing(path);
    btn.replaceChildren(svg(on ? ICON.pause : ICON.play));
    btn.classList.toggle("on", on);
    btn.title = t(on ? "common.pause" : "common.play");
  };
  sync();
  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    player.toggle(path);
  });
  if (!opts.silent) wave.el.addEventListener("click", (e) => {
    // seek (and let the click also select the card)
    const r = wave.el.getBoundingClientRect();
    player.seek(path, (e.clientX - r.left) / r.width);
  });
  const unsub = player.subscribe((p, frac) => {
    if (!btn.isConnected) {
      unsub();
      return;
    }
    sync();
    wave.setProgress(p === path ? frac : null);
  });
  wave.setProgress(null);
  return { btn, wave };
}

// -- score widgets -----------------------------------------------------------------------------
export const VERDICTS: Verdict[] = ["strong", "plausible", "weak", "miss"];
const LO = -1;
const HI = 3;

/** The calibrated score as a 0–100 "match": −1σ → 0, +3σ → 100. */
export function match(score: number): number {
  return Math.round(((Math.min(HI, Math.max(LO, score)) - LO) / (HI - LO)) * 100);
}

/** Sound-to-sound cosine as a 0–100 "similarity" (clips of any kind sit above ~0.6). */
export function similarity(cos: number): number {
  return Math.round(Math.min(1, Math.max(0, (cos - 0.6) / 0.4)) * 100);
}

const VERDICT_MARK: Record<Verdict, string> = { strong: "◎", plausible: "○", weak: "△", miss: "✕" };

export function meter(score: number, verdict: Verdict | null): HTMLElement {
  const pos = (Math.min(HI, Math.max(LO, score)) - LO) / (HI - LO);
  const ticks = [0.3, 1.1, 2.0].map((x) => h("i", { class: "tick", style: `left:${((x - LO) / (HI - LO)) * 100}%` }));
  return h(
    "div",
    { class: `meter v-${verdict ?? "none"}` },
    h("div", { class: "fill", style: `width:${(pos * 100).toFixed(1)}%` }),
    ...ticks,
  );
}

/** Pro: "plausible". Simple: "○ Close" (◎ ○ △ ✕). */
export function badge(verdict: Verdict): HTMLElement {
  if (store.pro) return h("span", { class: `badge v-${verdict}` }, t(`verdict.${verdict}`));
  return h(
    "span",
    { class: `badge plain v-${verdict}` },
    h("b", { class: "mark" }, VERDICT_MARK[verdict]),
    t(`plain.verdict.${verdict}`),
  );
}

/** Pro: "+1.42σ". Simple: "match 61". */
export function scoreLabel(score: number, verdict: Verdict | null): HTMLElement {
  if (store.pro) return h("span", { class: `score v-${verdict ?? "none"}` }, sigma(score));
  return h(
    "span",
    { class: `score plain v-${verdict ?? "none"}`, title: t("plain.matchTip") },
    h("small", null, t("plain.match")),
    String(match(score)),
  );
}

/** Plain-language technical findings: only what needs attention, then one line for the rest. */
export function qcPlain(checks: QcCheck[]): HTMLElement {
  const issues = checks.filter((c) => c.status === "warn" || c.status === "fail");
  const ok = checks.filter((c) => c.status === "pass" || c.status === "info").length;
  return h(
    "div",
    { class: "qc-plain" },
    issues.map((c) =>
      h(
        "div",
        { class: `qc-row s-${c.status}` },
        h("span", { class: "qc-sym" }, QC_SYM[c.status]),
        h("span", { class: "qc-say" }, tm("plain.qc", c.code, hintData(c)) || qcHint(c) || qcLabel(c)),
      ),
    ),
    h(
      "div",
      { class: "qc-row s-pass" },
      h("span", { class: "qc-sym" }, "✓"),
      h("span", { class: "qc-say" }, issues.length ? t("plain.qcRestOk", { n: ok }) : t("plain.qcAllOk")),
    ),
  );
}

function hintData(c: QcCheck): Record<string, unknown> {
  const data = { ...c.data } as Record<string, unknown>;
  if (typeof data.gain === "number") data.gain = `${data.gain > 0 ? "+" : ""}${data.gain}`;
  return data;
}

export function qcDots(checks: QcCheck[]): HTMLElement {
  return h(
    "span",
    { class: "qc-dots" },
    checks
      .filter((c) => c.status !== "info")
      .map((c) => h("i", { class: `dot s-${c.status}`, title: `${qcLabel(c)}: ${qcValue(c)}` })),
  );
}

export function qcLabel(c: QcCheck): string {
  return tm("qc.label", c.key) ?? c.label;
}

export function qcValue(c: QcCheck): string {
  return tm("qc.value", c.code, c.data) ?? c.value;
}

export function qcHint(c: QcCheck): string {
  if (!c.hint) return "";
  return tm("qc.hint", c.code, hintData(c)) ?? c.hint;
}

const QC_SYM = { pass: "✓", warn: "!", fail: "✕", info: "·" } as const;

export function qcList(checks: QcCheck[], opts: { onlyIssues?: boolean } = {}): HTMLElement {
  const rows = checks.filter((c) => !opts.onlyIssues || c.status === "warn" || c.status === "fail");
  return h(
    "div",
    { class: "qc-list" },
    rows.map((c) =>
      h(
        "div",
        { class: `qc-row s-${c.status}` },
        h("span", { class: "qc-sym" }, QC_SYM[c.status]),
        h("span", { class: "qc-label" }, qcLabel(c)),
        h("span", { class: "qc-value" }, qcValue(c)),
        qcHint(c) ? h("div", { class: "qc-hint" }, qcHint(c)) : null,
      ),
    ),
  );
}

export function spinner(): HTMLElement {
  return h("span", { class: "spinner" });
}

export function progressBar(done: number, total: number): HTMLElement {
  const pct = total > 0 ? Math.min(100, (done / total) * 100) : 0;
  return h(
    "div",
    { class: `progress${total > 0 ? "" : " indeterminate"}` },
    h("div", { class: "bar", style: `width:${pct.toFixed(1)}%` }),
  );
}
