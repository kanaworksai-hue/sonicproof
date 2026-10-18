"""Sonicproof command line: check / rank / qc / report / fit / index / search / dedupe / serve / mcp."""

from __future__ import annotations

import functools
import json
import math
import os
import sys
from pathlib import Path

# An empty HF_HOME / HF_HUB_CACHE (`-e HF_HOME="$HF_HOME"` when it was never set) would make huggingface_hub
# download into ./hub of whatever folder the process starts in: treat empty as unset, before any hub import.
for _var in ("HF_HOME", "HF_HUB_CACHE"):
    if os.environ.get(_var) == "":
        del os.environ[_var]
# Quiet third-party chatter before anything heavy is imported (progress bars come back for a first download).
_WE_HID_HUB_BARS = "HF_HUB_DISABLE_PROGRESS_BARS" not in os.environ
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import typer
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import __version__, service
from .errors import InputError, SonicproofError

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,
    help="[bold]Sonicproof[/] — the audio QA layer for generative audio.\n\n"
    "Does the clip sound like the prompt? Which seed is best? Is it technically clean? "
    "Does it suit the picture? And where in your library is that sound?",
)
console = Console()
err = Console(stderr=True)
_debug = os.environ.get("SONICPROOF_DEBUG", "") not in ("", "0")

EXIT_HELP = "Exit codes: 0 ok · 1 qc failed (or --strict and not ship) · 2 bad input · 3 setup problem."


def _utf8_stdio() -> None:
    """σ ✓ ━ and CJK must not crash on a cp1252 / cp932 pipe (Windows, captured output)."""
    for s in (sys.stdout, sys.stderr):
        if (
            hasattr(s, "reconfigure")
            and (getattr(s, "encoding", "") or "").lower().replace("-", "") != "utf8"
        ):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _fail(e: BaseException, as_json: bool = False):
    """One red line (and ``{"error": …}`` on stdout with --json), exit 2 for input / 3 for setup problems."""
    se = service.as_error(e)
    if as_json:
        print(_json({"error": se.to_dict()}))
    err.print(Text(se.message, style="red"), soft_wrap=True)
    if se.code == "internal":
        err.print("[dim]Run again with --debug (or SONICPROOF_DEBUG=1) for the traceback.[/]")
    raise typer.Exit(se.exit_code) from e


def _guarded(fn):
    """Expected failures become one line + an exit code instead of a traceback (unless --debug)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (typer.Exit, typer.Abort, KeyboardInterrupt):
            raise
        except Exception as e:
            import click

            if isinstance(e, click.exceptions.ClickException) or _debug:
                raise
            _fail(e, bool(kwargs.get("as_json")))

    return wrapper


def command(*args, **kwargs):
    """``app.command`` with the error guard."""

    def deco(fn):
        return app.command(*args, **kwargs)(_guarded(fn))

    return deco


def _lufs(value: float | None) -> float | None:
    if value is not None and not (math.isfinite(value) and -70 <= value <= -5):
        raise typer.BadParameter("give a loudness between -70 and -5 LUFS, e.g. -16 or -23.")
    return value


def _narrow() -> bool:
    return console.width < 100


VERDICT_STYLE = {"strong": "bold green", "plausible": "bold cyan", "weak": "bold yellow", "miss": "bold red"}
STATUS_STYLE = {"pass": ("✓", "green"), "warn": ("!", "yellow"), "fail": ("✗", "red"), "info": ("·", "dim")}


def _finite(o):
    """NaN / inf → None, so ``--json`` output is always strict JSON."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_finite(v) for v in o]
    return o


def _json(o) -> str:
    return json.dumps(_finite(o), ensure_ascii=False, indent=2, allow_nan=False)


def _judge(vision: bool = False):
    from .model import DOWNLOAD_SIZE, hub_cache_dir, model_cached, prefer_offline

    service.require_model()  # missing extra / offline without the model: one line, before anything heavy
    label = "EmbeddingGemma 2 + vision encoder" if vision else "EmbeddingGemma 2"
    if not model_cached():
        # first run: say what is happening and let the hub's progress bars (stderr) show instead of a spinner
        err.print(f"Downloading EmbeddingGemma 2 ({DOWNLOAD_SIZE}) to {hub_cache_dir()} — first run only.")
        if _WE_HID_HUB_BARS:
            os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
            try:
                from huggingface_hub.utils import enable_progress_bars

                enable_progress_bars()
            except ImportError:
                pass
        from .judge import Judge

        j = Judge(vision=vision)
        _ = j.V
        return j
    prefer_offline()
    with err.status(f"[dim]Loading {label}…[/]", spinner="dots"):
        from .judge import Judge

        j = Judge(vision=vision)
        _ = j.V  # vocabulary embeddings (cached after the first run)
    return j


def _meter(score: float, width: int = 28) -> Text:
    """−1σ … +3σ bar with the verdict thresholds marked."""
    from .judge import verdict

    lo, hi = -1.0, 3.0
    pos = round((min(max(score, lo), hi) - lo) / (hi - lo) * (width - 1))
    style = VERDICT_STYLE[verdict(score)].replace("bold ", "")
    bar = Text()
    for i in range(width):
        bar.append("━" if i <= pos else "─", style=style if i <= pos else "grey35")
    return bar


def _alt_lines(alts, mark: str) -> Text:
    out = Text()
    for i, a in enumerate(alts):
        if i:
            out.append("\n")
        out.append(f"{a.score:+.2f}σ  ", style="dim")
        out.append(a.text, style="green" if a.agrees else "")
        if a.agrees:
            out.append(f"  {mark}", style="green")
        if a.category:
            out.append(f"  {a.category.replace('_', ' ')}", style="dim italic")
    return out


def _check_panel(r, show_file: bool = True) -> Panel:
    from .judge import VERDICT_HINT

    head = Text()
    head.append(f" {r.verdict.upper()} ", style=f"reverse {VERDICT_STYLE[r.verdict]}")
    head.append(f"  {r.score:+.2f}σ  ", style="bold")
    head.append_text(_meter(r.score))
    head.append(f"\n{VERDICT_HINT[r.verdict]}", style="dim")

    meta = Table.grid(padding=(0, 2))
    meta.add_column(style="dim", justify="right")
    meta.add_column()
    src = " [dim](from sidecar)[/]" if r.prompt_source == "sidecar" else ""
    meta.add_row("prompt", f"“{r.prompt}”{src}")
    meta.add_row("ranking", f"#{r.vocab_rank} of {r.vocab_size + 1} descriptions for this clip")
    meta.add_row(
        "evidence", f"cos {r.cosine:.3f} · clip-side {r.z_clip:+.2f}σ · prompt-side {r.z_prompt:+.2f}σ"
    )

    body = Group(head, Text(""), meta, Text("\nHeard as", style="bold"), _alt_lines(r.heard_as, "✓"))
    title = f"[bold]{Path(r.file).name}[/]" if show_file else None
    return Panel(body, title=title, title_align="left", border_style="grey50", padding=(1, 2))


def _qc_table(checks) -> Table:
    """Status · label · value · hint; under 100 columns the hint goes under the value instead."""
    narrow = _narrow()
    t = Table(box=None, show_header=False, padding=(0, 1), expand=False)
    t.add_column(width=1, no_wrap=True)
    t.add_column(style="bold", no_wrap=True)
    t.add_column(max_width=None if narrow else 30)
    if not narrow:
        t.add_column(style="dim", min_width=24)
    for c in checks:
        c = c if isinstance(c, dict) else c.to_dict()
        sym, color = STATUS_STYLE[c["status"]]
        value = Text(c["value"], style=color if c["status"] != "pass" else "")
        if narrow:
            if c["hint"]:
                value.append("\n" + c["hint"], style="dim")
            t.add_row(Text(sym, style=color), c["label"], value)
        else:
            t.add_row(Text(sym, style=color), c["label"], value, c["hint"])
    return t


def _probe(path: Path) -> Path:
    """Fail fast (no model loaded yet) on a missing, empty, folder or undecodable input."""
    service.probe_audio(path)
    return path


ERROR_STYLE = ("✗", "red")


@command(epilog=EXIT_HELP)
def check(
    file: Path = typer.Argument(..., help="Audio file (any format ffmpeg can read)."),
    prompt: str = typer.Argument(
        None,
        help="What it should sound like. Any language. "
        "Omit to use the prompt in a Stable Audio .json sidecar.",
    ),
    top: int = typer.Option(
        5, "--top", "-n", min=0, max=50, help="How many 'heard as' descriptions to show."
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
):
    """Does this clip sound like the prompt?"""
    _probe(file)
    service.need_prompt(file, prompt)
    j = _judge()
    r = j.check(file, prompt, n_alt=top)
    if as_json:
        print(_json(r.to_dict()))
    else:
        console.print(_check_panel(r))


@command(epilog=EXIT_HELP + ' Unreadable takes are listed last (verdict "error") and give exit code 2.')
def rank(
    prompt: str = typer.Argument(..., help="The prompt all candidates were generated from."),
    files: list[Path] = typer.Argument(..., help="Candidate files (e.g. several seeds)."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
):
    """Rank several takes of the same prompt — best first."""
    if not prompt.strip():
        raise InputError("The prompt is empty.", code="no_prompt")
    errors = []
    for f in files:
        try:
            _probe(f)
        except SonicproofError as e:
            errors.append({"file": str(f), "verdict": "error", "error": e.to_dict()})
    if len(errors) == len(files):  # nothing to listen to: don't load the model
        results = errors
    else:
        j = _judge()
        with err.status(f"[dim]Listening to {len(files)} files…[/]", spinner="dots"):
            results = service.rank_records(j, prompt, files)
    code = 2 if any(r["verdict"] == "error" for r in results) else 0
    if as_json:
        print(_json(results))
        raise typer.Exit(code)
    t = Table(box=None, padding=(0, 2 if not _narrow() else 1), header_style="dim")
    t.add_column("#", justify="right", no_wrap=True)
    t.add_column("file", overflow="fold")
    t.add_column("score", justify="right", no_wrap=True)
    if not _narrow():
        t.add_column("", no_wrap=True)
    t.add_column("verdict", no_wrap=True)
    t.add_column("heard as", style="dim", overflow="fold")
    for i, r in enumerate(results, 1):
        name = Text(Path(r["file"]).name, style="bold" if i == 1 else "")
        if r["verdict"] == "error":
            row = [str(i), name, "", Text("error", style="red"), Text(r["error"]["message"], style="red")]
            if not _narrow():
                row.insert(3, "")
            t.add_row(*row)
            continue
        verdict = Text(r["verdict"], style=VERDICT_STYLE[r["verdict"]])
        if r.get("note"):
            verdict.append(f" ({r['note']})", style="dim")
        row = [
            str(i),
            name,
            f"{r['score']:+.2f}σ",
            verdict,
            r["heard_as"][0]["text"] if r["heard_as"] else "",
        ]
        if not _narrow():
            row.insert(3, _meter(r["score"], 16))
        t.add_row(*row)
    console.print(
        Panel(t, title=f"[bold]“{prompt}”[/]", title_align="left", border_style="grey50", padding=(1, 2))
    )
    best = results[0]
    if best["verdict"] == "error":
        console.print("[red]None of the files could be read.[/]")
    elif best["verdict"] in ("weak", "miss"):
        console.print("[yellow]None of the takes clearly match — try new seeds or rephrase the prompt.[/]")
    else:
        console.print(f"[green]Pick:[/] [bold]{Path(best['file']).name}[/]")
    if len(files) == 1:
        console.print("[dim]Only one take — `sp check` says the same with more detail.[/]")
    raise typer.Exit(code)


def _qc_panel(rec: dict, name: str) -> Panel:
    status = rec["status"]
    if status == "error":
        sym, color = ERROR_STYLE
        body, sub = Text(rec["error"]["message"], style="red"), ""
    else:
        sym, color = STATUS_STYLE[status]
        body = _qc_table(rec["checks"])
        sub = f"{rec['duration']:.2f} s · {rec['sample_rate'] / 1000:g} kHz · {rec['channels']} ch"
    return Panel(
        body,
        title=f"[bold]{name}[/]  [dim]{sub}[/]",
        title_align="left",
        subtitle=Text(f"{sym} {status}", style=color),
        subtitle_align="right",
        border_style="grey50",
        padding=(1, 2),
    )


@command(epilog=EXIT_HELP + " --json always prints a list, one record per file.")
def qc(
    files: list[Path] = typer.Argument(..., help="Audio files to inspect."),
    loop: bool = typer.Option(False, "--loop", help="The file is meant to loop: check the seam."),
    target: float = typer.Option(
        None, "--target-lufs", callback=_lufs, help="Required loudness, -70…-5, e.g. -16 or -23."
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
):
    """Technical checks that need no model: loudness, peaks, clipping, silence, clicks, phase…"""
    out = []
    for f in files:
        rec = service.qc_record_safe(f, loop, target)  # one bad file never sinks the batch
        out.append(rec)
        if not as_json:
            console.print(_qc_panel(rec, f.name))
    if as_json:
        print(_json(out))
    worst = service.worst_status(out)
    raise typer.Exit(2 if worst == "error" else 1 if worst == "fail" else 0)


@command(epilog=EXIT_HELP)
def report(
    file: Path = typer.Argument(..., help="Audio file."),
    prompt: str = typer.Argument(None, help="What it should sound like (defaults to the sidecar prompt)."),
    loop: bool = typer.Option(False, "--loop", help="The file is meant to loop."),
    target: float = typer.Option(None, "--target-lufs", callback=_lufs, help="Required loudness, e.g. -16."),
    strict: bool = typer.Option(
        False, "--strict", help="Exit 1 unless the overall verdict is ship (for CI)."
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
):
    """One scorecard: prompt adherence + technical QC."""
    _probe(file)
    service.need_prompt(file, prompt)
    q = service.qc_record(file, loop, target)
    j = _judge()
    r = j.check(file, prompt, n_alt=3)
    rec = service.report_record(r.to_dict(), q)
    code = 1 if strict and rec["overall"] != "ship" else 0
    if as_json:
        print(_json(rec))
        raise typer.Exit(code)
    overall, note, checks = rec["overall"], rec["note"], q["checks"]
    color = {"ship": "green", "review": "yellow", "redo": "red"}[overall]
    head = Text()
    head.append(f" {overall.upper()} ", style=f"reverse bold {color}")
    head.append(f"  {note}", style=color)
    sub = f"{q['duration']:.2f} s · {q['sample_rate'] / 1000:g} kHz · {q['channels']} ch"
    console.print(
        Panel(
            Group(
                head,
                Text(""),
                _check_panel(r, show_file=False),
                Panel(
                    _qc_table(checks),
                    title="[bold]Technical[/]",
                    title_align="left",
                    border_style="grey35",
                    padding=(1, 2),
                ),
            ),
            title=f"[bold]{file.name}[/]  [dim]{sub}[/]",
            title_align="left",
            border_style=color,
            padding=(1, 2),
        )
    )
    raise typer.Exit(code)


def _resolve_picture(file: Path | None, image: Path | None, video: Path | None) -> tuple[Path, str]:
    """--image / --video, or the video track of FILE itself when it is a video."""
    if image and video:
        raise InputError("Give either --image or --video, not both.", code="bad_value")
    if image or video:
        _, kind = service.resolve_picture(None, image or video, "video" if video else None)
        return Path(image or video), kind
    from .visual import VIDEO_EXTS

    if file is None or file.suffix.lower() not in VIDEO_EXTS:
        raise InputError(
            "Give a picture: --image frame.png or --video clip.mp4 (or pass a video file as FILE).",
            code="bad_value",
        )
    service.resolve_picture(file, None)
    return file, "video"


def _fit_panel(r) -> Panel:
    from .judge import FIT_HINT

    head = Text()
    head.append(f" {r.verdict.upper()} ", style=f"reverse {VERDICT_STYLE[r.verdict]}")
    head.append(f"  {r.score:+.2f}σ  ", style="bold")
    head.append_text(_meter(r.score))
    head.append(f"\n{FIT_HINT[r.verdict]}", style="dim")

    meta = Table.grid(padding=(0, 2))
    meta.add_column(style="dim", justify="right")
    meta.add_column()
    pic = Path(r.picture).name
    kind = f"video · {r.frames} frame{'s' if r.frames != 1 else ''}" if r.picture_kind == "video" else "image"
    if Path(r.picture) == Path(r.file):
        pic += " [dim](its own picture)[/]"
    meta.add_row("picture", f"{pic} [dim]· {kind}[/]")
    if r.scene:
        meta.add_row("scene", f"“{r.scene}”")
    ev = f"cos {r.cosine:.3f} · direct {r.z_direct:+.2f}σ · described {r.z_described:+.2f}σ"
    if r.z_scene is not None:
        ev += f" · scene {r.z_scene:+.2f}σ"
    meta.add_row("evidence", ev)

    parts = [
        head,
        Text(""),
        meta,
        Text("\nPictured as", style="bold"),
        _alt_lines(r.pictured_as, "✓ heard"),
        Text("\nHeard as", style="bold"),
        _alt_lines(r.heard_as, "✓ pictured"),
    ]
    if r.verdict in ("weak", "miss") and r.pictured_as:
        tip = Text("\nTry a prompt like ", style="dim")
        tip.append(f"“{r.pictured_as[0].text}”", style="bold")
        parts.append(tip)
    return Panel(
        Group(*parts),
        title=f"[bold]{Path(r.file).name}[/]",
        title_align="left",
        border_style="grey50",
        padding=(1, 2),
    )


@command(epilog=EXIT_HELP)
def fit(
    file: Path = typer.Argument(..., help="The sound (audio, or a video whose soundtrack to judge)."),
    scene: str = typer.Argument(None, help="Optional words about the scene, any language."),
    image: Path = typer.Option(None, "--image", "-i", help="A still: frame, concept art, screenshot."),
    video: Path = typer.Option(None, "--video", "-v", help="A clip; sampled at 1 fps (≤ 32 frames)."),
    top: int = typer.Option(5, "--top", "-n", min=0, max=50, help="How many descriptions to show."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
):
    """Does this sound suit the picture? Image, video, or a video's own soundtrack."""
    pic, kind = _resolve_picture(file, image, video)
    _probe(file)  # also catches a video without a soundtrack, before the model loads
    j = _judge(vision=True)
    with err.status(f"[dim]Looking at {pic.name} and listening to {file.name}…[/]", spinner="dots"):
        r = j.fit(file, pic, kind, scene=scene, n_alt=top)
    if as_json:
        print(_json(r.to_dict()))
    else:
        console.print(_fit_panel(r))


# -- library ------------------------------------------------------------------------------------
def _fmt_duration(sec: float) -> str:
    sec = round(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _open_library(directory: Path, in_cache: bool = False, judge=None, quiet: bool = False):
    """Load the index and bring it up to date. Returns (library, update stats, judge-or-None)."""
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn

    from .library import Library

    root = service.library_root(directory)
    lib = Library(root, in_cache=in_cache)
    holder = {"judge": judge}

    def embed(wave):
        if holder["judge"] is None:
            from .judge import Judge
            from .model import model_cached, prefer_offline

            if not model_cached():
                progress.stop()
                holder["judge"] = _judge()  # checks the extra itself
                progress.start()
            else:
                service.require_model()
                prefer_offline()
                progress.update(task, description="[dim]Loading EmbeddingGemma 2…[/]")
                holder["judge"] = Judge()
        return holder["judge"].embedder.embed_audio([wave])[0]

    progress = Progress(
        TextColumn("{task.description}"),
        BarColumn(bar_width=28, complete_style="cyan", finished_style="green"),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=err,
        transient=True,
        disable=quiet,
    )
    with progress:
        task = progress.add_task("[dim]Indexing…[/]", total=None)

        def on_progress(i, total, rel):
            name = rel if len(rel) <= 40 else "…" + rel[-39:]
            progress.update(task, completed=i, total=total, description=f"[dim]Indexing[/] {name}")

        stats = lib.update(embed, progress=on_progress)
    return lib, stats, holder["judge"]


def _failed_note(failed: list[str]) -> None:
    if failed:
        n = len(failed)
        err.print(
            f"[yellow]{n} file{'s' if n != 1 else ''} could not be read[/] [dim](see `sp index` for which)[/]"
        )


@command(epilog=EXIT_HELP)
def index(
    directory: Path = typer.Argument(..., help="Library folder (scanned recursively).", show_default=False),
    in_cache: bool = typer.Option(
        False, "--in-cache", help="Keep the index in ~/.cache/sonicproof instead of <folder>/.sonicproof."
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
):
    """Index a sound library for search and dedupe. Re-run any time: only new or changed files are read."""
    lib, st, _ = _open_library(directory, in_cache, quiet=as_json)
    if as_json:
        print(_json(service.index_record(lib, st)))
        return
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim", justify="right")
    t.add_column()
    t.add_row("sounds", f"[bold]{len(lib)}[/] · {_fmt_duration(lib.total_duration())} total")
    changes = []
    for n, label, style in (
        (st.added, "new", "green"),
        (st.changed, "changed", "cyan"),
        (st.removed, "removed", "yellow"),
    ):
        if n:
            changes.append(f"[{style}]{n} {label}[/]")
    changes.append(f"[dim]{st.unchanged} unchanged[/]")
    t.add_row("this run", " · ".join(changes))
    fresh = st.added + st.changed
    if fresh:
        t.add_row("listened", f"{st.embedded} embedded · {fresh - st.embedded} from cache")
    t.add_row("index", str(lib.location))
    body = [t]
    if st.failed:
        bad = Text("\nCould not read", style="bold red")
        for f in st.failed[:10]:
            bad.append(f"\n  {f}", style="red")
        if len(st.failed) > 10:
            bad.append(f"\n  … and {len(st.failed) - 10} more", style="red")
        body.append(bad)
    console.print(
        Panel(
            Group(*body),
            title=f"[bold]{lib.root.name}[/]",
            title_align="left",
            border_style="grey50",
            padding=(1, 2),
        )
    )


@command(epilog=EXIT_HELP)
def search(
    query: str = typer.Argument(None, help="What you are looking for, any language."),
    directory: Path = typer.Option(None, "--in", "-d", help="Library folder (required; indexed on the fly)."),
    like: Path = typer.Option(None, "--like", "-l", help="Find sounds like this file instead."),
    image: Path = typer.Option(None, "--image", "-i", help="Find sounds that suit this picture."),
    video: Path = typer.Option(None, "--video", "-v", help="Find sounds that suit this clip."),
    top: int = typer.Option(10, "--top", "-n", min=1, max=50, help="How many results (1-50)."),
    in_cache: bool = typer.Option(False, "--in-cache", help="Index lives in the cache (see `sp index`)."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
):
    """Search a library by words, by example (--like) or by picture (--image / --video)."""
    from . import library as libmod

    modes = [x for x in (like, image, video) if x is not None]
    if len(modes) > 1 or (like is not None and query):
        raise InputError(
            "Search by words, --like, --image or --video — one at a time (words may describe a picture).",
            code="bad_value",
        )
    if not modes and not query:
        raise InputError(
            'What should I look for? e.g. sp search "rain on a tin roof" --in ~/SFX', code="bad_value"
        )
    if directory is None:
        raise InputError('Which library? Add --in FOLDER, e.g. sp search "rain" --in ~/SFX', code="bad_value")
    service.library_root(directory)
    if like is not None:
        _probe(like)
    elif image is not None or video is not None:
        _resolve_picture(None, image, video)
    vision = image is not None or video is not None
    j = _judge(vision=vision)
    lib, st, _ = _open_library(directory, in_cache, judge=j, quiet=as_json)
    with err.status("[dim]Searching…[/]", spinner="dots"):
        if like is not None:
            hits, title, mode = libmod.search_like(lib, j, like, top), f"sounds like {like.name}", "like"
        elif vision:
            pic = image or video
            v = j.image(pic) if image is not None else j.video(pic)[0]
            hits, mode = libmod.search_picture(lib, j, v, query, top), "picture"
            title = f"sounds for {pic.name}" + (f" · “{query}”" if query else "")
        else:
            hits, title, mode = libmod.search_text(lib, j, query, top), f"“{query}”", "words"
    if as_json:
        out = service.search_record(lib, mode, query, hits, like, image or video, st.failed)
        print(_json(out))
        return
    _failed_note(st.failed)
    if not hits:
        console.print(f"[yellow]No sounds in {lib.root}.[/]")
        return
    t = Table(box=None, padding=(0, 1), header_style="dim", expand=True)
    t.add_column("#", justify="right", no_wrap=True, min_width=2)
    t.add_column("file", no_wrap=True, overflow="ellipsis", ratio=3)
    t.add_column("score" if mode != "like" else "similarity", justify="right", no_wrap=True, min_width=7)
    if mode != "like":
        t.add_column("", no_wrap=True, min_width=12)
        t.add_column("verdict", no_wrap=True, min_width=9)
    t.add_column("length", justify="right", style="dim", no_wrap=True, min_width=7)
    t.add_column("heard as", style="dim", no_wrap=True, overflow="ellipsis", ratio=2)
    for i, h in enumerate(hits, 1):
        name = Text(h.path, style="bold" if i == 1 else "")
        if mode == "like":
            t.add_row(str(i), name, f"{h.score:.3f}", f"{h.duration:.1f} s", h.heard_as or "")
        else:
            t.add_row(
                str(i),
                name,
                f"{h.score:+.2f}σ",
                _meter(h.score, 12),
                Text(h.verdict, style=VERDICT_STYLE[h.verdict]),
                f"{h.duration:.1f} s",
                h.heard_as or "",
            )
    console.print(
        Panel(
            t,
            title=f"[bold]{title}[/]",
            title_align="left",
            subtitle=f"[dim]{len(lib)} sounds in {lib.root.name}[/]",
            subtitle_align="right",
            border_style="grey50",
            padding=(1, 2),
        )
    )


@command(epilog=EXIT_HELP)
def dedupe(
    directory: Path = typer.Argument(..., help="Library folder.", show_default=False),
    match: float = typer.Option(
        None,
        "--match",
        min=0.5,
        max=1.0,
        help="Fingerprint correlation needed to call two files the same audio (0.5-1.0, default 0.98).",
    ),
    in_cache: bool = typer.Option(False, "--in-cache", help="Index lives in the cache (see `sp index`)."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
):
    """Find duplicate and near-duplicate files (re-encodes, gain changes, trims) and pick one to keep."""
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

    from .library import MATCH, find_duplicates

    lib, st, _ = _open_library(directory, in_cache, quiet=as_json)
    with Progress(
        TextColumn("[dim]Comparing[/] {task.description}"),
        BarColumn(bar_width=28, complete_style="cyan", finished_style="green"),
        MofNCompleteColumn(),
        console=err,
        transient=True,
        disable=as_json,
    ) as progress:
        task = progress.add_task("", total=None)
        groups = find_duplicates(
            lib,
            match_min=match if match is not None else MATCH,
            progress=lambda i, n, rel: progress.update(task, completed=i, total=n, description=rel[-40:]),
        )
    redundant = sum(len(g.members) - 1 for g in groups)
    reclaim = sum(g.reclaimable_bytes for g in groups)
    if as_json:
        print(_json(service.dedupe_record(lib, groups, st.failed)))
        return
    _failed_note(st.failed)
    if not groups:
        console.print(f"[green]✓ No duplicates among {len(lib)} sounds in {lib.root.name}.[/]")
        return
    rel_style = {
        "keep": "bold green",
        "identical": "red",
        "same audio": "yellow",
        "trimmed": "cyan",
        "same start": "magenta",
    }
    narrow = _narrow()
    for g in groups:
        t = Table(box=None, padding=(0, 1 if narrow else 2), show_header=False)
        t.add_column(no_wrap=True, min_width=10)
        t.add_column(overflow="fold", min_width=16)
        t.add_column(style="dim", no_wrap=True, overflow="ellipsis")
        t.add_column(justify="right", style="dim", no_wrap=True, min_width=11)
        for m in g.members:
            if narrow:
                fmt = f"{m.sample_rate / 1000:g}k {m.channels}ch " + ("lossless" if m.lossless else "lossy")
            else:
                fmt = f"{m.duration:.2f} s · {m.sample_rate / 1000:g} kHz · {m.channels} ch"
                fmt += " · lossless" if m.lossless else " · lossy"
            match_txt = (
                ""
                if m.relation == "keep"
                else ("same bytes" if m.relation == "identical" else f"match {m.match:.3f}")
            )
            t.add_row(Text(m.relation, style=rel_style[m.relation]), m.path, fmt, match_txt)
        console.print(
            Panel(
                t,
                title=f"[bold]{Path(g.keep).name}[/]  [dim]keep: {g.reason}[/]",
                title_align="left",
                subtitle=f"[dim]{_fmt_bytes(g.reclaimable_bytes)} reclaimable[/]",
                subtitle_align="right",
                border_style="grey50",
                padding=(0, 1),
            )
        )
    console.print(
        f"[bold]{len(groups)}[/] group{'s' if len(groups) != 1 else ''} · "
        f"[bold]{redundant}[/] redundant file{'s' if redundant != 1 else ''} · "
        f"[bold]{_fmt_bytes(reclaim)}[/] reclaimable [dim](nothing was deleted)[/]"
    )


@command()
def serve(
    port: int = typer.Option(7788, "--port", "-p", min=1, max=65535, help="Port on localhost."),
    no_open: bool = typer.Option(False, "--no-open", help="Don't open the browser."),
):
    """Open the workbench in your browser: drop files in, see scores, waveforms and fixes. Local only."""
    from .model import prefer_offline
    from .server import serve as run

    prefer_offline()
    run(port=port, open_browser=not no_open)


@command()
def mcp(
    http: bool = typer.Option(False, "--http", help="Streamable HTTP instead of stdio."),
    port: int = typer.Option(7789, "--port", "-p", min=1, max=65535, help="Port for --http (localhost)."),
):
    """Run the MCP server for coding agents (stdio by default). Tools never download the model."""
    from .mcp_server import run

    run("http" if http else "stdio", port=port)


@command()
def doctor(load: bool = typer.Option(False, "--load", help="Also load the model and embed a test tone.")):
    """Check the environment: Python, ffmpeg, PyTorch device, model cache (--load downloads it)."""
    import platform
    import shutil

    rows = []

    def row(ok, name, value, hint=""):
        rows.append(("pass" if ok is True else "warn" if ok is None else "fail", name, value, hint))

    row(sys.version_info >= (3, 11), "Python", platform.python_version())
    ff = shutil.which("ffmpeg")
    row(
        True if ff else None,
        "ffmpeg",
        ff or "not found",
        "" if ff else "Needed for M4A/AAC/MP4 and video input.",
    )
    MODEL_ID = None
    try:
        from .model import MODEL_ID as _id
        from .model import pick_device, require_model_extra

        require_model_extra()
        import torch

        dev = pick_device()
        row(True, "PyTorch", f"{torch.__version__} · device {dev}")
        MODEL_ID = _id
    except ImportError:
        row(False, "PyTorch", "missing", f"Install the model extra: {service.INSTALL_HINT}")
    except SonicproofError as e:
        row(False, "PyTorch", "missing" if e.code == "missing_extra" else "device", e.message)
    if MODEL_ID:
        from .model import DOWNLOAD_SIZE, hub_cache_dir, model_cached, prefer_offline

        prefer_offline()
        cached = model_cached()
        row(
            True if cached else None,
            "Model",
            f"{MODEL_ID} ({'cached' if cached else 'not downloaded'})",
            "" if cached else f"`sp doctor --load` downloads it ({DOWNLOAD_SIZE}, no account needed).",
        )
        row(True, "HF cache", str(hub_cache_dir()))
    from .cache import cache_dir

    row(True, "Cache", str(cache_dir()))
    if load and MODEL_ID:
        import time

        import numpy as np

        t0 = time.time()
        try:
            j = _judge()
            tone = (0.3 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).astype(np.float32)
            e = j.embedder.embed_audio([tone])
            ok = bool(np.isfinite(e).all())
            row(ok, "Inference", f"{time.time() - t0:.1f} s load + embed" if ok else "NaN in output")
        except Exception as ex:
            if _debug:
                raise
            row(False, "Inference", "failed", service.as_error(ex).message)
    t = Table(box=None, show_header=False, padding=(0, 1))
    t.add_column(no_wrap=True)
    t.add_column(no_wrap=True)
    t.add_column(overflow="fold")
    t.add_column(overflow="fold")
    for st, name, value, hint in rows:
        sym, color = STATUS_STYLE[st]
        t.add_row(Text(sym, style=color), Text(name, style="bold"), value, Text(hint, style="dim"))
    console.print(
        Panel(
            t,
            title=f"[bold]Sonicproof {__version__}[/] doctor",
            title_align="left",
            border_style="grey50",
            padding=(1, 2),
        )
    )
    if any(r[0] == "fail" for r in rows):
        raise typer.Exit(1)


# -- agent skill --------------------------------------------------------------------------------
skill_app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="The agent skill (SKILL.md) that ships with Sonicproof.",
)
app.add_typer(skill_app, name="skill")


def _skill_text() -> str:
    from importlib.resources import files

    return (files("sonicproof") / "skill" / "SKILL.md").read_text(encoding="utf-8")


@skill_app.command("install")
@_guarded
def skill_install(
    project: bool = typer.Option(
        False, "--project", help="Install into ./.claude/skills/ (this project) instead of ~/.claude/skills/."
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite a SKILL.md that differs from this version."),
):
    """Install the skill for Claude Code (and other agents that read .claude/skills)."""
    base = Path.cwd() if project else Path.home()
    dest = base / ".claude" / "skills" / "sonicproof" / "SKILL.md"
    text = _skill_text()
    if dest.exists():
        if dest.read_text(encoding="utf-8", errors="replace") == text:
            console.print(f"[green]✓[/] Already up to date: {dest}")
            return
        if not force:
            raise InputError(
                f"{dest} exists and differs from this version — add --force to replace it.", code="bad_value"
            )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    console.print(f"[green]✓[/] Installed the Sonicproof skill: {dest}")


@skill_app.command("show")
@_guarded
def skill_show():
    """Print SKILL.md (to paste into agents that don't read .claude/skills)."""
    sys.stdout.write(_skill_text())


@app.callback(invoke_without_command=True)
def _main(
    version: bool = typer.Option(False, "--version", help="Show the version."),
    debug: bool = typer.Option(False, "--debug", help="Show full tracebacks (also SONICPROOF_DEBUG=1)."),
):
    global _debug
    _utf8_stdio()
    if debug:
        _debug = True
    if version:
        console.print(f"sonicproof {__version__}")
        raise typer.Exit()


if __name__ == "__main__":
    app()
