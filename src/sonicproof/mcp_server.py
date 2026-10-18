"""MCP server (official Python SDK): Sonicproof as tools for coding agents. stdio or streamable HTTP.

Every tool returns the same records the matching CLI command prints with ``--json`` (``qc`` and ``rank``
return lists). The model is loaded on the first call and kept for the life of the server; it is never
downloaded inside a call (run ``sonicproof doctor --load`` once, or set ``SONICPROOF_ALLOW_DOWNLOAD=1``).
Failures come back as tool errors with a one-line reason the agent can act on.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from . import __version__, service
from .errors import InputError, SetupError

INSTRUCTIONS = """Sonicproof listens to audio files on this machine with EmbeddingGemma 2 and reports whether
they sound like what was asked for. Use absolute paths (relative paths resolve against the server's
working directory, which is usually not yours); `~` is expanded.

- check: does a clip sound like its prompt (any language)? Calibrated score in σ and a verdict:
  strong (≥2.0) / plausible (≥1.1) / weak (≥0.3) / miss. heard_as lists what it sounds like instead.
- rank: several takes (seeds) of one prompt, best first. Keep the top one if it is strong or plausible,
  then confirm with report (a take with "note": "silent" is never a pick; unreadable takes come last
  with verdict "error").
- qc: technical checks without the model (loudness, true peak, clipping, silence, cut-off tails, clicks,
  phase, loop seams). One record per file; each item in checks[] has status fail/warn/info/pass and a
  one-line fix in `hint`. A file that can't be read has status "error" and an `error` object.
- report: check + qc in one scorecard with overall ship / review / redo.
- fit: does a sound suit an image or video? pictured_as says what the picture shows.
- index / search / dedupe: a local sound library — search by words, by example sound or by picture.
  The first index of a big folder can take minutes; if the call times out, run
  `sonicproof index DIR` in a terminal once.

Generation loop: generate 2-4 variations → rank → report on the best → if every take is weak/miss,
rephrase the prompt (use heard_as to see what came out) or try new seeds; fix qc failures with the hints.
Short concrete prompts work best; very short transient sounds are the hardest to judge — say so.
If a tool says the model is not downloaded, ask the user to run `sonicproof doctor --load` once."""

mcp = MCPServer("sonicproof", instructions=INSTRUCTIONS, version=__version__, log_level="WARNING")
_engine = service.Engine(allow_download=False)

Top = Annotated[int, Field(ge=0, le=50, description="How many descriptions to list (0-50).")]
TargetLufs = Annotated[
    float | None, Field(ge=-70, le=-5, description="Required integrated loudness, e.g. -16 or -23 (LUFS).")
]
Match = Annotated[
    float | None, Field(ge=0.5, le=1.0, description="Fingerprint correlation that counts as the same audio.")
]


def _path(p: str | None, folder: bool = False) -> str | None:
    """``~`` expanded; a relative path that does not resolve gets an error naming the server's cwd."""
    if p is None:
        return None
    if not str(p).strip():
        raise InputError("Empty path.", code="bad_value")
    q = Path(p).expanduser()
    if not q.is_absolute() and not q.exists():
        what = "folder" if folder else "file"
        raise InputError(
            f"No such {what}: {p} (relative paths resolve against the server's working directory "
            f"{os.getcwd()}; pass an absolute path)",
            code="not_found",
            file=str(p),
        )
    return str(q)


def _tool(fn):
    """Register ``fn`` as a tool whose failures reach the agent as a readable one-line ToolError."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except Exception as e:
            err = service.as_error(e)
            raise ToolError(f"{err.code}: {err.message}") from e

    return mcp.tool()(wrapper)


@_tool
def check(file: str, prompt: str | None = None, top: Top = 5) -> dict[str, Any]:
    """Does this audio file (absolute path) sound like the prompt? Omit prompt to use a Stable Audio .json
    sidecar."""
    return _engine.check(_path(file), prompt, top)


@_tool
def rank(prompt: str, files: list[str]) -> list[dict[str, Any]]:
    """Rank several takes (e.g. seeds) of the same prompt, best first. Use absolute paths."""
    if not files:
        raise InputError("No files given.", code="bad_value")
    return _engine.rank(prompt, [_path(f) for f in files])


@_tool
def qc(files: list[str], loop: bool = False, target_lufs: TargetLufs = None) -> list[dict[str, Any]]:
    """Technical checks (no model): loudness, true peak, clipping, DC, silence, cut-off tail, clicks, phase.
    Set loop=true for loops (checks the seam); target_lufs e.g. -16 (games/web) or -23 (broadcast).
    Use absolute paths. Unreadable files come back with status "error" instead of failing the batch."""
    if not files:
        raise InputError("No files given.", code="bad_value")
    out = []
    for f in files:  # one bad path never sinks the batch; its record carries the reason (and the cwd hint)
        try:
            out.append(service.qc_record_safe(_path(f), loop, target_lufs))
        except InputError as e:
            out.append({"file": f, "status": "error", "error": e.to_dict(), "checks": []})
    return out


@_tool
def report(
    file: str, prompt: str | None = None, loop: bool = False, target_lufs: TargetLufs = None
) -> dict[str, Any]:
    """One scorecard — prompt adherence + technical QC — with overall ship / review / redo. Absolute path."""
    return _engine.report(_path(file), prompt, loop, target_lufs)


@_tool
def fit(file: str, picture: str | None = None, scene: str | None = None, top: Top = 5) -> dict[str, Any]:
    """Does the sound suit the picture (image or video)? scene = optional words about the scene, any language.
    Omit picture when file is a video to judge its own soundtrack against its own picture. Absolute paths."""
    return _engine.fit(_path(file), _path(picture), scene, top)


@_tool
def index(directory: str, in_cache: bool = False) -> dict[str, Any]:
    """Index (or refresh) a sound library folder (absolute path) for search and dedupe. Only new or changed
    files are read."""
    return _engine.index(_path(directory, folder=True), in_cache)


@_tool
def search(
    directory: str,
    query: str | None = None,
    like: str | None = None,
    picture: str | None = None,
    top: Annotated[int, Field(ge=1, le=50, description="How many results (1-50).")] = 10,
    in_cache: bool = False,
) -> dict[str, Any]:
    """Search a library folder (absolute path) by words (any language), by an example sound (like) or by a
    picture (image/video, optionally with words about the scene in query). The index is refreshed first."""
    return _engine.search(_path(directory, folder=True), query, _path(like), _path(picture), top, in_cache)


@_tool
def dedupe(directory: str, match: Match = None, in_cache: bool = False) -> dict[str, Any]:
    """Find duplicate / near-duplicate files (re-encodes, gain changes, trims) and which copy to keep.
    Reports only — nothing is deleted. Absolute folder path."""
    return _engine.dedupe(_path(directory, folder=True), match, in_cache)


def _port_free(host: str, port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def run(transport: str = "stdio", host: str = "127.0.0.1", port: int = 7789) -> None:
    from .model import prefer_offline

    prefer_offline()
    if transport == "stdio":
        mcp.run("stdio")
    else:
        if not _port_free(host, port):
            raise SetupError(f"Port {port} is in use — try --port {port + 10}.", code="port_busy")
        mcp.run("streamable-http", host=host, port=port)
