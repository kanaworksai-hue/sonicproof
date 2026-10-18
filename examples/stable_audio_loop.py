"""Generate → check → retry → pick with two MCP servers: Stable Audio (generates) and Sonicproof (listens).

    uv run python examples/stable_audio_loop.py --source "a cat" --action "meowing twice" \\
        --material "soft, close" --out out/loop_cat

Each round asks Stable Audio for several seeds, downloads them with their .json sidecars, and has Sonicproof
rank them against the prompt. The best take is kept when it sounds like the prompt and has no technical
failures; otherwise the next round tries new seeds (and, if given, the next phrasing from --rephrase).
An agent following the Sonicproof skill (`sp skill show`) does the same, but rephrases the prompt itself
by reading ``heard_as``.

Set STABLE_AUDIO_MCP to your Stable Audio MCP URL (streamable HTTP).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import urllib.request
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client

STABLE_AUDIO = os.environ.get("STABLE_AUDIO_MCP", "http://127.0.0.1:8766/mcp")
GOOD = ("strong", "plausible")


def payload(res) -> object:
    if getattr(res, "is_error", False) or getattr(res, "isError", False):
        raise RuntimeError(res.content[0].text if res.content else "tool error")
    return (
        json.loads(res.content[0].text)
        if len(res.content) == 1
        else [json.loads(c.text) for c in res.content]
    )


async def main(a: argparse.Namespace) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    phrasings = [a.action, *a.rephrase]
    # pass the full environment: stdio servers otherwise get a minimal one and miss e.g. a custom HF_HOME
    sp_server = StdioServerParameters(
        command=sys.executable, args=["-m", "sonicproof.cli", "mcp"], env=dict(os.environ)
    )
    async with (
        streamable_http_client(STABLE_AUDIO) as (sr, sw),
        ClientSession(sr, sw) as sa,
        stdio_client(sp_server) as (pr, pw),
        ClientSession(pr, pw) as proof,
    ):
        await sa.initialize()
        await proof.initialize()
        best = None
        for rnd in range(a.rounds):
            action = phrasings[min(rnd, len(phrasings) - 1)]
            prompt = f"{a.source}, {action}"  # what the take should sound like (no mixing words)
            print(f"\n— round {rnd + 1}: “{prompt}” x {a.variations}")
            gen = payload(
                await sa.call_tool(
                    "generate_audio",
                    {
                        "audio_type": "sfx",
                        "sound_source": a.source,
                        "sound_action": action,
                        "sound_material": a.material,
                        "sound_space": a.space,
                        "duration_seconds": a.seconds,
                        "variations": a.variations,
                        "seed": a.seed + 100 * rnd,
                        "title": f"{a.title}_r{rnd + 1}",
                    },
                )
            )
            paths = [c["path"] for c in gen["candidates"]] + [c["metadata"] for c in gen["candidates"]]
            links = payload(await sa.call_tool("download_urls", {"paths": paths, "hours": 1}))
            files = []
            for name, url in links["urls"].items():
                dest = out / Path(name).name
                urllib.request.urlretrieve(url, dest)
                if dest.suffix == ".wav":
                    files.append(str(dest.resolve()))
            ranked = payload(await proof.call_tool("rank", {"prompt": prompt, "files": sorted(files)}))
            ranked = ranked if isinstance(ranked, list) else [ranked]
            for r in ranked:
                if r["verdict"] == "error":  # unreadable take: listed last, with the reason
                    print(f"  {Path(r['file']).name:<28} error: {r['error']['message']}")
                    continue
                heard = ", ".join(h["text"] for h in r["heard_as"][:2])
                print(
                    f"  {Path(r['file']).name:<28} {r['score']:+.2f}σ  {r['verdict']:<9}  heard as: {heard}"
                )
            if ranked[0]["verdict"] == "error":
                continue
            top = payload(await proof.call_tool("report", {"file": ranked[0]["file"], "prompt": prompt}))
            issues = [c for c in top["qc"]["checks"] if c["status"] in ("warn", "fail")]
            print(f"  best: {Path(top['file']).name} → {top['overall'].upper()} ({top['note']})")
            for c in issues:
                print(f"    {c['status']}: {c['label']} {c['value']} — {c['hint']}")
            if best is None or _rank_key(top) > _rank_key(best):
                best = top
            if _shippable(top):
                break
        if best is None:
            print("\nNo readable take in any round.")
            return 1
        print(f"\nPick: {best['file']}  {best['adherence']['score']:+.2f}σ {best['adherence']['verdict']}")
        (out / "pick.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0 if _shippable(best) else 1


def _shippable(rep: dict) -> bool:
    return rep["adherence"]["verdict"] in GOOD and rep["qc"]["status"] != "fail"


def _rank_key(rep: dict) -> tuple:
    """A deliverable take beats any non-deliverable one; then a take without failing qc; then the score."""
    return (_shippable(rep), rep["qc"]["status"] != "fail", rep["adherence"]["score"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="What makes the sound.")
    p.add_argument("--action", required=True, help="How it is triggered / evolves.")
    p.add_argument(
        "--rephrase", nargs="*", default=[], help="Other phrasings of the action for later rounds."
    )
    p.add_argument("--material", default="")
    p.add_argument("--space", default="")
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--variations", type=int, default=3)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--seed", type=int, default=7000)
    p.add_argument("--title", default="take")
    p.add_argument("--out", default="out/loop")
    sys.exit(asyncio.run(main(p.parse_args())))
