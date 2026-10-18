"""Generate a clip set through the Stable Audio MCP server and download it (internal dev tool).

STABLE_AUDIO_MCP=http://<host>:<port>/mcp \\
    uv run --with mcp python eval/gen_clips.py eval/refbank_v1.json samples/refbank

Set STABLE_AUDIO_MCP to your Stable Audio MCP URL (streamable HTTP).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import urllib.request
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

URL = os.environ.get("STABLE_AUDIO_MCP", "")


async def main(spec_path: str, out_dir: str) -> None:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    async with streamable_http_client(URL) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        for i, clip in enumerate(spec["clips"]):
            title = clip["title"]
            if (out / f"{title}.wav").exists():
                continue
            args = {k: v for k, v in clip.items() if k != "title"}
            args |= {"title": title, "seed": spec.get("seed0", 0) + i}
            res = await s.call_tool("generate_audio", args)
            data = json.loads(res.content[0].text)
            paths = [c["path"] for c in data["candidates"]]
            paths += [c["metadata"] for c in data["candidates"]]
            links = await s.call_tool("download_urls", {"paths": paths, "hours": 2})
            for name, url in json.loads(links.content[0].text)["urls"].items():
                urllib.request.urlretrieve(url, out / Path(name).name)
            print(f"[{i + 1}/{len(spec['clips'])}] {title}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: gen_clips.py SPEC.json OUT_DIR")
    if not URL:
        sys.exit("Set STABLE_AUDIO_MCP to your Stable Audio MCP URL, e.g. http://127.0.0.1:8766/mcp")
    asyncio.run(main(sys.argv[1], sys.argv[2]))
