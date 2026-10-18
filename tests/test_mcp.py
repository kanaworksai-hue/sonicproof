"""MCP server: tool list, readable tool errors and path handling, through the SDK's in-memory client."""

import asyncio

import pytest
from mcp import Client
from test_service import FakeEngine, tone

from sonicproof import mcp_server, service

TOOLS = {
    "check": {"file"},
    "rank": {"prompt", "files"},
    "qc": {"files"},
    "report": {"file"},
    "fit": {"file"},
    "index": {"directory"},
    "search": {"directory"},
    "dedupe": {"directory"},
}


@pytest.fixture
def call(tmp_path, monkeypatch):
    monkeypatch.setenv("SONICPROOF_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(mcp_server, "_engine", FakeEngine())

    def invoke(name, **args):
        async def go():
            async with Client(mcp_server.mcp) as c:
                return await c.call_tool(name, args)

        return asyncio.run(go())

    return invoke


def _text(r) -> str:
    return " ".join(getattr(c, "text", "") for c in r.content)


def test_tool_list():
    async def go():
        async with Client(mcp_server.mcp) as c:
            return (await c.list_tools()).tools

    tools = {t.name: t for t in asyncio.run(go())}
    assert set(tools) == set(TOOLS)
    for name, required in TOOLS.items():
        assert set(tools[name].input_schema.get("required", [])) == required


def test_qc_matches_the_service_record(call, tmp_path):
    a = tone(tmp_path / "a.wav")
    r = call("qc", files=[str(a)])
    assert not r.is_error
    assert r.structured_content["result"] == [service.qc_record(a)]


def test_errors_reach_the_agent(call, tmp_path):
    a = tone(tmp_path / "a.wav")
    r = call("check", file=str(tmp_path / "nope.wav"), prompt="x")
    assert r.is_error and "not_found" in _text(r) and "nope.wav" in _text(r)
    r = call("check", file=str(a))  # no prompt, no sidecar
    assert r.is_error and "no_prompt" in _text(r)
    r = call("check", file="relative/nope.wav", prompt="x")
    assert r.is_error and "working directory" in _text(r) and "absolute path" in _text(r)
    r = call("rank", prompt="x", files=[])
    assert r.is_error and "No files given" in _text(r)
    r = call("search", directory=str(tmp_path), query="rain", like=str(a))
    assert r.is_error and "one at a time" in _text(r)
    assert call("check", file=str(a), prompt="x", top=-1).is_error
    assert call("qc", files=[str(a)], target_lufs=0).is_error


def test_bad_file_in_a_qc_batch_is_a_record(call, tmp_path):
    a = tone(tmp_path / "a.wav")
    r = call("qc", files=[str(a), "relative/nope.wav"])
    out = r.structured_content["result"]
    assert not r.is_error and out[0]["status"] != "error"
    assert out[1]["status"] == "error" and "working directory" in out[1]["error"]["message"]


def test_tilde_is_expanded(call, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Path.home() on Windows
    tone(tmp_path / "a.wav")
    r = call("qc", files=["~/a.wav"])
    assert not r.is_error and r.structured_content["result"][0]["status"] != "error"
    assert not call("check", file="~/a.wav", prompt="a beep").is_error


def test_model_is_never_downloaded_in_a_call(call, tmp_path, monkeypatch):
    import sonicproof.model as model_mod

    monkeypatch.setattr(model_mod, "require_model_extra", lambda: None)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("SONICPROOF_ALLOW_DOWNLOAD", raising=False)
    monkeypatch.setattr(mcp_server, "_engine", service.Engine(allow_download=False))
    r = call("check", file=str(tone(tmp_path / "a.wav")), prompt="a beep")
    assert r.is_error and "model_unavailable" in _text(r) and "doctor --load" in _text(r)
