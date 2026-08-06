import json
import logging

import pytest

import vedana_core.llm as llm_mod
from vedana_core.llm import MCPTool, load_mcp_tools


class FakeContent:
    def __init__(self, text): self.text = text


class FakeCallResult:
    def __init__(self, text): self.content = [FakeContent(text)]


class FakeRemoteTool:
    def __init__(self, name, description, schema):
        self.name = name
        self.description = description
        self.inputSchema = schema


class FakeClient:
    """Fake fastmcp.Client: context manager + list_tools/call_tool."""
    last_call = None
    last_init_kwargs = None

    def __init__(self, url, auth=None, timeout=None):
        self.url = url
        self.auth = auth
        self.timeout = timeout
        FakeClient.last_init_kwargs = {"url": url, "auth": auth, "timeout": timeout}
        if "unreachable" in url:
            raise ConnectionError("boom")

    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False

    async def list_tools(self):
        return [FakeRemoteTool("ask", "Ask about salaries", {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]})]

    async def call_tool(self, name, args):
        FakeClient.last_call = (name, args)
        return FakeCallResult(f"result of {name}")


@pytest.fixture(autouse=True)
def patch_client(monkeypatch):
    monkeypatch.setattr(llm_mod, "_mcp_client_cls", lambda: FakeClient)
    monkeypatch.setattr(llm_mod, "_mcp_tools_cache", None)


def test_mcp_tool_openai_def():
    t = MCPTool("salary_ask", "Ask", {"type": "object", "properties": {}}, "http://x", None, "ask")
    assert t.openai_def == {
        "type": "function",
        "function": {"name": "salary_ask", "description": "Ask", "parameters": {"type": "object", "properties": {}}},
    }


@pytest.mark.asyncio
async def test_mcp_tool_call_forwards():
    t = MCPTool("salary_ask", "Ask", {}, "http://x", "tok", "ask")
    out = await t.call(json.dumps({"question": "avg?"}))
    assert out == "result of ask"
    assert FakeClient.last_call == ("ask", {"question": "avg?"})


@pytest.mark.asyncio
async def test_mcp_tool_call_uses_timeout():
    t = MCPTool("salary_ask", "Ask", {}, "http://x", "tok", "ask")
    await t.call(json.dumps({"question": "avg?"}))
    assert FakeClient.last_init_kwargs == {
        "url": "http://x",
        "auth": "tok",
        "timeout": llm_mod.settings.mcp_call_timeout,
    }


@pytest.mark.asyncio
async def test_mcp_tool_call_invalid_json():
    t = MCPTool("salary_ask", "Ask", {}, "http://x", None, "ask")
    out = await t.call("{not json")
    assert "Invalid tool args" in out


@pytest.mark.asyncio
async def test_load_mcp_tools_from_settings(monkeypatch):
    monkeypatch.setattr(llm_mod.settings, "mcp_servers",
                        json.dumps([{"name": "salary", "url": "http://ok", "token": "t"}]))
    tools = await load_mcp_tools(logging.getLogger("test"))
    assert len(tools) == 1
    assert tools[0].name == "salary_ask"
    assert tools[0].description == "Ask about salaries"


@pytest.mark.asyncio
async def test_load_mcp_tools_graceful_on_unreachable(monkeypatch):
    monkeypatch.setattr(llm_mod.settings, "mcp_servers",
                        json.dumps([{"name": "bad", "url": "http://unreachable"}]))
    tools = await load_mcp_tools(logging.getLogger("test"))
    assert tools == []


@pytest.mark.asyncio
async def test_load_mcp_tools_empty_setting(monkeypatch):
    monkeypatch.setattr(llm_mod.settings, "mcp_servers", "")
    assert await load_mcp_tools() == []


@pytest.mark.asyncio
async def test_load_mcp_tools_malformed_json_returns_empty(monkeypatch):
    """A typo in MCP_SERVERS must not break answering: log and degrade to no MCP tools."""
    monkeypatch.setattr(llm_mod.settings, "mcp_servers", "{not valid json")
    tools = await load_mcp_tools(logging.getLogger("test"))
    assert tools == []
    # And it must actually be cached as an empty (but known) result - not retried forever,
    # since the setting itself is broken, not a transient server issue.
    assert llm_mod._mcp_tools_cache == []


@pytest.mark.asyncio
async def test_load_mcp_tools_does_not_cache_empty_result_when_server_configured(monkeypatch):
    """An unreachable-but-configured server should be retried on the next call, not stuck
    with a permanently empty cache."""
    monkeypatch.setattr(llm_mod.settings, "mcp_servers",
                        json.dumps([{"name": "bad", "url": "http://unreachable"}]))

    tools = await load_mcp_tools(logging.getLogger("test"))
    assert tools == []
    assert llm_mod._mcp_tools_cache is None  # not cached - eligible for retry

    # Second call retries (server recovers) instead of returning a stale empty cache.
    monkeypatch.setattr(llm_mod.settings, "mcp_servers",
                        json.dumps([{"name": "salary", "url": "http://ok", "token": "t"}]))
    tools = await load_mcp_tools(logging.getLogger("test"))
    assert len(tools) == 1
    assert llm_mod._mcp_tools_cache == tools
