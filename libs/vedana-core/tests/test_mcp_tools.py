import json
import logging

import pytest

import vedana_core.mcp_tools as mcp_tools_mod
from vedana_core.mcp_tools import MCPTool, load_mcp_tools
from vedana_core.rag_agent import CYPHER_TOOL_NAME, VTS_TOOL_NAME


class FakeContent:
    def __init__(self, text):
        self.text = text


class FakeCallResult:
    def __init__(self, text):
        self.content = [FakeContent(text)]


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

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def list_tools(self):
        return [
            FakeRemoteTool(
                "ask",
                "Ask about salaries",
                {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]},
            )
        ]

    async def call_tool(self, name, args):
        FakeClient.last_call = (name, args)
        return FakeCallResult(f"result of {name}")


@pytest.fixture(autouse=True)
def patch_client(monkeypatch):
    monkeypatch.setattr(mcp_tools_mod, "_mcp_client_cls", lambda: FakeClient)
    mcp_tools_mod._discover_mcp_tools.cache_clear()
    yield
    mcp_tools_mod._discover_mcp_tools.cache_clear()


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
        "timeout": mcp_tools_mod.settings.mcp_call_timeout,
    }


@pytest.mark.asyncio
async def test_mcp_tool_call_invalid_json():
    t = MCPTool("salary_ask", "Ask", {}, "http://x", None, "ask")
    out = await t.call("{not json")
    assert "Invalid tool args" in out


@pytest.mark.asyncio
async def test_load_mcp_tools_from_settings(monkeypatch):
    monkeypatch.setattr(
        mcp_tools_mod.settings, "mcp_servers", json.dumps([{"name": "salary", "url": "http://ok", "token": "t"}])
    )
    tools = await load_mcp_tools(logging.getLogger("test"))
    assert len(tools) == 1
    assert tools[0].name == "salary_ask"
    assert tools[0].description == "Ask about salaries"


@pytest.mark.asyncio
async def test_load_mcp_tools_discovery_uses_discovery_timeout(monkeypatch):
    """Discovery is a cheap handshake and must NOT use settings.mcp_call_timeout (that's for
    actual tool calls, which can legitimately run for minutes)."""
    monkeypatch.setattr(
        mcp_tools_mod.settings, "mcp_servers", json.dumps([{"name": "salary", "url": "http://ok", "token": "t"}])
    )
    await load_mcp_tools(logging.getLogger("test"))
    assert FakeClient.last_init_kwargs["timeout"] == mcp_tools_mod._MCP_DISCOVERY_TIMEOUT
    assert FakeClient.last_init_kwargs["timeout"] != mcp_tools_mod.settings.mcp_call_timeout


@pytest.mark.asyncio
async def test_load_mcp_tools_graceful_on_unreachable(monkeypatch):
    monkeypatch.setattr(
        mcp_tools_mod.settings, "mcp_servers", json.dumps([{"name": "bad", "url": "http://unreachable"}])
    )
    tools = await load_mcp_tools(logging.getLogger("test"))
    assert tools == []


@pytest.mark.asyncio
async def test_load_mcp_tools_empty_setting(monkeypatch):
    monkeypatch.setattr(mcp_tools_mod.settings, "mcp_servers", "")
    assert await load_mcp_tools() == []


@pytest.mark.asyncio
async def test_load_mcp_tools_malformed_json_returns_empty(monkeypatch):
    """A typo in MCP_SERVERS must not break answering: log and degrade to no MCP tools."""
    monkeypatch.setattr(mcp_tools_mod.settings, "mcp_servers", "{not valid json")
    tools = await load_mcp_tools(logging.getLogger("test"))
    assert tools == []
    # And it must actually be cached as an empty (but known) result - not retried forever,
    # since the setting itself is broken, not a transient server issue.
    assert mcp_tools_mod._discover_mcp_tools.cache_info().currsize == 1


@pytest.mark.asyncio
async def test_load_mcp_tools_does_not_cache_empty_result_when_server_configured(monkeypatch):
    """An unreachable-but-configured server should be retried on the next call, not stuck
    with a permanently empty cache."""
    monkeypatch.setattr(
        mcp_tools_mod.settings, "mcp_servers", json.dumps([{"name": "bad", "url": "http://unreachable"}])
    )

    tools = await load_mcp_tools(logging.getLogger("test"))
    assert tools == []
    assert mcp_tools_mod._discover_mcp_tools.cache_info().currsize == 0  # not cached - eligible for retry

    # Second call retries (server recovers) instead of returning a stale empty cache.
    monkeypatch.setattr(
        mcp_tools_mod.settings, "mcp_servers", json.dumps([{"name": "salary", "url": "http://ok", "token": "t"}])
    )
    tools = await load_mcp_tools(logging.getLogger("test"))
    assert len(tools) == 1
    assert mcp_tools_mod._discover_mcp_tools.cache_info().currsize == 1


# --- Item 1: valid JSON but wrong shape must never raise, and must degrade to [] ---------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        '{"name":"a","url":"http://x"}',  # dict, not a list
        '["http://x"]',  # list of strings, not server objects
        "123",  # not a list at all
        "null",  # not a list at all
    ],
)
async def test_load_mcp_tools_bad_shape_returns_empty_without_raising(monkeypatch, raw):
    monkeypatch.setattr(mcp_tools_mod.settings, "mcp_servers", raw)
    tools = await load_mcp_tools(logging.getLogger("test"))
    assert tools == []


@pytest.mark.asyncio
async def test_load_mcp_tools_dict_entry_missing_url_is_skipped(monkeypatch):
    monkeypatch.setattr(
        mcp_tools_mod.settings, "mcp_servers", json.dumps([{"name": "salary"}, {"name": "ok", "url": "http://ok"}])
    )
    tools = await load_mcp_tools(logging.getLogger("test"))
    # the entry without a url is skipped; the valid one still loads
    assert len(tools) == 1
    assert tools[0].name == "ok_ask"


# --- Item 5: tool-name sanitization / built-in collisions --------------------------------


@pytest.mark.asyncio
async def test_load_mcp_tools_skips_invalid_prefixed_name(monkeypatch):
    """Server names with spaces (or anything outside ^[a-zA-Z0-9_-]{1,64}$) would otherwise
    make litellm reject EVERY request in the conversation, not just this tool call."""
    monkeypatch.setattr(
        mcp_tools_mod.settings, "mcp_servers", json.dumps([{"name": "my salary server", "url": "http://ok"}])
    )
    tools = await load_mcp_tools(logging.getLogger("test"))
    assert tools == []


@pytest.mark.asyncio
async def test_load_mcp_tools_skips_name_colliding_with_builtin(monkeypatch):
    """A remote tool that resolves to a built-in name must not silently shadow it in
    tools_map."""
    # Build a server name + remote tool name such that "{name}_{remote_name}" collides
    # exactly with a built-in tool name (vector_text_search).
    prefix, remote_name = VTS_TOOL_NAME.split("_", 1)

    class CollidingFakeClient(FakeClient):
        async def list_tools(self):
            return [FakeRemoteTool(remote_name, "shadow", {"type": "object", "properties": {}})]

    monkeypatch.setattr(mcp_tools_mod, "_mcp_client_cls", lambda: CollidingFakeClient)
    monkeypatch.setattr(mcp_tools_mod.settings, "mcp_servers", json.dumps([{"name": prefix, "url": "http://ok"}]))
    tools = await load_mcp_tools(logging.getLogger("test"))
    assert VTS_TOOL_NAME not in [t.name for t in tools]
    assert CYPHER_TOOL_NAME not in [t.name for t in tools]
    assert tools == []


# --- Item 6: fastmcp is an optional extra -------------------------------------------------


@pytest.mark.asyncio
async def test_load_mcp_tools_warns_clearly_when_fastmcp_not_installed(monkeypatch, caplog):
    """A user with MCP_SERVERS configured but the `mcp` extra not installed must get a clear,
    actionable warning - not a crash."""

    def _raise_import_error():
        raise ImportError("No module named 'fastmcp'")

    monkeypatch.setattr(mcp_tools_mod, "_mcp_client_cls", _raise_import_error)
    monkeypatch.setattr(mcp_tools_mod.settings, "mcp_servers", json.dumps([{"name": "salary", "url": "http://ok"}]))
    with caplog.at_level(logging.WARNING):
        tools = await load_mcp_tools(logging.getLogger("test"))
    assert tools == []
    assert any("vedana-core[mcp]" in r.message for r in caplog.records)


def test_real_mcp_client_cls_seam_still_resolves(monkeypatch):
    """Sanity check independent of the FakeClient seam: with the `mcp` extra installed,
    `_mcp_client_cls()` (unpatched) must still successfully lazy-import the real
    `fastmcp.Client` - i.e. the optional-dependency refactor (item 6) didn't break the
    lazy-import indirection point itself. Skipped if the `mcp` extra isn't installed."""
    pytest.importorskip("fastmcp")
    monkeypatch.undo()  # revert the autouse patch_client fixture's monkeypatch for this test
    from fastmcp import Client

    assert mcp_tools_mod._mcp_client_cls() is Client
