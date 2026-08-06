"""MCP client transport/discovery: turns servers listed in `settings.mcp_servers` into
tools the RAG agent can call alongside the built-in `vector_text_search`/`cypher` tools.

Split out of `llm.py` (which is otherwise about LLM prompting) to keep that module focused.
"""

import json
import logging
import re
from dataclasses import dataclass, field

from async_lru import alru_cache

from vedana_core.rag_agent import CYPHER_TOOL_NAME, VTS_TOOL_NAME
from vedana_core.settings import settings

_module_logger = logging.getLogger(__name__)

# Tool names go straight into the LLM API request, which requires
# ^[a-zA-Z0-9_-]{1,64}$ (litellm/OpenAI function-name schema). A remote server name with a
# space, or a name+prefix combo over 64 chars, would otherwise make litellm reject EVERY
# request in the conversation, not just the one MCP call.
_VALID_TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

# Built-in tool names an MCP server must not be able to shadow in `tools_map`. Imported from
# rag_agent rather than redefined here (no circular import: rag_agent only depends on llm.py,
# not on this module) so the two can't drift apart.
_RESERVED_TOOL_NAMES = frozenset({VTS_TOOL_NAME, CYPHER_TOOL_NAME})

# Discovery (`list_tools`) is a cheap handshake against the MCP server, unlike an actual tool
# call which can run agentic remote work for a long time (see settings.mcp_call_timeout).
_MCP_DISCOVERY_TIMEOUT = 30


def _mcp_client_cls():
    """Indirection point so tests can substitute a fake fastmcp Client."""
    from fastmcp import Client

    return Client


class MCPTool:
    """A remote MCP server tool exposed through the same interface as Tool.

    openai_def is built straight from the MCP inputSchema — no pydantic model needed.
    """

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict,
        server_url: str,
        token: str | None,
        remote_name: str,
    ) -> None:
        self.name = name
        self.description = description
        self.server_url = server_url
        self.token = token
        self.remote_name = remote_name
        self.openai_def = {
            "type": "function",
            "function": {"name": name, "description": description, "parameters": input_schema},
        }

    async def call(self, args_json: str) -> str:
        try:
            args = json.loads(args_json)
        except ValueError:
            return f"Invalid tool args: {args_json}"
        # Agentic remote tools can take a minute to respond, especially the first call, which
        # may need to build up remote context before it can answer.
        client = _mcp_client_cls()(self.server_url, auth=self.token, timeout=settings.mcp_call_timeout)
        async with client:
            result = await client.call_tool(self.remote_name, args)
        texts = [c.text for c in getattr(result, "content", []) if getattr(c, "text", None)]
        return "\n".join(texts) if texts else str(result)


@dataclass
class _DiscoveryResult:
    tools: list[MCPTool] = field(default_factory=list)
    # True once we've actually attempted to reach at least one configured server (as opposed
    # to every entry being dropped for a config-shape problem before any network call). Used
    # to decide whether an empty result is a permanent (config error, cache it) or transient
    # (server down, don't cache - retry next call) outcome. See load_mcp_tools.
    attempted_any_server: bool = False
    # (level, message) pairs to be emitted by load_mcp_tools via the caller-supplied logger.
    # Discovery itself only runs once (cached) but its log output should still show up on
    # every call, using whichever logger the caller passed in that time.
    log_events: list[tuple[int, str]] = field(default_factory=list)


@alru_cache
async def _discover_mcp_tools() -> _DiscoveryResult:
    """Discover tools on all MCP servers from settings.mcp_servers (JSON list).

    Cached per process via alru_cache (also de-dupes concurrent cold-start callers, since
    async-lru shares the in-flight task between concurrent calls with the same key). Callers
    needing a retry after a transient failure must explicitly invalidate the cache entry
    (`_discover_mcp_tools.cache_invalidate()`) rather than call this directly — see
    load_mcp_tools, which owns that decision.
    """
    result = _DiscoveryResult()
    raw = settings.mcp_servers
    if not raw:
        return result

    try:
        servers = json.loads(raw)
    except json.JSONDecodeError as e:
        result.log_events.append((logging.WARNING, f"MCP_SERVERS setting is not valid JSON, ignoring: {e}"))
        return result

    if not isinstance(servers, list):
        result.log_events.append(
            (
                logging.WARNING,
                f"MCP_SERVERS must be a JSON list of server objects, got {type(servers).__name__}; ignoring",
            )
        )
        return result

    for srv in servers:
        if not isinstance(srv, dict):
            result.log_events.append((logging.WARNING, f"Ignoring non-object MCP server entry: {srv!r}"))
            continue

        display_name = srv.get("name") or srv.get("url") or "<unnamed>"
        url = srv.get("url")
        if not url:
            result.log_events.append((logging.WARNING, f"MCP server {display_name!r} is missing 'url', skipping"))
            continue

        result.attempted_any_server = True
        try:
            client = _mcp_client_cls()(url, auth=srv.get("token"), timeout=_MCP_DISCOVERY_TIMEOUT)
            async with client:
                remote = await client.list_tools()
            loaded = 0
            for t in remote:
                tool_name = f"{display_name}_{t.name}"
                if not _VALID_TOOL_NAME_RE.match(tool_name):
                    result.log_events.append(
                        (
                            logging.WARNING,
                            f"MCP tool name {tool_name!r} (server {display_name!r}) doesn't match "
                            f"{_VALID_TOOL_NAME_RE.pattern!r}, skipping",
                        )
                    )
                    continue
                if tool_name in _RESERVED_TOOL_NAMES:
                    result.log_events.append(
                        (
                            logging.WARNING,
                            f"MCP tool name {tool_name!r} (server {display_name!r}) collides with a "
                            f"built-in tool, skipping",
                        )
                    )
                    continue
                result.tools.append(
                    MCPTool(
                        name=tool_name,
                        description=t.description or "",
                        input_schema=t.inputSchema,
                        server_url=url,
                        token=srv.get("token"),
                        remote_name=t.name,
                    )
                )
                loaded += 1
            result.log_events.append((logging.INFO, f"MCP server {display_name}: loaded {loaded} tools"))
        except ImportError as e:
            result.log_events.append(
                (
                    logging.WARNING,
                    f"MCP server {display_name!r} is configured but the 'fastmcp' package is not installed; "
                    f"install the 'mcp' extra (`vedana-core[mcp]`) to use MCP_SERVERS. Skipping: {e}",
                )
            )
        except Exception as e:
            result.log_events.append((logging.WARNING, f"MCP server {display_name} unavailable, skipping: {e}"))

    return result


async def load_mcp_tools(logger: logging.Logger | None = None) -> list[MCPTool]:
    """Discover and return all MCP tools configured via settings.mcp_servers.

    A server that is down (or misconfigured, or missing the optional 'mcp' extra) is skipped
    with a warning — the pipeline keeps working with the built-in tools.

    Args:
        logger: Logger to emit discovery warnings/info to. Defaults to this module's logger.

    Returns:
        The discovered MCPTool instances (empty list if nothing configured or reachable).
    """
    log = logger or _module_logger
    result = await _discover_mcp_tools()
    for level, msg in result.log_events:
        log.log(level, msg)

    # Don't keep an empty result cached if we actually attempted to reach a configured server:
    # a configured-but-currently-unreachable server should be retried on the next call rather
    # than sticking to a permanently empty result for the life of the process. An empty/absent
    # config, or a config that's broken in a way json.loads/shape-validation catches, IS safe
    # to cache - that's a static error, not a transient one, so no point retrying it.
    if not result.tools and result.attempted_any_server:
        _discover_mcp_tools.cache_invalidate()

    return result.tools
