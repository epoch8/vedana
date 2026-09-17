---
title: External MCP Tools
section: Guides
order: 13
---

# External MCP Tools

Vedana can call tools hosted on remote [MCP](https://modelcontextprotocol.io/) servers — an HR system, a pricing service, another assistant. From the agent's side they work like any built-in tool: they appear in the tool list, their calls are recorded, and their results go back into the loop.

Use MCP when the capability already lives in another service. If you need logic inside the Vedana process, write a [custom tool](./custom-tools.md) instead.

## 1. Install the extra

The MCP client is an optional dependency of `vedana-core`:

```bash
uv add "vedana-core[mcp]"
```

or, inside the Vedana workspace:

```bash
uv sync --extra mcp
```

## 2. List the servers

```bash
VEDANA_MCP_TOOL_SERVERS='[{"name": "salary", "url": "https://hr.example.com/mcp", "token": "…"}]'
VEDANA_MCP_TOOL_CALL_TIMEOUT=120
```

| Field   | Required | Meaning                                                           |
| ------- | -------- | ------------------------------------------------------------------ |
| `name`  | no       | Short server name used in tool names. Defaults to the URL.         |
| `url`   | yes      | MCP endpoint.                                                      |
| `token` | no       | Bearer token sent to the server.                                    |

## 3. Allow the tools in a scenario

Each remote tool is registered as `mcp__<name>__<tool>` — for example, the `ask` tool on the `salary` server becomes `mcp__salary__ask`.

Add that exact name to `tools_allowed` on every [Queries](../data-model/queries.md) scenario that should use it:

| query_name              | tools_allowed                                   |
| ----------------------- | ------------------------------------------------ |
| Salary band question    | `mcp__salary__ask, cypher, select_scenario`      |

MCP tools are **not** available when no scenario is active, and never inside scenarios that don't list them. Tool names are case-sensitive and aren't validated at ETL time.

## How it behaves

- **Discovery.** On the first request, Vedana connects to each server (30-second `list_tools` handshake) and caches the tool list for the process. The cache is keyed by the data model's `dm_updated_at` stamp, so reloading the data model re-discovers the tools.
- **Failures don't break answering.** If a server is unreachable, the JSON is invalid, or the `mcp` extra isn't installed, a warning is logged and those tools are skipped.
- **Name rules.** Names that don't match `^[a-zA-Z0-9_-]{1,64}$` are skipped (LLM APIs reject them). A remote tool can't shadow a built-in tool name.
- **Calls.** Each call opens a client with `VEDANA_MCP_TOOL_CALL_TIMEOUT` seconds. The text content of the result goes back to the agent as `{"text": ...}`. If the server reports an error, the agent receives `{"text": ..., "error": ...}` as a normal result it can react to.
- **Tracing.** Every call is recorded in `rag.query_processed.technical_info.tool_calls` with its arguments and result.

## Checklist

- [ ] `vedana-core[mcp]` is installed in the image.
- [ ] `VEDANA_MCP_TOOL_SERVERS` is valid JSON (a list of objects).
- [ ] After the first request, the logs show `MCP server <name>: loaded N tools`.
- [ ] Every scenario that needs a tool lists its `mcp__<name>__<tool>` name in `tools_allowed`.
- [ ] The data model has been reloaded after changing `tools_allowed`.
- [ ] Golden dataset questions that depend on the tool pass.

## What's next

- [Tool choice & the agent loop](../architecture/agent-loop.md) — how allowed tools are applied.
- [Writing a Custom Tool](./custom-tools.md) — in-process tools.
