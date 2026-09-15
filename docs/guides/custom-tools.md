---
title: Writing a Custom Tool
section: Guides
order: 11
---

# Writing a Custom Tool

`vector_text_search` and `cypher` cover the vast majority of Vedana scenarios. But sometimes you want to give the LLM something else: calculations, calls to external APIs, domain logic. That's what custom tools are for.

## When you need a custom tool

- **Calculations:** currency conversions, taxes, shipping, prices including discounts.
- **External APIs:** exchange rates, weather, order status from ERP.
- **Domain logic:** product compatibility checks, configuration validation.
- **Actions (write):** create a ticket, send a notification, update a record.

> For read-only operations on your own graph, Cypher is usually enough. A custom tool is needed when the logic **goes beyond Memgraph/pgvector**. If the capability already lives in another service that speaks MCP, connect it through [MCP tools](./mcp-tools.md) — no code needed.

## Anatomy of a Tool

`vedana_core.llm.Tool` (see `libs/vedana-core/src/vedana_core/llm.py`) is a name, a description, an OpenAI-style function definition and an async handler. Build one with a constructor:

```python
class Tool:
    @classmethod
    def from_pydantic(
        cls,
        name: str,
        description: str,
        args_cls: type[BaseModel],
        fn: Callable[[BaseModel], Awaitable[str]],
    ) -> "Tool": ...

    @classmethod
    def from_json_schema(
        cls,
        name: str,
        description: str,
        schema: dict[str, Any],
        fn: Callable[[dict[str, Any]], Awaitable[str]],
    ) -> "Tool": ...

    async def call(self, args_json: str) -> str: ...
```

- **`name`** — the tool name, visible to the LLM. Must match `^[a-zA-Z0-9_-]{1,64}$`.
- **`description`** — what the tool does. Determines **when** the LLM calls it.
- **`args_cls`** / **`schema`** — a `pydantic.BaseModel` for arguments (validated before `fn` runs), or a raw JSON Schema (`fn` receives the parsed dict). The LLM gets the schema and produces JSON. Invalid arguments come back to the LLM as `Invalid tool args: ...`.
- **`fn`** — an async function; returns a string (which goes into the LLM context).

## Example: currency conversion

```python
import httpx
from pydantic import BaseModel, Field
from vedana_core.llm import Tool


class FxArgs(BaseModel):
    from_currency: str = Field(description="ISO currency code, e.g. USD")
    to_currency: str = Field(description="ISO currency code, e.g. EUR")
    amount: float = Field(default=1.0, description="Amount to convert (default: 1)")


async def fx_fn(args: FxArgs) -> str:
    async with httpx.AsyncClient(timeout=5) as client:
        resp = await client.get(
            "https://api.exchangerate.host/convert",
            params={"from": args.from_currency, "to": args.to_currency, "amount": args.amount},
        )
        if resp.status_code != 200:
            return f"FX API error: {resp.status_code}"
        data = resp.json()
        return f"{args.amount} {args.from_currency} = {data['result']:.2f} {args.to_currency}"


fx_tool = Tool.from_pydantic(
    name="fx_convert",
    description="Convert an amount from one currency to another using current exchange rates.",
    args_cls=FxArgs,
    fn=fx_fn,
)
```

## Where to wire it in

The toolset is assembled per request in `RagAgent.answer_question`: `build_tools(...)` returns the built-ins, then MCP tools are appended. Add your tool at the same point — subclass `RagAgent` and reproduce `answer_question` with one extra line:

```python
from vedana_core.rag_agent import RagAgent
from vedana_core.tools import build_tools


class MyRagAgent(RagAgent):
    async def answer_question(self, text_query, threshold, top_n, data_model_description, scenario=None):
        # ... same as RagAgent.answer_question: create ToolCallRecorder + RunState,
        # build the tools with build_tools(...) and build_mcp_tools(...) ...
        tools.append(fx_tool)
        # ... then call self.llm.answer_with_tools(...) as the base method does ...
```

If you want the call to appear in `technical_info.tool_calls`, record it inside `fn` with the request's `ToolCallRecorder` (`recorder.record("fx_convert", args.model_dump(), result)`) — build the tool inside `answer_question` so the handler can close over it.

Then override `RagPipeline.generate_answer` so it builds your agent:

```python
class MyRagPipeline(RagPipeline):
    async def generate_answer(self, query, ctx):
        # ... same logic as RagPipeline.generate_answer, but with MyRagAgent ...
```

And swap the class in `make_vedana_app`:

```python
import logging

pipeline = MyRagPipeline(
    graph=graph,
    vts=vts,
    data_model=data_model,
    logger=logging.getLogger(__name__),  # RagPipeline requires logger as a positional/keyword arg
)
```

> `RagPipeline.__init__` requires `logger` — there is no default. The Vedana factory `make_vedana_app` passes `loguru.logger`; in your own setup pass any logger you like.

## Allow it in the playbook

Registering a tool isn't enough: once a scenario is active, the agent only sees the tools in that scenario's `tools_allowed`. Add `fx_convert` to `tools_allowed` of every [Queries](../data-model/queries.md) scenario that should use it, then reload the data model.

To make the agent call your tool first, set the scenario's `tool_choice` to `required` and describe the call in `query_example`.

## Best practices

### Description

The LLM decides when to call the tool **only** based on the description. So:

- start with a verb: "Convert…", "Fetch…", "Calculate…";
- specify when **exactly** to use it ("Use when the user asks about exchange rates", "Use when the user wants the current weather");
- specify when **not** to use it ("Do NOT use for historical rates older than 30 days").

### Arguments

- Use `Field(description=...)` for every field — the LLM uses it.
- Use Enums for restricted values (currencies are better as Enums than free strings).
- Make required fields default-less, optional fields with sensible defaults.

### Return value

The LLM gets your string as the "tool result" and continues. So:

- return a structured, human-readable string (not JSON, unless the LLM is meant to parse it further);
- include units, context (`"3.42 EUR"`, not `"3.42"`);
- on error, return a short description (`"FX API error: 503"`) — the LLM will tell the user or retry.

### Security

- **Never** trust LLM-supplied arguments unconditionally. If the tool writes (creates a ticket, sends an email), validate against business rules.
- For write tools, prefer a two-step scheme: the tool prepares a "draft", the user confirms in the UI, a separate handler executes.
- Limit rate / cost: timeouts, exponential back-off retries, daily budgets.

### Async only

`Tool` handlers are async. If your logic is synchronous and CPU-heavy, wrap it with `await asyncio.to_thread(...)` inside the handler so it doesn't block the event loop.

### Testing

Before production:

- unit-test the `fn` itself;
- integration-test by running golden questions through the pipeline with the tool plugged in and comparing answers;
- monitor — the built-in `llm_calls_total{model}` counter only tracks LLM calls (labelled by model), **not tool calls**. There is no built-in per-tool counter. If you need one, register your own `prometheus_client.Counter` (e.g. `vedana_tool_calls_total{tool_name}`) and increment it inside your `fn`.

## What you can't do with a custom tool

- **Pick which of several allowed tools the LLM calls.** That's the model's decision based on the description. `tool_choice = required` guarantees *a* tool call on the scenario's first round; to guarantee *your* tool, make it the only one in that scenario's `tools_allowed`.
- **Modify context after a tool call.** A tool returns a string — that's all the LLM sees.

To keep the LLM away from your tool in a given scenario, leave it out of that scenario's `tools_allowed`. Without an active scenario, only the default tools and `select_scenario` are allowed.

## What's next

- [Tools concept](../concepts/tools-for-vedana.md) — the theory.
- [Vedana Core architecture](../architecture/vedana-core.md) — `LLM` and tool-loop details.
- [Tool choice & the agent loop](../architecture/agent-loop.md) — how `tools_allowed` and `tool_choice` are applied.
- [External MCP tools](./mcp-tools.md) — tools from other services.
