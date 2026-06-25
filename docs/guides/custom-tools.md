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

> For read-only operations on your own graph, Cypher is usually enough. A custom tool is needed when the logic **goes beyond Memgraph/pgvector**.

## Anatomy of a Tool

`vedana_core.llm.Tool` (see `libs/vedana-core/src/vedana_core/llm.py`):

```python
class Tool[T: BaseModel]:
    def __init__(
        self,
        name: str,
        description: str,
        args_cls: type[T],
        fn: Callable[[T], Awaitable[str]] | Callable[[T], str],
    ) -> None: ...

    async def call(self, args_json: str) -> str: ...
```

- **`name`** — the tool name, visible to the LLM.
- **`description`** — what the tool does. Determines **when** the LLM calls it.
- **`args_cls`** — a `pydantic.BaseModel` schema for arguments. The LLM gets the schema and produces JSON.
- **`fn`** — async or sync function; takes an instance of `args_cls`, returns a string (which goes into the LLM context).

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


fx_tool = Tool(
    name="fx_convert",
    description="Convert an amount from one currency to another using current exchange rates.",
    args_cls=FxArgs,
    fn=fx_fn,
)
```

## Where to wire it in

The simplest path is to subclass `RagAgent` and override `text_to_answer_with_vts_and_cypher`:

```python
from vedana_core.rag_agent import RagAgent

class MyRagAgent(RagAgent):
    async def text_to_answer_with_vts_and_cypher(self, text_query, threshold, top_n=5):
        # ... reproduce the base method's logic ...
        # but add your tool to the list:
        tools.append(fx_tool)
        ...
```

Less invasively — override `RagPipeline.process_rag_query` so it builds your agent:

```python
class MyRagPipeline(RagPipeline):
    async def process_rag_query(self, query, ctx):
        # ... same logic as RagPipeline, but with MyRagAgent ...
```

And then swap the class in `make_vedana_app`:

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

### Async vs sync

`Tool` supports both signatures. If your function is synchronous (a pure calculation), write it sync — `Tool.call` will run it via `asyncio.to_thread`. For I/O — async only.

### Testing

Before production:

- unit-test the `fn` itself;
- integration-test by running golden questions through the pipeline with the tool plugged in and comparing answers;
- monitor — the built-in `llm_calls_total{model}` counter only tracks LLM calls (labelled by model), **not tool calls**. There is no built-in per-tool counter. If you need one, register your own `prometheus_client.Counter` (e.g. `vedana_tool_calls_total{tool_name}`) and increment it inside your `fn`.

## What you can't do with a custom tool

- **Force the LLM to always call your tool.** That's the model's decision based on the description. If you want a guarantee, use the playbook (Queries) with an explicit instruction.
- **Modify context after a tool call.** A tool returns a string — that's all the LLM sees.
- **Forbid the LLM from using your tool.** Once registered, the LLM may pick it. Remove it from the `tools` list to disable temporarily.

## What's next

- [Tools concept](../concepts/tools-for-vedana.md) — the theory.
- [Vedana Core architecture](../architecture/vedana-core.md) — `LLM` and tool-loop details.
