---
title: Python API
section: API Reference
order: 5
---

# Python API

If you need programmatic access to Vedana without HTTP — for batch processing, tests, or embedding in your own Python service — use the Python API directly.

## Installation

In your project's `pyproject.toml`:

```toml
[project]
dependencies = [
  "vedana-core",
  "jims-core",
  # for ETL:
  # "vedana-etl",
]
```

When working inside the uv workspace, every internal package (`jims-core`, `vedana-core`, `vedana-etl`, `jims-api`, `jims-widget`, `jims-telegram`, `jims-tui`, `jims-backoffice`, `vedana-backoffice`) is installed editable from `libs/*` — see the `[tool.uv.sources]` block in the root `pyproject.toml`. The `Makefile` exposes `make build` targets that run `uv build` per package, and `make publish` uses `uv publish` with a GCP OAuth token. The published artifacts go to Epoch8's internal GCP Artifact Registry; **Vedana packages are not on public PyPI** at the moment. To consume them outside the workspace you need either to add them to your own `[tool.uv.sources]` (workspace mode), or to configure your installer against the internal registry.

## Minimal example: ask a question

```python
import asyncio
from uuid import UUID
from jims_core.util import uuid7
from vedana_core.app import make_jims_app


async def main():
    app = await make_jims_app()

    # create a new thread
    ctl = await app.new_thread(
        contact_id="batch-job",
        thread_id=uuid7(),
        thread_config={"interface": "python"},
    )

    # record the user message
    await ctl.store_user_message(
        event_id=uuid7(),
        content="What are Geneva Durben's interests?",
    )

    # run the main pipeline
    outgoing = await ctl.run_pipeline_with_context(app.pipeline)

    # assistant messages
    for ev in outgoing:
        if ev.event_type == "comm.assistant_message":
            print("ASSISTANT:", ev.event_data["content"])

asyncio.run(main())
```

## Continue an existing thread

```python
from jims_core.thread.thread_controller import ThreadController

ctl = await ThreadController.from_thread_id(app.sessionmaker, thread_id)
# or
ctl = await ThreadController.latest_thread_from_contact_id(app.sessionmaker, "batch-job")
```

## Access to low-level components

Sometimes you need direct access to Vedana's components — to run Cypher without the LLM, or build an embedding.

```python
from vedana_core.app import make_vedana_app

vedana = await make_vedana_app()

# Cypher directly
records = list(await vedana.graph.execute_ro_cypher_query(
    "MATCH (p:person) RETURN p.person_name LIMIT 10"
))

# vector search directly
from jims_core.llms.llm_provider import LLMProvider

llm = LLMProvider()  # reads MODEL / EMBEDDINGS_MODEL / EMBEDDINGS_DIM from env
emb = await llm.create_embedding("quokka")
hits = await vedana.vts.vector_search(
    label="interest",
    prop_type="node",
    prop_name="interest_name",
    embedding=emb,
    threshold=0.7,
    top_n=5,
)

# data model
anchors = await vedana.data_model.get_anchors()
links = await vedana.data_model.get_links()
queries = await vedana.data_model.get_queries()
```

## Custom pipeline

Implement the `Pipeline` protocol (`jims_core.schema.Pipeline`):

```python
from typing import Any
from jims_core.thread.thread_context import ThreadContext

class MyPipeline:
    async def __call__(self, ctx: ThreadContext) -> Any:  # the protocol allows Any
        msg = ctx.get_last_user_message()
        ctx.send_message(f"Echo: {msg}")
        # return value is unused by ThreadController; returning None is fine
```

Swap it into `JimsApp`:

```python
from jims_core.app import JimsApp

custom_app = JimsApp(
    sessionmaker=app.sessionmaker,
    pipeline=MyPipeline(),
    conversation_start_pipeline=app.conversation_start_pipeline,
)
```

And run the same `jims-api`:

```bash
uv run python -m jims_api.main --app my_module:custom_app
```

## Custom tool

See `vedana_core.llm.Tool` and the [Custom Tools guide](../guides/custom-tools.md). The key idea:

```python
from vedana_core.llm import Tool
from pydantic import BaseModel, Field

class CalcArgs(BaseModel):
    expression: str = Field(description="arithmetic expression")

async def calc_fn(args: CalcArgs) -> str:
    return str(eval(args.expression))  # demo, not for prod

calc_tool = Tool(
    name="calculator",
    description="Evaluate an arithmetic expression",
    args_cls=CalcArgs,
    fn=calc_fn,
)
```

Add it to the `tools` list in your pipeline / overridden `RagAgent`.

## ETL programmatically

```python
from vedana_etl.pipeline import get_pipeline, default_custom_steps
from datapipe.compute import build_compute, run_steps

pipeline = get_pipeline(custom_steps=default_custom_steps)
# datapipe API:
# build_compute / run_steps_changelist
```

See the [Datapipe](https://github.com/epoch8/datapipe) docs for API details.

## ThreadContext: reading

```python
ctx = await ctl.make_context()

# the last 20 comm. messages + related context.* events
history = ctx.context(20)

# the last user message
last_msg = ctx.get_last_user_message()

# thread state, if you previously called set_state
class MyState(BaseModel):
    selected_category: str

state = ctx.get_state("my_state", MyState)
```

## ThreadContext: writing

```python
ctx.send_message("Hello!")
ctx.send_event("custom.metric", {"value": 42})
ctx.set_state("my_state", MyState(selected_category="laptops"))
await ctx.update_agent_status("Working on it...")
```

When the pipeline finishes, `ThreadController.run_pipeline_with_context` writes `outgoing_events` to the DB.

## What's next

- [Architecture: JIMS Core](../architecture/jims-core.md) — `Pipeline` / `ThreadContext` details.
- [Architecture: Vedana Core](../architecture/vedana-core.md) — `RagPipeline` / `RagAgent` details.
- [Custom Tools](../guides/custom-tools.md) — adding your own tool.
