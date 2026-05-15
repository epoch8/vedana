---
title: Code Style
section: Contributing
order: 2
---

# Code Style

This document describes Vedana's code style. If something isn't covered, follow PEP 8 and general Python community practices.

## Linter: ruff

Configuration is in the root `pyproject.toml`:

```toml
[tool.ruff]
line-length = 120
```

Run:

```bash
uv run ruff check .
uv run ruff format .
```

CI fails ruff on any violation. PRs don't merge until green.

## Type checker: mypy

Configuration:

```toml
[tool.mypy]
ignore_missing_imports = true
follow_imports = "silent"
namespace_packages = true
enable_incomplete_feature = ["NewGenericSyntax"]
```

Type annotations are **required** for:

- public functions and methods (anything not `_private`);
- API endpoint signatures;
- classes in `dataclass`/`pydantic`.

Run:

```bash
uv run mypy libs/vedana-core/src
uv run mypy libs/jims-core/src
```

## Python version

Target — **3.12**. We use new features:

- `match` / `case` where appropriate;
- `type Foo = ...` (PEP 695) for type aliases;
- generic syntax `class Tool[T: BaseModel]` (PEP 695).

Don't use the 3.9 `from typing import Generic, TypeVar` style — write the new way.

## Import structure

```python
# 1. Standard library
import asyncio
from datetime import datetime

# 2. Third-party packages
import httpx
from pydantic import BaseModel

# 3. Internal Vedana packages
from jims_core.thread.thread_context import ThreadContext
from vedana_core.graph import Graph

# 4. Local imports
from .util import some_helper
```

Ruff sorts automatically (`isort` mode).

## Naming conventions

- `snake_case` for functions, methods, variables, modules.
- `PascalCase` for classes, types, Enums.
- `UPPER_SNAKE` for constants.
- `_leading_underscore` for private items.
- Names that go into the LLM context or into Cypher (anchors, links, attributes) follow the style described in [Data Model](../data-model/overview.md): anchor — lowercase singular, link sentence — `ANCHOR_verb_ANCHOR` UPPER_CASE.

## Docstrings

Use Google-style docstrings for functions with non-trivial semantics:

```python
async def process_rag_query(self, query: str, ctx: ThreadContext) -> tuple[str, list, dict]:
    """Run the full RAG pipeline for a single user query.

    Args:
        query: The user's question text.
        ctx: Thread context containing history and LLM provider.

    Returns:
        A tuple of (answer, agent_query_events, technical_info).

    Raises:
        Exception: Any unhandled exception from data model loading or tool execution.
    """
```

Don't duplicate type hints in prose (ruff/`pydocstyle` may complain).

For trivial functions (getters, simple wrappers) docstrings aren't needed.

## Async / sync

Vedana is async-first. Write functions as `async def` except when:

- the function is purely computational (no I/O);
- the function is called from a sync context (e.g. `Tool.fn` supports sync because `Tool.call` wraps it in `asyncio.to_thread`).

In async functions:

- don't block the loop. Long CPU work — through `asyncio.to_thread`.
- don't use `time.sleep` — use `asyncio.sleep`.
- httpx async client (`httpx.AsyncClient`), not `requests`.

## Error handling

- Outer pipeline boundaries (HTTP API, Telegram, widget) catch `Exception` and return a generic message to the user, while writing the traceback to the log.
- Internal modules raise **specific** exceptions (`ValueError`, `KeyError`, or your subclasses), not a global `Exception`.
- Don't swallow exceptions with `except: pass`. If you really need to — `except SpecificError: logger.warning(...)`.

Example from `RagPipeline.__call__`:

```python
try:
    answer, ... = await self.process_rag_query(user_query, ctx)
    ctx.send_message(answer)
except Exception as e:
    self.logger.exception(f"Error in RAG pipeline: {e}")
    ctx.send_message("An error occurred while processing the request")  # generic for the user
    ctx.send_event("rag.error", {"query": user_query, "error": str(e), "traceback": traceback.format_exc()})
```

## Logging

- Use the standard `logging` for libraries (`libs/`).
- Use `loguru` for applications (CLI wrappers in `apps/`).
- Levels:
  - `DEBUG` — detailed tool calls, data model selection, query parameters.
  - `INFO` — normal events (thread created, ETL step done).
  - `WARNING` — odd but non-fatal (filtering fell back to full DM, iteration limit).
  - `ERROR` / `EXCEPTION` — actual errors.
- Never log secrets (tokens, keys).

## Pydantic

- For configuration models — `pydantic_settings.BaseSettings`.
- For API / tool argument models — `pydantic.BaseModel`.
- Use `Field(description="...")` for fields that go to the LLM (it's part of the tool schema).
- For strict validation — `pydantic.types` or your own validators.

## SQL / SQLAlchemy

- Use the async API (`AsyncSession`, `async_sessionmaker`).
- Never build SQL via f-strings. Use parameterisation or the ORM.
- Transactions — through `async with sessionmaker() as session`.

## Cypher

Cypher strings are safe for read-only, but follow these rules:

- parameterise via `$param`, not string interpolation;
- escape labels via `escape_labels` (see `vedana_core/graph.py`);
- don't do unbounded traversals (`MATCH (a)-[*]-(b)`) without a limit.

## Tests

See [Testing](./contributing/testing.md). In short:

- unit tests are required for non-trivial logic;
- integration tests — where actual DBs / LLM are involved.
- LLM calls in tests — via VCR (`tests/cassettes`) or LiteLLM router mocks.

## What definitely not to do

- Use `print()` instead of the logger.
- Global state variables (except `LLMProvider` usage counters that are specifically per-process).
- Module-level imports that do I/O (e.g. read a file at import time).
- Magic numbers without a constant / comment.
- Long functions > 80 lines without decomposition.
- TODOs without an issue (if you leave a TODO — link it to an issue number).

## What's next

- [Testing](./contributing/testing.md)
- [Repository Structure](./contributing/repository-structure.md)
- [Contributing Guide](./contributing/contributing.md)
