---
title: Configuration Reference
section: API Reference
order: 6
---

# Configuration Reference

> **Two configuration documents — when to read which.** This page is the **authoritative complete reference**, grouped by the Python class that reads each variable — use it as a lookup table when wiring up your `.env`. For a purpose-grouped tour (LLM / RAG / DB / observability) read top-to-bottom, see the [Configuration guide](../getting-started/configuration.md). When the two pages disagree on a default, **this page wins** (it mirrors the source code directly).

A full reference of environment variables read by Vedana, grouped by the Python class that reads them.

> **All three classes below use `env_prefix=""` and read the same `.env` file.** That means a single env variable like `MODEL` is read by every class that declares it. The "default" column is the value used **only when the variable is absent from `.env` and the runtime environment** — in any real deployment `.env.example` sets it explicitly.

## Vedana Core (`VedanaCoreSettings`)

File: `libs/vedana-core/src/vedana_core/settings.py`.

| Variable                       | Type   | Default                  | Description                                                       |
| ------------------------------ | ------ | ------------------------ | ------------------------------------------------------------------ |
| `GRIST_SERVER_URL`              | str    | —                        | Grist base URL (`http://grist:8484` or `https://api.getgrist.com`).|
| `GRIST_API_KEY`                 | str    | —                        | Grist API key.                                                    |
| `GRIST_DATA_MODEL_DOC_ID`       | str    | —                        | DocId of the document with the data model.                        |
| `GRIST_DATA_DOC_ID`             | str    | —                        | DocId of the document with the data.                              |
| `DEBUG`                         | bool   | `false`                  | Verbose logs and dev features.                                    |
| `TIMEZONE`                      | str    | `UTC`                    | Timezone for `{current_date}` in the agent prompt.                 |
| `MODEL`                         | str    | `gpt-4.1-mini`           | Main LLM model used by `RagPipeline` / `RagAgent` for answer generation and Cypher. **This is the value that wins in the standard pipeline** — Vedana Core overrides the inner LLM provider on every call. |
| `REASONING_EFFORT`              | str    | `disable`                | Passed to the agent loop, finalisation and no-answer calls. Models that don't support it ignore it (LiteLLM runs with `drop_params=True`). |
| `ENABLE_DM_FILTERING`           | bool   | `false`                  | Enable the data model filtering step.                             |
| `FILTER_MODEL`                  | str    | `gpt-4.1-mini`           | Model used for filtering, the scenario selection fallback and source extraction. |
| `JUDGE_MODEL`                   | str    | `gpt-4.1-mini`           | Model used for evaluation (LLM-as-judge).                          |
| `EXTRACT_SOURCES`               | bool   | `false`                  | Extract and ground answer sources after the answer is sent. See [Sources & grounding](../architecture/sources.md). |
| `EMBEDDINGS_MODEL`              | str    | `text-embedding-3-large` | Embeddings model.                                                  |
| `EMBEDDINGS_DIM`                | int    | `1024`                   | Embedding dimensions.                                               |
| `EMBEDDINGS_THRESHOLD`          | float  | `0.8`                    | Fallback `embed_threshold` when an attribute has none.             |
| `EMBEDDINGS_TOP_N`              | int    | `10`                     | Fallback `embed_top_n` for attributes, and the pipeline's `top_n`. |
| `PIPELINE_HISTORY_LENGTH`       | int    | `20`                     | How many recent `comm.*` messages go into the context.            |
| `CYPHER_ROWS_LIMIT`             | int    | `1000`                   | Maximum rows returned by the `cypher` tool.                         |
| `MEMGRAPH_URI`                  | str    | —                        | Memgraph Bolt URI.                                                  |
| `MEMGRAPH_USER`                 | str    | —                        | Memgraph user.                                                     |
| `MEMGRAPH_PWD`                  | str    | —                        | Memgraph password.                                                  |

### Scenario routing

See [Scenario routing](../data-model/scenario-routing.md).

| Variable                            | Type   | Default | Description                                                                 |
| ----------------------------------- | ------ | ------- | ---------------------------------------------------------------------------- |
| `SCENARIO_ROUTING_THRESHOLD`        | float  | `0.8`   | Embedding similarity needed to accept a scenario without rerank; below it, rerank may run. |
| `SCENARIO_ROUTING_TOP_N`            | int    | `10`    | How many routing examples are fetched per request.                           |
| `SCENARIO_RERANK_THRESHOLD`         | float  | `0.3`   | Lowest rerank score the top scenario can have.                                |
| `SCENARIO_RERANK_MARGIN`            | float  | `0.1`   | How far the top scenario must lead the next one.                              |
| `SCENARIO_RERANK_SIMILARITY_FLOOR`  | float  | `0.5`   | Lowest embedding similarity for a scenario picked by rerank.                  |

### Agent loop

See [Tool choice & the agent loop](../architecture/agent-loop.md).

| Variable                        | Type | Default | Description                                                      |
| ------------------------------- | ---- | ------- | ----------------------------------------------------------------- |
| `AGENT_MAX_ITERS`               | int  | `5`     | Default round budget, and default `max_iters` for scenarios.     |
| `AGENT_MAX_ITERS_CEILING`       | int  | `12`    | Hard limit on the budget, including after scenario switches.     |
| `AGENT_MAX_SCENARIO_SWITCHES`   | int  | `2`     | Maximum `select_scenario` switches per request.                  |

### Web search

| Variable                    | Type                        | Default      | Description                                     |
| --------------------------- | --------------------------- | ------------ | ------------------------------------------------ |
| `ENABLE_WEB_SEARCH`         | bool                        | `false`      | Register the `web_search` tool.                  |
| `WEB_SEARCH_MODEL`          | str                         | `MODEL`      | Model that runs web searches (must support search). |
| `WEB_SEARCH_CONTEXT_SIZE`   | `low` / `medium` / `high`   | `medium`     | How much search context to use.                 |

### Reranker

| Variable              | Type                    | Default               | Description                                                     |
| --------------------- | ----------------------- | --------------------- | ---------------------------------------------------------------- |
| `RERANKER_BACKEND`    | `disabled` / `cohere`   | `disabled`            | Turns on reranking for tool results and scenario routing.        |
| `RERANKER_MODEL`      | str                     | `cohere/rerank-v3.5`  | Rerank model name in LiteLLM.                                    |
| `RERANKER_MAX_DOCS`   | int                     | `1000`                | Rows above this number are cut before reranking.                 |
| `COHERE_API_KEY`      | str                     | —                     | Cohere API key.                                                  |

### MCP client

See [External MCP tools](../guides/mcp-tools.md). Requires the `vedana-core[mcp]` extra.

| Variable                         | Type        | Default | Description                                                        |
| -------------------------------- | ----------- | ------- | ------------------------------------------------------------------- |
| `VEDANA_MCP_TOOL_SERVERS`        | JSON list   | empty   | `[{"name": "...", "url": "https://.../mcp", "token": "..."}]`       |
| `VEDANA_MCP_TOOL_CALL_TIMEOUT`   | float (s)   | `120`   | Timeout for each remote tool call.                                  |

## LLM Provider (`LLMSettings`)

File: `libs/jims-core/src/jims_core/llms/llm_provider.py`.

`LLMSettings` is the inner provider used by `jims-core`. It reads the same `MODEL` env var as `VedanaCoreSettings`, but if you instantiate `jims-core` standalone (without `vedana-core`), no override happens and the fallback default (`gpt-4.1-nano`) applies.

> **Which `MODEL` actually runs?** In the standard Vedana stack the two classes read the **same env var**, so whatever you put in `.env` (e.g. `MODEL=gpt-4.1-mini`) is used by both. The defaults differ only as a safety net for standalone `jims-core` users who don't set the env var: vedana-core falls back to `gpt-4.1-mini`, jims-core falls back to `gpt-4.1-nano` (cheap, OK for non-RAG flows). **For all production deployments — set `MODEL` explicitly in `.env`.**

| Variable                          | Type   | Default                          | Description                                                            |
| --------------------------------- | ------ | -------------------------------- | ----------------------------------------------------------------------- |
| `MODEL`                            | str    | `gpt-4.1-nano`                   | Fallback model for `jims-core` standalone use. Overridden by vedana-core in the standard pipeline. |
| `EMBEDDINGS_MODEL`                 | str    | `text-embedding-3-large`         | Embeddings model.                                                       |
| `EMBEDDINGS_DIM`                   | int    | `1024`                           | Dimensionality.                                                          |
| `EMBEDDINGS_MAX_BATCH_SIZE`        | int    | `2048`                           | Max texts per embeddings batch.                                         |
| `EMBEDDINGS_MAX_TOKENS_PER_BATCH`  | int    | `200000`                         | Max tokens per batch.                                                    |
| `MODEL_API_KEY`                    | str    | `None`                           | If set, overrides the key for the main model.                           |
| `EMBEDDINGS_MODEL_API_KEY`         | str    | `None`                           | Same for embeddings.                                                     |
| `OPENROUTER_API_BASE_URL`          | str    | `https://openrouter.ai/api/v1`   | OpenRouter endpoint.                                                     |

> Provider keys (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `GOOGLE_APPLICATION_CREDENTIALS`, etc.) are read by LiteLLM from standard env vars.

## ETL (`vedana_etl.settings.Settings`)

File: `libs/vedana-etl/src/vedana_etl/settings.py`.

| Variable                | Type | Default | Description                                                       |
| ----------------------- | ---- | ------- | ------------------------------------------------------------------ |
| `DB_CONN_URI`            | str  | —       | Postgres connection URI for Datapipe.                             |
| `GRIST_TEST_SET_DOC_ID`  | str  | empty   | DocId of the document with the golden dataset.                    |
| `GDS_TABLE_NAME`         | str  | `Gds`   | Table name in the test set doc.                                    |
| `TESTS_TABLE_NAME`       | str  | `Tests` | Test table name in the test set doc.                               |
| `TEST_ENVIRONMENT`       | str  | empty   | Environment label for evaluation results.                         |

## Database

| Variable                       | Type   | Default              | Description                                                   |
| ------------------------------ | ------ | -------------------- | -------------------------------------------------------------- |
| `JIMS_DB_CONN_URI`              | str    | —                    | Postgres URI for JIMS.                                         |
| `JIMS_DB_USE_NULL_POOL`         | bool   | `false`              | Disable the connection pool (for serverless).                 |
| `JIMS_DB_POOL_SIZE`             | int    | SQLA default         | Pool size.                                                      |
| `JIMS_DB_POOL_MAX_OVERFLOW`     | int    | SQLA default         | Extra connections above the pool.                               |
| `CREATE_PGVECTOR_EXTENSION`     | bool   | `true`               | Create the pgvector extension via migration.                    |

## Backoffice

| Variable                  | Type | Default                         | Description                                       |
| ------------------------- | ---- | ------------------------------- | -------------------------------------------------- |
| `VEDANA_BACKOFFICE_DEBUG`  | bool | `true` (in .env.example)       | Dev-only features in the backoffice.            |

## Telegram

| Variable             | Type | Description                       |
| -------------------- | ---- | --------------------------------- |
| `TELEGRAM_BOT_TOKEN` | str  | Bot token from BotFather.        |

## Sentry

| Variable             | Type | Description                                  |
| -------------------- | ---- | --------------------------------------------- |
| `SENTRY_DSN`         | str  | Sentry endpoint. Empty means off.            |
| `SENTRY_ENVIRONMENT` | str  | Environment name shown in Sentry.            |

## Datapipe

| Variable             | Type | Default              | Description                          |
| -------------------- | ---- | -------------------- | ------------------------------------- |
| `DATAPIPE_PIPELINE`   | str  | `vedana_etl.app`     | Datapipe pipeline entry point (`module[:attr]`, default attr `app`). Used by `datapipe run` / `datapipe api` and by the backoffice's `project_runtime.get_etl_bindings`. |

## Backoffice

| Variable                    | Type | Default                           | Description                          |
| --------------------------- | ---- | --------------------------------- | ------------------------------------- |
| `VEDANA_BACKOFFICE_DEBUG`   | bool | `false`                           | Enables dev-only features in the backoffice. |
| `VEDANA_APP`                | str  | `vedana_core.app:make_vedana_app` | `module[:attr]` of a factory returning `vedana_core.app.VedanaApp`. Resolved by `vedana_backoffice.project_runtime.get_vedana_app`. The factory may be sync or async; the result is isinstance-checked against `VedanaApp`. |

## Prefixes and read order

- `vedana_core.settings` uses `env_prefix=""` (no prefix) and reads `apps/vedana/.env` (working directory must be `apps/vedana`). The exceptions are the MCP client settings, which use the explicit `VEDANA_MCP_` names.
- `vedana_etl.settings` — same.
- `jims_core.llms.llm_provider.LLMSettings` — same.
- JIMS CLI wrappers use `auto_envvar_prefix="JIMS"` (`click`), so the `--api-key` flag can be provided as `JIMS_API_KEY=...`.

## .env.example vs .env.ci-cd

`apps/vedana/` ships with two ready profiles:

- **`.env.example`** — for local development; references `db`, `memgraph`, `grist` (the docker-compose service names).
- **`.env.ci-cd`** — for CI; uses `localhost` addresses and test keys.

For production, create your own profile and **never** commit secret keys (`.env` is in `.gitignore`).
