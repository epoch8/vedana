---
title: Configuration
section: Getting Started
order: 4
---

# Configuration

> **Two configuration documents — when to read which.** This page is a **purpose-grouped tour** of Vedana's environment variables — read top-to-bottom to understand what to set when bringing up the stack. The [Configuration Reference](../api/configuration-reference.md) is the **authoritative complete list** grouped by Python class — use it as a lookup table. Defaults shown in this guide are mirrored from the reference; when you spot a discrepancy, the reference (and ultimately the source code) wins.

Vedana is configured through environment variables. Everything important is read via `pydantic-settings` in three places:

- `vedana_core.settings.VedanaCoreSettings` — main settings of the RAG pipeline.
- `vedana_etl.settings.Settings` — ETL settings.
- `jims_core.llms.llm_provider.LLMSettings` — LLM provider parameters.

All three classes use `env_prefix=""` and read the same `.env` file at `apps/vedana/.env`, so any shared variable name (e.g. `MODEL`, `EMBEDDINGS_MODEL`) is read by every class that declares it.

## LLM

Vedana runs on top of [LiteLLM](https://www.litellm.ai/). Model names follow the LiteLLM format (`gpt-4.1-mini`, `openrouter/anthropic/claude-3.5-sonnet`, `vertex_ai/gemini-2.5-pro`, etc.). Provider access is configured through standard environment variables (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `GOOGLE_APPLICATION_CREDENTIALS`, etc.).

| Variable                          | Purpose                                                                                                | Default                       |
| --------------------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------ |
| `MODEL`                            | Main model: user answers and Cypher generation. Always set this explicitly in `.env`. ([why](../api/configuration-reference.md#llm-provider-llmsettings)) | `gpt-4.1-mini`                |
| `REASONING_EFFORT`                 | Reasoning effort for the main agent calls. Ignored by models that don't support it.                    | `disable`                     |
| `FILTER_MODEL`                     | Model for the data model filtering step (see [`RagPipeline.filter_data_model`](../architecture/vedana-core.md#data-model-filtering)), the scenario selection fallback and source extraction. Usually smaller and faster. | `gpt-4.1-mini`               |
| `JUDGE_MODEL`                      | Model for the evaluation pipeline (LLM-as-judge).                                                       | `gpt-4.1-mini`                |
| `EMBEDDINGS_MODEL`                 | Embeddings model.                                                                                       | `text-embedding-3-large`      |
| `EMBEDDINGS_DIM`                   | Embeddings dimensionality. Changing this requires a SQL migration.                                      | `1024`                        |
| `EMBEDDINGS_MAX_BATCH_SIZE`        | Maximum number of texts in one embeddings batch.                                                        | `2048`                        |
| `EMBEDDINGS_MAX_TOKENS_PER_BATCH`  | Maximum tokens in one embeddings batch.                                                                  | `200000`                      |
| `MODEL_API_KEY`                    | If set, overrides the key for the main model (otherwise the standard provider env var is used).         | `None`                        |
| `EMBEDDINGS_MODEL_API_KEY`         | Same for the embeddings model.                                                                           | `None`                        |
| `OPENROUTER_API_BASE_URL`          | OpenRouter endpoint (you can change it to your own gateway).                                            | `https://openrouter.ai/api/v1`|

> **Note on `MODEL`.** Vedana has two settings classes that both read the `MODEL` env var (`VedanaCoreSettings` and `LLMSettings` from jims-core). Their built-in fallback defaults differ (`gpt-4.1-mini` vs `gpt-4.1-nano`) but in any real deployment you should set `MODEL` explicitly in `.env` — both classes then read the same value. See [Configuration Reference → LLM Provider](../api/configuration-reference.md#llm-provider-llmsettings) for details.

## RAG pipeline

| Variable                    | Purpose                                                                                              | Default      |
| --------------------------- | ----------------------------------------------------------------------------------------------------- | ------------ |
| `ENABLE_DM_FILTERING`        | Enables the data model filtering step before the main agent. Reduces tokens for large data models.   | `false`      |
| `EXTRACT_SOURCES`            | Extracts and grounds the sources the answer relied on (needs `is_source` attributes).                 | `false`      |
| `PIPELINE_HISTORY_LENGTH`    | How many recent `comm.*` messages to feed into the agent's context.                                  | `20`         |
| `CYPHER_ROWS_LIMIT`          | Maximum rows returned by the `cypher` tool.                                                           | `1000`       |
| `TIMEZONE`                   | Timezone for `{current_date}` in the agent prompt.                                                    | `UTC`        |
| `DEBUG`                      | Turns on additional logging and dev features.                                                         | `false`      |

The threshold and `top_n` parameters for vector search are set per embeddable attribute in the data model (`embed_threshold`, `embed_top_n`); `EMBEDDINGS_THRESHOLD` (`0.8`) and `EMBEDDINGS_TOP_N` (`10`) are the fallbacks.

### Scenario routing and the agent loop

| Variable                          | Purpose                                                                                   | Default   |
| --------------------------------- | ------------------------------------------------------------------------------------------ | --------- |
| `SCENARIO_ROUTING_THRESHOLD`      | Embedding similarity needed to accept a scenario without further checks.                  | `0.8`     |
| `SCENARIO_ROUTING_TOP_N`          | Routing examples fetched per request.                                                      | `10`      |
| `SCENARIO_RERANK_THRESHOLD` / `SCENARIO_RERANK_MARGIN` / `SCENARIO_RERANK_SIMILARITY_FLOOR` | Acceptance rules when routing reranks candidates. | `0.3` / `0.1` / `0.5` |
| `AGENT_MAX_ITERS`                 | Default tool-calling round budget.                                                          | `5`       |
| `AGENT_MAX_ITERS_CEILING`         | Hard limit on rounds, including after scenario switches.                                   | `12`      |
| `AGENT_MAX_SCENARIO_SWITCHES`     | How many times the agent may switch scenarios per request.                                  | `2`       |

See [Scenario routing](../data-model/scenario-routing.md) and [Tool choice & the agent loop](../architecture/agent-loop.md).

### Extra tools

| Variable                          | Purpose                                                                                   | Default      |
| --------------------------------- | ------------------------------------------------------------------------------------------ | ------------ |
| `RERANKER_BACKEND`                | `cohere` enables reranking of tool results and routing candidates (`COHERE_API_KEY`, `RERANKER_MODEL`, `RERANKER_MAX_DOCS`). | `disabled` |
| `ENABLE_WEB_SEARCH`               | Registers the `web_search` tool (`WEB_SEARCH_MODEL`, `WEB_SEARCH_CONTEXT_SIZE`).            | `false`      |
| `VEDANA_MCP_TOOL_SERVERS`         | JSON list of remote MCP servers whose tools the agent can call (`VEDANA_MCP_TOOL_CALL_TIMEOUT`). | empty    |

See [External MCP tools](../guides/mcp-tools.md) and the [Configuration Reference](../api/configuration-reference.md) for every variable.

## Databases

### PostgreSQL (JIMS + pgvector + Datapipe)

| Variable                       | Purpose                                                                | Example                                  |
| ------------------------------ | ---------------------------------------------------------------------- | ---------------------------------------- |
| `JIMS_DB_CONN_URI`              | Connection URI for JIMS (threads, events, state).                      | `postgresql://postgres:postgres@db:5432` |
| `DB_CONN_URI`                   | Connection URI for Datapipe / ETL.                                      | `postgresql://postgres:postgres@db:5432` |
| `JIMS_DB_USE_NULL_POOL`         | Disable the connection pool (useful in serverless setups).             | `false`                                  |
| `JIMS_DB_POOL_SIZE`             | Pool size.                                                              | SQLAlchemy default                       |
| `JIMS_DB_POOL_MAX_OVERFLOW`     | Maximum extra connections above the pool.                               | SQLAlchemy default                       |
| `CREATE_PGVECTOR_EXTENSION`     | Whether the migration should run `CREATE EXTENSION pgvector`. Use `true` for self-hosted Postgres where you manage the cluster (default). Use `false` on managed Postgres (Yandex Cloud, Google Cloud SQL, Supabase, Neon, RDS) — those vendors enable extensions through their own control plane rather than via SQL inside a migration. | `true`                                   |

> By default, JIMS and Datapipe write to the same database. They can be split into two if you need isolation.

### Memgraph

| Variable         | Purpose                          | Example                  |
| ---------------- | -------------------------------- | ------------------------ |
| `MEMGRAPH_URI`   | Memgraph Bolt endpoint.          | `bolt://memgraph:7687`   |
| `MEMGRAPH_USER`  | User.                             | `neo4j`                  |
| `MEMGRAPH_PWD`   | Password.                         | set in `.env.example`    |

## Data source

### Grist (default)

| Variable                   | Purpose                                                |
| -------------------------- | ------------------------------------------------------- |
| `GRIST_SERVER_URL`         | Grist base URL (`http://grist:8484` or `https://api.getgrist.com`). |
| `GRIST_API_KEY`            | API key.                                                |
| `GRIST_DATA_MODEL_DOC_ID`  | DocId of the document with the data model (Anchors, Links, …). |
| `GRIST_DATA_DOC_ID`        | DocId of the document with the data itself.              |
| `GRIST_TEST_SET_DOC_ID`    | DocId of the document with the golden dataset for evaluation. |

Alternative sources are wired up via [Custom ETL](../data-ingestion/custom-etl.md).

## Interfaces

### Telegram

| Variable             | Purpose                                  |
| -------------------- | ----------------------------------------- |
| `TELEGRAM_BOT_TOKEN` | Bot token from BotFather.                |

### HTTP API (`jims-api`) — authentication

The API supports two modes:

- Bearer token via the `--api-key` flag.
- [Authentik](https://goauthentik.io/) verification (`--authentik-url`, `--authentik-app-slug`).

See [HTTP API](../api/http-api.md).

### Backoffice

| Variable                    | Purpose                                                              | Default                          |
| --------------------------- | --------------------------------------------------------------------- | --------------------------------- |
| `VEDANA_BACKOFFICE_DEBUG`   | Enables dev-only features in the backoffice (test stands, manual queries). | `false`                      |
| `VEDANA_APP`                | `module[:attr]` of a factory returning `vedana_core.app.VedanaApp`. Lets you point the backoffice at a tenant/deployment-specific app factory without forking. | `vedana_core.app:make_vedana_app` |

### Datapipe / ETL bindings

| Variable             | Purpose                                                                                                            | Default          |
| -------------------- | ------------------------------------------------------------------------------------------------------------------- | ----------------- |
| `DATAPIPE_PIPELINE`  | Datapipe app entry point (`module[:attr]`, default attr `app`). Used by `datapipe run` / `datapipe api` and by the backoffice ETL bindings (`project_runtime.get_etl_bindings`). The ETL config module is derived as `<package>.config`. | `vedana_etl.app` |

## Observability

| Variable             | Purpose                                                    |
| -------------------- | ----------------------------------------------------------- |
| `SENTRY_DSN`         | Sentry endpoint. If empty, Sentry is off.                  |
| `SENTRY_ENVIRONMENT` | Environment name shown in Sentry.                           |

OpenTelemetry traces and Prometheus metrics are exposed through the standard mechanisms (`--metrics-port` of each CLI and `OTEL_*` env vars). See [Observability](../architecture/observability.md).

## Environment profiles

`apps/vedana/` ships two profiles:

- `.env.example` — a template for local development and the quick start.
- `.env.ci-cd` — a CI/CD profile (used in GitHub Actions).

Create your own profiles using the same template. Never commit secret keys: `.env` is already in `.gitignore`.
