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
| `MODEL`                         | str    | `gpt-4.1`                | Main LLM model used by `RagPipeline` / `RagAgent` for answer generation and Cypher. **This is the value that wins in the standard pipeline** — Vedana Core overrides the inner LLM provider on every call. |
| `ENABLE_DM_FILTERING`           | bool   | `true`                   | Enable the data model filtering step.                             |
| `FILTER_MODEL`                  | str    | `gpt-4.1-mini`           | Model used for filtering.                                          |
| `JUDGE_MODEL`                   | str    | `gpt-4.1-mini`           | Model used for evaluation (LLM-as-judge).                          |
| `EMBEDDINGS_MODEL`              | str    | `text-embedding-3-large` | Embeddings model.                                                  |
| `EMBEDDINGS_DIM`                | int    | `1024`                   | Embedding dimensions.                                               |
| `PIPELINE_HISTORY_LENGTH`       | int    | `20`                     | How many recent `comm.*` messages go into the context.            |
| `MEMGRAPH_URI`                  | str    | —                        | Memgraph Bolt URI.                                                  |
| `MEMGRAPH_USER`                 | str    | —                        | Memgraph user.                                                     |
| `MEMGRAPH_PWD`                  | str    | —                        | Memgraph password.                                                  |

## LLM Provider (`LLMSettings`)

File: `libs/jims-core/src/jims_core/llms/llm_provider.py`.

`LLMSettings` is the inner provider used by `jims-core`. It reads the same `MODEL` env var as `VedanaCoreSettings`, but if you instantiate `jims-core` standalone (without `vedana-core`), no override happens and the fallback default (`gpt-4.1-nano`) applies.

> **Which `MODEL` actually runs?** In the standard Vedana stack the two classes read the **same env var**, so whatever you put in `.env` (e.g. `MODEL=gpt-4.1-mini`) is used by both. The defaults differ only as a safety net for standalone `jims-core` users who don't set the env var: vedana-core falls back to `gpt-4.1` (heavier, better tool-calling), jims-core falls back to `gpt-4.1-nano` (cheap, OK for non-RAG flows). **For all production deployments — set `MODEL` explicitly in `.env`.**

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

- `vedana_core.settings` uses `env_prefix=""` (no prefix) and reads `apps/vedana/.env` (working directory must be `apps/vedana`).
- `vedana_etl.settings` — same.
- `jims_core.llms.llm_provider.LLMSettings` — same.
- JIMS CLI wrappers use `auto_envvar_prefix="JIMS"` (`click`), so the `--api-key` flag can be provided as `JIMS_API_KEY=...`.

## .env.example vs .env.ci-cd

`apps/vedana/` ships with two ready profiles:

- **`.env.example`** — for local development; references `db`, `memgraph`, `grist` (the docker-compose service names).
- **`.env.ci-cd`** — for CI; uses `localhost` addresses and test keys.

For production, create your own profile and **never** commit secret keys (`.env` is in `.gitignore`).
