---
title: Local Development
section: Getting Started
order: 3
---

# Local Development

This guide is for developers who want to **work on Vedana itself** — debug a single service, add a feature to `vedana-core`, run a custom pipeline locally — and need their own Python environment instead of the all-in-one Docker stack.

> **If you just want to try Vedana**, follow the [Quick Start](./quick-start.md) — it brings the whole stack up via Docker Compose with the LIMIT demo dataset, and you don't need any of the steps below.
>
> **If you're deploying Vedana to production**, see [Operations → Deployment](../operations/deployment.md). It covers single-node and clustered topologies with managed Postgres, Memgraph, and Grist.

This page covers the middle ground: native Python on your machine, infrastructure (Postgres, Memgraph, Grist) in Docker.

## System requirements

| Component       | Version / requirement                                                          |
| --------------- | ------------------------------------------------------------------------------- |
| Python          | 3.12 (see `.python-version`)                                                    |
| Docker          | 24+ with Compose v2 (for the infra services)                                    |
| `uv`            | recent — [astral.sh/uv](https://docs.astral.sh/uv/)                             |
| LLM provider key | OpenAI, OpenRouter, Google/VertexAI, or any provider compatible with LiteLLM   |

The LLM-provider key is the only thing you need to obtain externally. Postgres, Memgraph, and Grist are brought up locally via Docker Compose in step 2.

## 1. Sync the workspace

The repository is a [`uv` workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) — every `libs/jims-*` and `libs/vedana-*` package is editable in place.

```bash
git clone https://github.com/epoch8/vedana
cd vedana

uv sync
```

`uv` creates `.venv/` at the repo root and installs all workspace packages. After this, edits to any `libs/*/src/...` are picked up immediately by every script — no reinstall needed.

## 2. Bring up the infrastructure only

You don't want the full `app` / `api` / `widget` services running in Docker — those are what you'll run natively. Bring up only the data services:

```bash
docker compose -f apps/vedana/docker-compose.yml up -d db memgraph memgraph-lab grist
```

This starts:

- `db` — Postgres 15 with `pgvector` on port `5432`
- `memgraph` — Memgraph on port `7687` (Bolt) and `7444` (HTTP monitoring)
- `memgraph-lab` — Web inspector on port `3000`
- `grist` — Grist on port `8484`

Wait ~30 seconds for healthchecks to settle, then check Postgres is up:

```bash
docker compose -f apps/vedana/docker-compose.yml exec db pg_isready -U postgres
```

## 3. Configure `.env`

```bash
cp apps/vedana/.env.example apps/vedana/.env
```

Open `apps/vedana/.env` and set at minimum:

```env
# LLM provider — pick one
OPENAI_API_KEY="sk-..."
# OPENROUTER_API_KEY="sk-or-..."
# GOOGLE_APPLICATION_CREDENTIALS="path-to-creds.json"

# Postgres (already correct in .env.example for local Docker)
JIMS_DB_CONN_URI="postgresql://postgres:postgres@localhost:5432"
DB_CONN_URI="postgresql://postgres:postgres@localhost:5432"

# Memgraph
MEMGRAPH_URI="bolt://localhost:7687"
MEMGRAPH_USER="neo4j"
MEMGRAPH_PWD="modular-current-bonjour-senior-neptune-8618"
```

> **Note on hostnames.** `.env.example` ships with Docker-network hostnames (`db`, `memgraph`, `grist`) because it's optimised for the Docker Compose path. Running natively, replace them with `localhost` — both `JIMS_DB_CONN_URI` / `DB_CONN_URI` and `MEMGRAPH_URI` need that change.
>
> **Grist credentials.** `GRIST_API_KEY`, `GRIST_DATA_MODEL_DOC_ID`, `GRIST_DATA_DOC_ID`, and `GRIST_TEST_SET_DOC_ID` are hardcoded in `apps/vedana/docker-compose.yml` (under `x-app-common.environment`) and point to a public Grist demo. When you run natively those env vars aren't injected — copy them from `docker-compose.yml` into your `.env`, or replace them with your own Grist values.

The full list of environment variables is in the [Configuration Reference](../api/configuration-reference.md); the [Configuration guide](./configuration.md) explains them grouped by purpose.

## 4. Apply migrations

```bash
cd apps/vedana
uv run alembic upgrade head
```

The migration `2dfad73e5cce_move_emb_to_pgvector` requires the `pgvector` extension. The local Docker `db` service is built on `pgvector/pgvector:pg15` and supports it out of the box, so the default `CREATE_PGVECTOR_EXTENSION=true` in `.env.example` will create the extension automatically.

If you've pointed `JIMS_DB_CONN_URI` at a managed Postgres that handles extensions for you (Supabase, Neon, RDS with the extension pre-enabled), set `CREATE_PGVECTOR_EXTENSION=false` to skip the `CREATE EXTENSION` statement.

## 5. Run the service you're working on

The repository defines several CLI scripts. Run the one you need; others can stay off, or you can run several in separate terminals.

| Command                                  | What it runs                                              |
| ---------------------------------------- | ---------------------------------------------------------- |
| `uv run vedana-backoffice-with-caddy`    | Reflex backoffice + Caddy reverse proxy (chat, ETL runner, eval — port 9000) |
| `uv run python -m jims_api.main --app vedana_core.app:app --host 0.0.0.0 --port 8080` | HTTP API on FastAPI (port 8080) |
| `uv run python -m jims_widget.main --app vedana_core.app:app --host 0.0.0.0 --port 8090` | Web widget (port 8090) |
| `uv run jims-telegram --app vedana_core.app:app`           | Telegram bot |
| `uv run jims-tui --app vedana_core.app:app`                | Terminal UI for interactive debugging |

All scripts accept `--app module:attr` (default `app`); for Vedana use `vedana_core.app:app`. Each also accepts `--metrics-port`, `--enable-sentry`, and `--verbose`. See the [API overview](../api/overview.md) for the full CLI flag list per service.

> **Note on `vedana-backoffice-with-caddy`.** This script spawns Caddy as a subprocess to reverse-proxy WebSocket events on port 9000 to the Reflex backend on port 8000, so you need the `caddy` binary on `PATH` (`brew install caddy` on macOS, `apt install caddy` on Debian). If you only want the Reflex part, run `uv run reflex run --env dev --backend-only` from `apps/vedana/` and point your browser at port 8000.

## 6. Run the ETL

Without ETL the graph is empty. Open the backoffice ([http://localhost:9000](http://localhost:9000)), go to the **ETL** page, click **Run Selected** on the main tab. Wait for every step to turn green.

Alternatively, from the command line:

```bash
cd apps/vedana
uv run python -m datapipe run --pipeline vedana_etl.app
```

## 7. Verify

```bash
curl http://localhost:8080/healthz
# {"status":"ok"}
```

Then ask a question via the chat at [http://localhost:9000/chat](http://localhost:9000/chat). The first request will hit your local pipeline and your LLM provider — watch the terminal of the backoffice for traces.

If something fails, see [Troubleshooting](../operations/troubleshooting.md).

## What's next

- [Architecture Overview](../architecture/overview.md) — how the code is structured.
- [Repository Structure](../contributing/repository-structure.md) — where every file lives.
- [Testing](../contributing/testing.md) — how to run tests, VCR cassettes, the `tests/` layout per package.
- [Custom Tools](../guides/custom-tools.md) and [Custom ETL](../data-ingestion/custom-etl.md) — extending Vedana.
