---
title: Vedana Backoffice
section: Architecture
order: 5
---

# Vedana Backoffice

`vedana-backoffice` is an admin application built on [Reflex](https://reflex.dev/) with a built-in [Caddy](https://caddyserver.com/) reverse proxy. In docker-compose it's started by `vedana-backoffice-with-caddy` and listens on ports 9000 (Caddy) and 8000 (Reflex backend).

> **`vedana-backoffice` is a Vedana-specific overlay on top of `jims-backoffice`.** Generic Chat / Eval / Thread-list UI lives in the `jims-backoffice` library and is registered into the Reflex app via `register_app` / `register_chat_state` / `register_eval_state` (see `vedana_backoffice.py`). The Caddy launch, the `start_services.py` entry point, and the JIMS-thread state classes (`ThreadListState`, `ThreadViewState`) are imported / re-exported from `jims_backoffice`.

## What's inside

Layout (`libs/vedana-backoffice/src/vedana_backoffice/`):

```
vedana_backoffice/
├── components/        # reusable Reflex UI components
├── graph/             # graph visualisation and parts
├── pages/             # Reflex pages (main_dashboard, chat, ETL, eval, jims_thread_list_page)
├── states/            # Reflex State classes (chat, common, etl, eval, main_dashboard)
├── ui.py              # common UI shell (sidebar, header)
├── start_services.py  # 3-line re-export of jims_backoffice.start_services.main
├── project_runtime.py # resolves VedanaApp + Datapipe bindings from env (VEDANA_APP / DATAPIPE_PIPELINE)
├── vedana_backoffice.py # Reflex app assembly (registers pages into jims_backoffice)
└── util.py             # helpers
```

> The Caddyfile lives in `jims_backoffice/` now, alongside the launcher.

## Main sections

### Main Dashboard

`pages/main_dashboard.py` (mounted at `/`) is the landing page. It surfaces ingest health and graph state against a sliding time window: per-anchor / per-link counts in Memgraph vs. Datapipe, recent additions / changes / deletions from the `<table>_meta` timestamps, and links into the per-anchor / per-link drill-downs.

State: `states/main_dashboard.py` (`DashboardState`). The dashboard reads `dm_*` tables, `nodes` / `edges`, the Memgraph `:label` counts, and the `_meta` columns (`create_ts`, `update_ts`, `delete_ts`) produced by Datapipe; the time window is parameterised on the page.

> The dashboard is **not** a diff against a previous Grist snapshot — it's a time-window delta on datapipe metadata. A snapshot-diff view remains on the roadmap.

### JIMS Thread list

`pages/jims_thread_list_page.py` (mounted at `/jims`) is the inspector for JIMS threads — list, filter, drill into events, replay messages. The state classes (`ThreadListState`, `ThreadViewState`) are imported from `jims_backoffice` — this page is a thin wrapper that adds the Vedana sidebar / shell.

### Chat

`pages/chat.py` is the conversation UI. It uses the same `JimsApp` (via `make_jims_app`) as the production API.

Key features:

- create a new thread or continue an existing one (by `contact_id`);
- status messages from `StatusUpdater` ("Analysing query structure…", "Searching knowledge base…");
- a **Details** block under each answer: what the assistant did internally (data model filtering, Cypher queries, vector search calls, model statistics).

### ETL Runner

`pages/etl.py` is the UI on top of Datapipe.

Capabilities:

- list of steps with labels;
- a **Run Selected** button for the chosen flow / stage;
- execution logs and a status table.

It uses exactly the same `vedana_etl.pipeline.get_pipeline(...)` as in CLI mode — the backoffice doesn't add any separate logic.

### Evaluation

`pages/eval.py` is the harness for running evaluation on the golden dataset.

Steps:

1. Pick questions from `eval_gds` (the golden dataset).
2. Pick configs: `Judge configuration` (judge model and prompt), `Pipeline config` (main pipeline model, filtering flag, `top_n`).
3. **Refresh Data Model** — guarantees you're using the latest data model.
4. **Run Selected** — starts the evaluation.
5. Get the aggregate **Pass Rate** metric (internally `pass_rate = passed / total`, see `states/eval.py`) — together with `avg_rating`, `cost_total`, and per-test answer times — and the per-question breakdown.

See [Quality Metrics & Evaluation](../product/evaluation.md).

### Data Model viewer

A view of the current data model the way the LLM sees it:

- list of anchors / links / attributes with descriptions;
- indices (vector / text).

> A "diff against the previous Grist state" is on the roadmap and not yet implemented in `pages/`.

### Prompts editor

View the prompt templates that come from the `Prompts` table.

> Write-back of edited prompts to Grist is **not yet implemented**. To change a template, edit the row directly in Grist > Data Model > Prompts, then re-run `data_model_steps` (or "Refresh Data Model").

> Important: the backoffice is a tool for configuration and monitoring. The "knowledge" itself lives in Grist and the graph; the backoffice only displays it / runs the ETL for it.

## Running

### Via docker-compose

```yaml
services:
  app:
    command: vedana-backoffice-with-caddy
    ports:
      - "9000:9000"  # Caddy / UI
      - "8000:8000"  # Reflex backend
```

### Locally

```bash
uv run vedana-backoffice-with-caddy
```

Caddy proxies the Reflex backend behind a single address — to the user the UI looks like a single service on port 9000.

> **Caddy must be on `PATH`.** `vedana-backoffice-with-caddy` spawns Caddy as a subprocess (`libs/jims-backoffice/src/jims_backoffice/start_services.py:21` calls `subprocess.Popen(["caddy", "run", ...])`), so the `caddy` binary has to be installed locally (`brew install caddy` on macOS, `apt install caddy` on Debian). If you don't need Caddy, run `uv run reflex run --env dev --backend-only` from `apps/vedana/` and point your browser at port 8000.

### Debug mode

`VEDANA_BACKOFFICE_DEBUG=true` (or the more general `DEBUG=true`) turns on additional dev-only features: manual queries, test scenarios, expanded UI logs.

## Integration with the rest of the stack

The backoffice uses the same libraries as the API/widget/Telegram, but resolves them indirectly so multi-tenant deployments (Vedana / Stell / Maytoni-style overlays) can swap the app and ETL bindings without forking the backoffice:

- `vedana_backoffice.project_runtime.get_vedana_app()` returns the configured `VedanaApp` (which exposes `.jims_app`, `.pipeline`, `.data_model`, `.graph`, `.vts`). The factory is taken from the `VEDANA_APP` env var (default `vedana_core.app:make_vedana_app`).
- `vedana_backoffice.project_runtime.get_etl_bindings()` returns the Datapipe `app` / `pipeline` / `dbconn` resolved from the `DATAPIPE_PIPELINE` env var (default `vedana_etl.app`). Same env var used by the `datapipe` CLI.
- `vedana_core.data_model.DataModel` — the data model (reached via `VedanaApp.data_model`).
- `vedana_core.graph.MemgraphGraph` — the graph (reached via `VedanaApp.graph`).
- `vedana_etl.*` — ETL (resolved through `DATAPIPE_PIPELINE`).

This means any change in `vedana_core` is immediately visible in the backoffice, and vice versa: the backoffice is a convenient debugging UI; nothing unique is hardcoded there. To run the backoffice against a different deployment (e.g. Stell, a tenant fork), point `VEDANA_APP` / `DATAPIPE_PIPELINE` at the right factories — no code change needed.

## Security

The backoffice has no built-in authentication on its own — it's normally hidden behind an external reverse proxy (Cloudflare Access, Authentik, IAP). Don't expose it to the public internet without authentication: through ETL and prompt editing it's possible to break production.
