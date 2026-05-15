---
title: API Reference Overview
section: API Reference
order: 1
---

# API Reference: Overview

Vedana provides several integration paths:

| Layer              | What it is                                            | Where                       |
| ------------------ | ----------------------------------------------------- | --------------------------- |
| **HTTP API**       | FastAPI with the `/api/v1/chat` endpoint              | `libs/jims-api`             |
| **Web Widget**     | Embeddable JS widget + backend                        | `libs/jims-widget`          |
| **Telegram Bot**   | Bot built with aiogram                                | `libs/jims-telegram`        |
| **Terminal UI**    | Interactive TUI on Textual                             | `libs/jims-tui`             |
| **Python API**     | `make_jims_app()`, `ThreadController`, `RagPipeline`  | `libs/vedana-core` + `libs/jims-core` |

All interfaces use **the same kernel** (`vedana_core.app:app`) — meaning the assistant behaves identically in Telegram, the widget, and the HTTP API; only the transport differs.

## Entry point: `JimsApp`

All CLI wrappers work the same way: they take a `--app` option in `module:attr` format. For Vedana that's `vedana_core.app:app`. The helper `jims_core.util.load_jims_app(app_str)` imports the object, awaits the coroutine if needed, and returns a `JimsApp`.

```python
# vedana_core/app.py
app = make_jims_app()  # this is a coroutine
```

## Common CLI configuration

Every CLI service has a shared set of flags:

| Flag                     | Default      | Description                                                                | Available on                                                                 |
| ------------------------ | ------------ | --------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `--app`                   | `app`        | which `JimsApp` to import (`vedana_core.app:app` for Vedana)              | all CLIs                                                                      |
| `--enable-sentry`         | off          | enable Sentry integration (reads `SENTRY_DSN`, `SENTRY_ENVIRONMENT`)      | all CLIs                                                                      |
| `--enable-healthcheck`    | on           | bring up a separate `aiohttp` `/healthz` server                            | **`jims-telegram` only.** `jims-api` and `jims-widget` expose `/healthz` on their main HTTP port. |
| `--healthcheck-port`      | 9000         | healthcheck port for the separate server                                    | **`jims-telegram` only.**                                                     |
| `--metrics-port`          | 8000 (8001 for `jims-widget`) | Prometheus port. Default differs per service to avoid collisions on one host. | all CLIs                                                                |
| `--verbose`               | off          | turn on debug logs                                                           | all CLIs                                                                      |

All CLIs also read environment variables with the `JIMS_` prefix (`auto_envvar_prefix="JIMS"`).

> Note: `--healthcheck-port` defaults to `9000`, which is the same port the backoffice (Caddy) binds to in `apps/vedana/docker-compose.yml`. If you run `jims-telegram` on the same host as the backoffice, override it (`--healthcheck-port 9100` or similar).

## What's next

- [HTTP API](./api/http-api.md) — for external integrations.
- [Widget API](./api/widget-api.md) — embedding in a web page.
- [Telegram](./api/telegram.md) — Telegram bot.
- [Python API](./api/python-api.md) — programmatic access.
- [Configuration Reference](./api/configuration-reference.md) — every ENV variable.
