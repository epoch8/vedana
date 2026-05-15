---
title: Widget API
section: API Reference
order: 3
---

# Widget API (`jims-widget`)

`jims-widget` is the backend + static assets for the embeddable web chat widget. In docker-compose it's the `widget` service on port 8090.

## Architecture

`libs/jims-widget/src/jims_widget/`:

- `main.py` — CLI and uvicorn wrapper.
- `server.py` — the FastAPI application with REST endpoints + a WebSocket for real time.
- `static/` — the built widget frontend (JS/CSS) included on the page via `<script>`.

## Running

```bash
uv run python -m jims_widget.main \
  --app vedana_core.app:app \
  --host 0.0.0.0 \
  --port 8090 \
  --cors-origins "*" \
  --metrics-port 8001
```

### CLI options

| Flag             | Default | Description                                                                 |
| ---------------- | ------- | ---------------------------------------------------------------------------- |
| `--app`          | `app`   | JIMS app in `module:attr` form (use `vedana_core.app:app` for Vedana).      |
| `--host`         | `0.0.0.0` | HTTP bind host.                                                            |
| `--port`         | `8090`  | HTTP port for both the WebSocket and the static assets.                     |
| `--cors-origins` | `*`     | Comma-separated CORS origins. Use `*` for development; restrict in production. |
| `--enable-sentry`| off     | Enable Sentry tracing.                                                       |
| `--metrics-port` | `8001`  | Prometheus port. **Different default from `jims-api` (8000)** so both can run on the same host. |
| `--verbose`      | off     | Debug logs.                                                                  |

## Embedding in a page

Once the service is running, drop on your page:

```html
<script src="https://your-vedana-host/static/jims-widget.js"
        data-server="https://your-vedana-host"
        async></script>
```

The widget creates a floating chat button in the bottom-right corner. Clicking it opens the conversation window.

`data-server` is **required** — it tells the script which origin to open the WebSocket against. All other attributes are optional.

| Attribute            | Default        | Description                                                                  |
| -------------------- | -------------- | ----------------------------------------------------------------------------- |
| `data-server`        | —              | **Required.** Origin of the widget backend.                                  |
| `data-contact-id`    | empty          | Persistent visitor identifier. If empty, the backend generates `widget:<uuid7>`. |
| `data-thread-id`     | empty          | Resume an existing thread.                                                    |
| `data-intro-message` | empty          | Initial AI greeting shown when the panel opens.                              |
| `data-position`      | `bottom-right` | `bottom-right` or `bottom-left`.                                              |
| `data-open`          | `false`        | `"true"` to start with the panel expanded.                                    |
| `data-title`         | `AI Assistant` | Header title.                                                                 |
| `data-accent`        | `#4f46e5`      | Accent hex colour.                                                            |

For the source of truth, see `libs/jims-widget/src/jims_widget/static/jims-widget.js`.

## Endpoints

The widget backend exposes:

- `GET /healthz` — healthcheck (`{"status":"ok"}`).
- `GET /` — a built-in demo page (`static/demo.html`).
- `GET /static/*` — static asset mount, serving the embed script and the demo HTML.
- `WS /ws/chat?thread_id=<uuid>&contact_id=<id>` — the only chat transport. Both query parameters are optional; if `thread_id` is missing or unknown, the backend creates a new thread.

There are **no REST endpoints for `POST /threads`, `POST /threads/{id}/messages`, `GET /threads/{id}/events`**. All chat flows go through the WebSocket.

## WebSocket protocol

Connection:

```
ws://your-vedana-host:8090/ws/chat?thread_id={uuid}&contact_id={id}
```

**Client → server** — a DeepChat frame. Any of these is accepted (`_extract_user_text` normalises them):

```json
{"messages": [{"role": "user", "text": "Hello"}]}
```

```json
"Hello"
```

```text
Hello
```

**Server → client** — a single flat JSON payload per pipeline run:

```json
{"text": "<assistant answer>"}
```

```json
{"error": "Empty message"}
```

```json
{"error": "Processing error: <message>"}
```

There is **no `{"type":"status|event","data":{...}}` envelope** at the moment, and no intermediate status events over the WebSocket — only the final assistant text (or an error). Token-by-token streaming is on the roadmap.

## Security

By default the widget has no authentication — if you need to restrict access, use:

- your own reverse proxy that issues a token only to authenticated users;
- pass `contact_id` and a signed token via `data-token` and have the backend validate it (requires modifying `server.py`).

In production, do not expose the widget directly on the public internet without authentication — that opens an unrestricted channel to your LLM provider (= money).

## Localisation

Widget texts (greeting, placeholder, send button) are configured in Grist via `ConversationLifecycle` (see [ConversationLifecycle](../data-model/conversation-lifecycle.md)) or directly in the `data-*` attributes when embedding.

## What's next

- [HTTP API](./api/http-api.md) — the underlying API (the widget is a thin wrapper over the same kernel).
- [ConversationLifecycle](../data-model/conversation-lifecycle.md) — configure greetings.
