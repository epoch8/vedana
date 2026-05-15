---
title: HTTP API
section: API Reference
order: 2
---

# HTTP API (`jims-api`)

`jims-api` is a FastAPI service that accepts requests over HTTP and forwards them to the JIMS pipeline. It's the primary integration point between Vedana and external systems.

## Running

```bash
uv run python -m jims_api.main \
  --app vedana_core.app:app \
  --host 0.0.0.0 \
  --port 8080 \
  --api-key "your-secret-bearer-token" \
  --metrics-port 8000
```

In docker-compose the `api` service runs the same command and exposes port 8080.

### CLI options

| Flag                       | Value                                                              |
| -------------------------- | ------------------------------------------------------------------ |
| `--app`                    | module:attr of the JIMS app                                        |
| `--host`                   | (default `0.0.0.0`)                                                |
| `--port`                   | (default `8080`)                                                   |
| `--api-key`                | Bearer token; if set, every request is checked                    |
| `--authentik-url`          | Authentik URL for permission checks via API                       |
| `--authentik-app-slug`     | Authentik application slug                                         |
| `--enable-sentry`          | enable Sentry                                                       |
| `--metrics-port`           | Prometheus port (default 8000)                                     |
| `--verbose`                | debug logs                                                          |

## Authentication

Two modes are supported:

1. **Bearer token.** If `--api-key` is set, every request must include `Authorization: Bearer <token>`.
2. **Authentik (optional).** If **both** `--authentik-url` **and** `--authentik-app-slug` are set, the user's token is checked via `GET /api/v3/core/applications/<slug>/check_access/`. Setting only one of the two will not enable Authentik verification (the check in `jims_api/main.py:60-75` requires both); the request will simply fail with `401`. Always pass them together.
3. **No auth.** If both are empty, the API is public (dev / closed network only).

You can use both at the same time: api-key is tried first, then Authentik.

## Endpoints

### `GET /healthz`

Healthcheck. Returns `{"status": "ok"}`.

### `POST /api/v1/chat`

The main endpoint. Accepts a user message, runs it through the pipeline, returns answers and events.

**Request body** (`ChatRequest`):

```json
{
  "contact_id": "user-123",
  "message": "What are Geneva Durben's interests?",
  "thread_id": "0192b5f2-2c03-7b91-9c6c-9a1234567890",
  "thread_config": {"interface": "api"},
  "event_type": "comm.user_message",
  "run_conversation_start_on_new_thread": false
}
```

| Field                                       | Type        | Required? | Description                                                                  |
| ------------------------------------------- | ----------- | --------- | ----------------------------------------------------------------------------- |
| `contact_id`                                 | string      | yes       | User identifier (from your system).                                          |
| `message`                                    | string      | yes       | Message text.                                                                  |
| `thread_id`                                  | UUID        | no        | Thread ID. If null, the latest by `contact_id` is found, or a new thread is created. |
| `thread_config`                              | dict        | no        | Arbitrary thread configuration (passed to `ThreadDB.thread_config`).        |
| `event_type`                                 | string      | no        | Type of the inbound event. Default `comm.user_message`.                       |
| `run_conversation_start_on_new_thread`       | bool        | no        | Whether to run `conversation_start_pipeline` when a new thread is created.    |

**Response** (`ChatResponse`):

```json
{
  "thread_id": "0192b5f2-2c03-7b91-9c6c-9a1234567890",
  "created_new_thread": false,
  "assistant_messages": [
    "Geneva Durben is interested in Quokkas, Slide Rules, Mosaic, Eating Disorders, Tantric, Marrakesh."
  ],
  "events": [
    {
      "event_type": "comm.assistant_message",
      "event_data": {"role": "assistant", "content": "Geneva Durben is interested in ..."}
    },
    {
      "event_type": "rag.query_processed",
      "event_data": {
        "query": "What are Geneva Durben's interests?",
        "answer": "...",
        "technical_info": {
          "vts_queries": [],
          "cypher_queries": ["MATCH (p:person)-[:PERSON_has_INTEREST]->(i:interest) WHERE p.person_name = 'Geneva Durben' RETURN i.interest_name"],
          "num_vts_queries": 0,
          "num_cypher_queries": 1,
          "model_used": "gpt-4.1-mini",
          "model_stats": {"gpt-4.1-mini": {"requests_count": 3, "prompt_tokens": 5421, ...}}
        },
        "threshold": 0.8
      }
    }
  ]
}
```

| Field                  | Description                                                                       |
| ---------------------- | ---------------------------------------------------------------------------------- |
| `thread_id`            | Thread ID (new or existing).                                                       |
| `created_new_thread`   | Whether a thread was created in this request.                                      |
| `assistant_messages`   | List of assistant messages from this run (usually one).                            |
| `events`               | All outgoing events: `comm.assistant_message`, `rag.query_processed`, `rag.data_model_filtered`, etc. |

**Errors**:

- `401 Unauthorized` — auth is on and the token is invalid.
- `404 Not Found` — `thread_id` was provided but no such thread exists.
- `500 Internal Server Error` — pipeline error (with details in `detail`).

## Execution flow

1. If `thread_id` is provided, look it up via `ThreadController.from_thread_id(...)`. If not found, return `404`.
2. Otherwise, find the latest thread by `contact_id` (`ThreadController.latest_thread_from_contact_id`).
3. If neither exists, create a new one (`new_thread`) and set `created_new_thread=true`. If `run_conversation_start_on_new_thread=true` and `conversation_start_pipeline` is set, run it.
4. Record the inbound message as an event of type `event_type` with `{"role":"user","content":message}`.
5. Run the main pipeline (`run_pipeline_with_context`).
6. Collect `assistant_messages` from the outgoing `comm.assistant_message` events.
7. Return a `ChatResponse`.

## Examples

### curl

```bash
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "contact_id": "user-123",
    "message": "What are Geneva Durben'\''s interests?"
  }'
```

### Python (httpx)

```python
import httpx

async def ask(message: str, contact_id: str, thread_id=None):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://localhost:8080/api/v1/chat",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"contact_id": contact_id, "message": message, "thread_id": thread_id},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
```

### Node.js (fetch)

```js
const resp = await fetch("http://localhost:8080/api/v1/chat", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "Authorization": `Bearer ${process.env.VEDANA_API_KEY}`,
  },
  body: JSON.stringify({
    contact_id: userId,
    message: text,
    thread_id: threadId,
  }),
});
const data = await resp.json();
```

## Streaming

The `/api/v1/chat` endpoint **does not stream** at the time of writing — the client gets the final set of messages once the pipeline completes. For real-time progress, use:

- the WebSocket interface of the widget (`jims-widget`);
- Telegram (which has intermediate "typing…" status messages);
- the Reflex backoffice (`StatusUpdater`).

## Performance

A typical request (one cypher tool call) lands in 2–6 seconds with `gpt-4.1-mini` on a simple domain. Heavy queries (complex playbook, many vector searches) can take 10–20 seconds.

Watch metrics in `jims_pipeline_run_duration_seconds{pipeline="RagPipeline"}`.

## What's next

- [Configuration Reference](./api/configuration-reference.md) — every ENV variable.
- [Widget API](./api/widget-api.md) — embeddable widget.
- [Python API](./api/python-api.md) — programmatic access.
