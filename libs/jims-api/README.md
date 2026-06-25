# jims-api

`jims-api` exposes any JIMS application as a FastAPI HTTP service.


## Run

```bash
jims-api --app my_project.app:app --port 8080
```

Environment variables:

- `JIMS_APP` - JIMS app import path (`module:attr`)
- `JIMS_PORT` - HTTP port
- `JIMS_HOST` - HTTP host
- `JIMS_API_KEY` - optional static bearer token for auth
- `JIMS_AUTHENTIK_URL` - Authentik instance base URL (e.g. `https://authentik.epoch8.dev`)
- `JIMS_AUTHENTIK_APP_SLUG` - Authentik application slug to check access against

## Authentication

Auth is checked in the following order:

1. If `JIMS_API_KEY` is set and the provided bearer token matches — allow
2. If `JIMS_AUTHENTIK_URL` and `JIMS_AUTHENTIK_APP_SLUG` are set — validate the bearer token against Authentik's `/api/v3/core/applications/{slug}/check_access/` endpoint. If 200 — allow
3. If no auth is configured (neither static key nor Authentik) — open access
4. Otherwise — 401 Unauthorized

Send the token as `Authorization: Bearer <token>`.

## Endpoints

- `GET /healthz`
- `POST /api/v1/chat`
- `POST /api/v1/chat/stream`

### `POST /api/v1/chat`

Request:

```json
{
  "contact_id": "customer:42",
  "message": "Hello",
  "thread_id": null,
  "thread_config": {"interface": "api"},
  "event_type": "comm.user_message",
  "run_conversation_start_on_new_thread": false
}
```

Response:

```json
{
  "thread_id": "0194f0f3-d88d-7cca-8f37-ff44f911f539",
  "created_new_thread": true,
  "assistant_messages": ["Hi! How can I help?"],
  "events": [
    {
      "event_type": "comm.assistant_message",
      "event_data": {"role": "assistant", "content": "Hi! How can I help?"}
    }
  ]
}
```

Send the API token as `Authorization: Bearer <token>` (see [Authentication](#authentication) above).

### `POST /api/v1/chat/stream`

Same request body as `/api/v1/chat`, but the response is a `text/event-stream` (SSE) instead of a single JSON object.
Use this to show the user live progress (e.g. "Searching knowledge base...") while the pipeline is running.

Request: same as `/api/v1/chat`.

The stream emits the following event types:

- `status` - an agent status update, emitted while the pipeline is running:
  ```
  event: status
  data: {"status": "Searching knowledge base...", "type": "searching_knowledge_base"}
  ```
  `type` is a stable identifier; `status` is a string that may change between releases.
- `result` - the final response, with the same shape as the `/api/v1/chat` response body:
  ```
  event: result
  data: {"thread_id": "...", "created_new_thread": true, "assistant_messages": ["Hi!"], "events": [...]}
  ```
- `error` - sent instead of `result` if the pipeline raised an exception:
  ```
  event: error
  data: {"detail": "Pipeline error: ..."}
  ```

A stream always ends with either a `result` or an `error` event.

Note: browsers' `EventSource` API cannot send custom headers, so it can't be used directly with
`Authorization: Bearer <token>` auth. Consume this endpoint with `fetch` and read the response body as a stream instead.
