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
- `JIMS_API_KEY` - optional bearer token for auth

## Endpoints

- `GET /health`
- `POST /api/v1/chat`

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

If `JIMS_API_KEY` is set, send it as `Authorization: Bearer <token>`
