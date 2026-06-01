---
title: Telegram
section: API Reference
order: 4
---

# Telegram (`jims-telegram`)

`jims-telegram` is a Telegram bot built on [aiogram](https://docs.aiogram.dev/) wrapping the JIMS pipeline.

## Running

```bash
TELEGRAM_BOT_TOKEN="123456:ABC..." \
uv run jims-telegram \
  --app vedana_core.app:app \
  --enable-sentry \
  --metrics-port 8000
```

In docker-compose the `tg` service is commented out (lines `# tg: ...`), because in production it's typically deployed separately with its own secret and scaling. Enable it by analogy with the other services:

```yaml
tg:
  <<: *app-common
  command: uv run python -m jims_telegram.main
  depends_on:
    db-migrate:
      condition: service_completed_successfully
    memgraph:
      condition: service_started
    db:
      condition: service_healthy
      
```

### CLI

| Flag                  | Default | Description                                |
| --------------------- | -------- | ----------------------------------------- |
| `--app`                | `app`    | JIMS app                                  |
| `--enable-sentry`      | off      | Sentry                                    |
| `--enable-healthcheck` | on       | bring up `/health` on `--healthcheck-port` |
| `--healthcheck-port`   | 9000     | healthcheck port                          |
| `--metrics-port`       | 8000     | Prometheus                                 |
| `--verbose`            | off      | debug logs                                 |

## Getting a token

1. Find `@BotFather` on Telegram.
2. `/newbot` → set a name → receive a token.
3. `/setdescription`, `/setabouttext`, `/setuserpic` — branding.
4. `/setcommands` — set commands (`start - restart`, `help - help`).

Save the token in `.env`:

```env
TELEGRAM_BOT_TOKEN="123456789:AA..."
```

## Default behaviour

When the bot is started (`TelegramController.create(jims_app).run()`):

- Each Telegram chat becomes a separate JIMS thread.
- `contact_id` is built from the Telegram user id.
- `thread_config` contains `{"interface": "telegram", ...}` — you can use that in your pipeline for channel-specific behaviour.
- The `/start` command **always creates a new JIMS thread** (it does not reuse the existing one) and then runs the `conversation_start_pipeline` (`StartPipeline`), which reads the welcome message from `ConversationLifecycle["/start"]`. The literal text of the command (`"/start"` plus any payload after a space) is also stored on the new thread as the first user message via `store_user_message`, so it appears in the thread's event log. Previous chat history for that contact stays in the database but is no longer used as conversation context. If you want `/start` to resume an existing thread, you'll need to subclass `TelegramController.command_start`.
- Every text message runs the main pipeline (`RagPipeline`).

Intermediate status updates (`update_agent_status`) are surfaced as "typing..." via the Telegram Bot API (`sendChatAction`).

## Markdown to Telegram

`md2tgmd.py` is a utility that converts Markdown answers from the LLM into Telegram's MarkdownV2 format. Without it, Telegram chops off asterisks/underscores. The utility escapes special characters and keeps `**bold**`, `_italic_`, inline code, and quotes working.

## Healthcheck

Alongside aiogram the bot brings up an aiohttp server with `/health` and `/healthz` on `--healthcheck-port` (default 9000). Useful for:

- Kubernetes liveness/readiness;
- load balancers;
- "is the bot alive?" monitoring.

## Security

- Never commit `TELEGRAM_BOT_TOKEN` to the repository.
- For corporate bots, use a `chat_id` allow-list (you can add it in your `Pipeline` and ignore disallowed users).
- Before going to production, enable rate limiting on the Vedana side (via aiogram middleware or an external reverse proxy).

## What's next

- [Architecture: JIMS Core](../architecture/jims-core.md) — thread internals.
- [HTTP API](./api/http-api.md) — for non-Telegram integrations.
