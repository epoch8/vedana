---
title: ConversationLifecycle
section: Data Model
order: 7
---

# ConversationLifecycle

The **ConversationLifecycle** table stores responses to conversation lifecycle events — for example, what the assistant should reply on `/start`, on the user's first message, or on a session reset.

It's the simplest mechanism in Vedana: a clean lookup by event name.

## Fields

| Field   | Description                                                            |
| ------- | ---------------------------------------------------------------------- |
| **event** | Event name (PK). For example `/start`, `welcome`, `session_reset`.  |
| **text**  | Message or instruction the assistant returns.                       |

## How it works

`StartPipeline` (`vedana_core.rag_pipeline.StartPipeline`) is a separate JIMS pipeline wired up as `conversation_start_pipeline` in `JimsApp`. On the first message in a new thread or on `/start` it:

1. Reads `ConversationLifecycle` via `DataModel.conversation_lifecycle_events()`.
2. Picks the value for the `/start` key.
3. If found — sends it as a `comm.assistant_message`.
4. If not found — sends the default message "Bot online. No response for /start command in LifecycleEvents".

There are no LLM calls, no tools, no Cypher — it's a pure table lookup.

## When to use

- **`/start`** — a welcome message; tell the user what the assistant can do and how to start.
- **`welcome`** — for channels without an explicit `/start` command (a Telegram button, a widget on a page).
- **`session_reset`** — what to reply when the user starts a new thread with the same `contact_id`.
- **`fallback_unknown_command`** — for unknown commands (if you want to distinguish them from regular questions).
- **`legal_disclaimer`** — for legal assistants: show a disclaimer on first contact.

## Example

| event             | text                                                                                                                                                                                                                                                                                                                                  |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/start`          | Hi! I'm an assistant for the Acme product catalog. I can help you:<br/>• find a product by description or SKU;<br/>• check stock at our stores;<br/>• answer questions about the return and delivery policy.<br/><br/>Just ask in your own words — for example, "Do you have any black sleeper sofas under £600 in stock?". |
| `welcome`         | Hello! How can I help?                                                                                                                                                                                                                                                                                                                |

## How to update

1. Open **Grist > Data Model > ConversationLifecycle**.
2. Add or update a row.
3. In the backoffice → ETL → `data_model_steps`.
4. Verify in chat (`/start` from a fresh user).

Changes take effect right after the data model is refreshed.

## Best practices

- **A short greeting** beats a wall of text — the user shouldn't have to scroll.
- **Provide 2–3 example questions** — users often don't know how to phrase things.
- **Don't use lifecycle events as a playbook substitute.** If a user asks "what can you do?", that's a Queries scenario (with examples and instructions for the LLM), not a single fixed message.
- **No marketing copy** — the user just opened the bot; brevity beats "selling".

## Extension

If you need more complex scenarios (e.g. "show a survey on first message", "after 5 minutes of inactivity, send a follow-up"), that's not lifecycle anymore but a full JIMS pipeline. Write your own `Pipeline` class and replace `JimsApp.conversation_start_pipeline` — see [JIMS Core](../architecture/jims-core.md).
