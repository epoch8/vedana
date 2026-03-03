# Name

JIMS stands for "Just an Integrated Multiagent System"

# Concepts

## Thread

Thread is a conversation between user (or users) and agentic system
Each interaction between agents and users happens in thread

## Event

Something that happens in a thread
Each event has id, type and data

Example:

{
    "event_id": "...",
    "event_type": "comm.user_message",
    "event_data": {
        "content": "Hello!"
    }
}

### event_type and event_domain

`event_type` is structured as follows:

- **`<event_domain>.<event_name>[.<subname>...]`**

`event_domain` is the first segment and is used for routing and filtering. The
rest of the string identifies the concrete event.

`event_domain` values include:

#### `comm.*` - user–assistant communication

Visible to the end user.

  - **`comm.user_message`**: a user text message (input);
  - **`comm.assistant_message`**: a text message produced by an agent/pipeline.

These events are:

- Exposed as `CommunicationEvent` (`role`, `content`).
- Collected into `ThreadContext.history`.


#### Pipeline‑specific domains 

Defined in a Pipeline, these events are produced via `ThreadContext.send_event(...)` and consumed elsewhere.

example: `rag.*`
- **`rag.query_processed`**
- **`rag.error`**

#### `context.*` - `comm.*` extended with pipeline context

Conversational AI Pipelines can benefit from preserving some extra data (context) about the conversation in between messages (previous reasoning, extra data, etc.). For this, we use a superset of `comm.` domain called `context.`. It includes everything in `comm.` (everything in `ThreadContext.history`), and adds `context.` events passed from Pipeline. `context.` events are internal and not sent/shown to the end user.


#### `state.*` - internal state of the pipeline

Used to persist some data between pipeline calls. 

- `ThreadContext.set_state(state_name, state)` emits `state.set.<state_name>` with a dict payload.
- `ThreadContext.get_state(state_name, state_type)` scans events in reverse for `state.set.<state_name>`
  and validates the payload into a Pydantic model.

Define your own state schema inside the Pipeline


#### `jims.*` - lifecycle and system domain: 

Lifecycle and system‑level events

Example:

- **`jims.lifecycle.thread_created`**: emitted when a thread is first created
  by `ThreadController.new_thread(...)`.