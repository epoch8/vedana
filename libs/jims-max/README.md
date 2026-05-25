MAX interface for JIMS
======================

`jims-max` provides a MAX messenger interface for JIMS applications

It mirrors the architecture of `jims-telegram`:

- Uses the `maxapi` Python SDK for interacting with the MAX platform.
- Bridges MAX updates into JIMS threads and events.
- Emits JIMS `comm.*` events from incoming user messages and button clicks.
- Renders `comm.assistant_message` and `comm.assistant_buttons` events back into MAX messages.

