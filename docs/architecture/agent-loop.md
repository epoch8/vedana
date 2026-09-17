---
title: Tool Choice & Agent Loop
section: Architecture
order: 8
---

# Tool Choice & the Agent Loop

The agent answers a question in rounds: in each round the LLM either calls tools or writes the answer. Which tools it may call, whether the first round must call one, and how many rounds it gets are all decided by the active **scenario** from the [playbook](../data-model/queries.md).

Each request keeps this state in a `RunState` (`libs/vedana-core/src/vedana_core/run_state.py`). The loop (`LLM.create_completion_with_tools`) and the `select_scenario` tool both read and update it, so a scenario switch changes what the next round is allowed to do.

## `RunState`

| Field                  | Starts as                          | Updated when                                                                  |
| ---------------------- | ---------------------------------- | ------------------------------------------------------------------------------ |
| `scenario`             | routed scenario or `None`           | `activate()` runs — at the start, or from `select_scenario`                    |
| `budget`               | `AGENT_MAX_ITERS`                   | on activation: `min(rounds_used + scenario.max_iters, AGENT_MAX_ITERS_CEILING)` |
| `pending_tool_choice`  | scenario's `tool_choice`            | used up by the next round, which then reverts to `auto`                         |
| `switches`             | `0`                                 | goes up by one per real switch, never past `AGENT_MAX_SCENARIO_SWITCHES`        |
| `rounds_used`          | `0`                                 | every round                                                                      |

## Allowed tools

| State               | Tools the LLM sees                                                                         |
| ------------------- | ------------------------------------------------------------------------------------------- |
| No scenario active   | `vector_text_search`, `cypher`, `inspect_data_model`, `select_scenario`                    |
| Scenario active      | the scenario's `tools_allowed`, plus `inspect_data_model`                                  |

Only tools that are actually registered for the request are sent. `web_search` and MCP tools (`mcp__<server>__<tool>`) are never available without an active scenario that lists them.

> `inspect_data_model` is defined in `vedana_core.tools` and appears in the default allow-list, but it is not registered by `build_tools` at the moment, so the LLM doesn't see it.

## Each round

1. Work out the allowed tools from `RunState`.
2. Call the LLM with **only** those tool definitions, `tool_choice = state.take_tool_choice()` and `reasoning_effort = REASONING_EFFORT`.
3. If the reply has no tool calls, stop — its text is the answer.
4. Otherwise run the tool calls in parallel (`asyncio.gather`) and append the results. A call to a tool that isn't allowed doesn't run; its result is `Tool X is not available in the current scenario.`
5. When the round count reaches `budget`, add the `finalize_answer_tmplt` system message and make one last call with `tool_choice="none"`. The same tool definitions are still sent so the provider's prompt cache still matches.

The finished message trace is scanned from the end for the last non-empty assistant message; if there is none, `generate_no_answer_tmplt` produces the "nothing found" reply.

## `tool_choice`

Set per scenario in the `tool_choice` column of Queries:

| Value      | First round of the scenario                                  | Later rounds |
| ---------- | ------------------------------------------------------------- | ------------ |
| `auto`     | the LLM decides whether to call tools                         | `auto`       |
| `required` | the LLM must call at least one tool                           | `auto`       |
| `none`     | the LLM answers without tools; the loop ends after this round | —            |

A scenario with `tool_choice = none` also gets no graph description in the system prompt — only its own guidance (see [Vedana Core → Data model filtering](./vedana-core.md#data-model-filtering)). Use it for smalltalk and other scenarios that never need data.

If a scenario is activated mid-loop through `select_scenario`, its `tool_choice` applies from the next round.

## Switching scenarios

`select_scenario(id, reason)` lets the agent switch to another scenario when the first one turns out not to fit.

- A scenario can only switch if its `tools_allowed` includes `select_scenario`. Before any scenario is chosen, the tool is always available.
- `id` is the 1-based scenario ID shown in the system prompt (`- [id] name:`). An unknown ID returns an error listing the available scenarios.
- Choosing the scenario that's already active does nothing and doesn't count as a switch.
- Once `AGENT_MAX_SCENARIO_SWITCHES` (default `2`) is reached, the tool returns an error telling the agent to carry on with the current scenario.
- A successful switch returns the new scenario's `guidance` (its `query_example`) and `tools_available`, and raises the budget — up to `AGENT_MAX_ITERS_CEILING` (default `12`). The guidance replaces any earlier scenario guidance.

After `AGENT_MAX_ITERS` rounds the user sees the status message *Taking longer than expected…*.

## Where to see it

`technical_info.scenario` in the `rag.query_processed` event records which scenario ended up running:

```json
{
  "active": "Legal question with statute lookup",
  "switches": 1,
  "rounds_used": 6,
  "budget": 9
}
```

Compare `active` with `rag.selected_scenario.selected_scenario` (what routing picked) to separate two different failures: routing picked the wrong scenario, or the agent switched and still got it wrong. Every `select_scenario` call is also listed in `technical_info.tool_calls`.

## Settings

| Variable                        | Default | Meaning                                                            |
| ------------------------------- | ------- | ------------------------------------------------------------------- |
| `AGENT_MAX_ITERS`               | `5`     | Default round budget, and default `max_iters` for scenarios.       |
| `AGENT_MAX_ITERS_CEILING`       | `12`    | Hard limit on the budget, including after switches.                 |
| `AGENT_MAX_SCENARIO_SWITCHES`   | `2`     | Maximum `select_scenario` switches per request.                     |
| `REASONING_EFFORT`              | `disable` | Passed to the loop, finalisation and no-answer calls. Ignored by models that don't support it. |

## What's next

- [Queries (playbook)](../data-model/queries.md) — the `tool_choice`, `tools_allowed` and `max_iters` columns.
- [Scenario routing](../data-model/scenario-routing.md) — how the first scenario is chosen.
- [Tools for Vedana](../concepts/tools-for-vedana.md) — what each tool does.
