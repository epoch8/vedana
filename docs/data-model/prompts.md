---
title: Prompts
section: Data Model
order: 6
---

# Prompts

The **Prompts** table stores text templates that Vedana uses across the pipeline. The `name` column is the **key** that the code looks up the template by. If a row for the name isn't found, the built-in default is used (see `vedana_core.llm` and `vedana_core.data_model`).

## Fields

| Field   | Description                          |
| ------- | -------------------------------------- |
| **name** | Template name (key).                 |
| **text** | Template body (with placeholders).    |

## Available templates

| Key                                  | Where it's used                                                       | Placeholders                                       |
| ------------------------------------ | --------------------------------------------------------------------- | --------------------------------------------------- |
| `generate_answer_with_tools_tmplt`   | Main system prompt for the agent (`LLM.generate_cypher_query_with_tools`) | `{tools}`, `{graph_description}`              |
| `dm_filter_prompt`                   | System prompt for the data model filtering step                       | —                                                   |
| `dm_filter_user_prompt`              | User prompt for the data model filtering step                          | `{user_query}`, `{compact_data_model}`            |
| `finalize_answer_tmplt`              | System prompt for finalisation after the tool-calling loop is exhausted | —                                                 |
| `generate_no_answer_tmplt`           | Prompt for the "nothing found" scenario                                | —                                                   |
| `dm_descr_template`                  | Template that renders the entire data model                            | `{anchors}`, `{anchor_attrs}`, `{links}`, `{link_attrs}`, `{queries}` |
| `dm_anchor_descr_template`           | Template for one anchor row                                             | `{anchor.noun}`, `{anchor.description}`, `{anchor.id_example}`, `{anchor.query}` |
| `dm_attr_descr_template`             | Template for one attribute row                                          | `{anchor.noun}`, `{attr.name}`, `{attr.description}`, `{attr.example}`, `{attr.query}` |
| `dm_link_descr_template`             | Template for one link row                                               | `{link.sentence}`, `{link.description}`, `{link.query}` |
| `dm_link_attr_descr_template`        | Template for one link attribute row                                     | `{link.sentence}`, `{attr.name}`, `{attr.description}`, `{attr.example}`, `{attr.query}` |
| `dm_query_descr_template`            | Template for one Queries row                                             | `{query.name}`, `{query.example}`                  |

## Defaults

Every template has a default hardcoded in the code. That means you can run Vedana without entries in `Prompts` — it'll just work with the built-in prompts (in Russian by default for Vedana's defaults).

Defaults:

- `generate_answer_with_tools_tmplt` — the main system prompt for the agent; see `vedana_core/llm.py`.
- `dm_filter_*` — prompts for data model filtering; see `vedana_core/rag_pipeline.py`.
- `finalize_answer_tmplt` — finalisation.
- `generate_no_answer_tmplt` — "nothing found" message.
- `dm_*_descr_template` — render the data model into human-readable sections.

The full text of the defaults is in the source:

- `libs/vedana-core/src/vedana_core/llm.py`
- `libs/vedana-core/src/vedana_core/rag_pipeline.py`
- `libs/vedana-core/src/vedana_core/data_model.py`

## When to override

### Main system prompt (`generate_answer_with_tools_tmplt`)

The most common — override for:

- **tone and style** of the assistant (formal/friendly, formal "you" / informal "you");
- **domain vocabulary**: "we issue orders", not "we hand over goods";
- **hard rules**: "never answer about competitor prices", "decline when personal data is mentioned";
- **answer format**: always return a list, always include a link, always mention the ID.

### Filtering prompt (`dm_filter_prompt` / `dm_filter_user_prompt`)

Rarely overridden, but sometimes useful:

- if you have a special structure of model description;
- if you want the model to be more aggressive or more lenient in selecting attributes.

### Finalisation (`finalize_answer_tmplt`)

Triggers when the tool-call iteration limit is reached. Worth overriding to:

- give an honest "I couldn't get an answer, here's what I found" with the limitations.

### No answer (`generate_no_answer_tmplt`)

When nothing was found. Worth overriding to:

- propose relevant alternative questions;
- give a contact point (support, feedback form);
- avoid over-apologising — sounds fake.

### Data model rendering templates

Override if you want:

- to shrink the context (drop `query` for attributes);
- to add more structure (markdown tables instead of lists);
- to localise section headers.

## How to update

1. Open **Grist > Data Model > Prompts**.
2. Create or update a row with the right `name`.
3. In the backoffice → ETL run `data_model_steps` (or "Refresh Data Model").
4. Run an evaluation to confirm metrics didn't drop.

Changes take effect immediately after refresh.

## Best practices

- **Don't write a prompt-novel.** The longer the system prompt, the more expensive every request is (it goes into every LLM call in the thread).
- **Clear "do / don't" rules** work better than vague guidelines.
- **List tools explicitly** if you want to constrain choices: "Use only cypher; vector_text_search — only if cypher returns an empty result".
- **Test prompts through evaluation**, not "by feel" in chat — that gives you a hit-rate metric, not a subjective feeling.
- **Version**: give the prompt a date/version suffix (`vedana_main_v3`) and keep older versions next to it so you can roll back.

## Debugging a prompt

If something went wrong, in the backoffice → Chat → Details next to the answer you can see:

- the final system prompt sent to the LLM (with `{graph_description}` and `{tools}` substituted);
- the message set (including the `dm_filter_*` instructions if filtering is on);
- the list of executed tool calls;
- the finalisation step (if it was applied).

That helps you understand whether the LLM is **really** receiving what you wrote.

## What NOT to change unnecessarily

- **Template key names.** They are hard-coded in source; a wrong name means the code falls back to the default, not your text.
- **Placeholders (`{tools}`, `{graph_description}`, `{user_query}`, `{compact_data_model}`).** Without them the templates won't work.
- **Final "verify" instructions.** If you want to check answer correctness, do it via evaluation — don't bake "double-check yourself" into the prompt.

## What's next

- [Prompts reference](../data-model/prompts.md) — full list of templates.
- [Evaluation](../product/evaluation.md) — how to measure the impact of changes.
