---
title: Customizing Prompts
section: Guides
order: 10
---

# Customizing Prompts

Vedana is wired to **Grist > Data Model > Prompts**: any template the code uses can be overridden by adding a row with the matching name.

See the full list of keys in the [Prompts reference](../data-model/prompts.md).

## When to override

| Template                            | When to change                                                       |
| ----------------------------------- | --------------------------------------------------------------------- |
| `generate_answer_with_tools_tmplt`   | Tone, domain vocabulary, answer formats, hard rules.                |
| `dm_filter_prompt`                  | If you need to change how anchors/links are picked for a question.  |
| `finalize_answer_tmplt`             | Iteration limit hit — what to say to the user.                       |
| `generate_no_answer_tmplt`          | "Nothing found" — you want a precise and useful wording.            |
| `dm_descr_template` and row templates | If you want a more compact / more structured model description in the LLM context. |

## Example: main system prompt

In Prompts add a row:

| name                                | text |
| ----------------------------------- | ---- |
| `generate_answer_with_tools_tmplt`  | (see below) |

```text
You are an Acme assistant. You help employees and customers find precise answers about the Acme product catalog, policies, and processes.

Style:
- Use a polite "you" form, no excessive familiarity.
- Brief and to the point. When useful, structure the answer as a list.
- Don't make things up. If data isn't available, say so honestly and offer a clarification.

Use the {tools} tools to work with the knowledge base.

graph_composition:
{graph_description}

If the question is about specific values (prices, stock, opening hours), try cypher first.
If the question is about policies, regulations, or explanations, try vector_text_search over document_chunk first.
If neither path produces a result, honestly say so.

Always include the source when using document data: "(Source: <document title>, <link>)".
```

`{tools}` and `{graph_description}` are required placeholders — they're substituted automatically.

## Example: "no answer found"

| name                          | text |
| ----------------------------- | ---- |
| `generate_no_answer_tmplt`    | (see below) |

```text
You write the response when we couldn't find a precise answer in the knowledge base.

Tell the user briefly and in a friendly tone that there's no direct answer.
Based on the conversation context, offer 1–2 clarifying questions.
Don't apologise more than once.
Don't add support contact details — that's done elsewhere.
```

## Example: compact data model rendering

If your data model is large and the context "eats" many tokens, override the rendering templates to drop `query` from the description (the LLM will still generate its own Cypher):

| name                                    | text                                                                |
| --------------------------------------- | -------------------------------------------------------------------- |
| `dm_anchor_descr_template`              | `- {anchor.noun}: {anchor.description}`                              |
| `dm_attr_descr_template`                | `- {anchor.noun}.{attr.name} ({attr.dtype}): {attr.description}`     |
| `dm_link_descr_template`                | `- {link.sentence}: {link.description}`                              |
| `dm_link_attr_descr_template`           | `- {link.sentence}.{attr.name}: {attr.description}`                  |

That can shrink the context 2–3x for large data models.

## Best practices

- **Test via golden dataset.** Pass Rate (the aggregate `pass_rate` metric in the eval state, surfaced in the backoffice) is the only objective indicator.
- **Version.** Give the prompt a date suffix (`vedana_main_20260301`) and keep the previous version in a comment column in Grist for rollback.
- **Don't write a prompt-novel.** Every extra paragraph goes into every request. Shorter is cheaper.
- **Separate "how to behave" from "what to answer".** Behaviour → main prompt. What exactly to answer → the Queries playbook (more precise and reproducible).

## Debugging a prompt

If something went wrong, in the backoffice → Chat → Details next to the answer you can see:

- the final system prompt that was sent to the LLM (with `{graph_description}` and `{tools}` substituted);
- the message set (including `dm_filter_*` instructions if filtering is on);
- the list of executed tool calls;
- the finalisation step (if it was applied).

That helps you understand whether the LLM is **really** receiving what you wrote.

## What NOT to change unnecessarily

- **Template key names.** They're hardcoded in the source; a wrong name means the code uses the default, not your text.
- **Placeholders (`{tools}`, `{graph_description}`, `{user_query}`, `{compact_data_model}`).** Without them the templates don't work.
- **Final "verify" instructions.** If you want to verify answer correctness, do it through evaluation — don't bake "double-check yourself" into the prompt.

## What's next

- [Prompts reference](../data-model/prompts.md) — full list of templates.
- [Evaluation](../product/evaluation.md) — how to measure the impact of changes.
