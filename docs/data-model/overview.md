---
title: Data Model Overview
section: Data Model
order: 1
---

# Data Model: Overview

The data model in Vedana is the contract between your domain and the assistant. It describes **which entities exist**, **what properties they have**, and **how they are connected**. Without it the assistant only sees text; with it, it sees the structure of knowledge.

The data model is **not the data**, it's the schema. The data itself (products, documents, contracts, branches) lives in the Grist Data doc and is loaded into the graph through ETL. The data model lives in the Grist Data Model doc and describes how that data is interpreted.

## The seven tables

| Table                    | Required?                              | Purpose                                                                  |
| ------------------------ | --------------------------------------- | ------------------------------------------------------------------------- |
| **Anchors**               | yes                                     | graph node types (domain entities)                                        |
| **Anchor_attributes**     | yes                                     | properties of nodes                                                        |
| **Links**                 | yes                                     | edge types (relationships)                                                 |
| **Link_attributes**       | only if edges have attributes           | properties of edges                                                        |
| **Queries**               | no                                      | the playbook — scenarios for typical questions                            |
| **Prompts**               | no                                      | overrides for prompt templates                                             |
| **ConversationLifecycle** | no                                      | responses to lifecycle events (`/start`, etc.)                            |

The minimum working data model is `Anchors` + `Anchor_attributes` + `Links`. Everything else extends behaviour but isn't required for the system to start.

## How the model gets into the LLM

For each request:

1. ETL has already synced Grist → the `dm_*` tables in Postgres.
2. `DataModel` reads the tables and builds in-memory `Anchor`, `Link`, `Attribute`, `Query`, `Prompt` objects.
3. (Optional) The data model filtering step (`RagPipeline.filter_data_model`) picks only the elements relevant to the current question.
4. `DataModel.to_text_descr(...)` renders the chosen elements into text using the `dm_*_descr_template` templates (or their overrides from the `Prompts` table).
5. That text is inserted into the system prompt for the main agent.

## Rendering templates

By default the data model is rendered into five sections. Section headers in the built-in template (`vedana_core.data_model.dm_descr_template`) are in Russian:

- `## Узлы:` — anchors;
- `## Атрибуты узлов:` — anchor attributes;
- `## Связи между узлами:` — links;
- `## Атрибуты связей:` — link attributes;
- `## Типичные вопросы:` — typical questions (playbook).

You can override these by setting `dm_descr_template` in the `Prompts` table — e.g. to translate them to English:

```text
## Nodes:
- Product: A sellable product...; example ID: product_id: "p-001"; query: MATCH (p:product) WHERE ...

## Node attributes:
- Product.price: Price in EUR; example: 999.0; query: ...

## Links:
- PRODUCT_belongs_to_CATEGORY: a product belongs to a category; example query: MATCH ...

## Link attributes:
- ...

## Typical questions:
- Who likes <interest>?
  1) ...
  2) ...
```

Templates (with names from `Prompts`):

| Template                       | What it renders                                |
| ------------------------------ | ----------------------------------------------- |
| `dm_descr_template`            | the wrapper for all sections                   |
| `dm_anchor_descr_template`     | a row about an anchor                          |
| `dm_attr_descr_template`       | a row about an anchor attribute                |
| `dm_link_descr_template`       | a row about a link                             |
| `dm_link_attr_descr_template`  | a row about a link attribute                   |
| `dm_query_descr_template`      | a row from the playbook                        |

You can override any of these in the `Prompts` table to change the format the LLM sees the model in.

## Principles of a good data model

1. **Describe only what's needed.** Extra anchors/attributes inflate the context and reduce filtering accuracy.
2. **Anchor names: singular, Latin script.** `product`, not `products`, not `product_catalog`.
3. **Clear `description`s.** This is the primary signal to the LLM telling it *when* to apply an anchor / link. "Represents a product" — bad. "A sellable product in the catalog with a price, availability status, and category" — good.
4. **Fill in `query` for anchor / attribute / link.** Without that, the assistant can't reliably retrieve the data.
5. **Embeddable — only for text.** Identifiers and numbers should be regular attributes.
6. **Tune `embed_threshold` through evaluation.** Too low — irrelevant results, too high — misses. Start at `0.7`.
7. **Write Queries (the playbook).** That's the cheapest way to make behaviour predictable.

## What's next

- [Anchors](./data-model/anchors.md)
- [Attributes](./data-model/attributes.md)
- [Links](./data-model/links.md)
- [Queries](./data-model/queries.md)
- [Prompts](./data-model/prompts.md)
- [ConversationLifecycle](./data-model/conversation-lifecycle.md)
