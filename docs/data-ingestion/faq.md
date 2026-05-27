---
title: FAQ
section: Data Ingestion
order: 4
---

# FAQ

## Overview

Vedana includes a built-in FAQ mechanism that works out of the box. It's the simplest way to give short, consistent, high-confidence answers to known questions — no extra modeling required.

FAQ entries are stored in a dedicated Grist table and retrieved directly at runtime. Because the answers are predefined, they **don't vary** between users or sessions. When stable wording matters (for support, policies, operational answers), FAQ is the most reliable ingestion mechanism.

## What an FAQ entry is

A predefined question/answer pair stored in Grist. When a user asks a question that matches an FAQ entry closely enough, the stored answer is returned **directly** — no graph traversal, no document search, no retrieval logic beyond similarity matching.

## Table structure

| Column     | Content                                       |
| ---------- | ----------------------------------------------- |
| **question** | A representative phrasing of the question.   |
| **answer**   | The exact answer to return.                  |

Each row is one FAQ entry. **The attribute names (`question`, `answer`) are your choice** — Vedana doesn't hardcode them. We use `question` / `answer` throughout these docs because they read naturally; the test fixture in `libs/vedana-core/tests/test_data_model.py` happens to use `faq_question_text` / `faq_answer_text` instead. Whatever you pick, just make sure the names in your Grist `Anchor_attributes` table match what you reference in `vector_text_search(... property=...)` and in any Cypher you write.

## How retrieval works

FAQ is **not a hardcoded pre-step in the pipeline** — there's no special "is this an FAQ question?" branch in `RagPipeline`. Instead, FAQ entries are loaded into the knowledge graph as a regular anchor (`faq`) whose `question` attribute is embeddable. The agent finds the matching FAQ entry the same way it finds any other answer: by issuing a `vector_text_search` tool call.

```mermaid
flowchart TD
    Q[User:<br/>"What's the return policy?"] --> RP[RagPipeline]
    RP --> DMF[Data model filtering<br/>e.g. faq + return_policy anchors]
    DMF --> AG[RagAgent decides<br/>which tool to use]
    AG --> VTS["vector_text_search<br/>label=faq, property=question"]
    VTS --> S{High similarity to<br/>a stored faq.question?}
    S -- "Yes" --> ANS[Agent returns the<br/>matching faq.answer]
    S -- "No" --> OTHER[Agent continues with<br/>cypher / other vts calls]
    OTHER --> ANS2[Grounded answer<br/>from the graph + documents]
```

What actually happens at runtime:

1. The user's question enters `RagPipeline.process_rag_query` (see [Vedana Core architecture](../architecture/vedana-core.md#rag-pipeline)).
2. Data model filtering may include or exclude the `faq` anchor depending on the question.
3. The agent looks at the available data model and decides which tool to invoke. For a known intent it issues `vector_text_search(label="faq", property="question", text=<user question>)`.
4. pgvector returns rows from the `faq` anchor whose `question` similarity ≥ the attribute's `embed_threshold`.
5. The agent reads the `answer` field of the matching row and returns it (often verbatim, sometimes lightly rephrased depending on the system prompt).

If no row matches above the threshold, the agent simply doesn't find an FAQ answer and proceeds with other tool calls (Cypher over structured anchors, vector search over `document_chunk`, etc.) — the same as for any other question type.

The threshold controlling "close enough" is the `embed_threshold` of the `faq.question` attribute in the data model. Recommended starting range: **0.70–0.78** (see [Tuning Embeddings & Thresholds](../guides/tuning-embeddings.md#starting-values)). If FAQ matches too broadly or too narrowly, that's the first place to look.

## How FAQ differs from documents and structured data

Each type serves a different purpose. Picking the wrong one is one of the most common causes of inconsistent answers.

**Use FAQ** when the answer is short, fixed, and shouldn't vary: opening hours, return policy, contact details, standard support replies. The answer is authoritative and should be returned verbatim.

**Use documents** when the answer is long, contextual, requires interpretation: policies, manuals, contract texts. The user is asking what something *says*, not which value.

**Use structured data** when the answer depends on a specific attribute, count, filter, or relationship: product prices, branch locations, contract dates. The user is asking for a specific value computable from the graph.

The most common mistake is using FAQ for questions that should be structured data, or documents for questions that should be FAQ. Branch opening hours stored as an FAQ entry work, but they **can't** be filtered, compared, or updated without manual editing of the row. The same data modelled as a `Branch` anchor with an `opening_hours` attribute can be updated in one place and queried precisely for any branch.

## What FAQ doesn't do

FAQ is a lookup table, not a reasoning layer. It:

- doesn't synthesise answers from multiple entries,
- doesn't follow relationships,
- doesn't adapt the answer based on context.

If a question doesn't match a stored entry above the threshold, FAQ returns nothing and the system continues with other retrieval methods.

FAQ also **doesn't scale well** as the primary answer mechanism for a large domain. As the table grows, entries with overlapping intents start competing with each other, producing unstable matching. FAQ works best as a **small, curated** set of high-confidence entries — not as a substitute for a properly modeled domain.

## What's next

- [Adding FAQ Entries guide](../guides/adding-faq-entries.md) — practical guide.
- [Data for Vedana](../concepts/data-for-vedana.md) — the three data types in context.
