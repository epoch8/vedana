---
title: Introduction
section: Getting Started
order: 1
---

# Introduction

**Vedana** is an open-source framework for building AI assistants whose answers can be verified and reproduced. Vedana stores domain knowledge as a knowledge graph, and the assistant explores this graph step by step through explicitly described tools — vector search and Cypher queries against the graph database.

Unlike classic RAG, Vedana:

- doesn't guess the answer based on top-K similar text fragments — it builds a structured query;
- returns complete sets (not samples), exact values (not approximations), and explicit sources (not "somewhere in the documents");
- lets you describe your domain as a data model (anchors, attributes, links, queries) and get deterministic behaviour.

## Who needs this

Vedana is designed for domains where mistakes are not allowed or are expensive:

- **e-commerce** — product catalogs, compatibility, in-store availability, delivery policies;
- **legal and compliance** — which requirements apply to a product, which documents regulate a category;
- **internal knowledge bases** — answers based on documentation, regulations, organisational structure;
- **customer support** — stable answers to FAQs, intent-based routing, exact values instead of "roughly that".

If your assistant's answers must be verifiable and aligned with real business logic — Vedana is your case. If "summarize this PDF" is enough, classic RAG will do.

## What's in the box

- **JIMS** — a framework for managing threads, events, and pipelines. Supports several interfaces: Telegram, Terminal UI, HTTP API, web widget, Reflex backoffice.
- **Vedana Core** — a RAG pipeline with data model filtering, an agent, and the `cypher` and `vector_text_search` tools.
- **Vedana ETL** — an incremental pipeline on top of [Datapipe](https://github.com/epoch8/datapipe) for loading the data model and data from Grist into Memgraph and pgvector.
- **Vedana Backoffice** — a Reflex admin UI with a chat, an ETL pipeline inspector, an evaluation harness, and prompt settings.
- **LiteLLM support**: you can use OpenAI, OpenRouter, Google/VertexAI, and any compatible providers.
- **Observability**: OpenTelemetry traces, Prometheus metrics, Sentry integration.

## What's next

- [Quick Start](./getting-started/quick-start.md) — bring Vedana up locally and ask your first question.
- [Concepts](../concepts/what-is-vedana.md) — the theory: what Semantic RAG is, why classic RAG breaks, how the data model is organised.
- [Architecture Overview](../architecture/overview.md) — how the code is structured and how the components fit together.
