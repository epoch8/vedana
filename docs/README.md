---
title: Vedana Documentation
---

# Vedana Semantic RAG Documentation

> Licensed under the **[Apache License 2.0](https://github.com/epoch8/vedana/blob/main/LICENSE)** — free for commercial and non-commercial use, including modification and redistribution.

**Vedana** is an open-source multi-agent system for building AI assistants on top of **Semantic RAG** (Retrieval-Augmented Generation over a knowledge graph). Unlike classic RAG, Vedana doesn't guess answers based on text similarity — it explores the knowledge base step by step: it issues Cypher queries against the graph, runs vector search, verifies sources, and assembles an answer from real data.

Vedana is built around three components:

- **JIMS** (*Just an Integrated Multiagent System*) — a framework for managing message flows, events, threads, and LLM providers.
- **Vedana Core** — the RAG pipeline with data model filtering, Cypher generation, and vector search.
- **Vedana ETL** — an incremental pipeline built with [Datapipe](https://github.com/epoch8/datapipe) that syncs the data model and the data itself from [Grist](https://github.com/gristlabs/grist-core) into [Memgraph](https://memgraph.com/) and [pgvector](https://github.com/pgvector/pgvector).

## Where to start

Depending on your role:

- **I'm a developer and want to run Vedana locally** — start with [Quick Start](./getting-started/quick-start.md) to see Vedana working in Docker. If you then want to develop *on* Vedana itself (debug a service, add a feature), see [Local Development](./getting-started/local-development.md), then [Architecture Overview](./architecture/overview.md).
- **I'm a product manager and want to understand what this product is** — start with [What is Vedana](./concepts/what-is-vedana.md) and [Why Classic RAG Fails](./concepts/why-classic-rag-fails.md), then [Use Cases](./product/use-cases.md).
- **I want to describe my domain and load data** — read [Data Model](./data-model/overview.md) and then [Data Ingestion](./data-ingestion/overview.md).
- **I want to contribute** — see the [Contributing Guide](./contributing/contributing.md).

## Documentation map

### Getting Started

- [Introduction](./getting-started/introduction.md) — what Vedana is in one paragraph and who needs it.
- [Quick Start](./getting-started/quick-start.md) — bring up the stack in Docker and ask your first question in 10 minutes.
- [Local Development](./getting-started/local-development.md) — develop on Vedana itself: native Python with `uv`, infrastructure in Docker. (For first-run demo see Quick Start; for production see [Operations → Deployment](./operations/deployment.md).)
- [Configuration](./getting-started/configuration.md) — every key ENV variable and what it means.

### Concepts (for all roles)

- [What is Vedana](./concepts/what-is-vedana.md)
- [Why Classic RAG Fails](./concepts/why-classic-rag-fails.md)
- [Semantic RAG Overview](./concepts/semantic-rag-overview.md)
- [Data for Vedana](./concepts/data-for-vedana.md)
- [Data Model for Vedana](./concepts/data-model-for-vedana.md)
- [Tools for Vedana](./concepts/tools-for-vedana.md)
- [Playbook for Vedana](./concepts/playbook-for-vedana.md)

### Architecture (for developers)

- [Overview](./architecture/overview.md) — high-level diagram, repository, component relationships.
- [JIMS Core](./architecture/jims-core.md) — threads, events, pipelines, ThreadController and ThreadContext.
- [Vedana Core](./architecture/vedana-core.md) — `RagPipeline`, `RagAgent`, `LLM`, `Graph`, `VectorStore`.
- [Vedana ETL](./architecture/vedana-etl.md) — Datapipe catalog, steps, incremental loading.
- [Vedana Backoffice](./architecture/vedana-backoffice.md) — Reflex admin UI, chat, ETL runner, metrics.
- [Storage Model](./architecture/storage-model.md) — Postgres, Memgraph, pgvector, Grist.
- [Observability](./architecture/observability.md) — OpenTelemetry, Prometheus, Sentry.

### Data Model

- [Overview](./data-model/overview.md)
- [Anchors (nodes)](./data-model/anchors.md)
- [Attributes](./data-model/attributes.md)
- [Links (relationships)](./data-model/links.md)
- [Queries (playbook)](./data-model/queries.md)
- [Prompts](./data-model/prompts.md)
- [ConversationLifecycle](./data-model/conversation-lifecycle.md)

### Data Ingestion

- [Overview](./data-ingestion/overview.md)
- [Documents and Chunks](./data-ingestion/documents-and-chunks.md)
- [Structured Data](./data-ingestion/structured-data.md)
- [FAQ](./data-ingestion/faq.md)
- [Custom ETL](./data-ingestion/custom-etl.md)

### API Reference

- [Overview](./api/overview.md)
- [HTTP API (`jims-api`)](./api/http-api.md)
- [Widget API (`jims-widget`)](./api/widget-api.md)
- [Telegram (`jims-telegram`)](./api/telegram.md)
- [Python API (`vedana-core`)](./api/python-api.md)
- [Configuration Reference (`.env`)](./api/configuration-reference.md)

### Guides

- [Setting Up Data Model](./guides/setting-up-data-model.md)
- [Test Dataset (LIMIT)](./guides/test-dataset.md)
- [Adding Anchors](./guides/adding-anchors.md)
- [Adding Attributes](./guides/adding-attributes.md)
- [Adding Links](./guides/adding-links.md)
- [Adding Documents](./guides/adding-documents.md)
- [Adding Structured Data](./guides/adding-structured-data.md)
- [Adding FAQ Entries](./guides/adding-faq-entries.md)
- [Tuning Embeddings & Thresholds](./guides/tuning-embeddings.md)
- [Customizing Prompts](./guides/customizing-prompts.md)
- [Writing a Custom Tool](./guides/custom-tools.md)
- [Multi-tenancy](./guides/multi-tenancy.md)

### Product (for PMs)

- [Use Cases](./product/use-cases.md)
- [Comparison with Classic RAG](./product/comparison.md)
- [Quality Metrics & Evaluation](./product/evaluation.md)
- [Limitations](./product/limitations.md)
- [FAQ](./product/faq.md)
- [Open roadmap items on GitHub](https://github.com/epoch8/vedana/issues)

### Operations

- [Deployment](./operations/deployment.md)
- [Monitoring & Metrics](./operations/monitoring.md)
- [Troubleshooting](./operations/troubleshooting.md)
- [Cost Management](./operations/costs.md)
- [Security](./operations/security.md)

### Contributing

- [Contributing Guide](./contributing/contributing.md)
- [Code Style](./contributing/code-style.md)
- [Testing](./contributing/testing.md)
- [Repository Structure](./contributing/repository-structure.md)

## License and community

The project is distributed under the **[Apache License 2.0](https://github.com/epoch8/vedana/blob/main/LICENSE)** — a permissive license that allows commercial use, modification, and redistribution, including in proprietary products, with attribution and a notice of changes. Source code and issue tracker — at [github.com/epoch8/vedana](https://github.com/epoch8/vedana). Product website: [vedana.tech](https://vedana.tech).
