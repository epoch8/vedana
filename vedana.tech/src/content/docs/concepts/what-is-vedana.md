---
title: What is Vedana
section: Concepts
order: 1
---

# What is Vedana

**Vedana is a system that makes AI outputs reliable and verifiable by grounding them in structured knowledge.**

Vedana combines vector search, structured retrieval, and multi-step reasoning over data. The result is answers that are:

- **traceable** — every reasoning step can be inspected and reproduced;
- **reproducible** — the same question triggers the same operations;
- **aligned with real business logic** — the assistant operates within an explicitly described data model.

Unlike typical RAG solutions, Vedana doesn't rely on raw vector search or single-pass generation. It enables the AI to explore data **step by step**, following an explicit data model and a controlled reasoning process.

## What this gives a company

- **Control over how answers are produced** — not "LLM magic", but an explicit playbook: which tool, in which order, for which kind of question.
- **Verify every reasoning step** — which nodes and links were retrieved, which Cypher queries were executed, which document chunks contributed to the answer.
- **Reduce hallucinations** — the answer relies on actual graph data, while the LLM acts as an interpreter, not a "source of truth".

## Where Vedana applies

Vedana is designed for domains where correctness is critical:

- **legal / compliance** — which documents regulate a product category, which requirements apply to a contract;
- **e-commerce** — product catalogs, compatibility, in-store stock, delivery policies;
- **internal knowledge bases** — exact answers about documentation, regulations, organisational structure, processes;
- **B2B support** — stable answers to typical client and partner questions.

## What Vedana isn't

- It is **not a replacement** for classic RAG when it comes to summarization or vague "what is this document about?" questions — there, ordinary RAG is simpler and sufficient.
- It is **not a "smart" generative agent** — Vedana deliberately constrains the LLM with an explicit set of tools so behaviour is predictable.
- It is **not a turn-key business product** — you have to describe the data model and reasoning rules for your domain.

## What's next

- [Why Classic RAG Fails](./concepts/why-classic-rag-fails.md) — where ordinary RAG breaks down and why this is a structural problem, not a "bad embeddings" problem.
- [Semantic RAG Overview](./concepts/semantic-rag-overview.md) — the four parts of Semantic RAG.
- [Data Model for Vedana](./concepts/data-model-for-vedana.md) — how to describe a domain.
