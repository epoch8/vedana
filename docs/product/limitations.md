---
title: Limitations
section: Product
order: 4
---

# Limitations

Vedana is a powerful tool, but it's not a silver bullet. This section is an honest list of limitations to keep in mind when picking and operating it.

## Dependence on data model quality

Vedana is never better than its data model and the data itself.

- if anchors are described vaguely → the assistant gets confused;
- if links aren't described → multi-hop questions don't work;
- if the data in Grist is incomplete / outdated → answers are too.

That's not a bug — it's by design. Vedana **deliberately** doesn't try to "guess" structure. If you want emergent behaviour, take classic RAG.

## Setup cost

For a working assistant in a non-trivial domain you need to:

- describe anchors / attributes / links — that's a domain expert + DS engineer's work, several days minimum;
- compile a golden dataset (50–200 questions);
- configure the playbook;
- do 5–10 tune→eval iterations.

It pays back on big and important domains. For a weekend prototype Vedana is overkill.

## Latency

A single Vedana request is typically 2–10 seconds:

- 1 LLM call for data model filtering (optional);
- up to 5 tool-calling iterations (LLM + Cypher / vector search);
- finalisation (if iterations are exhausted).

Compared to "one LLM call with RAG context" (1–3 sec), that's slower. Acceptable for an interactive chat, can be tight for a real-time API.

What helps:

- smaller models (`gpt-4.1-mini` vs `gpt-4.1`);
- caching (LiteLLM caching);
- parallel tool calls (Vedana already does this).

## Per-request cost

Each request includes:

- 1 LLM call for filtering (~1–3k prompt tokens);
- 1–5 main agent LLM calls (each — 5–20k prompt tokens + 200–2000 completion tokens);
- embeddings as needed.

On `gpt-4.1-mini` a typical request is ~ $0.001–$0.005. On `gpt-4.1` — an order of magnitude more expensive.

The `llm_usage_*_tokens_total` metrics will help track this.

What helps:

- a small model for filtering;
- a sensible `pipeline_history_length` (don't drag the entire thread);
- compact data model rendering templates (see [Customizing Prompts](../guides/customizing-prompts.md));
- keep data model filtering on.

## Dependence on Grist

By default the data model lives in Grist. If Grist is down — updates can't go through. The assistant itself keeps running on the latest snapshot synced into Postgres.

If Grist isn't acceptable for you, move to your own source via [Custom ETL](../data-ingestion/custom-etl.md).

## Memgraph operational nuances

- `--storage-mode=IN_MEMORY_ANALYTICAL` (the compose default) — all data in RAM. Big datasets need a big instance.
- restarting Memgraph without a snapshot = data loss. Production configs always include snapshot/replication.
- complex Cypher queries can block. The 30-row limit (`RagAgent.execute_cypher_query`) caps the result, not the execution time.

## LLM hallucinations don't fully disappear

Vedana **reduces** hallucinations but doesn't eliminate them:

- the LLM may misinterpret a tool result;
- the LLM may invent an attribute or link not in the data model, especially when the description is vague;
- the LLM may "complete" the answer from world knowledge instead of retrieved context.

What helps:

- a strict system prompt with "use only tool results";
- evaluation on adversarial questions;
- post-validation in the playbook ("verify each value via cypher before answering").

## Security

- Vedana **has no built-in** authn/authz inside requests. Data access is controlled only at the API gateway / reverse proxy layer.
- Cypher runs in read-only mode but **in one graph space**. If multiple tenants share the graph, user A may receive user B's data (if the corresponding query goes through).
- The LLM may attempt to generate malicious Cypher — read-only mode protects writes, but `OOM` queries are possible (though `rows_limit=30` minimises that).
- For multi-tenancy: either separate graphs per tenant, or strict filtering by `tenant_id` in every anchor (via cypher templates).

## Not suitable for

- tasks without a structured domain (just a chat over a PDF — classic RAG is simpler);
- generating creative content (Vedana is rigidly tied to data);
- real-time low-latency tasks (< 1 sec response);
- domains with no structured data and nowhere to take it (structure the data first, then Vedana).

## Possible bottlenecks at scale

- **Large graph: Memgraph** must be monitored for memory and CPU. Clustering is possible (see Memgraph docs).
- **Lots of embeddings: pgvector** — add HNSW indexes (`CREATE INDEX ON ... USING hnsw (embedding vector_cosine_ops)`).
- **Many parallel requests: LLM provider** — bottlenecks on provider rate limits. OpenRouter with multi-model fallback can help.

## What's next

- [Use Cases](./product/use-cases.md) — where Vedana shines.
- [FAQ](./product/faq.md) — common questions.
- [GitHub Issues](https://github.com/epoch8/vedana/issues) — what's planned and in progress.
