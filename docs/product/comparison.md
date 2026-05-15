---
title: Comparison with Classic RAG
section: Product
order: 2
---

# Comparison with Classic RAG

Vedana isn't an "improved" RAG. It's a **different approach** optimised for a different class of tasks.

## High-level comparison

| Parameter                                       | Classic RAG                              | Vedana (Semantic RAG)                                    |
| ---------------------------------------------- | ----------------------------------------- | --------------------------------------------------------- |
| Storage                                         | flat collection of chunks + embeddings    | knowledge graph + structured data + embeddings           |
| Retrieval                                       | top-K by vector similarity                 | Cypher / vector / hybrid                                  |
| Answer completeness                             | a sample                                  | a complete set                                            |
| Exact values                                    | approximation (text from top-K)           | exact (via Cypher)                                       |
| Multi-hop reasoning                             | bad                                       | good (when links are described)                          |
| Determinism                                     | no                                        | yes (with a strict playbook)                              |
| Source citation                                 | conditional (chunk metadata)              | deterministic (graph nodes, document.url)                 |
| Setup cost                                      | low                                       | medium (data model required)                              |
| Handling data changes                           | re-embedding                              | incremental ETL (Datapipe)                                |
| Explainability                                  | low                                       | high (Cypher + reasoning are visible)                    |
| Speed on simple questions                       | faster (1 LLM call)                       | comparable or slower (filtering + tool calls)             |
| Suited for                                      | summarization, text search                | catalogs, regulations, hybrid domains                    |

## Where Vedana clearly wins

### Exact values

"How much is X?" → Vedana fetches `product.price` via Cypher. Classic RAG will find text like "X currently costs 999, but there are sometimes promos" — and you'll get an answer with possible errors / stale data.

### Complete lists

"Show me every category under 1000" → Vedana returns **all** records. Classic RAG returns the top-5 chunks that mention prices — and that's not "all of them".

### Complex relationships

"Which documents regulate products in category X?" → Vedana traverses two graph hops. Classic RAG, at best, finds a paragraph that happens to mention the relationship.

### Audit

In Vedana every answer can be broken down by operations: which Cypher ran, which chunks were retrieved, how data model filtering picked the context. Classic RAG usually limits you to "these chunks went into the context".

## Where classic RAG is simpler

### "Tell me about this document"

If the task is to summarise a document briefly, classic RAG (or even just the full text in the LLM) is simpler and cheaper.

### A purely text domain

If there's no data, just PDFs, and the assistant is expected to "find and retell" — Vedana is overkill. ChromaDB + a prompt will do.

### MVP over a weekend

Bringing up classic RAG over ChromaDB / Pinecone with one LLM is an evening's work. Describing Vedana's data model is a working week minimum, and ROI shows up on complex domains.

## Hybrid

Vedana still has vector search — it doesn't ditch classic RAG, it **extends** it. That means you can move gradually:

1. Start with the default model (only document/document_chunk/faq) — that's already advanced RAG.
2. Add structured anchors over time for question types where classic RAG fails.
3. Extend the playbook as you spot failure patterns on the golden dataset.

Unlike "RAG vs graph", Vedana enables **gradual migration** — without redeploys or rewrites.

## When to choose what

If your business case looks like this:

- "We have PDF regulations and need search" → start with classic RAG; might be enough.
- "We have PDFs + a structured catalog and accuracy matters" → Vedana.
- "We have a complex domain — legal/compliance/medical/finance" → Vedana, no question.
- "We just need a text helper for the team" → classic RAG, more economical.
- "Everything at once and growing" → Vedana, scaling on its data model is easier.

## What to choose in any case

Whether RAG or Vedana, **definitely** set up a golden dataset and run eval regularly. Without it you can't objectively compare which approach works better for **your** case. See [Quality Metrics & Evaluation](./evaluation.md).

---

## Vedana vs "LangChain + pgvector + Memgraph" (build vs buy)

The most common pushback we hear from technical leaders is: *"We can glue this together ourselves with LangChain, pgvector, and a graph DB — why adopt your framework?"*

You can. Vedana doesn't do anything magical that's impossible to reproduce. The honest question is: **how many engineer-weeks does that take, and will the in-house version still work in eight months when the team that built it has moved on?**

Below is a feature-by-feature checklist of what Vedana ships out of the box and a rough estimate of how much engineering it takes to build the equivalent yourself with LangChain (or LlamaIndex) + pgvector + Memgraph. Estimates assume one mid-to-senior engineer comfortable with Python, async, LLM tool-calling, and basic DB ops; numbers are calendar weeks for a "production-ready" version that includes tests, monitoring, and incremental ETL — not a 200-line prototype.

| #  | Capability                                                                 | Vedana | DIY estimate                | Why it's harder than it looks                                                                                                                                  |
| -- | -------------------------------------------------------------------------- | ------ | --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1  | **Data model with anchors / attributes / links**, declarative              | ✅      | 2–3 weeks                    | Schema, validators, version tracking, mapping into graph DDL, error messages that aren't "hash mismatch in adjacency table".                                  |
| 2  | **Data model filtering** before the main agent                             | ✅      | 1–1.5 weeks                  | Compact JSON renderer, Pydantic structured-output schema, eval harness to make sure filtering doesn't cut needed entities, fallback path on filter errors.    |
| 3  | **Playbook (`Queries`)** — example-based per-intent steering                | ✅      | 1–2 weeks                    | Storage format, Cypher / VTS step composition, integration into the agent prompt, governance over keeping examples up-to-date with the schema.                |
| 4  | **Datapipe ETL**: incremental, label-driven, partial reruns                | ✅      | 3–4 weeks                    | A naive `pandas → write_graph` job is a day. An incremental ETL that detects changed rows, recomputes only their embeddings, and reruns only affected steps is weeks of fiddly state management. |
| 5  | **Reflex backoffice** with chat, ETL runner, eval browser, manual queries  | ✅      | 3–5 weeks                    | Even a "minimal admin" turns into auth, threads list, message details, ETL trigger UI, eval result browser, manual query console. Most teams underbudget this by 4×. |
| 6  | **Eval pipeline** with golden dataset + LLM-as-judge                       | ✅      | 1.5–2 weeks                  | Dataset format, run scheduler, LLM-judge prompt, score aggregation, regression alerts, history.                                                                |
| 7  | **OTel traces + Prometheus metrics** for every LLM/Cypher/VTS call         | ✅      | 1 week                       | Instrumenting every tool call, propagating thread_id through async context, defining the right cardinality (`model`, `tool`, `outcome`) so it doesn't blow up. |
| 8  | **Cypher tool with Enum-constrained args** (label/property come from data model) | ✅      | 1 week                       | Dynamic Pydantic schema generation, OpenAI/Anthropic/LiteLLM compatibility checks, error reporting back to the agent.                                          |
| 9  | **`vector_text_search` tool with per-attribute thresholds**                | ✅      | 0.5–1 week                   | Per-attribute thresholds + `top_n`, plumbing through to pgvector, distance/similarity inversion, label-aware tables.                                          |
| 10 | **Thread/event model (JIMS)** — durable, replayable, plug-in pipelines     | ✅      | 2–3 weeks                    | Threads, events, ThreadController, state, event types — a small framework on its own. LangChain memory does ~30% of this.                                    |
| 11 | **Multiple interfaces from the same core** (HTTP, widget, Telegram)        | ✅      | 1–2 weeks each interface     | Auth, transport-specific event delivery (WebSocket vs polling), idempotency.                                                                                  |
| 12 | **LiteLLM-based provider abstraction** with per-call usage counters         | ✅      | 0.5–1 week                   | LiteLLM gives you the API; you still have to track token counters and propagate model-stats into your `query_processed` events.                              |
| 13 | **Grist as a default editorial UI** for non-technical content owners       | ✅      | 2–3 weeks if you build a CRUD | If you don't use Grist or Notion you'll build a small admin UI. PMs and content owners *will* ask for one.                                                    |
| 14 | **Ready-to-use `document` / `document_chunk` / `faq` anchors**             | ✅      | included in the schema work above | —                                                                                                                                                              |
| 15 | **Reproducible Quick Start in Docker Compose**                             | ✅      | 0.5–1 week                   | Trivial — but needed for onboarding and CI.                                                                                                                   |

### Total

| Path                                       | Calendar weeks (1 engineer) | Calendar weeks (2 engineers) |
| ------------------------------------------ | --------------------------- | ----------------------------- |
| **Adopt Vedana**                           | 1–2 weeks to first answer; 4–6 weeks to production on your domain | Same — the bottleneck is your data model, not the platform |
| **DIY on LangChain + pgvector + Memgraph** | 18–28 weeks to feature parity, plus ongoing maintenance | 11–17 weeks to feature parity |

These numbers match what most teams report after they've actually tried both paths. If you're an engineering shop that *enjoys* maintaining frameworks and you have a single canonical use case forever — DIY is fine. If you have a domain that will keep evolving (new entities, new playbook steps, new data sources, new interfaces), the maintenance tail of a hand-rolled stack is what kills it.

### What you don't get from Vedana that you'd build anyway

To be clear, Vedana is **not** a replacement for:

- **Your domain modelling** — somebody still has to decide what the anchors and links are. Vedana enforces the discipline; it doesn't invent the entities for you.
- **Your golden dataset** — Vedana ships an eval harness, but the questions and reference answers are yours.
- **Your Cypher index strategy** — Vedana auto-creates structural indexes, but high-traffic filter columns (`price`, `status`) need manual `CREATE INDEX`.
- **Production hardening** for your specific deployment — backups, multi-tenant guarantees, custom auth: see [Operations / Deployment](../operations/deployment.md).

### When DIY is genuinely the right call

- You have a one-shot prototype that will live for ≤3 months and never need a backoffice.
- Your domain is purely textual (no structured catalog, no relationships) — classic RAG over pgvector is enough; Vedana's graph-first design doesn't earn its weight.
- You have hard constraints that make Vedana's stack a non-starter (no Docker allowed in prod, no pgvector permitted, must use a specific vector DB Vedana doesn't yet support).
- You're the rare team with a dedicated platform group already maintaining a similar internal framework. Then your "DIY" is really "extend our existing platform" — different math.

For everyone else, the realistic question isn't *can we build this* (you can), it's *do we want this on our maintenance plate for the next two years*.

---

## What's next

- [Use Cases](./use-cases.md)
- [Quality Metrics & Evaluation](./evaluation.md)
- [Limitations](./limitations.md)
