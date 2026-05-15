---
title: Product FAQ
section: Product
order: 6
---

# FAQ (product)

A collection of frequently asked questions about Vedana — for product managers, engineers, and folks just discovering Semantic RAG.

## What is Vedana in one sentence?

An open-source framework for AI assistants whose answers can be verified: instead of "guessing from text", Vedana queries a structured knowledge graph and knows where every fact came from.

## What licence is Vedana released under? Can we use it commercially?

Vedana is released under the **[Apache License 2.0](https://github.com/epoch8/vedana/blob/main/LICENSE)**. This is a permissive licence that explicitly allows:

- **Commercial use** — including in closed-source / proprietary products.
- **Modification** — fork it, change it, ship it.
- **Redistribution** — bundle it into your own product or distribution.
- **Private use** — use it inside your company without publishing anything.

Requirements are minimal: keep the copyright and licence notice, and state any significant changes you make. Apache 2.0 also includes an **explicit patent grant** from contributors, which is what most enterprise legal teams care about (it's the main reason Apache 2.0 is preferred over MIT for B2B adoption).

Vedana itself bundles a few third-party components (Memgraph, Grist, Postgres, pgvector, Reflex) — each ships under its own licence; review them for your deployment if you redistribute the full stack.

## How is Vedana different from LangChain / LlamaIndex / Haystack?

These are tools of different kinds. LangChain / LlamaIndex / Haystack are **libraries for classic RAG** — you assemble the pipeline yourself. Vedana is a **ready framework for Semantic RAG** with a data model, knowledge graph, ETL, and evaluation. Put another way, Vedana = a pre-built pattern of "document RAG + structural graph + playbook" with production infrastructure around it.

## Can Vedana be used without Memgraph / Neo4j?

Technically — no, you do need a graph DB. But `Graph` is an abstract class, and you can implement your own backend on top of any Cypher-compatible DB. For Neo4j the existing `CypherGraph` should largely work (the Bolt driver is shared with Memgraph). AWS Neptune supports only the [openCypher subset](https://docs.aws.amazon.com/neptune/latest/userguide/opencypher.html) — not full Cypher — so plugging Neptune in realistically means subclassing `Graph` (not just `CypherGraph`) and adapting the query patterns Vedana's `data_model.query` fields generate. See [Vedana Core architecture](../architecture/vedana-core.md).

## Can it run without Postgres?

JIMS tables and Datapipe require Postgres in production. Migrations are written for Postgres + pgvector. SQLite is **not** a tested production target — the only SQLite usage in the repo is the in-memory test database in `libs/vedana-core/tests/test_data_model.py` for unit tests of `DataModel`. The `jims_core/db.py` schema has a `JSON.with_variant(sa.JSON, "sqlite")` shim but the migrations and `vedana_etl.store` aren't validated against SQLite, and vector search requires pgvector. Treat "works on SQLite for an MVP" as an aspiration, not a guarantee.

## Can it run without Grist?

Yes. Grist is the default data model and data source, but it's not a required part. See [Custom ETL](../data-ingestion/custom-etl.md). For example, you can keep the data model in a YAML file in the repo.

## Which LLM provider is best?

Vedana runs on LiteLLM, so theoretically — any. In practice:

- **OpenAI (gpt-4.1, gpt-4.1-mini)** — a stable baseline; the project was assembled on it.
- **Anthropic Claude** — excellent at tool-calling, especially for Cypher generation.
- **Google Gemini** — large context, cheap, sometimes worse at following step-by-step playbooks.
- **Local models via Ollama** — workable for on-prem, but tool-calling quality is much lower.

For production we recommend `gpt-4.1-mini` or `claude-3.5-haiku` as the main model, plus a small model (`gpt-4.1-nano`, `gemini-2.0-flash`) for filtering. The shipped `apps/vedana/.env.example` sets `MODEL="gpt-4.1-mini"`, so the recommended default and the file you copy from match. The class-level fallbacks in `VedanaCoreSettings` (`gpt-4.1`) and `LLMSettings` (`gpt-4.1-nano`) are safety nets for when the env var isn't set at all — for any real deployment, set `MODEL` explicitly in `.env`.

## How much data can Vedana handle?

Depends on Memgraph and Postgres, not on Vedana itself:

- 100k nodes + 1M edges — comfortable on a single instance.
- 1M nodes + 10M edges — Memgraph with ~ 32–64 GB RAM is needed.
- 10M+ — cluster Memgraph configs + sharded embeddings.

Vedana isn't designed for billions of nodes. If you have Big Data, use it as a metadata layer on top of your DWH, not as the primary store.

## Can I get a streaming response?

In the HTTP API today — no, the response is returned all at once after the full pipeline completes (`POST /api/v1/chat` returns a `ChatResponse` JSON in one shot; there is no SSE / chunked transfer). The web widget (`jims-widget`) currently behaves the same way: the WebSocket on `/ws/chat` returns a single `{"text": ...}` (or `{"error": ...}`) frame after the pipeline finishes — there are no intermediate status events and no token-by-token streaming yet. Token-by-token streaming is tracked in [GitHub Issues](https://github.com/epoch8/vedana/issues).

## How much does a request cost?

Depends on model and complexity. On `gpt-4.1-mini`:

- a simple structural query (1 cypher) — 0.001–0.002 USD;
- a document query (1 vector search + retrieval) — 0.002–0.005 USD;
- a complex multi-hop with data model filtering — 0.005–0.015 USD.

On `gpt-4.1` — an order of magnitude more expensive. See `llm_usage_*_tokens_total`.

## What does a typical adoption roadmap look like?

**Week 1:**
- Bring up the stack via Quick Start.
- Load documents.
- Get a working "document Q&A".

**Weeks 2–3:**
- Describe the first 3–5 anchors of your domain.
- Load structured data.
- Write a playbook for the main intents.
- Compile a golden dataset (50–100 questions).

**Week 4:**
- Run eval, analyse failures.
- Tune prompts and thresholds.
- Roll out a beta to the team.

**Weeks 5–8:**
- Extend the data model for growing scenarios.
- Regular eval iterations.
- Plug in additional interfaces (Telegram / widget).

**Month 3+:**
- Stable production operation.
- Custom tools for domain-specific scenarios.
- Multi-tenancy if needed.

## How much does adoption cost?

In the open-source variant — only infrastructure (LLM API + hosting ~ $50–500 per month at the start) and team time. On a small domain, one person can do it in 2–4 weeks.

For commercial deployments with Epoch8 team support — contact them directly (contacts on [vedana.tech](https://vedana.tech)).

## Common Mistakes

### Q: The assistant doesn't use graph search and gives vague answers.

**A:** The most common cause is empty `query` fields in Anchors / Attributes / Links. If they're empty, the assistant can't reliably navigate the graph and falls back to less precise methods. Fill in the Cypher queries on every anchor and attribute that should be queryable.

### Q: I described Queries but the assistant doesn't follow the expected steps.

**A:** Check the `query_example` field — it must be **concrete and step-by-step**, not a generic problem statement. Instead of "find the interest and return connected people", write: "1) Use vector_text_search(label="interest", property="interest_name", text='<interest>'), retrieve node_id. 2) Use Cypher: MATCH ...". The more precise the steps, the more stable the behaviour.

### Q: Answers are correct, but the tone / format doesn't fit our product.

**A:** That's controlled by the system prompt, not the data model. Open **Data Model > Prompts** and override `generate_answer_with_tools_tmplt`. Small change, noticeable effect.

### Q: ETL ran but Memgraph is empty.

**A:** Check the backoffice → ETL logs. Common causes:

- Grist API key is wrong → no data.
- A foreign key in the source table references an ID that doesn't exist in the target table → the edge isn't built.
- `dtype` in the model doesn't match the actual format → the row is skipped.

### Q: Vector search returns nothing, but data clearly exists.

**A:** The embed_threshold is probably too high. Lower it to 0.6–0.65 to test and watch whether anything is returned. Also check that the attribute is really `embeddable=true` and that ETL ran the `generate_embeddings` step.

## Anything else?

If your question isn't covered, open an issue at [github.com/epoch8/vedana](https://github.com/epoch8/vedana/issues) tagged `question` or join a discussion.

## What's next

- [Use Cases](./product/use-cases.md)
- [Limitations](./product/limitations.md)
- [GitHub Issues](https://github.com/epoch8/vedana/issues) — open roadmap items.
