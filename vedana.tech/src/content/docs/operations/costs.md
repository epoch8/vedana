---
title: Cost Management
section: Operations
order: 4
---

# Cost Management

Vedana is mostly LLM. The lion's share of costs is tokens at the LLM provider. This document covers where the budget leaks, how to measure, and how to reduce spend.

## Where the money goes

In a typical production case, costs are roughly:

| Item                                       | Share        |
| ------------------------------------------ | ------------ |
| LLM (main model + filtering)               | 80–90%       |
| Embeddings                                  | 5–15%       |
| Hosting (Postgres / Memgraph / app)        | 1–5%         |

So start optimisation with LLM cost.

## Where to look

Prometheus metrics:

- `llm_calls_total{model}` — number of calls;
- `llm_usage_prompt_tokens_total{model}` — prompt tokens consumed;
- `llm_usage_completion_tokens_total{model}` — completion tokens.

In each answer's technical info (`rag.query_processed.event_data.technical_info.model_stats`) you have full per-model usage stats including `cached_tokens` and `requests_cost` (when LiteLLM gets it from the provider — the field name on `ModelUsage` has a trailing `s`, see `jims_core/llms/llm_provider.py:54-60`).

In Sentry / Datadog you can build a "cost per request" dashboard via recording rules.

## Reduction strategies

### 1. A small model for filtering

The cheapest optimisation. Set `FILTER_MODEL=gpt-4.1-nano` or `gemini-2.0-flash`. This model doesn't answer the user — it only picks relevant anchors / links for the main model's context. Simple task → cheap model.

### 2. Two-tier system

- Simple / frequent questions — cheap model (mini/nano).
- Complex / rare questions — expensive model (gpt-4.1, claude-sonnet).

Routing — through an explicit classification step (you can add it as a pre-step in your `Pipeline`).

### 3. Caching

LiteLLM supports caching via `use_cache=True`. In Vedana it's off by default in the main pipeline (RAG answers must rely on fresh data), but for static answers (smalltalk, FAQ) you can enable it by overriding the pipeline.

Provider-side caching:

- OpenAI prompt caching (auto, when the first part of the messages is shared) — already works for Vedana because the system prompt is large and shared.
- Anthropic prompt caching — supported by LiteLLM; gives a 90% discount on cached parts.

The `cached_tokens` metric tells you whether it's working.

### 4. Compress the data model

See [Customizing Prompts → compact rendering](../guides/customizing-prompts.md#example-compact-data-model-rendering). Drop `query` fields from the description (the LLM generates Cypher itself anyway). Can shrink the context 2–3x.

### 5. Keep data model filtering on

`ENABLE_DM_FILTERING=true` (default). A small model quickly picks relevant anchors → the main model sees a compact context. The bigger the data model, the bigger the savings.

### 6. Lower `pipeline_history_length`

`PIPELINE_HISTORY_LENGTH=20` by default. If sessions are short and history isn't needed — set 10. Each segment of history goes into every request.

### 7. Trim the playbook

The shorter the `query_example`, the fewer tokens. Don't write a README in the playbook — write concrete steps.

### 8. Use OpenRouter with auto-fallback

OpenRouter lets you:

- use multiple providers under one key;
- configure fallback chains (if the main model is down, fall back to the next);
- get prices below OpenAI on some models.

Configuration: `OPENROUTER_API_KEY` + `MODEL=openrouter/anthropic/claude-3.5-sonnet`.

### 9. Local models for non-critical tasks

Through LiteLLM you can plug in Ollama / vLLM / TGI. Suited to:

- embeddings (`text-embedding-3-large` is expensive; a local `nomic-embed-text` or `bge-large` saves a lot);
- filtering (if you have a GPU and the `qwen-2.5-7b-instruct` model).

Not suited to the main production model — tool-calling quality is still lower.

### 10. Per-user limits

For public APIs, add rate limits and a budget per `contact_id`. Otherwise one bad actor can eat your monthly budget.

## Embeddings

| Action                  | Cost                                      |
| ----------------------- | ------------------------------------------ |
| One embedding request   | ~$0.0001 on text-embedding-3-large         |
| Full graph re-encode    | num_chunks × 0.0001                        |
| Embeddings model change | re-encode everything                       |

Embeddings are usually pennies during ingest. The bill becomes noticeable only when you have millions of chunks or change the model often.

What helps:

- text-embedding-3-small (3x cheaper, slightly worse quality);
- local models (nomic, bge, e5);
- incremental ETL — Datapipe re-encodes only what changed;
- turn off embeddable on attributes that don't need vector search.

## Hosting

| Component   | Cheaper                                                                |
| ----------- | ---------------------------------------------------------------------- |
| Postgres    | Managed (Supabase / Neon / RDS) — free tier for prototyping.          |
| Memgraph    | Self-hosted on a single VPS — bottlenecked by RAM. Managed is more expensive. |
| Backoffice  | Spot/preemptible instances, since it's not prod-critical.             |
| API / widget | Auto-scaling under load.                                               |

## Budget baseline

A minimal starting setup:

- Postgres: managed free / VPS $20/month.
- Memgraph: VPS 16GB RAM ~ $40/month.
- Application + widget + api: $20–40/month.
- LLM: $50–200/month for a 10–30 person team using the assistant daily.

Total: $130–300/month for ~ 1000 requests/day.

## Optimisation checklist

- [ ] `FILTER_MODEL` is small (nano/mini).
- [ ] Prompt caching is on (via the provider).
- [ ] Data model rendering templates are compact.
- [ ] `pipeline_history_length` is tuned.
- [ ] embed_threshold is tuned (no extra results).
- [ ] `llm_usage_*` metrics are graphed and alerted on growth.
- [ ] Per-user rate limit / budget is set on the API.
- [ ] The golden dataset is stable — no regressions that require re-tuning.

## What's next

- [Monitoring & Metrics](./operations/monitoring.md)
- [Customizing Prompts](../guides/customizing-prompts.md)
- [Configuration Reference](../api/configuration-reference.md)
