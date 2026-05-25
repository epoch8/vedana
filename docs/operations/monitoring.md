---
title: Monitoring & Metrics
section: Operations
order: 2
---

# Monitoring & Metrics

Vedana exposes Prometheus metrics and OpenTelemetry traces. This document covers the key metrics, recommended dashboards, and the alerts to set.

See also: [Architecture → Observability](../architecture/observability.md).

## Key metrics

### LLM (cost & throughput)

| Metric                                | What it shows                                |
| -------------------------------------- | --------------------------------------------- |
| `llm_calls_total{model}`                | number of LLM calls per model                |
| `llm_usage_prompt_tokens_total{model}`  | how many prompt tokens were used              |
| `llm_usage_completion_tokens_total{model}` | completion tokens                          |

**Dashboard:**

- request rate per minute per model;
- USD cost (via a recording rule: `tokens × $/1k_tokens`);
- share of cached_tokens (when the provider supports it);
- top-5 models by traffic.

**Alerts:**

- `rate(llm_usage_prompt_tokens_total[1h]) > X` — sudden cost spike;
- a >80% drop over 1h — your provider may have died or you're missing traffic.

### Pipeline (latency & reliability)

| Metric                                              | What it shows                              |
| ---------------------------------------------------- | -------------------------------------------- |
| `jims_pipeline_runs_total{status,pipeline}`           | number of pipeline runs                      |
| `jims_pipeline_run_duration_seconds_bucket{status,pipeline}` | duration histogram                  |

**Dashboard:**

- p50 / p95 / p99 latency for `RagPipeline` (success);
- error rate: `rate(jims_pipeline_runs_total{status="failure"}[5m])`;
- breakdown by pipeline type (`RagPipeline`, `StartPipeline`, custom ones).

**Alerts:**

- `error_rate > 1%` over 5 minutes;
- `p95_latency > 15s` over 10 minutes;
- `success_rate == 0` for 1 minute (zero traffic — something fell over).

### Database

> `postgres_exporter` and `memgraph_exporter` are **recommended optional add-ons**, not part of Vedana's repo or `apps/vedana/docker-compose.yml`. To get the metrics below you need to install and scrape them yourself.

Standard Postgres metrics through `postgres_exporter` (e.g. [prometheus-community/postgres_exporter](https://github.com/prometheus-community/postgres_exporter)):

- connections;
- query duration;
- size of `thread_events`, `rag_anchor_embeddings`;
- bloat and autovacuum activity.

Memgraph through a Memgraph metrics exporter (if used):

- query duration;
- memory usage;
- vector / text index size.

## Application logs

All components write to stdout via `loguru` / standard `logging`. In production, aggregate through Loki / ELK / Datadog.

Levels:

- `INFO` — normal operation.
- `WARNING` — something suspicious (data model filtering fell back to full DM, tool-call iteration limit hit).
- `ERROR` / `EXCEPTION` — something failed.

Useful patterns for grep / Loki queries:

- `RagPipeline` — every main pipeline run;
- `Data model filter selection` — what filtering picked;
- `Reached tool call iteration limit` — the pipeline hit 5 iterations;
- `Error executing tool` — a tool call failed;
- `vts_fn(on=...)` / `cypher_fn(...)` — debug for tool calls.

## OpenTelemetry traces

A typical trace flow (LLM spans below are illustrative; in Jaeger/Tempo you'll see span names produced by `openinference.instrumentation.litellm` — typically `litellm.completion` — not the `llm.chat_completion_*` labels):

```
jims.run_pipeline_with_context (12.4s)
├── memgraph.execute_ro_cypher_query (0.1s)
├── litellm.completion (1.2s) [filtering, structured]
├── litellm.completion (3.5s) [iter 1, with tools]
│   ├── memgraph.execute_ro_cypher_query (0.2s)
│   └── pgvector.vector_search (0.1s)
├── litellm.completion (4.1s) [iter 2, with tools]
│   └── memgraph.execute_ro_cypher_query (0.3s)
└── litellm.completion (3.0s) [iter 3 - final answer]
```

This shows where time is actually spent. The bottleneck is usually LLM calls; sometimes a slow Cypher.

Use Jaeger / Tempo / Datadog APM for visualisation.

## Business metrics

In addition to the technical metrics, it's worth counting:

- **rate of unique users** per day (`COUNT(DISTINCT contact_id)` in `threads`);
- **average thread length** (`COUNT(thread_events) / COUNT(threads)`);
- **share of `rag.error` events** — quality drops or LLM provider failures;
- **share of smalltalk** vs real questions — helps assess whether the assistant is being used as intended;
- **CSAT** — if you give the user a "useful/not" button, log it as `comm.user_feedback`. Note: this is **not a built-in event type** — Vedana doesn't produce `comm.user_feedback` itself. You add the rating capture in your interface (widget / Telegram / your API) and write it through `ctx.send_event("comm.user_feedback", {...})` or `ctl.store_event_dict(...)`. Pick any event_type you like; `comm.user_feedback` is just a convention.

## Sampling real conversations

The most valuable thing for the product team — once a week, take a random sample of 20–50 threads and do manual review:

- did the assistant answer correctly?
- did it use the right tool?
- was the answer useful for the user?

This ritual gives qualitative feedback that the golden dataset can't.

## Alerting

A minimal alert set for production:

| Alert                                                     | Severity | Condition                                                       |
| --------------------------------------------------------- | -------- | --------------------------------------------------------------- |
| Pipeline error rate > 1%                                   | high     | `rate(jims_pipeline_runs_total{status="failure"}[5m]) > 0.01` |
| Pipeline p95 > 15s                                         | medium   | `histogram_quantile(0.95, jims_pipeline_run_duration_seconds_bucket) > 15` |
| LLM tokens spike (>2x baseline)                            | medium   | `rate(llm_usage_prompt_tokens_total[15m]) > 2 * baseline`     |
| Postgres connections > 80%                                  | high     | `pg_stat_activity_count > 0.8 * max_connections`               |
| Memgraph memory > 80%                                       | high     | `process_resident_memory_bytes > 0.8 * limit`                  |
| API healthcheck down                                       | critical | `up{job="vedana-api"} == 0` for 1 minute                       |
| Sentry: rate of new errors > 10/min                         | medium   | via Sentry alert rules                                           |

## Dashboards (recommendations)

### Ops dashboard

- request rate (per service);
- latency p50/p95/p99 (per service);
- error rate;
- DB connections / query latency;
- memory / CPU.

### LLM dashboard

- tokens per minute (per model);
- cost per minute (per model);
- cache hit ratio;
- top scenarios by cost.

### Product dashboard

- DAU / MAU;
- average thread length;
- `rag.error` share;
- smalltalk share;
- CSAT (if available).

## What's next

- [Troubleshooting](./operations/troubleshooting.md)
- [Cost Management](./operations/costs.md)
