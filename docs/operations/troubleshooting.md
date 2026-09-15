---
title: Troubleshooting
section: Operations
order: 3
---

# Troubleshooting

A collection of common problems and how to fix them.

## Startup

### `docker compose up` — `db-migrate` fails on pgvector

**Symptom:** the pgvector Alembic migration fails with `extension "vector" is not available`.

**Cause:** the Postgres provider doesn't allow auto-creating extensions.

**Fix:**

- Self-hosted Postgres → make sure the image is `pgvector/pgvector:pg15` (or another with pgvector pre-installed). Set `CREATE_PGVECTOR_EXTENSION=true`.
- Managed (Supabase / Neon / RDS) → create the extension manually via the provider console, set `CREATE_PGVECTOR_EXTENSION=false`.

### `Memgraph: Authentication failed`

**Symptom:** at startup the `app` logs say it can't log into Memgraph.

**Cause:** the password in `.env` doesn't match what's stored in Memgraph.

**Fix:** verify `MEMGRAPH_USER`, `MEMGRAPH_PWD` in `.env` — they must match the values used on the first Memgraph start (Memgraph stores auth across restarts).

If you've recreated Memgraph with a different password — drop the `mg_graph` volume (`docker compose down -v`); it'll be re-created with the new password.

### Grist doesn't open on 8484

**Symptom:** `http://localhost:8484` is empty or complains.

**Fix:**

- Check `docker compose ps` — is `grist` healthy?
- In the logs (`docker compose logs grist`) check for errors. Often it's volume permissions: `/persist` must be writable.
- Restart the service: `docker compose restart grist`.

## ETL

### ETL ran but Memgraph is empty

**Things to check:**

1. Backoffice → ETL logs: which steps succeeded, which failed?
2. In Postgres check `nodes` and `edges`:
   ```sql
   SELECT COUNT(*) FROM nodes;
   SELECT COUNT(*) FROM edges;
   ```
   If empty — data didn't reach from Grist (error in `get_grist_data` / `prepare_*`).
3. If Postgres has data but Memgraph doesn't — the issue is in the `pass_df_to_memgraph` step. Check Memgraph logs; the `bolt://` URI may be wrong.

### ETL works but embeddings aren't built

**Symptom:** rows don't appear in `rag_anchor_embeddings`.

**Causes:**

- No attribute in the data model has `embeddable=true`.
- The LLM provider doesn't respond (check `OPENAI_API_KEY`).
- Provider rate limit exceeded (LiteLLM logs this).
- The attribute is embeddable but values in the data are empty / NULL.

### `Data model filtering failed` in the logs

**Symptom:** `Data model filtering failed: ...`. The pipeline fell back to the full model.

**Check:** `FILTER_MODEL` is reachable; if you overrode `dm_filter_prompt` / `dm_filter_user_prompt`, they still format. To disable the step, set `ENABLE_DM_FILTERING=false`.

### Data model load fails with `Unsupported tool_choice values` / `Unsupported max_iters values`

A Queries row has an invalid `tool_choice` or `max_iters`. Correct the cells (blank is the default) and re-run `data_model_steps`. See [Queries](../data-model/queries.md#fields).

## Scenario routing and playbook

Wrong scenario, empty `rag.scenario_routing.matches`, or every request falling through to the LLM step: see [Scenario routing → Tuning](../data-model/scenario-routing.md#tuning).

### The agent says a tool is "not available in the current scenario"

The active scenario's `tools_allowed` doesn't list that tool. Names are case-sensitive (`mcp__catalog__ask`). Add it and reload the data model.

Vague, wrong-format, or invented answers are playbook / prompt issues — see [Queries](../data-model/queries.md) and [Quality Metrics & Evaluation](../product/evaluation.md).

## Vector search

### Vector search never returns results

**Checklist:**

1. Is the attribute `embeddable=true`?
2. Did ETL run `generate_embeddings` without errors?
3. Are there rows for that attribute in `rag_anchor_embeddings`?
4. `embed_threshold` isn't cranked to 0.95+ (try 0.5 to rule out the threshold).
5. Embedding dimensions match (`EMBEDDINGS_DIM` in `.env` vs the model's actual dimensionality).

### Vector search returns junk

**Causes:**

- threshold too low → raise to 0.7+;
- an attribute is embeddable that shouldn't be (e.g. a numeric code) → drop embeddable;
- embeddings were built on the previous model after `EMBEDDINGS_MODEL` was changed → rebuild embeddings.

See [Tuning Embeddings & Thresholds](../guides/tuning-embeddings.md).

## Cypher

### The LLM generates broken Cypher

**Symptom:** in the logs `Error executing tool cypher: ...`.

**Causes:**

- Cypher templates in the `query` fields of anchors/links are wrong → the LLM learned from them.
- Label names don't match what's in Memgraph (e.g. `Product` vs `product`).
- The LLM gets confused by named edge labels — verify the link `sentence` matches what's in Memgraph.

**What to do:**

1. Open Memgraph Lab → run the query manually to see the syntax error.
2. Fix the `query` in the data model so the LLM has the right template.
3. If the LLM keeps producing the same wrong pattern — add a Queries scenario with an explicit instruction.

### Cypher works in Memgraph Lab but fails in chat

**Probable cause:** the LLM puts single quotes inside a string Cypher doesn't accept (or wrong escaping).

**Fix:** in the playbook, specify that parameters are passed via `$param`, not interpolated into the string.

## Performance

### Requests became very slow

**Checklist:**

1. `jims_pipeline_run_duration_seconds` metrics — where does it grow?
2. If it's LLM latency — look at traces. Extra tool-call rounds are a common cause.
3. If it's Memgraph latency — add/check indexes on frequently filtered fields (`CREATE INDEX ON :Label(property)`).
4. If it's pgvector latency — add/check an HNSW index (see [Limitations](../product/limitations.md#possible-bottlenecks-at-scale)).

Token spend: [Cost Management](./costs.md).

## Sentry / Logs

### Sentry isn't getting errors

**Causes:**

- `SENTRY_DSN` is empty or wrong.
- The `--enable-sentry` CLI flag wasn't passed.

### Prometheus metrics are empty

**Causes:**

- `--metrics-port` isn't exposed.
- Prometheus scrape config isn't set up.
- The service hasn't received any requests yet (metrics appear after the first call).

## What's next

- [Monitoring & Metrics](./monitoring.md)
- [Cost Management](./costs.md)
- [Scenario routing](../data-model/scenario-routing.md)
