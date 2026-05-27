---
title: Troubleshooting
section: Operations
order: 3
---

# Troubleshooting

A collection of common problems and how to fix them.

## Startup

### `docker compose up` — `db-migrate` fails on pgvector

**Symptom:** the Alembic migration `2dfad73e5cce_move_emb_to_pgvector` fails with `extension "vector" is not available`.

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

**Symptom:** the logs show `Data model filtering failed: ...`. Chat answers become slower and worse.

**Cause:** the data model filtering step crashed; the pipeline fell back to the full model.

**Things to check:**

- the `FILTER_MODEL` is reachable and didn't error out;
- the data model isn't too big for structured output (if there are too many descriptions, the LLM may fail to comply);
- check `dm_filter_prompt` / `dm_filter_user_prompt` — if you overrode them, there may be a syntax error.

**Workaround:** set `ENABLE_DM_FILTERING=false`. That makes requests more expensive but more stable.

## Assistant answers

### The assistant gives a vague / generic answer

**Checklist:**

1. Open Details for the answer in the backoffice. Which tool calls did it make?
2. If only `vector_text_search` — the playbook is the issue: it should use cypher for a structural question. Check `Queries`.
3. If cypher ran but returned an empty result — verify there's data in the graph (Memgraph Lab).
4. If cypher didn't return the expected attribute — add/refine the `query` field in `Anchor_attributes`.

### The assistant uses the wrong anchor

**Cause:** the anchor's description is ambiguous or incomplete; the LLM doesn't know when to use it.

**Fix:** improve `description` — add specific scenarios "use this anchor when …".

### The assistant invents an attribute / link

**Cause:** the system prompt isn't strict enough about "don't go beyond the data model".

**Fix:** override `generate_answer_with_tools_tmplt` and add: "If the needed attribute/link isn't in graph_composition, don't make it up — say so".

### The assistant gives the right answer but in the wrong format

**Cause:** the format isn't pinned in the playbook or the system prompt.

**Fix:** in the Queries for the corresponding intent, specify a step "Format the answer as: …" — a concrete template.

## Vector search

### Vector search never returns results

**Checklist:**

1. Is the attribute `embeddable=true`?
2. Did ETL run `generate_embeddings` without errors?
3. Are there rows for that attribute in `rag_anchor_embeddings`?
4. embed_threshold isn't cranked to 0.95+ (try 0.5 to rule out the threshold).
5. Embedding dimensions match (`EMBEDDINGS_DIM` in .env vs the model's actual dimensionality).

### Vector search returns junk

**Causes:**

- threshold too low → raise to 0.7+;
- an attribute is embeddable that shouldn't be (e.g. a numeric code) → drop embeddable;
- embeddings were built on the previous model after `EMBEDDINGS_MODEL` was changed → rebuild embeddings.

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
2. If it's LLM latency — look at traces. Maybe additional tool-call iterations crept in.
3. If it's Memgraph latency — add indexes (`CREATE INDEX ON :Product(price)` for frequently-filtered fields).
4. If it's pgvector latency — add an HNSW index on `rag_anchor_embeddings.embedding` manually (`CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)`). Vedana's Alembic migrations do **not** create this index automatically — its optimal parameters depend on your row count, dimensionality, and recall/latency target, so it's left as an explicit operator decision.

### Very expensive requests (lots of tokens)

**Causes and fixes:**

- **Data model filtering is disabled.** `ENABLE_DM_FILTERING=true` is the default and is the right setting in production — keep it on. Only turn it off (`false`) if you're debugging the agent's behaviour with the full data model in context, or if your data model is so small that the filtering step costs more than it saves. If you've explicitly set it to `false`, set it back to `true`. (Same recommendation appears in [Cost Management](./costs.md#5-keep-data-model-filtering-on) — the two pages agree: keep it on by default.)
- Large data model in context → override rendering templates to drop `query` fields.
- `pipeline_history_length` too large → drop to 10–15.
- Many retry iterations → improve the playbook so the agent picks the right path immediately.

## Sentry / Logs

### Sentry isn't getting errors

**Causes:**

- `SENTRY_DSN` is empty or wrong.
- The `--enable-sentry` CLI flag wasn't passed.
- Errors are swallowed inside the pipeline (`RagPipeline` catches exceptions and writes via `logger.exception`, but Sentry sees them via the handler).

**Check:** temporarily add `raise Exception("test")` to the pipeline and verify it reaches Sentry.

### Prometheus metrics are empty

**Causes:**

- `--metrics-port` isn't exposed.
- Prometheus scrape config isn't set up.
- The service hasn't received any requests yet (metrics appear after the first call).

## What's next

- [Monitoring & Metrics](./operations/monitoring.md)
- [Cost Management](./operations/costs.md)
