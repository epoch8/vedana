---
title: Tuning Embeddings & Thresholds
section: Guides
order: 9
---

# Tuning Embeddings & Thresholds

The quality of semantic search in Vedana depends on three things:

1. which embeddings model you use;
2. what you make embeddable;
3. what threshold is set on each embeddable field.

This guide is about practical tuning of those parameters.

## Embeddings model

Set globally via `EMBEDDINGS_MODEL` (default `text-embedding-3-large`). Dimension — `EMBEDDINGS_DIM` (default 1024).

Important:

- **Don't change the dimension on the fly.** It requires a SQL migration and recomputing all embeddings.
- If budget matters, `text-embedding-3-small` is faster and cheaper but quality is lower.
- Local models (via VertexAI or OpenRouter) can give wildly different quality — after a model change, always run the golden dataset.

## What to make embeddable

| Field type                  | embeddable | Why                                                                  |
| --------------------------- | ----------- | --------------------------------------------------------------------- |
| Entity names                 | ✅          | Users often write with typos / abbreviations                          |
| Descriptions, titles         | ✅          | Meaning-based search is the whole point of vector search             |
| Addresses                    | ✅          | "store on Nevsky" instead of an exact address                         |
| FAQ questions                | ✅          | The main field for matching                                            |
| Document chunks (`content`)  | ✅          | The whole point of document RAG                                       |
| Numeric values               | ❌          | An embedding for "999.00" is meaningless                              |
| ID, SKU, article numbers    | ❌          | Need exact match                                                       |
| boolean                      | ❌          | Two values aren't enough for vector space                             |
| dates                        | ❌          | Semantically meaningless — you need filters/comparisons               |
| enum (category, status)      | usually ❌  | Better as a link / strictly typed attribute                           |
| URL, file path               | ❌          | No semantics                                                            |

## Picking thresholds

Threshold is the cosine similarity above which a vector search result is considered relevant. Set **per attribute** in Anchor_attributes / Link_attributes.

### Starting values

| Scenario                            | Start     |
| ------------------------------------ | --------- |
| Names / exact identifiers            | 0.75–0.85 |
| Descriptions                          | 0.65–0.75 |
| Document chunks                       | 0.50–0.65 |
| FAQ                                   | 0.70–0.78 |

### Iterative tuning

1. **Run the golden dataset** at current thresholds. Record the Hit Rate and the failure list.
2. **Failure analysis:**
   - **False positive** (returned an irrelevant result) → threshold is too low.
   - **False negative** (didn't find a valid result) → threshold is too high.
3. **Change one field at a time** by ±0.05.
4. **Re-run the eval.** Compare Hit Rate.
5. Repeat until the metric plateaus.

### Special cases

- **One very common term appears in multiple anchors** → raise the threshold for that field, otherwise you get permanent conflicts.
- **Long texts vs short queries** → lower the threshold: the embedding of a long text is "diffuse".
- **Short texts vs long queries** → raise the threshold: too easy to land on a coincidental match.

## Cosine similarity vs cosine distance

In Vedana we operate in **similarity** (`1 - distance`). Closer to 1.0 means a tighter match. In pgvector this is computed via the `<=>` operator (cosine distance), and the inversion happens on the SQLAlchemy side.

In the code (`PGVectorStore.vector_search`):

```python
similarity = (1 - rag_anchor_embeddings.c.embedding.cosine_distance(embedding)).label("similarity")
stmt = (
    ...
    .where(similarity > threshold)
    .order_by(distance)
    .limit(top_n)
)
```

## `top_n` limits

`top_n` (default 5) is how many results to return per tool call. Too many — bloats LLM context and pays for extra tokens. Too few — risks missing the right one.

Start: `top_n=5`. For document questions with very specific queries — sometimes `top_n=3`. For broad exploratory questions — `top_n=10`.

Tune at the `RagPipeline(top_n=...)` level or inside the tool call (if you've rewritten the agent).

## Budget

| Action                            | Cost (order of magnitude)                                              |
| --------------------------------- | --------------------------------------------------------------------- |
| Embedding one chunk                | ~0.0001 USD on text-embedding-3-large                                 |
| Vector search                      | free (only Postgres compute)                                          |
| Reprocessing the full dataset     | `total_chunks × 0.0001` — usually pennies for thousands of documents |
| Changing `EMBEDDINGS_MODEL`        | recompute all embeddings from scratch                                  |
| Changing `EMBEDDINGS_DIM`          | recompute + SQL migration (downtime!)                                  |

The `llm_calls_total{model}` and `llm_usage_prompt_tokens_total{model}` metrics will tell you how many embeddings you actually run.

## Tuning checklist

- [ ] Golden dataset is collected and updated (at least 50 questions).
- [ ] Eval was run before changes — you have a baseline.
- [ ] One parameter at a time.
- [ ] Eval after every change.
- [ ] Track historical Hit Rate to see the trend.
- [ ] Don't confuse threshold with embeddable: `embeddable=false` doesn't use a threshold at all.

## What's next

- [Evaluation](../product/evaluation.md) — how to measure.
- [Customizing Prompts](./guides/customizing-prompts.md) — improve answers beyond retrieval.
