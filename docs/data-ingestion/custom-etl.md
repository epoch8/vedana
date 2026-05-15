---
title: Custom ETL
section: Data Ingestion
order: 5
---

# Custom ETL

When Grist + the default ETL aren't a fit — large volumes, streaming, sources from external systems — you write your own ETL. Vedana is designed so this is a natural extension point.

## What to change, and what not to

✅ You can change:

- the source of the raw data (instead of Grist — your CRM, ERP, S3, API);
- normalisation logic (`prepare_nodes`, `prepare_edges`);
- the embeddings store (if you want Pinecone / Weaviate / Qdrant instead of pgvector);
- triggers (cron, event queue, webhooks).

❌ Don't change:

- the **data model schema** (anchors / links / attributes / queries / prompts) — it stays the source of truth and describes the contract between the data and the LLM;
- the `Graph` / `VectorStore` interface (or change them in lockstep with `vedana-core`);
- the format of writes to Memgraph (`GENERIC_NODE_DATA_SCHEMA`, `GENERIC_EDGE_DATA_SCHEMA`).

## Extension points

### 1. Custom data source

Replace `grist_steps` with your own `BatchGenerate` steps. Return DataFrames matching the catalog schemas — see `GENERIC_NODE_DATA_SCHEMA` / `GENERIC_EDGE_DATA_SCHEMA` in `vedana_etl/schemas.py`. Edges require **six columns** (`from_node_id`, `to_node_id`, `from_node_type`, `to_node_type`, `edge_label`, `attributes`); the four `*_type` and `*_id` columns together form the primary key and Vedana does **not** infer `from_node_type` / `to_node_type` for you.

```python
import pandas as pd
from datapipe.compute import Pipeline
from datapipe.step.batch_generate import BatchGenerate

from vedana_etl.catalog import grist_nodes, grist_edges

def get_my_data():
    nodes_df = pd.DataFrame([
        {"node_id": "p-001", "node_type": "product", "attributes": {"name": "Laptop", "price": 999.00}},
        ...
    ])
    edges_df = pd.DataFrame([
        {
            "from_node_id": "p-001",
            "to_node_id": "cat-01",
            "from_node_type": "product",
            "to_node_type": "category",
            "edge_label": "PRODUCT_belongs_to_CATEGORY",
            "attributes": {},
        },
        ...
    ])
    yield nodes_df, edges_df

my_steps = [
    BatchGenerate(
        func=get_my_data,
        outputs=[grist_nodes, grist_edges],
        labels=[("flow", "regular"), ("source", "MyCRM")],
    ),
]
```

### 2. Custom normalisation

If your source returns data in a special format, replace `default_custom_steps` (see `vedana_etl.pipeline`):

```python
from datapipe.step.batch_transform import BatchTransform
from vedana_etl.catalog import nodes, edges

def my_prepare_nodes(grist_nodes_df):
    df = grist_nodes_df.copy()
    df["node_id"] = df["node_id"].str.lower()
    # ... your logic ...
    return df

custom_steps = [
    BatchTransform(
        func=my_prepare_nodes,
        inputs=[grist_nodes],
        outputs=[nodes],
        transform_keys=["node_id"],
    ),
    BatchTransform(
        func=my_prepare_edges,
        inputs=[grist_edges],
        outputs=[edges],
        transform_keys=["from_node_id", "to_node_id", "edge_label"],
    ),
]
```

### 3. Assembling a custom pipeline

```python
from vedana_etl.pipeline import data_model_steps, memgraph_steps, eval_steps
from datapipe.compute import Pipeline

def get_my_pipeline() -> Pipeline:
    return Pipeline([
        *data_model_steps,   # data model still from Grist
        *my_steps,            # your data ingest
        *custom_steps,        # your normalisation
        *memgraph_steps,      # standard load into Memgraph + embeddings
        *eval_steps,          # evaluation stays
    ])
```

And register your `app`:

```python
# my_etl/app.py
from datapipe.compute import build_compute, run_steps
from my_etl.pipeline import get_my_pipeline

pipeline = get_my_pipeline()
```

Run:

```bash
DATAPIPE_PIPELINE=my_etl.app uv run python -m datapipe run
```

### 4. Alternative vector store

If you're moving from pgvector to another solution (Pinecone, Qdrant, Weaviate, Milvus, Memgraph vector index), do two things:

**a)** Implement `VectorStore` (`libs/vedana-core/src/vedana_core/vts.py`):

```python
from vedana_core.vts import VectorStore

class MyVectorStore(VectorStore):
    async def vector_search(self, label, prop_type, prop_name, embedding, threshold, top_n=5):
        # your client
        ...
        return results  # format: list of records with similarity, node_id, attributes keys
```

**b)** Replace `PGVectorStore` in `make_vedana_app()`:

```python
@alru_cache
async def make_vedana_app() -> VedanaApp:
    ...
    vts = MyVectorStore(...)
    pipeline = RagPipeline(graph=graph, vts=vts, ...)
    ...
```

And in ETL drop the `generate_embeddings` steps (the last two `BatchTransform`s in `memgraph_steps`, see `libs/vedana-etl/src/vedana_etl/pipeline.py`) and replace them with writes to your store. Either redefine `memgraph_steps` from scratch, or build a custom list that reuses Vedana's index/load steps and adds your own embedding writer:

```python
from vedana_etl.pipeline import (
    nodes, edges,
    dm_anchor_attributes, dm_link_attributes,
    memgraph_anchor_indexes, memgraph_link_indexes,
    memgraph_nodes, memgraph_edges,
)
from vedana_etl import steps
from datapipe.compute import BatchTransform

custom_memgraph_steps = [
    BatchTransform(
        func=steps.ensure_memgraph_node_indexes,
        inputs=[dm_anchor_attributes],
        outputs=[memgraph_anchor_indexes],
        transform_keys=["attribute_name"],
    ),
    BatchTransform(
        func=steps.ensure_memgraph_edge_indexes,
        inputs=[dm_link_attributes],
        outputs=[memgraph_link_indexes],
        transform_keys=["attribute_name"],
    ),
    BatchTransform(
        func=steps.pass_df_to_memgraph,
        inputs=[nodes],
        outputs=[memgraph_nodes],
        transform_keys=["node_id", "node_type"],
    ),
    BatchTransform(
        func=steps.pass_df_to_memgraph,
        inputs=[edges],
        outputs=[memgraph_edges],
        transform_keys=["from_node_id", "to_node_id", "edge_label"],
    ),
    # Replaces the two generate_embeddings steps:
    BatchTransform(
        func=write_to_my_vector_store,           # your function
        inputs=[nodes, dm_anchor_attributes],
        outputs=[my_vts_marker],                 # your output table
        transform_keys=["node_id", "node_type"],
    ),
]
```

Pass `custom_memgraph_steps` instead of `memgraph_steps` when constructing the `Pipeline`. The first four `BatchTransform`s are the unchanged Vedana steps; only the embedding writer is replaced.

### 5. Streaming ETL

Datapipe supports incremental computation: a re-run of the pipeline recomputes only changed rows. For continuous loading:

- run `datapipe run` on cron or via queue events;
- keep a trigger table with `last_processed_at` and filter the source by it;
- for true real-time — set up your own worker that listens to Kafka/SQS/RabbitMQ and writes to `grist_nodes`/`grist_edges` in real time, while periodically running `datapipe run` to update Memgraph.

### 6. Alternative data model source

By default, the data model also comes from Grist. If you have your own source (a YAML file in the repo, a separate API), replace `data_model_steps`:

```python
def get_my_data_model():
    # return a tuple of DataFrames as in vedana_etl.steps.get_data_model
    ...

my_dm_steps = [
    BatchGenerate(
        func=get_my_data_model,
        outputs=[dm_anchors, dm_anchor_attributes, dm_link_attributes, dm_links, dm_queries, dm_prompts, dm_conversation_lifecycle],
        labels=[("flow", "regular"), ("source", "git")],
    ),
]
```

That's it. `vedana_core.data_model.DataModel` will read the `dm_*` tables as usual — it doesn't care who populated them.

## Best practices

- **Version the schemas.** If you change the format of `node_type` or `edge_label`, write a migration for old data — don't just say "this is how it works now".
- **Log drift.** Count how many rows you loaded today vs yesterday. Sudden drops should trigger an alert.
- **Test on a golden dataset** before and after ETL changes — that's your main safety net.
- **Don't try to auto-generate the data model.** "Auto-inferring" anchors from columns leads to vague descriptions, which lead to bad LLM answers. Better to describe a little less but well.
- **Keep idempotency.** Datapipe gives you that by default, but if you write to Memgraph directly via `add_node`, use `MERGE`, not `CREATE`, or you'll get duplicates.

## What's next

- [Vedana ETL architecture](../architecture/vedana-etl.md) — how the default pipeline is built.
- [Data Model overview](../data-model/overview.md) — what the data model is.
