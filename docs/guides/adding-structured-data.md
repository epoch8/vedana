---
title: Adding Structured Data
section: Guides
order: 7
---

# Adding Structured Data

A step-by-step scenario: how to load structured data (products, contracts, branches) into Vedana.

## 1. Describe the data model

Before loading data, describe the schema in **Data Model**:

- anchors (`product`, `category`, `branch`);
- their attributes;
- the links between them.

If you don't, the data lands in the graph but the assistant won't know about it.

See [Setting Up Data Model](./guides/setting-up-data-model.md).

## 2. Prepare tables

In the **Grist Data doc**, create one table per anchor. **The table name must follow the `Anchor_<noun>` convention** (`Anchor_product`, `Anchor_category`, `Anchor_branch`, …) — `GristDataProvider` discovers anchor data by that prefix (`vedana_core/data_provider.py:69`, `anchor_table_prefix = "Anchor_"`). The prefix is hard-coded; tables not matching it are ignored. In Grist's UI tabs you can keep nicer display names alongside the underlying table id.

### `Anchor_product`

| product_id | name      | description           | price | in_stock | category_id   |
| ---------- | --------- | --------------------- | ----- | -------- | -------------- |
| p-001      | MacBook Air | M2 chip laptop, 13"   | 1199  | true     | cat-laptops    |
| p-002      | Dell XPS    | Premium ultrabook     | 1299  | true     | cat-laptops    |

The columns map to the `product` anchor's attributes. The `category_id` column will become a `PRODUCT_belongs_to_CATEGORY` edge (see step 3).

### `Anchor_category`

| category_id   | name    |
| -------------- | ------- |
| cat-laptops   | Laptops |
| cat-monitors  | Monitors|

### `Anchor_branch`

| branch_id     | name           | address                                  | opening_hours          |
| -------------- | -------------- | ----------------------------------------- | ---------------------- |
| b-vno-01       | Acme Vilnius   | Gedimino pr. 1, Vilnius                  | Mon-Fri 09-21, Sat 10-18, Sun closed |

## 3. Connect tables via Links

In **Data Model > Links**, set `anchor1_link_column_name` / `anchor2_link_column_name` for each link:

| anchor1  | anchor2  | sentence                       | anchor1_link_column_name |
| -------- | -------- | ------------------------------ | ------------------------ |
| product  | category | `PRODUCT_belongs_to_CATEGORY`  | category_id              |

This tells ETL: "Take the `category_id` column in the products table, find the node in `category` by that id, and create an edge."

## 4. Run ETL

Backoffice → ETL → **Run Selected**.

ETL sequentially:

1. Reads the data model from Grist (`data_model_steps`).
2. Reads the data from Grist (`grist_steps`).
3. Prepares nodes / edges (`default_custom_steps`).
4. Creates indexes and loads into Memgraph (`memgraph_steps`).
5. Builds embeddings for embeddable attributes (`memgraph_steps` / `generate_embeddings`).

## 5. Verify in Memgraph Lab

```cypher
// node counts per type
CALL llm_util.schema() YIELD schema RETURN schema

// product → category links
MATCH (p:product)-[:PRODUCT_belongs_to_CATEGORY]->(c:category)
RETURN p.name, c.name LIMIT 10

// a structural query check
MATCH (p:product) WHERE p.price < 1500 RETURN p.name, p.price ORDER BY p.price
```

## 6. Verify in chat

Ask questions that require structured data:

> "Show me all laptops cheaper than 1500 euros"
> "Which branch has MacBook Air in stock?"
> "Which category does Dell XPS belong to?"

In Details, Cypher should run with the right labels and filters.

## 7. Hybrid approach with documents

If you have both structured attributes and related descriptive documentation (contracts, technical specs), use both paths simultaneously:

| Anchor `contract` | Document `contract_text` |
| ------------------ | ------------------------- |
| `contract_id`, `start_date`, `end_date`, `counterparty` (attributes) | content in chunks, embeddable |

Connect them with the `CONTRACT_has_TEXT → document` link.

Then:

- "When does contract C-123 expire?" → Cypher on `contract.end_date`;
- "What does clause 5.2 of contract C-123 say?" → vector search on `document_chunk.content` + Cypher on `CONTRACT_has_TEXT`.

See [Structured Data](../data-ingestion/structured-data.md).

## 8. Update support

Datapipe is incremental: the next ETL run recomputes only changed rows. That means:

- if you change a product's price → only it (and its embedding if the field is embeddable) is updated;
- if you add a new branch → only that node is created;
- if you delete a category → that node and its edges are removed.

Run ETL on cron (e.g. once an hour) — Vedana will stay current without full recomputes.

## Checklist

- [ ] Anchors / Attributes / Links are described and reviewed.
- [ ] Data tables are set up in Grist.
- [ ] Table columns match `attribute_name`s in the model.
- [ ] FK columns reference IDs in the other tables correctly.
- [ ] ETL ran without errors.
- [ ] Cypher checks in Memgraph Lab return data.
- [ ] Test questions in chat work.
- [ ] The golden dataset includes structural questions.

## Common mistakes

- **The FK column isn't listed in `anchor1_link_column_name`** → ETL doesn't build the edge.
- **The target anchor with that ID doesn't exist.** ETL builds "dangling" edges; Cypher doesn't find pairs.
- **Typo in the column name.** `category_Id` ≠ `category_id`. ETL skips mismatches.
- **`dtype` in the model doesn't match Grist.** E.g. price as a string in Grist, `dtype=float` in the model. ETL may fail or write `null`.
- **Forgot to run `memgraph_steps`.** Data model and node tables are updated, but Memgraph still has old data.

## What's next

- [Adding Anchors](./guides/adding-anchors.md), [Adding Attributes](./guides/adding-attributes.md), [Adding Links](./guides/adding-links.md)
- [Custom ETL](../data-ingestion/custom-etl.md) — for large volumes or external sources.
