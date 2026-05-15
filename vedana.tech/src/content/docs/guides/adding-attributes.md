---
title: Adding Attributes
section: Guides
order: 4
---

# Adding Attributes

A step-by-step scenario: how to add a new attribute to an existing anchor (or link).

## 1. Decide: attribute or link

See [Attributes vs Link](../data-model/attributes.md#attribute-vs-link). In short:

- if the value is scalar and used only for description → **attribute**;
- if the value references an entity that has its own attributes or other relationships → **link**.

## 2. Make sure the data exists

An attribute is the **schema** for a value that already lives in the data. If the source table doesn't have the corresponding column, the attribute won't work (the LLM will see the description but Cypher won't return values).

Check that:

- the Grist Data doc or your source table has the right column;
- values are non-empty and the format matches the expected `dtype`.

## 3. Fill in the row in Anchor_attributes (or Link_attributes)

In **Grist > Data Model > Anchor_attributes**:

| attribute_name | anchor   | description                                | data_example | dtype | embeddable | embed_threshold | query                                                                                          |
| -------------- | -------- | ------------------------------------------ | ------------ | ----- | ---------- | --------------- | ----------------------------------------------------------------------------------------------- |
| `description`  | product  | Detailed product description for users.    | "MacBook Air, M2 chip..." | str | true     | 0.65            | `MATCH (p:product {product_id: $id}) RETURN p.description`                                     |
| `weight_kg`    | product  | Product weight in kg.                      | 1.24         | float | false      | —               | `MATCH (p:product {product_id: $id}) RETURN p.weight_kg`                                       |
| `release_date` | product  | Release date.                               | "2024-06-10" | date  | false      | —               | `MATCH (p:product {product_id: $id}) RETURN p.release_date`                                    |

For an edge attribute, do the same in **Link_attributes**, putting the edge `sentence` in the `link` column.

## 4. Pick `embeddable`

| Data type        | embeddable? |
| ---------------- | ------------ |
| name, title       | ✅           |
| description       | ✅           |
| address           | ✅ (carefully — comparisons may beat exact matches) |
| FAQ.question      | ✅           |
| any numeric       | ❌           |
| date / time       | ❌           |
| boolean           | ❌           |
| code, SKU, article | ❌          |
| URL, file path    | ❌           |
| enum (structured set) | usually ❌ |

## 5. Pick `embed_threshold`

For embeddable attributes only. Use the canonical starting table from [Tuning Embeddings & Thresholds](./tuning-embeddings.md#starting-values):

| Attribute kind                                              | Start     |
| ----------------------------------------------------------- | --------- |
| Names / exact identifiers (product, employee, brand)        | 0.75–0.85 |
| Descriptions and explanatory text                           | 0.65–0.75 |
| Long documents / chunks                                     | 0.50–0.65 |
| FAQ entries                                                  | 0.70–0.78 |

After running on the golden dataset, adjust. Too many false positives → raise; missed valid matches → lower. Full tuning loop and edge cases (long-vs-short text, common terms across anchors) — see [Tuning Embeddings & Thresholds](./tuning-embeddings.md).

## 6. Run ETL

Backoffice → ETL → **Run Selected** on `memgraph_steps` (or do a full run). Datapipe is incremental: `generate_embeddings` is keyed by `(node_id, node_type)` for anchors and `(from_node_id, to_node_id, edge_label)` for edges. When a row in `nodes`/`edges` or in `dm_anchor_attributes`/`dm_link_attributes` changes (including flipping `embeddable` or editing `embed_threshold`/`attribute_name`), Datapipe re-runs the step only for the affected keys. A bulk change like adding an embeddable attribute on an anchor will recompute every node of that type once; subsequent runs are cheap.

## 7. Verify

In Memgraph Lab:

```cypher
MATCH (p:product) RETURN p.description LIMIT 5
```

Values for the new attribute should be returned.

In chat, ask a question that requires the new attribute:

> "Which product weighs less than a kilogram?"

If it works — great. If not — see Details:

- is the attribute visible in the system prompt? (no → ETL didn't apply / `description` is empty)
- was the Cypher generated correctly? (no → improve `description` and the `data_example`)

## Checklist

- [ ] The data column exists and is populated.
- [ ] `dtype` matches the actual format.
- [ ] `embeddable` is set correctly (only for text).
- [ ] `embed_threshold` is set (if embeddable).
- [ ] `query` works — verified in Memgraph Lab.
- [ ] ETL ran without errors.
- [ ] A test question gives the correct answer.

## Common mistakes

- **`dtype` doesn't match the data.** ETL fails or writes incorrectly.
- **`embeddable=true` for numbers/SKUs.** Useless — embedding "999.00" carries no semantics.
- **Empty `query`.** The assistant knows the attribute exists but can't fetch it.
- **Description is too generic.** "Property of product" — the LLM won't know when to use it.
- **One attribute "does it all".** Don't pack JSON into one column if queries on it matter — split into multiple attributes.

## What's next

- [Tuning Embeddings & Thresholds](./guides/tuning-embeddings.md)
- [Adding Anchors](./guides/adding-anchors.md), [Adding Links](./guides/adding-links.md)
