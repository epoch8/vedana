---
title: Adding Anchors
section: Guides
order: 3
---

# Adding Anchors

A step-by-step scenario: how to add a new anchor to a working data model.

## 1. Decide whether you need an anchor at all

Create an anchor if **at least one** of these is true:

- the entity participates in several relationships (`Category → regulates`, `Category → contains products`);
- the entity has its own attributes (description, parent_id, code);
- the entity must be queryable on its own ("Show me all categories").

Otherwise — keep it as a regular string attribute (`Product.category = "Laptops"`).

## 2. Fill in the row in Anchors

Open **Grist > Data Model > Anchors**.

Fill in 4 fields:

| Field          | Example                                                                        |
| -------------- | ----------------------------------------------------------------------------- |
| `noun`         | `category` (lowercase, singular, Latin script)                                 |
| `description`  | `"A catalog category. Groups products by type. Used to filter products and to attach regulatory documents."` |
| `id_example`   | `category_id: "cat-laptops"`                                                  |
| `query`        | `MATCH (c:category) WHERE c.category_id = $id RETURN c`                        |

## 3. Add required attributes

In **Anchor_attributes**, describe at least one attribute — usually a human-readable name:

| attribute_name | anchor   | description     | data_example | dtype | embeddable | embed_threshold | query                                                              |
| -------------- | -------- | --------------- | ------------ | ----- | ---------- | --------------- | ------------------------------------------------------------------- |
| `name`         | category | Category name.  | "Laptops"    | str   | true       | 0.7             | `MATCH (c:category {category_id: $id}) RETURN c.name`              |

Without `name` (or another embeddable field), the assistant can't find the category by user query.

## 4. Add links to other anchors

If the category is connected to products, regulatory documents, etc., add rows to **Links**:

```
product   | category | PRODUCT_belongs_to_CATEGORY  | ...
document  | category | DOCUMENT_regulates_CATEGORY  | ...
```

See [Adding Links](./guides/adding-links.md).

## 5. Load the data

If the anchor is your new entity (e.g. `category`), make sure the Data doc has a `categories` table with `category_id` and `name` columns. If you're using a custom ETL — make sure the corresponding node is loaded.

## 6. Run ETL

Backoffice → ETL → **Run Selected** for every step (data model + grist + memgraph + embeddings).

## 7. Verify

In Memgraph Lab (`http://localhost:3000`):

```cypher
MATCH (c:category) RETURN c LIMIT 10
```

Nodes with the `category` label should be returned.

In the chat (`http://localhost:9000/chat`):

> "Show me all categories"

The assistant should return a list. If not — check the Details:

- if it didn't use the `cypher` tool → an issue in the playbook or the anchor description;
- if Cypher returned an empty result → an issue with the data or ETL;
- if it failed with a Cypher error → an issue in the `query` fields.

## 8. Optional: add scenarios to Queries

If you have typical questions about the new anchor ("Which products are in category X?"), describe them in **Queries** — that makes behaviour more stable.

## Checklist after adding

- [ ] anchor is described in Anchors (4 fields filled in).
- [ ] At least one embeddable attribute is described in Anchor_attributes.
- [ ] All links are described in Links.
- [ ] Data exists in the source (Grist Data doc / your ETL).
- [ ] ETL ran without errors.
- [ ] Memgraph Lab shows nodes with the right label.
- [ ] A test question gives the correct answer.
- [ ] If you have a golden dataset — questions about the new anchor are added and eval has been run.

## Common mistakes

- **Description = repetition of the name.** "Category is a category". Useless to the LLM.
- **No `query`.** The assistant can't reliably fetch the entity.
- **No embeddable attribute.** If no attribute is embeddable, vector search over this anchor doesn't work (Cypher only).
- **Plural in `noun`.** `categories` breaks Cypher and matching against the data.
- **Forgot to run ETL.** The anchor only appears in Memgraph after ETL.

## What's next

- [Adding Attributes](./guides/adding-attributes.md)
- [Adding Links](./guides/adding-links.md)
