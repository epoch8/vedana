---
title: How to Add Structured Data
section: "Guides"

---


_This guide assumes you understand what anchors, attributes, and links are. If not, read [Structured Data](https://vedana.tech/docs/preparing-data-for-vedana/structured-data/) and the [Data Model] section first._

## Before You Start

Make sure your source data is in a tabular format where each row represents one entity and each column represents a property or a reference to another entity. Data that doesn't meet this structure needs to be cleaned or reorganized before ingestion.

If your structured data is embedded inside unstructured documents (price tables in a PDF, address lists in a handbook, contract terms in a policy file) extract it into a clean table first. See [Extracting Structure from Documents] below.

## Step 1 — Decide the Mapping

Before uploading anything, map each column in your table to one of three roles. This is the most important step: errors here cause ETL failures or unreliable retrieval later.

For each table, answer three questions:

**What is the anchor?** The anchor is the entity each row represents. A products table → Product anchor. A branches table → Branch anchor. One row becomes one node.

**Which columns are attributes?** Attributes are scalar properties of the entity — strings, numbers, booleans, dates. A column like `price (float)` or `city (string)` is an attribute.

**Which columns are links?** A column is a link if it holds a reference to another entity (typically a foreign key). A `category_id` column referencing a categories table becomes a `belongs_to` link rather than a string attribute.

A completed mapping for a products table looks like this:

|Column|Role|Type|
|---|---|---|
|`product_id`|Anchor primary key|string|
|`name`|Attribute|string|
|`price`|Attribute|float|
|`in_stock`|Attribute|boolean|
|`category_id`|Link → Category anchor|—|

When you're unsure whether something should be an attribute or a link, ask: does the target have its own attributes, or will it participate in other relationships? If yes, model it as a link. If it's just a plain value that will only ever be read, an attribute is simpler and sufficient.

## Step 2 — Clean Your Table

Before uploading to Grist, make sure your table has genuine tabular structure. The most common causes of failed ingestion are:

- Merged cells across rows or columns
- Multi-row headers (column names split across two rows)
- Mixed data types within a single column (numbers and strings in the same price field)
- Visual formatting used as structure (color-coded rows, manually aligned text)
- Comma-separated values inside a single cell (multiple categories in one field)

Each column must contain one data type, consistently, in every row. Each row must represent exactly one entity. If a column contains a comma-separated list, it needs to be split into a separate relationship table before ingestion.

## Step 3 — Upload the Table to Grist

Open **Grist → Data** and create a new table. If you have an existing CSV or Excel file, you can import it directly. Grist accepts both formats.

Name the table clearly and consistently with the anchor type it will represent. A table named `products` maps to a Product anchor; a table named `branches` maps to a Branch anchor. Naming matters because ETL uses the table name to identify the anchor type.

## Step 4 — Define the Data Model

Open **Grist → Data Model** and declare the structure you decided in Step 1.

**In the Anchors table**, add one row for the anchor type this table represents. Fill in the noun (singular: `Product`, not `Products`), a clear description, an ID example, and the Cypher query for retrieval. See [Anchors](https://vedana.tech/docs/data-model/anchors/) for full field definitions.

**In the Attributes table**, add one row per column that is an attribute. For each row, specify the attribute name, the data type, the anchor it belongs to, and whether the value should be embeddable for semantic search. See [Attributes](https://vedana.tech/docs/data-model/attributes/) for full field definitions.

**In the Links table**, add one row per column that is a link. Specify the source anchor, the target anchor, the edge label (e.g. `belongs_to`), and the Cypher query for traversal. See [Links](https://vedana.tech/docs/data-model/links/) for full field definitions.

Click **Update Data Model** when done. ETL will validate the data against this schema. If the schema is missing or incomplete, ingestion will fail.

## Step 5 — Run ETL

Open the Backoffice at `http://localhost:8000` (or `http://localhost:9000`), navigate to the ETL section, and run the pipeline. Confirm that the following steps complete successfully:

- Data model load
- Data load
- Embedding generation (if any attributes are marked embeddable)

After ETL completes, every row in your table exists as a typed node in Memgraph with the declared attributes and links. Run a Cypher query in Memgraph Lab to confirm:

cypher

```cypher
MATCH (p:Product)
RETURN p.product_id, p.name, p.price
LIMIT 10
```

If nodes are missing or attributes are wrong, re-check the data model definition and re-run ETL.

## Extracting Structure from Documents

If structured data is embedded inside a document rather than already in tabular form, extract it before ingesting.

Common cases: a PDF containing a price table, a handbook with a branch address list, a contract with key terms presented in a list rather than a spreadsheet.

The process is:

1. Open the source document and identify the structured elements: tables, repeating patterns, lists with consistent fields.
2. Copy or export the content into a clean spreadsheet with one entity per row.
3. Ensure each column contains one data type consistently.
4. Upload to Grist and follow Steps 1–5 above.

Do not skip this step and rely on document chunks instead. A price table left in a PDF can be retrieved through vector search, but only approximately. Extracted into a structured anchor, it becomes precisely queryable.

## Combining Structured Data with Documents

For most domains, the best outcome comes from keeping explanatory content as document chunks and extracting measurable, queryable fields as structured anchors.

For a contract: keep the full text in document chunks for clause retrieval, and extract `contract_id`, `start_date`, `end_date`, and `counterparty` as a structured anchor. The assistant can then answer "when does this contract expire?" via Cypher, and "what does the indemnity clause say?" via vector search. Both from the same source document.

For a product: keep the full product description in document chunks, and store `name`, `price`, `category`, and `in_stock` as structured attributes. The assistant can filter by price, check availability, and still retrieve descriptive text when needed.

The rule is simple: if users will ask for the value directly, structure it. If they will ask what the document says about it, keep it in chunks.
