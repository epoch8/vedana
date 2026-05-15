---
title: "Adding Structured Data"
section: "Preparing data for Vedana"

---

## Why Structured Data Matters

Document chunks are a good starting point, but chunk-based retrieval has limits. 
Vector search works well for explanatory content: policies, manuals, or any other long-form documents where the answer is somewhere in the text and needs to be found by meaning. But it is not reliable when the answer needs to be exact. 

If you need precise, deterministic answers, your data must be structured.

Take for example questions like:
- "Which products cost less than 500?"
- "What time does the Moscow branch close on Sunday?" 
- "Which contracts expire this month?"

These questions cannot be answered reliably by finding similar text. 

They require structured data – typed attributes that can be filtered, compared, and traversed deterministically.
If your users will ask questions that depend on specific values, dates, numbers, or relationships, that data needs to be modeled as anchors and attributes, not left inside document chunks.

### Benefits of Structured Data

With structured modeling, Vedana can:

- Use Cypher for exact filtering.
- Perform multi-hop traversal.
- Combine numeric comparisons.
- Guarantee attribute correctness.

This is not possible with pure vector search.

## When You Should Add Structured Data

You should structure data when working with:

- Products and product properties
- Price lists
- Opening hours
- Branch addresses and locations
- Contracts and effective dates
- Regulatory mappings
- Inventory data
- Service catalogs

A useful test: if a user might ask a question where the correct answer is a specific value (a number, a date, a name, a yes or no) that data should be structured.
Any domain where attributes matter should be structured.

## How to Add Datta to Grist
### Option 1 – Upload Existing Tables

Structured data often already exists in Excel files, CSV exports, ERP exports, or internal reporting tables. These can be uploaded into Grist directly.
Before uploading, make sure your tables are clean:
- No merged cells
- No multi-row headers
- No mixed data types within a single column
- No visual formatting used in place of real structure (e.g. color-coded rows, manually merged fields)

Grist expects genuine tabular structure where each row represents one entity. Messy formatting leads to broken ingestion and unreliable data in the graph.

Once uploaded, you need to **map the table to the data model**. For each table, decide what the anchor type is, which columns become attributes, and whether any columns represent relationships to other anchors that should become links.

When uploading structured tables, you must ask:
- What is an anchor type?
- What is an attribute?
- What is a link?
  
For example, a products table with columns product_id, name, price, and category maps to:

**Anchor**: Product
**Attributes**: name (string), price (float)
**Link**: Product → belongs_to → Category

Update the data model in Grist to reflect this before running ETL.

### Option 2 – Extract Structure from Text

Sometimes structured data is embedded inside unstructured documents: price tables in a PDF, contract terms in a policy file, address lists in a handbook. In these cases the data exists, but it is not yet in a form Vedana can query directly.
The correct approach is:
- Extract the text from the source document.
- Identify the structured elements (tables, lists, repeated patterns).
- Convert them into a clean tabular format.
- Upload into Grist.
- Update the data model accordingly.

This hybrid approach significantly improves answer accuracy for data that would otherwise only be retrievable through approximate text matching.


### Combining Both Approaches
The most effective strategy for most domains is to use both document chunks and structured anchors together, each for what it does best.

Keep explanatory content as document chunks. Extract measurable, queryable attributes into structured anchors. The two approaches are complementary, not mutually exclusive.

Take a contract as an example. Keep the full contract text in document_chunks so the assistant can retrieve and explain its contents. At the same time, extract the key structured fields — contract_id, start_date, end_date, counterparty — and store them as a structured anchor. 

For example:
Contract document:
- Keep full contract text in document_chunks so the assistant can retrieve and explain its contents.
- Extract:
    - contract_id
    - start_date
    - end_date 
    - counterparty
        
Store those as structured anchors.

Now Vedana can answer "When does this contract expire?" deterministically via Cypher, and still retrieve the relevant clause text via vector search when needed.
