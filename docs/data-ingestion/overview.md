---
title: Data Ingestion Overview
section: Data Ingestion
order: 1
---

# Data Ingestion: Overview

Vedana supports three types of domain data. Choosing the right type is the most important modeling decision: it determines which kinds of questions the assistant can answer precisely, and which only approximately or not at all.

| Type                 | When to use                                                                  | Where it lives        |
| -------------------- | ----------------------------------------------------------------------------- | ---------------------- |
| **Documents**        | Explanatory text. "What does the policy say about X?", "Tell me about…"      | Memgraph + pgvector   |
| **Structured data**  | Specific values, filters, relationships. "How many…", "All… cheaper than X", "Who is connected to…" | Memgraph (nodes + edges) |
| **FAQ**              | Short, fixed answers. "Opening hours", "Warranty", "How to return"            | Grist `Anchor_faq` table (same mechanism as any other anchor — see [FAQ](./faq.md)) |

## How they complement each other

The same object can be ingested through several mechanisms at once. A contract is a great example:

- **Document chunks** for the contract text — to answer "What does the termination clause say?".
- **Structured anchor `Contract`** with `start_date`, `end_date`, `counterparty` attributes — to answer "Which contracts expire this quarter?".
- **FAQ** for typical client questions about contracts — "How do I extend my contract?".

That's the ideal scenario: each data type covers a class of questions the others can't.

## Two loading paths

### Grist + ETL (default)

```
Data → Grist → ETL (Datapipe) → Memgraph + pgvector
```

Suited to:

- manual data entry by non-engineers;
- moderate volumes (tens/hundreds of thousands of rows);
- frequent data model changes.

### Custom ETL

```
External system → Custom ETL → Memgraph + pgvector
```

Suited to:

- large volumes (millions of rows);
- automated pipelines from external systems (CRM, ERP, S3 buckets);
- continuous ingestion;
- complex transformations.

See [Custom ETL](./data-ingestion/custom-etl.md).

## Critical note

Any source that writes to Memgraph **must** conform to the data model. Anchors / Links / Anchor_attributes / Link_attributes are the single source of truth for the schema. Custom ETL bypasses Grist as an entry point but not the data model as a contract.

If you load nodes with labels not described in Anchors, the assistant won't know about them and won't be able to use them.

## What's next

- [Documents and Chunks](./data-ingestion/documents-and-chunks.md)
- [Structured Data](./data-ingestion/structured-data.md)
- [FAQ](./data-ingestion/faq.md)
- [Custom ETL](./data-ingestion/custom-etl.md)
