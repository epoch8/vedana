---
title: Adding Documents
section: Guides
order: 6
---

# Adding Documents

Vedana ingests documents through the same Anchor / Link / Attribute mechanism as any other entity — there is no special "document" code path. The convention in this guide (and in test fixtures) is to declare anchors `document` and `document_chunk` and a link between them, then point an embeddable `content` attribute at the chunk text.

> **Read first:** [Documents and Chunks](../data-ingestion/documents-and-chunks.md) — there is **no built-in chunking step** in the default ETL (`prepare_nodes` is a pass-through). You either pre-chunk the document text before loading it into Grist, or you add a custom step in your own ETL.

## 1. Prepare the files

Supported:

- PDF, DOCX, TXT, Markdown, HTML, exported Google Docs, CSV (as text).

Before uploading:

- check that the text is extracted correctly (especially from PDF — many parsers mangle tables and columns);
- remove boilerplate pages (cover pages, tables of contents) if they hurt semantic search;
- split very large files into logical sections if they're too heterogeneous.

## 2. Upload to Grist > Data > Anchor_document

`GristDataProvider` discovers anchor data by table-name prefix: every table named `Anchor_<noun>` is treated as the data for the matching anchor (`vedana_core/data_provider.py:69`). So for a `document` anchor, create a table called `Anchor_document` with the columns that map to the anchor's attributes:

| id      | title                       | source_url                                | content      |
| ------- | --------------------------- | ----------------------------------------- | ------------ |
| doc-001 | Returns and exchanges       | https://acme.example.com/policy/refund    | (full text) |
| doc-002 | Warranty policy 2026        | https://acme.example.com/policy/warranty  | (full text) |

The `content` field is the full extracted text. **You** are responsible for splitting it into chunks before storing — either by pre-chunking and writing rows into a separate `Anchor_document_chunk` table, or by adding a chunking step to your custom ETL.

Alternatively, if there are many documents:

- store them in an S3 bucket and put the link in `source_url`, while extracting `content` in custom ETL;
- keep the texts in another DB and load them through a custom ETL step.

## 3. Configure chunking (if needed)

There is no built-in chunking step in the default ETL — `prepare_nodes` returns the input DataFrame unchanged. Recommended chunk sizes (300–800 tokens, with 0–50 token overlap for documents where context across paragraphs matters) are a target for **your own** pre-processing or a custom Datapipe step you add via [Custom ETL](../data-ingestion/custom-etl.md).

When to tune:

- very short documents (FAQ-style) → smaller chunks, no overlap;
- very long structured documents (contracts, regulations) → more overlap so heading terms appear in detail chunks.

## 4. Run ETL

Backoffice → ETL → **Run Selected** for:

- `data_model_steps` (if you changed the default model);
- `grist_steps` (load documents);
- `default_custom_steps` (chunk them);
- `memgraph_steps` (load into the graph + build embeddings).

## 5. Verify in Memgraph Lab

```cypher
// edge label below depends on the `sentence` you declared in Grist > Links.
// The recommended form is ANCHOR1_verb_ANCHOR2 — e.g. DOCUMENT_has_DOCUMENT_CHUNK.
// If you declared it differently, substitute your label here.
MATCH (d:document)-[:DOCUMENT_has_DOCUMENT_CHUNK]-(c:document_chunk)
RETURN d.title, count(c) AS num_chunks
ORDER BY num_chunks DESC
```

This should show that documents have been split into chunks.

```cypher
MATCH (c:document_chunk) RETURN c.content LIMIT 3
```

The chunk content should be human-readable.

## 6. Verify in chat

Ask a document question:

> "What does our return policy say about returns after 14 days?"

In Details a tool call `vector_text_search(label="document_chunk", property="content", text="...")` should appear. The assistant's answer should be grounded in the retrieved chunks.

## 7. If answers are bad

| Symptom                                            | What to fix                                                          |
| -------------------------------------------------- | -------------------------------------------------------------------- |
| The assistant doesn't find a document that exists  | embed_threshold too high → lower to 0.55–0.65 for chunk content.    |
| The assistant finds a lot of irrelevant material   | embed_threshold too low → raise it.                                  |
| The assistant gets facts confused                   | Chunks are too big — chunk smaller.                                   |
| Context is lost between chunks                      | Add overlap (10–20% of chunk size).                                   |
| It doesn't call vector search at all              | Playbook problem — add a "document question" scenario.               |

## 8. Source URLs / citations

To let the assistant cite sources, in the playbook (Queries) write:

```
3) Format the answer as: "<answer text> (Source: <document.title>, <document.source_url>)"
```

The LLM will then automatically add the link to the answer.

## Best practices

- **Always pair documents with FAQ.** Users ask basic questions — let FAQ answer them deterministically. Documents stay for deeper / specific questions.
- **Don't dump the whole knowledge base into one file.** Better to have dozens of documents with meaningful titles — improves vector search results.
- **Run the golden dataset on document questions regularly** — you'll quickly notice if a new document broke existing scenarios.

## What's next

- [Tuning Embeddings](./guides/tuning-embeddings.md) — how to choose thresholds.
- [Adding FAQ Entries](./guides/adding-faq-entries.md) — for canonical answers.
- [Adding Structured Data](./guides/adding-structured-data.md) — hybrid approach (document + structured attributes).
