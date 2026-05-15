---
title: How to Ingest Documents
section: "Guides"

---

_This guide assumes you understand what document chunks are and how Vedana uses them. If not, read [Documents and Document Chunks](https://vedana.tech/docs/preparing-data-for-vedana/documents-and-document-chunks/) first._

## Before You Start

Make sure your source files are in a supported format (PDF, DOCX, TXT, Markdown, HTML, or exported Google Docs) and that text extraction has been reviewed for quality. Garbled or poorly extracted text will degrade retrieval regardless of how well the rest of the pipeline is configured.

If your documents contain tables, price lists, or other structured content that users might query with specific values or filters, consider extracting that content as structured data alongside the document chunks. 
See [Adding Structured Data] for guidance on when and how to do this.

## Document Ingestion Process
### Step 1 — Register Your Documents

Open **Grist → Data → anchor_document** and create one row per file.

Each row requires the following fields:

| Field                              | Description                                                       |
| ---------------------------------- | ----------------------------------------------------------------- |
| `document_id`                      | A unique, stable identifier for this document (e.g. `policy-001`) |
| `document_name`                    | A human-readable name                                             |
| `document_summary`                 | The description of document (e.g. `policy`, `manual`, `contract`) |
| `link_document_has_document_chunk` | Chunks that will be tied to this document.                        |
| ...                                | Additional fields you deem necessary                              |

You can add any domain-specific metadata that will be useful for filtering or attribution later — for example, an `owner` field, a `department` field, or an `effective_date`. These are optional but recommended if users are likely to ask questions that reference these properties.

### Step 2 — Create the Chunks

Open **Grist → Data → anchor_document_chunks** and add one row for each chunk of text.

Each row requires:

| Field         | Description                                                             |
| ------------- | ----------------------------------------------------------------------- |
| `chunk_id`    | A unique identifier for this chunk (e.g. `policy-001-chunk-03`)         |
| `document_id` | The ID of the parent document — must match a row in the Documents table |
| `chunk_text`  | The raw text content of this chunk                                      |

Optional but useful: `section`, `page_number`, or any other positional metadata that will help the assistant attribute retrieved content accurately.

**Chunking guidance:**

Aim for 300–800 tokens per chunk. If you are working with long, continuous documents where meaning spans multiple paragraphs — such as legal policies or technical specifications — add overlap between adjacent chunks so that retrieved chunks carry enough surrounding context to be understood on their own. A 10–15% overlap (roughly 50–100 tokens at the boundary) is a reasonable starting point.

Avoid splitting mid-sentence or at arbitrary character counts. Split at natural boundaries: paragraph breaks, section headings, or numbered list items. The goal is for each chunk to be independently meaningful when read in isolation.

### Step 3 – Update the Data Model

Go to **Grist → Data Model** and confirm that the `document` and `document_chunk` anchor types are declared. If you are working with the default Vedana setup, these are pre-configured and no changes are needed. If you have added custom metadata fields to your anchor_documents or document_chunks tables, add the corresponding attribute definitions before running ETL.

Click **Update Data Model**  to apply any changes.

### Step 4 – Configure Retrieval Behavior

Open **Grist → Data Model → Queries** and confirm there is a query entry covering document-related questions. This entry tells the assistant which tool to use, how many chunks to retrieve, how to format the response, and whether to include source citations.

A minimal query entry for document retrieval looks like this:

- **Query type:** Document questions (e.g. "What does the policy say about X?")
- **Tool:** `VTS_tool` (vector text search)
- **Steps:** Embed the query, retrieve top-N chunks by similarity, synthesize answer from chunk content
- **Output:** Include source document name and chunk reference

If this entry is missing or incomplete, the assistant will not reliably route document questions to vector search. See [Configuring the Playbook] for full guidance on query entries.

### Step 5 — Run ETL

Open the Backoffice at `http://localhost:8000`, navigate to the ETL section, and run the pipeline. Confirm that the following steps complete successfully:

- Data model load
- Data load
- Embedding generation

After ETL completes, each document chunk exists as a node in Memgraph with an embedding attached. The assistant can now retrieve and use them.

## Scaling Beyond Manual Entry

For large document volumes, adding chunks row by row in Grist is not practical. In that case, use an automated chunking pipeline to process files in bulk and either batch-upload the results to Grist or write directly to Memgraph via custom ETL.

If writing directly to Memgraph, the pipeline must: create a document anchor node for each file, create a chunk anchor node for each chunk, generate and attach embeddings, and link each chunk back to its parent document. The data model must be defined in Grist before the pipeline runs — direct writes to Memgraph do not bypass schema validation.

For implementation guidance, see [Custom ETL for Vedana].

## Google Drive and Google Docs

Vedana does not include a built-in connector for Google Drive or Google Docs. To ingest content from either source, export the documents using the Google Drive API or Google Docs API, convert the exported content to plain text, chunk the text, and upload it to Grist or ingest it via custom ETL using the steps above.

See [Custom ETL for Vedana] for implementation details.
