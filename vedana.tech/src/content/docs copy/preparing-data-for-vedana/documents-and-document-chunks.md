---
title: "Documents and Document Chunks"
section: "Preparing data for Vedana"

---

## What Document Ingestion Does

Vedana can make any text-based document searchable and usable in responses. The mechanism is straightforward: a document is split into smaller pieces called chunks, each chunk is stored as a node in the knowledge graph, and each node receives an embedding that allows it to be retrieved by meaning rather than exact keyword.

When a user asks a question, Vedana searches the chunk embeddings for content that matches the intent of the question, retrieves the most relevant pieces, and uses them to construct a grounded answer. The assistant only draws on what was actually retrieved — it does not invent or infer beyond it.

This approach works well for explanatory content: policies, manuals, contracts, knowledge base articles, or any material where the answer is somewhere in the text and needs to be found by meaning. It is not the right tool when the answer requires precise filtering, counting, or traversal — those cases call for structured data instead. If you are unsure which approach fits your content, see [Adding Structured Data].

## What Files Can Be Ingested

Any file that can be reliably converted to plain text can be ingested. Supported source formats include PDF, DOCX, TXT, Markdown, HTML, exported Google Docs, and CSV files treated as text documents.

Raw binary formats such as images, audio, and video cannot be ingested directly. These need to be converted to plain text first, typically through transcription or OCR, before entering the pipeline.

The format matters less than the quality of the conversion. Poorly extracted text — garbled characters, broken sentences, merged columns — produces poor chunks. And poor chunks produce poor answers. Before uploading, review the extracted text before uploading to confirm it is clean and readable.

## How Chunking Works

A chunk is a contiguous excerpt of a document's text, stored as its own node in the graph. Each chunk is linked back to its parent document, so the assistant can always identify which document a retrieved passage came from.

Chunk size affects retrieval quality in both directions. Chunks that are too small lose the context needed to answer questions coherently. Chunks that are too large dilute the semantic signal, making it harder to retrieve the right passage for a specific question. As a general guideline, 300–800 tokens per chunk works well for most document types.

For documents where meaning depends heavily on continuity across paragraphs — long policy documents, multi-section contracts — adding overlap between adjacent chunks helps preserve that context at retrieval time. Overlapping chunks share some text with their neighbors, so a retrieved chunk carries enough surrounding context to be understood on its own.

## How the Assistant Uses Chunks at Runtime

When a user asks a document-related question, the sequence is:

1. The assistant detects that the question is document-related and selects the vector search tool.
2. The query is embedded and matched against stored chunk embeddings.
3. The top-N most relevant chunks are retrieved.
4. The chunks are passed to the LLM, which synthesizes a response from their contents.
5. If citations are configured, the source chunk IDs or document names are included in the response.

The assistant does not read entire documents. It reads the chunks that were retrieved. This means the quality of chunking and the relevance of retrieval directly determine the quality of the answer.`

## Relationship to the Rest of the Data Model

Documents and chunks are a specific type of anchor in the graph. They follow the same rules as other anchors — they must be defined in the data model, they are created during ETL, and their behavior at query time is governed by the playbook.

For a complete picture of how documents fit into the broader architecture, see [Data for Vedana].

**Next step:** 
[How to Ingest Documents] — a step-by-step guide to registering documents, creating chunks, and configuring retrieval behavior in Grist.
