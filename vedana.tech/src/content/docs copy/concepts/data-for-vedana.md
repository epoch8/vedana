---
title: "Data for Vedana"
section: "Concepts"
order: 2
---

## Core Principle
Most question-answering systems work by finding text that looks similar to a question and asking a language model to produce an answer from it. Vedana works differently. It stores your domain knowledge as a structured knowledge graph and answers questions by navigating that graph directly: counting, filtering, traversing relationships, and tracing every answer back to its source. You will always know where the answer came from.

This distinction matters because the kind of questions real users ask rarely fit neatly into "find me text that looks like this." They ask for specific values, exact counts, complete lists, and answers that depend on how entities relate to each other. A system built on text similarity can approximate those answers. A system built on a structured graph can compute them.

<img src="/images/Screenshot%202026-02-19%20at%2008.10.03.png" alt="Hero" width="800" class="center-image" />

The quality of answers Vedana produces is determined first by the quality of the data in the graph, and only then by the model and tools that query it. A well-designed data model with incomplete, outdated, or poorly structured data will still produce wrong answers. Data coverage, accuracy, and freshness matter as much as the architecture that surrounds them.

This is why the quality of data is crucial.
The better your data is, the better assistan't performance.

## How Data Is Stored

All domain data in Vedana lives in Memgraph as a knowledge graph, not as a flat collection of text chunks. Data is stored as typed nodes called **anchors**, connected by typed relationships called **links**, with typed properties called **attributes** attached to each node. Embeddings are stored alongside the graph where needed, but they are supplementary — the foundation is structure.

This storage model is what makes deterministic retrieval possible. A Cypher query run against the graph returns the same result every time, because it follows explicitly defined relationships rather than estimating similarity. If you load only documents without a structured model, Vedana works like an advanced RAG system — useful, but limited to text retrieval. If you model your domain with anchors, attributes, and links, it becomes a system capable of precise, verifiable, relationship-driven answers.

## Types of Data

Vedana supports three types of domain data, each suited to a different kind of question.

<img src="/images/data-model-for-vedana.png" alt="Hero" width="800" class="center-image" />

### Documents

Documents are unstructured text content — policies, manuals, contracts, knowledge base articles, or any material where the answer is somewhere in the text and needs to be found by meaning. A document is split into chunks at ingestion time, and each chunk is stored as a node in the graph with an embedding attached. When a user asks a document-related question, the assistant retrieves the most relevant chunks by semantic similarity and synthesizes a response from their contents.

Documents are the right choice when the answer is explanatory or contextual. They are not the right choice when the answer requires a specific value, a count, or a relationship traversal — those cases call for structured data.

→ See [Documents and Document Chunks] for a full explanation of how document ingestion works, including chunking strategy and retrieval behavior.

### Structured Data

Structured data is domain knowledge expressed as typed entities with queryable attributes and explicit relationships — product catalogs, price lists, branch locations, contracts with effective dates, compliance mappings, and similar records. Each row in a structured table becomes an anchor in the graph, and relationships between tables become links.

Structured data enables the retrieval that documents cannot provide: exact filtering by value, numeric comparisons, exhaustive lists, and multi-hop traversal across related entities. When a user asks which products cost less than 500, or which contracts expire this month, or which requirements apply to a given product, the answer comes from structured graph queries, not from text similarity.

→ See [Adding Structured Data] for guidance on when to structure data and how to model it correctly.

### FAQ

FAQ entries are predefined question-and-answer pairs stored in a dedicated Grist table. They are the simplest ingestion method and require no data modeling: you write the question, write the answer, and the system retrieves the right entry when a matching question is asked. Because the answers are predefined, they are stable and consistent regardless of how the question is phrased.

FAQ is appropriate for short, canonical responses that should not vary — business hours, return policies, standard support answers. It is not a replacement for documents or structured data; it is a fast, reliable layer for known question-answer pairs where consistent wording matters.

→ See [Questions and Answers] for a full explanation of how FAQ retrieval works and when to use it.

## How Retrieval Works

Vedana supports two retrieval mechanisms, and the assistant selects between them based on the type of question and the rules defined in the playbook.

**Vector search** is used for document-based questions. The user's query is embedded and matched against the stored chunk embeddings. The closest matches are retrieved and passed to the language model, which synthesizes a response from their contents. This works well for open-ended, descriptive questions where the answer lives in a body of text.

**Cypher queries** are used for structured questions. The assistant generates a Cypher query based on the user's question and the declared data model, runs it against Memgraph, and returns results directly from the graph. This is exact and exhaustive — every matching record is returned, not just the most similar ones. Cypher is the right tool whenever the question has a definite answer that depends on structure, relationships, or specific attribute values.

For questions that span both types of data, the assistant can combine both mechanisms: vector search retrieves candidate content from documents, and a Cypher query enriches or filters the results using graph structure.

## How Data Gets Into the Graph

There are two ingestion paths.
| Path | Best for |
|---|---|
| Grist-based ingestion | Manual entry, moderate-scale datasets |
| Custom ETL → Memgraph | Large-scale, automated pipelines |

### Grist

**Grist** is the default interface for data entry and configuration. You write data into Grist tables, define the data model there, and ETL transforms the Grist tables into graph structures in Memgraph. Grist is well-suited to manual entry, moderate-scale datasets, and any situation where the people managing data are not engineers. The transformation follows a fixed pipeline:

```
Data → Grist → ETL → Memgraph
```

### Custom ETL

**Custom ETL** writes directly to Memgraph and is appropriate for large volumes, automated pipelines, or data that originates in external systems and needs to be ingested continuously or near-real-time. Custom ETL bypasses the Grist interface but not the data model — any pipeline writing to Memgraph must still conform to the anchor, attribute, and link schema defined in Grist. The data model is always defined first; the ingestion method is a separate concern.

**Next steps:**

The data model, a the schema that defines what anchors, attributes, and links exist, – is covered in the [Data Model] section. Understanding it is a prerequisite for ingesting any structured data correctly.

To understand the principles of data ingestion, read [Documents and chunks](https://vedana.tech/docs/preparing-data-for-vedana/documents-and-document-chunks/).  

For step-by-step ingestion instructions, see the [Guides] section, starting with [Setting Up Your Data Model].
