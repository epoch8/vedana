---
title: "Tools for Vedana"
section: "Concepts"
order: 6
---

## Why Tools Exist

Vedana does not let the LLM generate answers freely from text. Instead, the assistant operates through explicit tools – functions it can call to inspect the graph, retrieve data, and produce answers that are grounded in what actually exists in the system.

Think of the LLM as an agent. 
It receives a question, decides which tool is appropriate, executes it, and uses the result to form an answer. 
It does not guess. It looks things up.

Vedana's assistant operates as a constrained agent. When it receives a question, it inspects the available tools, selects the one appropriate for the question, executes it, and uses the result to form an answer. If the first result is insufficient, it can continue exploring (calling additional tools or refining its approach) until it has what it needs.

What it cannot do is equally important. The assistant cannot invent tools that do not exist, bypass tool restrictions defined in the playbook, or access data outside the declared data model. Its behavior is bounded by the structure you define. This is by design – it is what makes the system predictable, auditable, and safe to deploy in production.

## Built-in Tools

Vedana ships with two core tools out of the box: vector search and Cypher.

### 1. Vector Search Tool

Vector search handles **semantic retrieval over text**. It works by embedding the user's query and finding the nearest matching content stored in Memgraph: document chunks, FAQ entries, or other embeddable fields.

This tool is designed for questions where the answer lives somewhere in a body of text and needs to be found by meaning rather than by exact match. It handles natural language well, tolerates variation in phrasing, and does not require you to know exactly where the answer is or how it is worded.

Use vector search when is well-suited for questions like "What does the policy say about overtime?" where the answer lives in a document and needs to be found by meaning rather than by structure.

How it works:
1. Embeddings are stored in Memgraph.
2. Query is embedded.
3. Nearest neighbors are retrieved.

Vector search is useful when:
- The query is text-heavy.
- Exact structure is unknown.
- Semantic matching is required.

The limitation of vector search is that it does not understand structure and relationships. It cannot reliably traverse the graph, count entities, enforce strict filters, or follow multi-hop paths. For those, you need Cypher.

### 2. Cypher Tool

The Cypher tool handles **structured graph queries**. When called, the LLM generates a Cypher query based on the user's question and the declared data model, runs it against Memgraph, and returns structured results directly from the graph.

Cypher operates on what has been explicitly modeled – anchors, attributes, and links. It does not approximate or infer. If you ask how many products belong to a given category, it counts them. If you ask which branches carry a specific item, it traverses the relationship and returns every match. The answer is exact, complete, and traceable back to the data.

Use Cypher when the question has a definite answer that depends on structure. 

For example: 

"What are the opening hours of the Vilnius branch?"

"Which requirements apply to this product?"

"Which documents regulate category X?" 

These are questions where correctness matters more than approximation, and where the answer should never be guessed.

Cypher is what makes Vedana more than a document search system. It enables deterministic reasoning: the factual correctness of an answer comes from the graph, not from the model's best guess.
Cypher is the mechanism through which the knowledge graph is actually used.

Used for:
- Attribute filtering.
- Relationship traversal.
- Deterministic domain queries.
- Multi-hop graph exploration.
    
How it works:
1. LLM generates Cypher query.
2. Query runs against Memgraph.
3. Structured results are returned.
    
Cypher is used when:
- Structure matters.
- Relationships are important.
- Filters must be exact.

## Tool Selection

The assistant selects tools dynamically based on the type of question, the available anchor types, and the structure of the data model. It may choose **vector search**, **Cypher**, or a **combination of both**.

Tool selection can also be explicitly governed by a **playbook**. Playbooks let you define which tool should be used for which type of question, in what order, and what fallback behavior is allowed. 
For example, you can instruct Vedana to always prefer Cypher for product queries, use vector search only for document retrieval, or never answer a question without tool evidence.

The LLM evaluates:
- Type of question.
- Available anchor types.
- Data model structure.
    
**Example:**

**Question:** *"What are the opening hours of the Moscow branch?"*

**Correct behavior:** *Use Cypher (structured lookup)*.
    
**Question:** *"What does the policy say about overtime?"*

**Correct behavior:** *Use vector search (document retrieval).*

For **hybrid retrieval**, a typical flow looks like this:
- Vector search retrieves candidate content from documents.
- A Cypher query narrows or enriches the results using graph structure.
- The LLM formats the final answer from the combined output.

This gives the system both semantic flexibility and structural precision.

## Extending Tools

Tools are extensible.
If your domain requires capabilities beyond vector search and Cypher, you can implement custom tools to:
- Query external APIs.
- Perform calculations.
- Integrate ERP systems.
- Execute domain-specific logic.

Custom tools must:
- Have clear input/output schema.
- Be deterministic.
- Be registered in the tool registry.
    
See [How to Write a Custom Tool] for implementation details.

## Determinism Boundary

Tools are deterministic. The LLM is probabilistic.
This distinction matters. The correctness of an answer in Vedana comes from the tools, from what the graph actually contains and what the query actually returns. The LLM's role is to select the right tool and format the result into readable language. It does not determine what is true. The data does.
As long as tool results are stable and the data model is well-defined, the structural correctness of answers is stable too.

Tool usage defines how Vedana explores the world.
