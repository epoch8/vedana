---
title: "Playbook for Vedana"
section: "Concepts"
order: 4
---

## What Is a Playbook

While the data model defines what exists, the playbook defines how to respond. It is the behavioral configuration for Vedana,  controlling how the assistant detects intent, selects tools, structures its responses, and constrains its behavior.
Playbooks make the agent predictable. Without one, the LLM may choose suboptimal tools, mix retrieval strategies, or answer too generically. With one, behavior becomes structured, constrained, and auditable.

## Where the Playbook Lives

The playbook is stored in **Grist** > **Data Model** > **Queries**.
It is part of configuration and loaded during runtime.

Because it lives in Grist:
- It can be edited without redeploy
- It is versionable
- It is environment-specific

Playbook updates affect agent behavior immediately after reload.

## Playbook Structure

Each playbook is organized around intents — discrete types of user requests. For each intent, the playbook defines:

- The intent name and description
- Which tools to use and in what order
- How to format and constrain the response
- Any additional behavioral rules

Each intent corresponds to a type of user request.

## Intent-Based Behavior

Playbook defines how Vedana reacts to different intents.

Example intents:

- "Give product information"
- "Give delivery information"
- "Find branch address"
- "Explain policy"
    
Each intent may require a different retrieval strategy. The playbook makes this explicit.

Product information is structured data, so the playbook directs the assistant to prefer Cypher — querying the Product anchor, filtering by name or SKU, and traversing related attributes. Vector search is used only as a fallback. Cypher guarantees deterministic results, while similarity search does not.

Delivery information typically lives in text documents, so the playbook directs the assistant to use vector search over document chunks, retrieve the relevant policy sections, and format the answer from what it finds.

Branch locations, policy explanations, compliance mappings — each intent gets its own clearly defined retrieval path and response format.

### Example: Product Information

Intent: "Give information about a product"

Behavior defined in playbook:
- Prefer Cypher tool
- Query Product anchor
- Filter by product name or SKU
- Traverse related attributes
- Only use vector search as fallback
    
Why:
Product information is structured.  
Cypher guarantees deterministic results.

### Example: Delivery Information

Intent: "Give delivery information"

Behavior defined in playbook:
- Use vector search
- Search document_chunks
- Retrieve relevant policy sections
- Format answer from retrieved chunks
    
Why:
Delivery details are often located in text documents.  
Vector search is appropriate.

## What Playbook Controls
**Tool selection**. You can specify that an intent must always use Cypher, prefer vector search, use hybrid retrieval, or never answer without tool results. This prevents random tool selection.

**Response instructions**. You can define exactly what each response should contain: price and availability for product queries, address and phone number for location queries, cited document chunks for policy queries. Responses become consistent and domain-specific rather than generically helpful.

**Behavioral constraints**. The playbook acts as a guardrail. It prevents the assistant from improvising logic that should be deterministic. Without Playbook LLM may choose suboptimal tools, mix retrieval strategies and give a generic answer. With Playbook behaviour becomes structured, constrained and auditable.

# Relationship to Data Model

Data model defines what can be queried.  
Playbook defines how it should be queried.
Together, they fully specify runtime behavior: one defines the structure of the domain, the other defines the rules for navigating it.

# How to add new scenario to Playbook

To add a new instruction for the assistant:

1. Go to **Grist** > **Data Model** > **Queries**.
2. Create a new row describing the following:
   - **Query type** – What kind of user question this instruction handles.
   - **Tool selection** – Which tools the assistant should use:
     - VTS_tool for semantic search
     - Cypher_tool for structured graph queries
   - **Step-by-step logic** – The reasoning steps the assistant should follow.
   - **Output format** – How the final response should be structured.

<img src="/images/Playbook.png" alt="Hero" width="800" class="center-image" />

In the `query_name` column, specify the type of user question.
In the `query_example` column, write the assistant instruction (prompt) that explains how to answer that type of query.

**Example structure**

- When to apply: Define the use case (e.g., "document-related questions", "product compatibility checks").
- Search strategy: Choose whether to start with vector search, follow with a structured query, or combine both.
- Answer construction: Explain how the assistant should generate the final answer.

The assistant uses these instructions to select the appropriate strategy and tools when responding to user questions.

3. Update the Data Model.

