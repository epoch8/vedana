---
title: "Data Model for Vedana"
section: "Concepts"
order: 3
---

## What is a Data Model and why it exists

The Data Model is a structured description of your domain that helps the assistant understand and work with your data in a clear and consistent way. The key idea of Vedana is that the assistant does not guess answers based on text similarity. Instead, it uses a clear data structure to build accurate, logic-based answers.

The data model explains:

- What entities exist in the system.
- What attributes each entity has.
- How different entities are connected to each other.

The model defines how the assistant perceives and uses knowledge, allowing it to answer questions in a structured and accurate way. Without a data model, the LLM sees only text. With a data model, the LLM understands the structure of knowledge domain.
Vedana does not rely on emergent reasoning. It relies on structure you define explicitly.

### Key Principle

If you want the LLM to reason about something, you must first describe it in the data model.
No structure → no deterministic reasoning.

Data model is the contract between:
- Domain knowledge
- Graph storage
- Tools
- LLM reasoning
    
Understanding the data model means understanding Vedana.

## Data Model Structure

The data model follows the Minimal Modeling approach and consists of **three core tables** are required for the assistant to function correctly:

- **Anchors** - Key entities in your domain (e.g., Product, Document, Product/Service code, Category)
- **Attributes** - Properties and characteristics of each entity (e.g., name, code, description)
- **Links** - Defined relationships between entities (e.g., "Product belongs to Category", "Product code requires Document")

These three tables form the minimal structure needed for the assistant to understand and reason over the knowledge graph.

In addition to these, the model can include optional extension tables that enhance functionality.

## Default Data Model

Vedana ships with a minimal working data model that supports RAG-style behavior out of the box. It includes three anchor types (documents, document_chunks, and faq) and is fully configured and operational from the start.

With this default model, Vedana can retrieve relevant document chunks, perform vector search, answer from FAQ entries, and return grounded responses with sources. This is sufficient for basic document Q&A.

<img src="/images/data-model-for-vedana.png" alt="Hero" width="800" class="center-image" />

## What Must Be Defined in a Data Model

To extend the data model, you must explicitly define:

### 1. Anchors

Anchors represent domain entities, the core objects of your domain.

Each row in the Anchors table describes one type of entity that the assistant can recognize and retrieve from the knowledge base. 

Each row consists of the following fields:

| Field           | What it contains                         |
| --------------- | ---------------------------------------- |
| **Noun**        | Entity name (English, unique)            |
| **Description** | Human-readable description of the entity |
| **ID example**  | Example of a real ID                     |
| **Query**       | Graph query used to fetch the data       |

- **noun** — the name of the entity type, written in singular form (e.g. person, product, contract). This becomes the node label in the graph. The assistant uses it to understand what kinds of things it can look up.
- **description** — a plain-language explanation of what this entity represents. This is included in the LLM context, so writing a clear description directly helps the assistant understand when to query this entity and what it means.
- **id_example** — a sample primary key value for this entity type (e.g. person:1). This shows the assistant what a valid identifier looks like, which helps it construct correct Cypher queries at runtime.
- **query** — a Cypher query that retrieves nodes of this type from the graph. This is one of the most important fields: without a valid query here, the assistant cannot reliably fetch this entity from the graph and will fall back to less precise retrieval.
    
Each anchor type must:
- Have a unique primary key
- Have a declared type
- Be registered in the data model

### 2. Attributes

Attributes describe structured properties of anchors. Attributes are the properties (fields) of the entities defined in the Anchors table.

If Anchors define what an object is, Attributes define what data the object contains, how that data is retrieved, and whether it can be used in vector (semantic) search.

| Field               | Description                                                   |
| ------------------- | ------------------------------------------------------------- |
| **Attribute Name**  | System name of the attribute                                  |
| **Description**     | Human-readable description                                    |
| **Anchor**          | The entity this attribute belongs to                          |
| **Link**            | Whether the field is a reference to another object            |
| **Data example**    | Example value                                                 |
| **Embeddable**      | Whether the value can be vectorized (used in semantic search) |
| **Query**           | Query used to retrieve the value                              |
| **dtype**           | Data type (str, int, url, file, bool, etc.)                   |
| **embed_threshold** | Semantic similarity threshold for returning results           |

Examples:
- Product.price (float)
- Branch.address (string)
- Branch.opening_hours (string or structured)
- Product.category (enum).
    
Attributes must be explicitly declared, typed and validated during ingestion.

The Attributes table defines:
- Which specific data is available to the assistant.
- Which fields participate in semantic search.
- How the model should retrieve attributes from the knowledge base.
- Which fields are shown to the user in the final response.
- How Anchors are correctly linked to document content and FAQs.

In Grist, attributes are declared across two tables depending on what they describe. Anchor_attributes covers properties that belong directly to an entity node. Link_attributes covers properties that belong to the relationship between two nodes. Both tables share the same column structure — the distinction is only in what they are attached to.

#### Anchor_attributes

The Anchor_attributes table defines what you know about each entity — its properties. Each column controls a specific aspect of how that property behaves:

- **attribute_name** — the name of the property as it is stored in the graph (e.g. person_name, interest_id). Must match the actual field name in the data.
- **anchor** — links this attribute to its parent entity type. This tells the system which node the attribute belongs to.
- **description** — explains what this attribute represents. Like the anchor description, this is included in the LLM context and helps the assistant understand what the field contains and when it is relevant.
- **data_example** — a sample value for this attribute (e.g. Geneva Durben). Helps the assistant understand the format and scale of the data, which improves query generation accuracy.
- **query** — a Cypher query that retrieves this specific attribute. Required for any attribute that should be directly queryable. Missing queries here are a common cause of the assistant failing to return specific field values.
- **dtype** — the data type of the attribute value: str, int, float, bool, url, file, etc. Must match the actual data stored in the graph. Type mismatches cause ETL failures or unpredictable query results.
- **embeddable** — a flag indicating whether the value of this attribute should be vectorized for semantic search. Set this to true for human-readable text fields where a user might search by meaning rather than exact match (e.g. names, descriptions, labels). Leave it false for identifiers, numeric values, and structured codes.
- **embed_threshold** — the minimum similarity score required for a vector search result to be returned, expressed as a float between 0 and 1. Only applies to embeddable fields. Setting it too low returns loosely related results; setting it too high may miss valid matches. Tune this per field based on how precise the expected queries are.

#### Link_attributes
Follows the same structure as Anchor_attributes, but applies to properties on the relationship itself rather than on a node. Use this when the edge between two entities carries its own data — for example, a start_date on a person → assigned_to → project relationship.
  
### 3. Links

Links define relationships between anchors. 

These relationships allow the assistant to:

- Navigate from a document → to its parts (chunks).
- Move from a chunk → to related FAQs.
- Combine data from different entity types.
- Correctly answer complex queries (e.g. “Show FAQs for this document”, “What chunks does this document have?”).

Examples:
- Product → available_at → Branch
- Product → belongs_to → Category
- Department → owns → Service

<img src="/images/Links-Vedana.png" alt="Hero" width="800" class="center-image" />
    
Each row in the Links table contains:

| Field                          | Description                                            |
| ------------------------------ | ------------------------------------------------------ |
| **Anchor1**                    | The first entity                                       |
| **Anchor2**                    | The second entity                                      |
| **Sentence**                   | A machine- and human-readable name of the relationship |
| **Description**                | What the relationship means                            |
| **Query**                      | How to retrieve the related objects                    |
| **anchor1_link_column_name**   | Link column name on the Anchor1 side                   |
| **anchor2_link_column_name**   | Link column name on the Anchor2 side                   |
| **has_direction** *(optional)* | Relationship direction (if applicable)                 |

The Links table defines how entities connect to each other. Each column describes one aspect of the relationship:

- **anchor1/anchor2** — the two entity types being connected. The relationship runs from anchor1 to anchor2.
- **sentence** — the edge label as it appears in the graph (e.g. PERSON_has_INTEREST). This is the relationship type used in Cypher queries.
- **description** — a plain-language explanation of what the relationship means (e.g. "person has an interest"). Included in the LLM context so the assistant understands when to traverse this edge.
- **query** — the Cypher query used to traverse this relationship. This is critical: without it, the assistant cannot perform multi-hop reasoning across connected entities.
- **anchor1_link_column_name/anchor2_link_column_name** — optional fields that specify which column on each anchor table holds the foreign key reference. Used when the link needs to be resolved through a specific column rather than inferred from the node ID.
- **has_direction** — specifies whether the relationship is directional. When set, traversal only follows the edge in the declared direction.

Links are directional and typed.
Without links, the graph has no structure.

### 4. Queries
The Queries table defines named retrieval strategies for specific question types the assistant is expected to handle:
- **query_name** — a description of the question pattern this strategy covers (e.g. Who likes <interest>?). The assistant uses this to match incoming user questions to the right retrieval plan.
- **query_example** — step-by-step instructions for how to answer this type of question: which tool to call first, what parameters to use, and what to do with the result. These instructions must be concrete and sequential — abstract descriptions of the desired outcome are not sufficient. The more precise the steps, the more reliably the assistant will follow them.

### 5. ConversationLifecycle
Defines system-level messages triggered at specific points in a conversation:
- **event** — the name of the lifecycle event (e.g. conversation start, session end).
- **text** — the message or instruction to inject at that point. Useful for greeting messages, session summaries, or fallback responses.

### 6. Prompts
Stores named prompt templates that the system uses for specific tasks:
- **name** — a unique identifier for the prompt (e.g. eval_judge_prompt, system_prompt).
- **text** — the full prompt text. The system prompt controls the assistant's overall tone, behavior, and constraints. The eval judge prompt controls how evaluation results are scored. Both can and should be customized for your specific domain and use case.

## Extending the Data Model

Vedana is not limited to documents. The system is domain-agnostic. What matters is how the domain is described. You can extend the model with any structured business data: product catalogs, price lists, branch addresses, opening hours, service catalogs, regulatory mappings, organizational structures, or any other typed entity your domain requires.
To extend the model, you must explicitly define each new anchor type, its attributes, and its links to other anchors. Nothing is inferred automatically.


## Where the Data Model is Stored

The data model is typically stored in **Grist** > **Data Model**, which contains the tables:

| Table                   | Purpose                                                             | 
| ----------------------- | ------------------------------------------------------------------- | 
| `Anchors`               | Defines node types (entities) in the graph                          | 
| `Anchor_attributes`     | Describes properties of node types, including embeddable fields     | 
| `Links`                 | Defines relationship types between nodes                            | 
| `Link_attributes`       | Describes properties of relationships                               | 
| `Queries`               | Example query scenarios that guide LLM behavior                     | 
| `Prompts`               | Customizable prompt templates for different query types             | 
| `ConversationLifecycle` | Predefined responses for assistant lifecycle events (e.g. `/start`) | 


This model defines how the assistant understands and uses knowledge, enabling it to generate structured and accurate responses.

### Important Note

The data model must fully represent all relevant columns and attributes from your actual data sources.
If the assistant gives incomplete or incorrect answers, the first thing to check is whether the necessary attributes and relationships are properly described in the data model.
    
**During ETL:**

1. Data model is loaded.
2. Graph schema is applied.
3. Anchors and links are validated.
4. Data is written to Memgraph.

## How the LLM Uses the Data Model

The data model is included directly in the LLM context. The LLM sees what anchor types exist, what attributes they carry, and what links connect them. Based on this, it selects tools, forms Cypher queries, filters by attributes, and traverses relationships.
The LLM does not invent structure. It operates within what has been declared. If something is not described in the data model, the tools cannot reliably query it — the data model defines the search space.

## Tools

The data model enables tool usage (SQL, Cypher, VTS, etc.).

Tools operate over:
- Anchors
- Links
- Attributes
    
If something is not described in the data model,  
the tools cannot reliably query it.
The data model defines the search space.

The tool library can be extended with:

- API queries
- Web search
- External reasoning tools

## Minimal vs Advanced Usage

At its most basic, Vedana with the default model behaves like an advanced RAG system — document retrieval with vector search and grounded answers.
When the model is extended with structured entities, Vedana becomes a domain knowledge engine capable of deterministic graph traversal, attribute filtering, multi-hop reasoning, and hybrid vector and structural retrieval.
The difference is entirely determined by how thoroughly the domain is modeled.
