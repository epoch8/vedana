---
title: "Links"
section: "Data model"

---

## What Is a Link

A **link** is a typed relationship between two anchors. 

If anchors define _entities_, and attributes define _properties_, then links define _relationships_.

Links turn isolated nodes into a graph and make traversal, multi-hop reasoning, and structural queries possible. Without links, Vedana cannot navigate the graph. It can only look things up in isolation.

Examples:

- Product → belongs_to → Category
- Product → available_at → Branch
- Contract → signed_with → Company
- Requirement → applies_to → Product

## Where Links Are Defined

Links are defined in Grist inside the **Data Model** document.
There is a dedicated table (often named something like `links`) that defines:

- link_type
- source_anchor
- target_anchor
- description
- optional flags

This table defines the schema of relationships, not the actual relationships themselves.
Actual relationships are created during ETL based on structured data tables you provide.

## How to Describe a Link

Each link definition typically includes:

- link_name — a verb-like identifier (e.g., `belongs_to`).
- source_anchor — the anchor the relationship originates from (e.g., `Product`).
- target_anchor — the anchor the relationship points to (e.g., `Category`).
- direction — always explicit, even if the relationship feels symmetric.
- description — a human-readable explanation of what the relationship means.
    
Links are directional by design.
Even if the relationship is conceptually symmetric,the graph direction must be explicit.

### Example Link Definition

Example: Product → belongs_to → Category

Link definition row in data-model:

- link_name: belongs_to
- source_anchor: Product
- target_anchor: Category
- description: "Indicates which category a product belongs to"
    
During ETL:

If `products` table contains a `category_id` column,  
ETL creates edges:

```css
(:Product { product_id: "123" })-[:belongs_to]->(:Category { category_id: "laptops" })
```

Now Cypher queries can traverse:

```sql
MATCH (p:Product)-[:belongs_to]->(c:Category)
WHERE c.name = "Laptops"
RETURN p
```

This is impossible with document chunks alone.

## Link vs Attribute

A common modeling question:

> Should something be an attribute or a link?

The answer depends on how the target will be used.

Use a **string attribute** when the value is simple and will only ever be read — for example, storing `Product.category = "Laptops"` as a plain string.

Use a **link** when:

- The target entity has its own attributes.
- You may query it independently.
- It participates in multiple relationships.

Example:

- Product.category as string → attribute (simpler).
- Product → belongs_to → Category → anchor (more powerful).
    
The second approach (modeling Category as a separate anchor connected via `belongs_to`) enables category-level filtering, hierarchy modeling, and multi-hop queries that a string attribute cannot support.
The more the target participates in the graph, the stronger the case for making it an anchor with a link.

## Multi-Hop Reasoning

Links enable multi-hop queries.

Example:
Product → belongs_to → Category  
Category → regulated_by → LegalDocument

With this structure, Vedana can answer: **"Which legal documents regulate products in category X?"**

It's a question that requires traversing two relationships in sequence. 
Vector search cannot reliably perform this. Graph traversal can.

## How Links Affect LLM Behavior

Link definitions are included in the LLM context. The LLM sees which relationships exist, which anchor types are connected, and how to traverse the graph. This allows it to generate valid Cypher queries, perform multi-hop reasoning, and avoid inventing relationships that do not exist.

If links are not defined, the LLM has no graph to explore and falls back to text-based guessing.

## Best Practices

- Use clear verb-like names for links (belongs_to, applies_to, located_in).
- Keep direction consistent across model.
- Avoid redundant reciprocal links unless necessary.
- Model real-world relationships explicitly.

Clean link modeling directly improves reasoning quality.


## Common Mistakes

The most frequent mistakes include:

- Encoding relationships as string attributes instead of links.
- Creating too many generic links (e.g., "related_to").
- Not defining link direction clearly.
- Creating circular structures without clear semantics.

All of these reduce the determinism and clarity that make Semantic RAG work.
