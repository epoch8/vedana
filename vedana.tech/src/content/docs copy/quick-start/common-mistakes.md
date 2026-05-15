---
title: Common Mistakes and FAQ
section: "Quick Start"
order: 3
---

## Common Mistakes and FAQ

### Q: The assistant is not using graph search and is giving vague or incomplete answers. What is wrong?

A: The most common cause is missing Cypher queries in the data model. If the **`query`** fields in the Anchors, Attributes, and Links tables are empty or incomplete, the assistant has no reliable way to traverse the graph and will fall back to less precise retrieval methods. Make sure every anchor and every attribute that should be queryable has a valid Cypher query defined. This is one of the highest-impact things you can do to improve answer quality.


### Q: I have defined queries in the Queries table but the assistant still does not follow the expected retrieval steps. What am I missing?

A: The **`query_example`** field in the **Data Model > Queries** table needs concrete, step-by-step instructions, not a general description of what the answer should contain. Instead of writing "find the relevant interest and return the people associated with it", specify exactly which tool to call first, what parameters to pass, and what to do with the result before calling the next tool. The assistant follows these instructions literally, so the more precise the steps, the more predictable the behavior.

**Example:** 
```
1) Use vector_text_search to get the interest node: 
vector_text_search(on="node", label="interest", property="interest_name", text='<interest>') 
retrieve "node_id"

2) Use Cypher to get all persons who are connected via PERSON_has_INTEREST link:
MATCH (p:person)-[:PERSON_has_INTEREST]->(i:interest) WHERE i.id=$node_id RETURN p.person_name, i.interest_name
```

<img src="/images/Queries.png" alt="Hero" width="800" class="center-image" />

### Q: The assistant gives reasonable answers but the tone, format, or style does not fit our product. Do we need to rebuild the data model?

A: No. The data model controls what the assistant can retrieve, not how it communicates. To adjust tone, format, or behavior, edit the system prompt in the **Data Model > Prompts** table. The default prompts work well as a starting point, but for any production deployment it is strongly recommended to customize the system prompt for your specific domain, audience, and use case. This is usually a small change with a noticeable impact on the overall experience.
