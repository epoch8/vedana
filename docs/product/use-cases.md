---
title: Use Cases
section: Product
order: 1
---

# Use Cases

Vedana is built for domains where **answer correctness** matters more than "fluency of text". Below are the typical scenarios where it outperforms classic RAG.

## E-commerce and catalogs

**What it can do**

- Answer questions about specific products with current prices and stock;
- filter by attributes ("all frost-free fridges under 100k");
- compare products by attributes;
- check availability at specific branches;
- map products to regulatory requirements (category → requirements → required certificates).

**Why classic RAG isn't a fit**

Prices, stock, and characteristics are structured data that change. Vector search will find old mentions, not actual values. Vedana, via Cypher, goes directly to the graph and returns current values.

**Example**

> "Which euro-book sleeper sofas under £600 do you have in stock at the Nevsky store?"

→ Vedana builds Cypher with filters `product.mechanism="euro-book"`, `product.price < 600`, `(p)-[:PRODUCT_available_at_BRANCH]->(:branch {name: "Nevsky"})`, and returns the full list with prices and SKUs.

## Legal and compliance

**What it can do**

- Which regulations / requirements apply to a product / contract / category;
- which documents regulate which area;
- contract dates and deadlines;
- approval statuses and blockers;
- multi-hop reasoning across documents and regulators.

**Example**

> "Which documents do we need to show the customer when selling built-in appliances in the EU?"

→ Vedana traverses `product → category → regulation → required_documents` and returns the requirements list with citations from the documents.

## Internal knowledge bases

**What it can do**

- search policies, regulations, instructions;
- answer org structure questions ("who is responsible for X", "who do I talk to about Y");
- explain processes with citations from the source;
- support employee onboarding.

**Example**

> "What's the approval process for an equipment purchase budget over 500k?"

→ Vedana finds chunks of the regulation + walks the graph "process → stages → owners", then gives a step-by-step answer with a quote from the document.

## B2B and dealer support

**What it can do**

- stable answers to typical dealer questions;
- export specs, price lists, partner terms;
- product compatibility checks, configurator;
- routing requests to the right manager.

**Example**

> "Can product X be bundled with Y, and what would the final price be for our dealer tier?"

→ Vedana checks compatibility through `(x)-[:COMPATIBLE_WITH]-(y)` and computes the price including the dealer tier through a structured query.

## HR and org structure

**What it can do**

- who on the team works on which project;
- who's responsible for a process / system / customer;
- who reports to whom and who to escalate to;
- schedules, vacations, on-call rotations.

**Example**

> "Who handles the integration with customer Foo, and who is their sales manager?"

→ `(:client {name: "Foo"})-[:HANDLED_BY]->(:person)-[:REPORTS_TO]->(:person)` — Vedana returns two people with contacts.

## Finance and reporting

**What it can do (with care)**

- answers about reference info (policies, limits, procedures);
- structured queries about deal statuses, contracts;
- helping navigate reports.

**What you shouldn't do**

- compute revenue directly through the LLM. Numbers come from BI/DWH through manual dashboards or via a custom tool.

## Less suitable scenarios

Vedana **won't replace**:

- **Creative writing.** Vedana is rigidly tied to data; the LLM has little freedom.
- **Free-form summarization.** If the task is "read it and tell me in your words", classic RAG is simpler.
- **Pure conversational AI without a domain.** With no structured data and complex questions, Vedana is overkill.
- **Code / API doc search.** Better to use specialised tools (e.g. Cursor / GitHub Copilot / Cody).
- **Real-time analytics on huge volumes.** Memgraph is analytical, but it's still a graph database, not Clickhouse.

## A template for evaluating a new case

Before taking on a case, answer three questions:

1. **Can the answer be expressed as a structured query or a graph traversal?** If yes, Vedana fits best.
2. **Does the answer live in text documents, and does it need meaning-based search?** If yes, Vedana via vector search + chunking.
3. **Can a combination of both be used?** If yes, that's the best fit for Vedana.

If all three are "no", look at other tools.

## What's next

- [Comparison with Classic RAG](./product/comparison.md)
- [Quality Metrics & Evaluation](./product/evaluation.md)
- [Limitations](./product/limitations.md)
