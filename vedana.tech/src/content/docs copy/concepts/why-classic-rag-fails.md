---
title: "Why Classic RAG fails"
section: "Concepts"
order: 7
---

Before diving into Vedana’s architecture, it’s important to understand where classic RAG breaks down.

## What is Classic RAG?

Classic RAG connects an LLM to documents:


```
Chunks → Embeddings → Top-K → LLM → Answer
```


It works well when the answer is contained within a few text fragments.

## Where it works

Classic RAG is effective for:

- Summarization  
- Simple factual questions  
- Small document search  
- Approximate answers  

If the answer lives inside one or two paragraphs, RAG is usually enough.

## Where it breaks

Problems start when queries require completeness, structure, or logic.

### 1. Aggregation (“How many?”)

RAG cannot count across a dataset — it guesses from top-K results.

### 2. Exhaustive queries (“Show me all”)

RAG returns a sample, not the full set. Missing items are invisible.

### 3. Relationship queries

Questions requiring joins, graph traversal, or compatibility checks cannot be answered reliably.

### 4. Domain logic

RAG does not execute rules. It predicts text.

## Root cause

LLMs do not build a structured model of the domain.  
They operate on text patterns, not on data or logic.

As a result:

- No guaranteed consistency  
- No ability to enforce rules  
- No reliable reasoning over systems  

Classic RAG amplifies this by asking the model to reason over incomplete text fragments.

## What this means

These failures are structural.

They cannot be fixed by:

- better embeddings  
- larger context windows  
- increasing top-K  

Reliable answers require:

- access to full data (not samples)  
- structured queries (not similarity)  
- explicit relationships  
- executable logic  

Fluency is not correctness.

## Why Vedana exists

Reliable AI requires more than text retrieval.  
It requires a system that can:

- query structured data  
- execute logic  
- ensure completeness  
- and attach evidence to every answer  

That is the problem Vedana is built to solve.