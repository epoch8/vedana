---
title: "Semantic RAG overview"
section: "Concepts"
order: 5
---

A Semantic RAG system consists of four parts:

**Data + Data Model + Tools + Assistant**

- **Data** - documents and structured records  
- **Data Model** - entities and relationships  
- **Tools** - operations (search, filter, count, traverse)  
- **Assistant** - selects and orchestrates tools  

Without this structure, the system guesses.  
With it, the system can produce correct answers.

## What This Enables

The system can:

- return complete sets (not samples)  
- compute exact values (not estimates)  
- evaluate relationships (not guess them)  
- enforce domain logic (not approximate it)  

## How It Works

<img src="/images/Screenshot%202026-02-19%20at%2016.42.33.png" alt="Hero" width="800" class="center-image" />


1. User asks a question
2. Assistant identifies the query type
3. Assistant selects the appropriate tool
4. Tool executes on data
5. Assistant returns the result with sources

## Required Properties

A real Semantic RAG system must have:

Deterministic execution — the same question triggers the same operations
Verifiable sources — every answer is traceable
Guaranteed completeness — when “all” is requested, all results are returned

If these are missing, it is still classic RAG.

## What Vedana Does

Vedana implements Semantic RAG to make AI outputs:

- verifiable
- aligned with real data and logic.