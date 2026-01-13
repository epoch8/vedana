# ai-assistants-oss

**JIMS** - **J**ust an **I**ntegrated **M**ultiagent **S**ystem;

**Vedana** - (multi)agentic AI chatbot system built on top of JIMS with semantic RAG and knowledge graph as tools.

## Overview

This is a complete framework for building conversational AI systems. Key features include:

- **Thread-based conversation management** with persistent event storage
- **Semantic RAG** using [Memgraph](https://github.com/memgraph/memgraph) (knowledge graph) + [pgvector](https://github.com/pgvector/pgvector) (vector search)
- **Business-defined data model** managed through [Grist](https://github.com/gristlabs/grist-core) spreadsheets
- **Multiple interfaces**: Telegram bot, Terminal UI, Web backoffice
- **Incremental ETL** built with [Datapipe](https://github.com/epoch8/datapipe)

## Repository Structure

```
ai-assistants-oss/
├── libs/                    # Reusable libraries
│   ├── jims-core/           # Core JIMS framework
│   ├── jims-backoffice/     # FastAPI backoffice for JIMS
│   ├── jims-telegram/       # Telegram bot adapter
│   ├── jims-tui/            # Terminal UI for testing
│   ├── vedana-core/         # Core Vedana framework
│   ├── vedana-backoffice/   # Reflex-based admin UI
│   └── vedana-etl/          # ETL pipeline (Datapipe-based)
├── projects/
│   ├── vedana/              # Main Vedana deployment
│   └── jims-demo/           # JIMS demo project
└── pyproject.toml           # UV Workspace configuration
```

## Components

### JIMS (Just an Integrated Multiagent System)

JIMS is a framework for building conversational AI systems with persistent thread management.

#### Key Concepts

- **Thread**: A conversation between user(s) and the agentic system
- **Event**: Something that happens in a thread (messages, actions, state changes)
- **Pipeline**: An async function that processes a `ThreadContext` and produces events
- **ThreadContext**: Spawned by ThreadController, an object that stores and handles all thread-related data during a `Pipeline` execution
- **ThreadController**: Manages thread lifecycle, event storage, and `Pipeline` execution

#### Example Event Structure
```json
{
  "event_id": "...",
  "event_type": "comm.user_message",
  "event_data": {
    "role": "user",
    "content": "Hello!"
  }
}
```

### Vedana

Vedana is an agentic AI system built on JIMS that provides Graph RAG capabilities.

#### Features

- **Graph-based knowledge retrieval** using Cypher queries on Memgraph
- **Semantic vector search** using pgvector embeddings
- **Dynamic data model filtering** to optimize token usage
- **Configurable prompts and query templates** via Grist
- **Conversation lifecycle management** (custom /start responses, etc.)

#### RAG Pipeline Flow

1. User sends a message
2. (Optional) Data model filtering selects relevant anchors/links
3. LLM generates Cypher and/or vector search queries using tools
4. Results are retrieved from Graph + Vector stores
5. LLM synthesizes final answer from retrieved context

### Vedana ETL

The ETL pipeline ingests data from Grist into the graph and vector databases.

#### Pipeline Stages

1. **Extract**: Load data model and data. In the most basic form data is loaded from Grist, 
but the pipeline can be easily extended to incorporate other sources
2. **Transform**: Process data into nodes and edges, generate embeddings
3. **Load**: Update knowledge graph and store pgvector embeddings

## Requirements

- Python 3.12
- PostgreSQL with pgvector extension
- Memgraph
- Grist (for data model and data source)
- OpenAI API key (or compatible LLM provider)

## Quick Start

**JIMS** manages conversations as **threads** containing **events** (messages, actions, state changes). A **pipeline**, provided by Vedana in this case, processes user input and produces response events.

**Vedana** provides a RAG pipeline that:
1. Receives user query
2. LLM generates Cypher / vector search queries as tool calls
3. Retrieves context from graph + vector stores
4. LLM synthesizes the answer

The **data model** (node types, relationships, attributes) is defined in Grist spreadsheets and synced via ETL.

## Configure

### Data Model (Grist)

The data model is configured via tables in Grist workspace:

| Table                   | Purpose                                               |
|-------------------------|-------------------------------------------------------|
| `Anchors`               | Node types (entities) in the graph                    |
| `Anchor_attributes`     | Properties of node types, including embeddable fields |
| `Links`                 | Relationship types between nodes                      |
| `Link_attributes`       | Properties of relationships                           |
| `Queries`               | Example query scenarios for the LLM                   |
| `Prompts`               | Customizable prompt templates                         |
| `ConversationLifecycle` | Responses for lifecycle events (e.g., /start)         |

### LLM Models

Models are handled via [LiteLLM](https://www.litellm.ai/) 
(with [OpenRouter](https://openrouter.ai/) support inside for easier usage and access management), 
configurable via environment variables for production and in backoffice UI for testing:

| Variable           | Purpose                                                               |
|--------------------|-----------------------------------------------------------------------|
| `MODEL`            | Main question answering model                                         |
| `FILTER_MODEL`     | Data model filtering (smaller, faster model for a preprocessing step) |
| `EMBEDDINGS_MODEL` | Text embeddings generation                                            |
| `EMBEDDINGS_DIM`   | Embedding dimensions                                                  |

### Run

This repository is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/):
```bash
uv sync
```

Fill the `.env` based on the `.env.example` [here](projects/vedana/.env.example)

```bash
cd projects/vedana
docker-compose up -d 
```

## Observability

- **OpenTelemetry** tracing for pipeline execution
- **Prometheus** metrics for LLM usage, pipeline duration
- **Sentry** integration for error tracking (optional)

## Contributing & Development

GitHub Actions pipelines are generated automatically using [uv-workspace-codegen](https://github.com/epoch8/uv-workspace-codegen)

TODO

## License

TODO
