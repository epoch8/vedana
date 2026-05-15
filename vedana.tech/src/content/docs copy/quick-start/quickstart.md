---
title: "Quick Start"
section: "Quick Start"
order: 1
next: first-data-model
---

In this guide, you will:

- Start the Vedana stack
- Load a data model + data
- Run ETL (build graph + embeddings)
- Ask your first question

By the end, you will have a working system that:

- understands your domain
- queries structured data
- returns grounded answers


## Before you start

You need:
- Docker
- LLM API key (OpenAI / OpenRouter / Google)

## Step 1 – Run the system

```bash
git clone https://github.com/epoch8/vedana
cd vedana

cp apps/vedana/.env.example apps/vedana/.env

```
Open `apps/vedana/.env` and set your LLM API key:

```env
OPENAI_API_KEY="your_api_key_here"
OPENROUTER_API_KEY="your_api_key_here"
```

or if you want to run Google/VertexAI models:

```
GOOGLE_APPLICATION_CREDENTIALS="path-to-creds.json"
```

Then start everything:

```
docker compose -f apps/vedana/docker-compose.yml up --build -d
```

## Step 2 – Check that everything is alive

Open:

- Vedana API (FastAPI) → http://localhost:8080/docs
- Vedana Backoffice → http://localhost:9000
- Memgraph Lab → http://localhost:3000
- Grist → http://localhost:8484

Also these services will run:
- Postgres + pgvector → localhost:5432
- Memgraph → localhost:7687


## Step 3 – Run ETL (this is where the system actually “comes alive”)

Go to backoffice:

→ http://localhost:9000/etl

Click Run Selected

<img src="/images/docs/etl-run-selected.png" alt="ETL Run Selected" width="800" class="center-image" />


This step does three things:

- Loads your data model
- Loads your data
- Builds graph (Memgraph) and embeddings (pgvector)

If ETL didn’t run – Vedana is not working, even if UI opens.

## Step 4 – Ask your first question

Go to chat:

→ http://localhost:9000/chat


Ask something like:

- Who likes tapirs?
- Who likes quokkas?
- Who likes spinach?


<img src="/images/docs/chat-question.png" alt="Chat question" width="800" class="center-image" />


### How to inspect Vedana's answers

Near Vedana's answer press 'Details'.

<img src="/images/docs/chat-details.png" alt="Chat question" width="800"  class="center-image" />


Inspect the logs to see how Vedana reasons.

## Step 5 – Stop the System

Stop containers:

```bash
docker compose -f apps/vedana/docker-compose.yml down
```

Stop and remove volumes (destructive reset):

```bash
docker compose -f apps/vedana/docker-compose.yml down -v
```