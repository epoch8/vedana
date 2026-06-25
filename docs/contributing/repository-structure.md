---
title: Repository Structure
section: Contributing
order: 4
---

# Repository Structure

This document describes the structure of the Vedana repository so it's easier to find your way around.

## Root

```
vedana/
├── apps/                     # runnable applications
├── libs/                     # reusable libraries
├── pyproject.toml             # uv workspace config
├── uv.lock                    # uv lock file
├── Makefile                   # build/publish commands
├── README.md                  # short overview
├── LICENSE                    # licence
├── .python-version            # 3.12
├── .pre-commit-config.yaml    # pre-commit hooks
├── .gitignore                 # ignored
├── .gitattributes             # attributes for linters
├── .github/                   # GitHub Actions workflows
└── .devcontainer/             # VS Code Dev Container
```

## `apps/`

Runnable applications using the libraries.

```
apps/
├── vedana/                   # the main Vedana production service
│   ├── CHANGELOG.md
│   ├── README.md              # quick orientation for the deployable app
│   ├── Dockerfile
│   ├── Makefile
│   ├── alembic.ini            # migrations
│   ├── docker-compose.yml     # stack: app + api + widget + db + memgraph + grist
│   ├── data/                  # mounted into containers (currently empty; reserved for ad-hoc seed data / artifacts)
│   ├── infra/                 # auxiliary configs (Caddy, Grist init seeds)
│   ├── migrations/            # Alembic migrations
│   ├── pyproject.toml
│   ├── .env.example
│   └── .env.ci-cd
└── jims-demo/                 # a minimal JIMS-only example without Vedana
    ├── jims_demo/
    │   ├── app.py              # JimsApp wiring (sessionmaker + custom pipeline)
    │   ├── db.py               # SQLAlchemy session/engine setup
    │   └── simple_pipeline.py  # tiny example Pipeline implementation
    └── migrations/             # Alembic migrations for the demo's Postgres tables
```

## `libs/`

Workspace libraries. Each one is a separate package with its own `pyproject.toml`.

```
libs/
├── jims-core/                # JIMS kernel: threads, events, LLMProvider
│   └── src/jims_core/
│       ├── app.py             # JimsApp
│       ├── db.py              # ThreadDB, ThreadEventDB
│       ├── schema.py          # Pipeline Protocol
│       ├── util.py            # uuid7, helpers, load_jims_app
│       ├── llms/
│       │   └── llm_provider.py
│       └── thread/
│           ├── schema.py       # CommunicationEvent, EventEnvelope
│           ├── thread_context.py
│           └── thread_controller.py
│
├── jims-api/                 # FastAPI HTTP API
│   └── src/jims_api/
│       └── main.py            # CLI and create_api
│
├── jims-widget/              # web widget
│   └── src/jims_widget/
│       ├── server.py
│       ├── main.py
│       └── static/            # widget frontend
│
├── jims-telegram/            # Telegram bot
│   └── src/jims_telegram/
│       ├── main.py
│       └── md2tgmd.py
│
├── jims-tui/                 # Terminal UI (Textual)
│   └── src/jims_tui/
│       ├── chat_app.py
│       └── main.py
│
├── jims-backoffice/          # minimal FastAPI backoffice (for JIMS demos)
│   └── src/jims_backoffice/
│       ├── app.py             # route setup, healthcheck, HTML landing
│       ├── main_app.py        # FastAPI() instance
│       ├── forms.py           # FastUI form schemas
│       ├── routes/            # event / home routes
│       ├── settings.py
│       └── utils.py
│
├── vedana-core/              # Vedana kernel: RAG pipeline, agent, tools
│   └── src/vedana_core/
│       ├── app.py             # VedanaApp, make_*_app
│       ├── data_model.py      # Anchor, Link, Attribute, DataModel
│       ├── data_provider.py   # Grist API/CSV providers
│       ├── db.py              # get_sessionmaker
│       ├── graph.py           # Graph, MemgraphGraph
│       ├── llm.py             # LLM, Tool, prompt templates
│       ├── rag_agent.py       # RagAgent
│       ├── rag_pipeline.py    # RagPipeline, StartPipeline
│       ├── settings.py        # VedanaCoreSettings
│       ├── start_pipeline.py
│       ├── utils.py
│       └── vts.py             # VectorStore, PGVectorStore, MemgraphVectorStore
│
├── vedana-etl/               # ETL on Datapipe
│   └── src/vedana_etl/
│       ├── catalog.py          # all Datapipe tables
│       ├── config.py           # DBCONN, MEMGRAPH_CONN_ARGS
│       ├── pipeline.py         # Pipeline assembly
│       ├── schemas.py          # GENERIC_NODE/EDGE_DATA_SCHEMA
│       ├── settings.py
│       ├── steps.py            # step implementations
│       └── store.py            # custom Datapipe stores
│
└── vedana-backoffice/        # Reflex admin UI
    └── src/vedana_backoffice/
        ├── components/
        ├── graph/
        ├── pages/
        ├── state.py
        ├── states/
        ├── ui.py
        ├── start_services.py
        ├── vedana_backoffice.py
        ├── Caddyfile
        └── util.py
```

## `pyproject.toml` (root)

Configures the uv workspace, mypy, ruff:

```toml
[tool.uv.workspace]
members = [
    "libs/jims-*",
    "libs/vedana-*",
    "apps/vedana",
    "apps/vedana-*",
    "apps/jims-*",
]

[tool.uv.sources]
jims-backoffice = { workspace = true }
jims-core = { workspace = true }
# ... every internal dep is { workspace = true }
```

This means in workspace mode, `uv sync` installs every internal library **from local paths**, not PyPI. So you can edit one library's code and immediately see the effect in another.

## Makefile

```
make build_jims_core         # build a specific package
make build                    # build every package
make build-vedana-project     # build the Vedana Docker image
make publish                  # publish (with a GCP token)
make clean                    # clean dist/
```

## CI/CD

GitHub Actions workflows are auto-generated through [uv-workspace-codegen](https://github.com/epoch8/uv-workspace-codegen). Configuration on the library side is in each library's `pyproject.toml`.

On a PR, it runs:

- linters;
- tests;
- type checker;
- Docker image build.

On merge to `main`:

- packages are published to a private PyPI;
- Docker images are built and pushed.

## Where to look for what

| I want to...                                          | Go to...                                                       |
| ---------------------------------------------------- | -------------------------------------------------------------- |
| Change how a user question is answered               | `libs/vedana-core/src/vedana_core/rag_pipeline.py` (`RagPipeline`) |
| Add a new tool                                        | `libs/vedana-core/src/vedana_core/rag_agent.py` + `llm.py`     |
| Change how the data model is read                     | `libs/vedana-core/src/vedana_core/data_model.py`               |
| Change thread / event structure                       | `libs/jims-core/src/jims_core/thread/`                         |
| Add a new ETL data source                             | `libs/vedana-etl/src/vedana_etl/steps.py` + `pipeline.py`     |
| Change the REST API                                   | `libs/jims-api/src/jims_api/main.py`                            |
| Change the backoffice UI                               | `libs/vedana-backoffice/src/vedana_backoffice/pages/`          |
| Change a default prompt template                       | `libs/vedana-core/src/vedana_core/llm.py` or `rag_pipeline.py` |
| Add a Postgres schema migration                       | `apps/vedana/migrations/versions/`                             |
| Change docker-compose                                 | `apps/vedana/docker-compose.yml`                                |
| Change the default Memgraph password                   | `apps/vedana/.env.example` + your `.env`                       |

## What's next

- [Code Style](./contributing/code-style.md)
- [Testing](./contributing/testing.md)
- [Contributing Guide](./contributing/contributing.md)
