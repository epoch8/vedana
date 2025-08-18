## Basic pipeline for all vedana projects.

This pipeline:

- Parses Grist Data & Data Model
- Ensures that Memgraph index/vector index structure is in sync with data model
- Updates Memgraph database in incremental fashion

To add steps:
1. pass extra transformations to [get_pipeline](src/pipeline.py)
2. pass extra tables to [init_catalog](src/catalog.py)
3. create new app configuration from [app.py](src/app.py)