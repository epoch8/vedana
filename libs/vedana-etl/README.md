# Basic pipeline for all vedana projects.

This pipeline:

- Parses Grist Data & Data Model
- Ensures that Memgraph index/vector index structure is in sync with data model
- Updates Memgraph database in incremental fashion

To add steps:
1. Pass extra transformations to [get_pipeline](src/pipeline.py)
2. Create new app configuration from [app.py](src/app.py)

## Pipeline Labels Hierarchy

### Pipeline

`labels=("pipeline", "pipeline_name")` defines a set of operations as standalone, sort of like a DAG in Airflow 
or a Dagster Job. Its purpose is to be able to render it as a separate tab on the ETL page of Backoffice in order to 
look at it independently of other transformations

### Stage

`labels=("stage", "stage_name")` defines a stage of `pipeline`. Currently, stages are useful for creating and managing 
observability features, such as [main dashboard's](/libs/vedana-backoffice/vedana_backoffice/pages/main_dashboard.py) 
Ingest table, which displays DataTable's of all transformations with `labels=("stage", "extract")`. 
Stages are also useful when running the pipeline manually.

### Flow

`labels=("flow", "flow_name")` helps execute a `pipeline` (or possibly several pipelines) in a nice fashion, 
used in defining cron jobs, etc.