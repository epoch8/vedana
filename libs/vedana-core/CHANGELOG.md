# 2025.11.25

* `RagPipeline`: Return `agent_query_events` from `process_rag_query`
* `LifecycleEvents`: Simplify `/start` command response logic

# 2025.10.29

* Rm `temperature` parameter from all LLM-related methods

# 2025.09.05

* Add `JimsApp` and `VedanaApp` construction to `vedana_core`, now we can run
  Vedana with standard `jims-tui` and `jims-telegram` runners

# 2025.08.29

* Make conversation history always present instead of a tool call

# 0.5.0

* (vedana) Refactor Vedana into packages

# 2025.07.19-dev.7

* (vedana) Rename memgraph-rag to vedana

# 2025.07.18-dev.5

* (jims-backoffice) Add feedback form

# 0.4.0

* Add jims-backoffice to deployment
* Add helm chart

# 0.3.1

* Execute LLM tool calls in parallel

# 0.3.0

* Add start conversation pipeline
* Remove top_n query result limit in cypher_fn
* Improve Grist parsing logic - filter out meta-columns

# 0.2.0

* Add Sentry and opentelemetry tracing
* Remove embedding cache files
* fix: exit tool call loop if no tool calls found

## Migration

Add env variables:
* `SENTRY_DSN`
* `SENTRY_ENVIRONMENT`
