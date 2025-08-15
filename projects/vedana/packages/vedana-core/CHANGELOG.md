# WIP

* (vedana-core) More tool call iterations + add final prompt on tool call limit

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
