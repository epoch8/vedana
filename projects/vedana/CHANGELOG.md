# 2025.08.06-dev.14

* (deploy) Enable GCP LLM settings in vedana deploy
* (deploy) Switch from `datapipe` to `simple-cronjob` helm chart for datapipe
  deployment
* (vedana-core) Fix bug in vector search tool generation for LLM
* (vedana-core) Fix bug data model read
* (vedana-etl) Add better exception logging in `get_grist_data`

# 2025.08.05-dev.13

* (vedana) Use jims_core.LLMProvider directly for embeddings

# 2025.08.04-dev.12

* (vedana) Get system prompts from Grist's DataModel, with fallback to hardcoded
  prompts

# 2025.08.01-dev.11

* (jims-core) Switch to LiteLLM for LLMProvider functioning

# 0.5.0

* (vedana) Refactor Vedana into packages: 
  * `vedana-core`
  * `vedana-gradio`
  * `vedana-tg`
  * `vedana-etl`

# 2025.07.22-dev.9

* (vedana) Turn off built-in password auth

# 2025.07.21-dev.8

* (vedana) Translate UI fields and pipeline logs in English

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
