# 2025.09.22-dev.21

* (vedana-eval/etl) Refactored evaluation script into a Datapipe pipeline
* (jims/vedana) Introduce and use `JimsApp`/`VedanaApp`, switch from `vedana-tg`
  to `jims-telegram`

# 2025.09.04-dev.20

* (vedana-etl) Refactor to var-style Table declaration in pipeline
* (vedana-etl) covered get_grist_data / filter_grist_data with tests
* (vedana-etl) more fixes in processing "reference" type columns for both links
  and attrs
* (vedana-etl) mv data filtering logic into get_grist_data, rm separate filter steps

# 2025.08.29-dev.19

* (vedana-etl) fixed loading bidirectional links in base pipeline
* (vedana-etl) use Grist API, fixing parsing formulas and reference fields
* (vedana-core) Make conversation history always present instead of a tool call
* (vedana-core) More tool call iterations + add final prompt on tool call limit

# 2025.08.21-dev.18

* (src) Reshuffle folders in monorepo
* (vedana) Consolidate databases (jims + datapipe), now it's one database
* (vedana-eval) Implement eval functionality

# 2025.08.15-dev.17

* (vedana-gradio) Remove obsolete UI components
* (vedana-core) add tests for DataModel
* (vedana-core) rm obsolete importers code
* (vedana-core) update DataModel graph update method use
* (vedana-etl) Filter edges against datamodel on import
* (vedana-core) handle LLM temperature in LiteLLM
* (vedana-etl) Fix links import from Grist
* (vedana-gradio) Display version in demo UI

# 2025.08.13-dev.16

* (vedana-etl) Implement parsing links from Grist links
* (vedana-etl) Add datamodel node to graph
* (vedana-gradio) Add GPT-5 to model selection
* (vedana-gradio) Fix chat completion for gpt-5
* (vedana-gradio) Remove obsolete UI components
* (jims-core) Use LiteLLM response_cost
* (vedana-gradio) Localized data model representation

# 2025.08.07-dev.15

* (vedana-core) Ported work with Memgraph to async
* (vedana-core) Do not use attr.dtype to determine eligibility for VTS
* (vedana-gradio) Migrate to pure async

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
