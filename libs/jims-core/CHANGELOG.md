# 2026.01.29 - 0.4.1

* Republish to PyPI

# 2025.11.25 - 0.3.1

* `ThreadContext`: Add `thread_config` field
* `ThreadContext`: Add `get_last_user_action` method
* `ThreadContext`: Update `send_message` to use `EventEnvelope[CommunicationEvent]`
* `ThreadContext`: Update `set_state` and `get_state` to support named states
* `Pipeline` protocol: Change return type to `Any`

# 2025.10.29 - 0.3.0

* Add `contact_id` field to threads
* Rm `temperature` parameter from `LLMProvider` interface

# 2025.09.05

* Add `jims_core.JimsApp` abstraction

# 2025.08.29

* Switch from litellm tracing to openinference
* Add utility command `setup_monitoring_and_tracing_with_sentry`

# 2025.08.06

* Enable litellm otel tracing

# 0.2.0

* `LLMProvider` now works on `LiteLLM`, allowing for different LLM providers

# 0.1.0

Initial commit
