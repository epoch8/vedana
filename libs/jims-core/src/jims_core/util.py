import logging
from uuid import UUID

from openinference.instrumentation.litellm import LiteLLMInstrumentor
from uuid_extensions import uuid7 as uuid7_impl


def uuid7() -> UUID:
    return uuid7_impl(as_type="UUID")  # type: ignore


def setup_verbose_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def setup_monitoring_and_tracing_with_sentry() -> None:
    from opentelemetry import trace
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.sdk.trace import TracerProvider

    try:
        import sentry_sdk
        from sentry_sdk.integrations.opentelemetry import SentryPropagator, SentrySpanProcessor
    except ImportError:
        raise ImportError("install sentry sdk")

    from prometheus_client import start_http_server

    sentry_sdk.init(
        send_default_pii=True,
        traces_sample_rate=1.0,
        instrumenter="otel",
    )

    provider = TracerProvider()
    provider.add_span_processor(SentrySpanProcessor())
    trace.set_tracer_provider(provider)

    LiteLLMInstrumentor().instrument(tracer_provider=provider)

    set_global_textmap(SentryPropagator())

    start_http_server(8000)
