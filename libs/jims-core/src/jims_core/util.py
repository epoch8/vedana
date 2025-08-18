from typing import TYPE_CHECKING
from uuid import UUID

from uuid_extensions import uuid7 as uuid7_impl

if TYPE_CHECKING:
    from jims_core.app import JimsApp


def uuid7() -> UUID:
    return uuid7_impl(as_type="UUID")  # type: ignore


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
    set_global_textmap(SentryPropagator())

    start_http_server(8000)


def load_jims_app(app_name: str) -> "JimsApp":
    from jims_core.app import JimsApp

    app_split = app_name.split(":")

    if len(app_split) == 1:
        module_name = app_split[0]
        app_attr = "app"
    elif len(app_split) == 2:
        module_name, app_attr = app_split
    else:
        raise Exception(f"Expected APP in format 'module:app' got '{app_name}'")

    from importlib import import_module

    # sys.path.append(os.getcwd())

    app_mod = import_module(module_name)
    app = getattr(app_mod, app_attr)

    assert isinstance(app, JimsApp)

    return app
