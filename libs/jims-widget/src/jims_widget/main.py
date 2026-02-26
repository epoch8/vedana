import asyncio
from typing import Awaitable

import click
import uvicorn
from jims_core.app import JimsApp
from jims_core.util import (
    load_jims_app,
    setup_monitoring_and_tracing_with_sentry,
    setup_prometheus_metrics,
    setup_verbose_logging,
)
from loguru import logger

from jims_widget.server import create_widget_app


async def _resolve_jims_app(app_name: str) -> JimsApp:
    loaded_app = load_jims_app(app_name)
    if isinstance(loaded_app, Awaitable):
        loaded_app = await loaded_app
    return loaded_app


@click.command()
@click.option("--app", type=click.STRING, default="app", help="JIMS app in module:attr format.")
@click.option("--host", type=click.STRING, default="0.0.0.0")
@click.option("--port", type=click.INT, default=8090)
@click.option("--cors-origins", type=click.STRING, default="*", help="Comma-separated CORS origins, or * for all.")
@click.option("--enable-sentry", is_flag=True, help="Enable tracing to Sentry", default=False)
@click.option("--metrics-port", type=click.INT, default=8001)
@click.option("--verbose", is_flag=True, default=False)
def cli(
    app: str,
    host: str,
    port: int,
    cors_origins: str,
    enable_sentry: bool,
    metrics_port: int,
    verbose: bool,
) -> None:
    if verbose:
        setup_verbose_logging()

    setup_prometheus_metrics(port=metrics_port)

    if enable_sentry:
        setup_monitoring_and_tracing_with_sentry()

    origins = [o.strip() for o in cors_origins.split(",") if o.strip()]

    async def run() -> None:
        jims_app = await _resolve_jims_app(app)
        widget_app = create_widget_app(jims_app, cors_origins=origins)
        server = uvicorn.Server(uvicorn.Config(widget_app, host=host, port=port, log_level="info"))
        await server.serve()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Widget server stopped by user")
    except Exception as exc:
        logger.exception(f"Widget server crashed: {exc}")
        raise SystemExit(1) from exc


def main() -> None:
    cli(auto_envvar_prefix="JIMS")


if __name__ == "__main__":
    main()
