import asyncio

import click
from aiohttp import web
from jims_core.util import load_jims_app, setup_monitoring_and_tracing_with_sentry, setup_verbose_logging
from loguru import logger

from jims_telegram import TelegramController


async def healthcheck(host: str = "0.0.0.0", port: int = 8000):
    async def health(request: web.Request):
        return web.json_response({"status": "ok"})

    app = web.Application()
    app.add_routes([web.get("/health", health)])
    app.add_routes([web.get("/healthz", health)])

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    return runner


@click.command()
@click.option("--app", type=click.STRING, default="app")
@click.option("--enable-sentry", is_flag=True, help="Enable tracing to Sentry", default=False)
@click.option("--enable-healthcheck", is_flag=True, help="Enable healthcheck endpoint", default=True)
@click.option("--host", type=click.STRING, default="0.0.0.0")
@click.option("--port", type=click.INT, default=8000)
@click.option("--verbose", is_flag=True, default=False)
def cli(app: str, enable_sentry: bool, enable_healthcheck: bool, host: str, port: int, verbose: bool) -> None:
    if verbose:
        setup_verbose_logging()

    if enable_sentry:
        setup_monitoring_and_tracing_with_sentry()

    jims_app = load_jims_app(app)

    async def run_telegram_bot():
        if enable_healthcheck:
            await healthcheck(host=host, port=port)

        telegram_controller = await TelegramController.create(jims_app)
        await telegram_controller.run()

    try:
        asyncio.run(run_telegram_bot())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
        exit(1)


def main():
    cli(auto_envvar_prefix="JIMS")


if __name__ == "__main__":
    main()
