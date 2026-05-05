import asyncio

import click
from jims_core.util import load_jims_app, setup_monitoring_and_tracing_with_sentry
from loguru import logger

from jims_tui.chat_app import ChatApp


@click.command()
@click.option("--app", type=click.STRING, default="app")
@click.option("--enable-sentry", is_flag=True, help="Enable tracing to Sentry", default=False)
def cli(app: str, enable_sentry: bool) -> None:
    if enable_sentry:
        setup_monitoring_and_tracing_with_sentry()

    jims_app = load_jims_app(app)

    async def run_chat_app():
        chat_app = await ChatApp.create(jims_app)
        await chat_app.run_async()

    logger.remove()
    asyncio.run(run_chat_app())


def main():
    cli(auto_envvar_prefix="JIMS")


if __name__ == "__main__":
    main()
