import asyncio

import click
from jims_core.util import load_jims_app, setup_monitoring_and_tracing_with_sentry
from loguru import logger

from jims_telegram import TelegramController


@click.command()
@click.option("--app", type=click.STRING, default="app")
@click.option("--enable-sentry", is_flag=True, help="Enable tracing to Sentry", default=False)
def cli(app: str, enable_sentry: bool) -> None:
    if enable_sentry:
        setup_monitoring_and_tracing_with_sentry()

    jims_app = load_jims_app(app)

    async def run_telegram_bot():
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
