import asyncio

import click
from jims_core.util import load_jims_app, setup_monitoring_and_tracing_with_sentry

from jims_telegram import TelegramController


@click.command()
@click.option("--app", type=click.STRING, default="app")
def cli(app: str) -> None:
    setup_monitoring_and_tracing_with_sentry()

    jims_app = load_jims_app(app)

    async def run_telegram_bot():
        telegram_controller = TelegramController(jims_app)
        await telegram_controller.run()

    asyncio.run(run_telegram_bot())


def main():
    cli(auto_envvar_prefix="JIMS")


if __name__ == "__main__":
    main()
