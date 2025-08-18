import asyncio

from jims_core.util import setup_monitoring_and_tracing_with_sentry
from jims_telegram import TelegramController
from jims_cli_demo.app import app


async def main():
    setup_monitoring_and_tracing_with_sentry()

    controller = TelegramController(app)
    await controller.run()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
