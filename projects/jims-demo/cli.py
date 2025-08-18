import asyncio

from jims_core.util import setup_monitoring_and_tracing_with_sentry
from jims_demo.app import app
from jims_tui.chat_app import ChatApp


async def main():
    setup_monitoring_and_tracing_with_sentry()

    chat_app = await ChatApp.create(app)
    await chat_app.run_async()


if __name__ == "__main__":
    asyncio.run(main())
