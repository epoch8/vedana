import asyncio
import uuid
from contextlib import suppress
from typing import Any, Awaitable, overload
import sqlalchemy as sa
import sqlalchemy.ext.asyncio as sa_aio

from aiogram import Bot, Dispatcher
from aiogram.filters import CommandStart
from aiogram.types import Message
from jims_core.app import JimsApp
from jims_core.db import ThreadDB
from jims_core.schema import Pipeline
from jims_core.thread.thread_context import StatusUpdater
from jims_core.thread.thread_controller import ThreadController
from jims_core.util import uuid7
from loguru import logger
from opentelemetry import trace
from pydantic_settings import BaseSettings, SettingsConfigDict

from jims_telegram.md2tgmd import escape

tracer = trace.get_tracer("jims_tui")


class TelegramSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TELEGRAM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    bot_token: str


settings = TelegramSettings()  # type: ignore

uuid_namespace = uuid.UUID("00000000-0000-0000-0000-000000000000")


def uuid_from_int(value: int) -> uuid.UUID:
    if not 0 <= value < 2**64:
        raise ValueError(f"Integer value must be between 0 and 2^64-1, got {value}")

    # Convert the integer to bytes in big-endian order
    name = value.to_bytes(8, byteorder="big")

    # Generate a version 5 UUID
    return uuid.uuid5(uuid_namespace, name)


class TelegramStatusUpdater(StatusUpdater):
    def __init__(self, bot: Bot, chat_id: int) -> None:
        self.bot = bot
        self.chat_id = chat_id

    async def update_status(self, status: str) -> None:
        await self.bot.send_chat_action(
            chat_id=self.chat_id,
            action="typing",
        )
        logger.debug(f"Status updated to: {status}")


class TelegramController:
    def __init__(
        self,
        app: JimsApp,
    ) -> None:
        self.app = app
        self.bot = Bot(token=settings.bot_token)

        self.dispatcher = Dispatcher()

        self.dispatcher.message.register(self.command_start, CommandStart())
        self.dispatcher.message.register(self.handle_message)

    @overload
    @classmethod
    async def create(cls, app: JimsApp) -> "TelegramController": ...

    @overload
    @classmethod
    async def create(cls, app: Awaitable[JimsApp]) -> "TelegramController": ...

    @classmethod
    async def create(cls, app: JimsApp | Awaitable[JimsApp]) -> "TelegramController":
        if isinstance(app, Awaitable):
            app = await app
        return cls(app)

    @classmethod
    async def thread_from_user_id(
        cls,
        sessionmaker: sa_aio.async_sessionmaker[sa_aio.AsyncSession],
        from_user_id: int | None,
    ) -> "ThreadController | None":
        """
        Retrieve the latest thread associated with a given telegram user_id.
        """
        if from_user_id:
            async with sessionmaker() as session:
                stmt = (
                    sa.select(ThreadDB)
                    .where(ThreadDB.thread_config["telegram_user_id"].astext == str(from_user_id))
                    .order_by(ThreadDB.created_at.desc())
                    .limit(1)
                )
                thread = (await session.execute(stmt)).scalar_one_or_none()

            if thread:
                return ThreadController(sessionmaker, thread)
        return None

    async def _run_pipeline(self, ctl: ThreadController, chat_id: Any, pipeline: Pipeline) -> None:
        ctx = await ctl.make_context()
        ctx = ctx.with_status_updater(TelegramStatusUpdater(self.bot, chat_id))

        async def status_updater():
            try:
                while True:
                    await ctx.update_agent_status("thinking")
                    await asyncio.sleep(5)
            except asyncio.CancelledError:
                pass

        updater_task = asyncio.create_task(status_updater())

        try:
            events = await ctl.run_pipeline_with_context(pipeline, ctx)
        finally:
            updater_task.cancel()
            with suppress(asyncio.CancelledError):
                await updater_task

        for event in events:
            if event.event_type == "comm.assistant_message":
                await self.bot.send_message(
                    chat_id=chat_id,
                    text=escape(event.event_data["content"]),
                    parse_mode="MarkdownV2",
                )

    async def command_start(self, message: Message) -> None:
        logger.debug(f"Received command start from {message.chat.id}")

        ctl = await self.app.new_thread(
            # uuid_from_int(message.chat.id),  # makes session_id persist for from_user.id
            uuid7(),
            {
                "interface": "telegram",
                "telegram_chat_id": message.chat.id,
                "telegram_user_id": message.from_user.id if message.from_user else None,
            },
        )

        if message.text:
            await ctl.store_user_message(
                event_id=uuid_from_int(message.message_id),
                content=message.text,
            )

        if self.app.conversation_start_pipeline is not None:
            await self._run_pipeline(ctl, message.chat.id, self.app.conversation_start_pipeline)

    async def handle_message(self, message: Message) -> None:
        with tracer.start_as_current_span("jims_telegram.handle_message"):
            logger.debug(f"Received message {message.text=} from {message.chat.id=}")
            ctl = await self.thread_from_user_id(
                self.app.sessionmaker,
                message.from_user.id if message.from_user else None,
            )

            if ctl is None:
                logger.warning(f"Thread with id {message.chat.id} not found, recreating")
                ctl = await ThreadController.new_thread(
                    self.app.sessionmaker,
                    # uuid_from_int(message.chat.id),  # makes session_id persist for from_user.id
                    uuid7(),
                    {
                        "interface": "telegram",
                        "telegram_chat_id": message.chat.id,
                        "telegram_user_id": message.from_user.id if message.from_user else None,
                    },
                )

            if message.text:
                await ctl.store_user_message(
                    event_id=uuid_from_int(message.message_id),
                    content=message.text,
                )

            await self._run_pipeline(ctl, message.chat.id, self.app.pipeline)

    async def run(self):
        logger.debug("Starting Telegram bot")
        await self.dispatcher.start_polling(self.bot)
