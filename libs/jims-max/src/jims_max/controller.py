import asyncio
import uuid
from contextlib import suppress
from typing import Any, Awaitable, overload

from jims_core.app import JimsApp
from jims_core.schema import Pipeline
from jims_core.thread.thread_context import StatusUpdater
from jims_core.thread.thread_controller import ThreadController
from jims_core.util import uuid7
from loguru import logger
from maxapi import Bot, Dispatcher, F
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import TypedDict


class MaxSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MAX_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    bot_token: str


settings = MaxSettings()  # type: ignore

uuid_namespace = uuid.UUID("00000000-0000-0000-0000-000000000000")


def uuid_from_int(value: int) -> uuid.UUID:
    if not 0 <= value < 2**64:
        raise ValueError(f"Integer value must be between 0 and 2^64-1, got {value}")
    return uuid.uuid5(uuid_namespace, value.to_bytes(8, byteorder="big"))


class MaxStatusUpdater(StatusUpdater):
    def __init__(self, bot: Bot, chat_id: int) -> None:
        self.bot = bot
        self.chat_id = chat_id

    async def update_status(self, status: str) -> None:
        logger.debug(f"Status updated to: {status} for chat_id={self.chat_id}")


class MaxButton(TypedDict):
    text: str
    id: str


class MaxController:
    def __init__(self, app: JimsApp) -> None:
        self.app = app
        self.bot = Bot(settings.bot_token)
        self.dispatcher = Dispatcher()

        self.dispatcher.message.register(self.command_start, F.text == "/start")
        self.dispatcher.message.register(self.handle_message)
        self.dispatcher.callback_query.register(self.handle_callback, F.data.startswith("btn:"))

    @overload
    @classmethod
    async def create(cls, app: JimsApp) -> "MaxController":
        ...

    @overload
    @classmethod
    async def create(cls, app: Awaitable[JimsApp]) -> "MaxController":
        ...

    @classmethod
    async def create(cls, app: JimsApp | Awaitable[JimsApp]) -> "MaxController":
        if isinstance(app, Awaitable):
            app = await app
        return cls(app)

    async def _run_pipeline(self, ctl: ThreadController, chat_id: Any, pipeline: Pipeline) -> None:
        ctx = await ctl.make_context()
        ctx = ctx.with_status_updater(MaxStatusUpdater(self.bot, chat_id))

        async def status_updater() -> None:
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
                await self.bot.send_message(chat_id=chat_id, text=str(event.event_data["content"]))
            elif event.event_type == "comm.assistant_buttons":
                await self.bot.send_message(
                    chat_id=chat_id,
                    text=str(event.event_data.get("content", "")),
                    reply_markup=self._build_inline_keyboard(event.event_data.get("buttons", [])),
                )

    async def command_start(self, message: Any) -> None:
        from_user = getattr(message, "from_user", None)
        chat = getattr(message, "chat", None)
        if from_user is None or chat is None:
            logger.error("from_user or chat not found in message")
            return

        from_id = getattr(from_user, "id", None)
        if from_id is None:
            logger.error("from_user.id not found in message")
            return

        ctl = await self.app.new_thread(
            contact_id=f"max:{from_id}",
            thread_id=uuid7(),
            thread_config={
                "interface": "max",
                "max_chat_id": chat.id,
                "max_user_id": from_id,
                "max_user_name": getattr(from_user, "username", None),
            },
        )

        if getattr(message, "text", None):
            await ctl.store_user_message(
                event_id=uuid_from_int(getattr(message, "message_id", 0)),
                content=message.text,
            )

        if self.app.conversation_start_pipeline is not None:
            await self._run_pipeline(ctl, chat.id, self.app.conversation_start_pipeline)

    async def handle_message(self, message: Any) -> None:
        from_user = getattr(message, "from_user", None)
        chat = getattr(message, "chat", None)
        if from_user is None or chat is None:
            logger.error("from_user or chat not found in message")
            return

        from_id = getattr(from_user, "id", None)
        if from_id is None:
            logger.error("from_user.id not found in message")
            return

        ctl = await ThreadController.latest_thread_from_contact_id(self.app.sessionmaker, f"max:{from_id}")
        if ctl is None:
            ctl = await ThreadController.new_thread(
                self.app.sessionmaker,
                contact_id=f"max:{from_id}",
                thread_id=uuid7(),
                thread_config={
                    "interface": "max",
                    "max_chat_id": chat.id,
                    "max_user_id": from_id,
                    "max_user_name": getattr(from_user, "username", None),
                },
            )

        if getattr(message, "text", None):
            await ctl.store_user_message(
                event_id=uuid_from_int(getattr(message, "message_id", 0)),
                content=message.text,
            )

        await self._run_pipeline(ctl, chat.id, self.app.pipeline)

    async def handle_callback(self, callback: Any) -> None:
        from_user = getattr(callback, "from_user", None)
        if from_user is None or getattr(from_user, "id", None) is None:
            return
        from_id = from_user.id

        ctl = await ThreadController.latest_thread_from_contact_id(self.app.sessionmaker, f"max:{from_id}")
        message = getattr(callback, "message", None)
        chat = getattr(message, "chat", None) if message else None

        if ctl is None:
            ctl = await ThreadController.new_thread(
                self.app.sessionmaker,
                contact_id=f"max:{from_id}",
                thread_id=uuid7(),
                thread_config={
                    "interface": "max",
                    "max_chat_id": getattr(chat, "id", None),
                    "max_user_id": from_id,
                    "max_user_name": getattr(from_user, "username", None),
                },
            )

        await ctl.store_event_dict(
            event_id=uuid7(),
            event_type="comm.user_button_click",
            event_data={
                "role": "user",
                "content": getattr(callback, "data", ""),
                "message_id": getattr(message, "message_id", None) if message else None,
            },
        )

        chat_id = getattr(chat, "id", from_id)
        await self._run_pipeline(ctl, chat_id, self.app.pipeline)

    @staticmethod
    def _build_inline_keyboard(buttons: list[list[MaxButton]] | list[MaxButton] | list[Any]) -> Any | None:
        if not buttons:
            return None
        return buttons

    async def run(self) -> None:
        await self.dispatcher.start_polling(self.bot)

