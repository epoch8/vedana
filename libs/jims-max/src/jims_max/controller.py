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
from maxapi.enums.sender_action import SenderAction
from maxapi.filters.command import CommandStart
from maxapi.types import ButtonsPayload, CallbackButton
from maxapi.types.updates import MessageCallback, MessageCreated
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


def uuid_from_str(value: str) -> uuid.UUID:
    return uuid.uuid5(uuid_namespace, value)


class MaxStatusUpdater(StatusUpdater):
    def __init__(self, bot: Bot, chat_id: int) -> None:
        self.bot = bot
        self.chat_id = chat_id

    async def update_status(self, status: str) -> None:
        # MAX uses sender actions instead of "typing..." status messages.
        with suppress(Exception):
            await self.bot.send_action(chat_id=self.chat_id, action=SenderAction.TYPING_ON)
        logger.debug(f"Status updated to: {status} for chat_id={self.chat_id}")


class MaxButton(TypedDict):
    text: str
    id: str


class MaxController:
    def __init__(self, app: JimsApp) -> None:
        self.app = app
        self.bot = Bot(settings.bot_token)
        self.dispatcher = Dispatcher()

        self.dispatcher.message_created.register(self.command_start, CommandStart())
        self.dispatcher.message_created.register(self.handle_message)
        self.dispatcher.message_callback.register(self.handle_callback, F.callback.payload.startswith("btn:"))

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
                    attachments=self._build_inline_keyboard(event.event_data.get("buttons", [])),
                )

    async def command_start(self, event: MessageCreated) -> None:
        from_user = event.from_user or event.message.sender
        chat_id = event.chat.chat_id if event.chat is not None else event.message.recipient.chat_id
        if from_user is None:
            logger.error("MessageCreated must include user information")
            return
        from_id = from_user.user_id

        ctl = await self.app.new_thread(
            contact_id=f"max:{from_id}",
            thread_id=uuid7(),
            thread_config={
                "interface": "max",
                "max_chat_id": chat_id,
                "max_user_id": from_id,
                "max_user_name": from_user.username,
            },
        )

        if event.message.body is not None and event.message.body.text:
            await ctl.store_user_message(
                event_id=uuid_from_str(event.message.body.mid),
                content=event.message.body.text,
            )

        if self.app.conversation_start_pipeline is not None:
            await self._run_pipeline(ctl, chat_id, self.app.conversation_start_pipeline)

    async def handle_message(self, event: MessageCreated) -> None:
        from_user = event.from_user or event.message.sender
        chat_id = event.chat.chat_id if event.chat is not None else event.message.recipient.chat_id
        if from_user is None:
            logger.error("MessageCreated must include user information")
            return
        from_id = from_user.user_id

        ctl = await ThreadController.latest_thread_from_contact_id(self.app.sessionmaker, f"max:{from_id}")
        if ctl is None:
            ctl = await ThreadController.new_thread(
                self.app.sessionmaker,
                contact_id=f"max:{from_id}",
                thread_id=uuid7(),
                thread_config={
                    "interface": "max",
                    "max_chat_id": chat_id,
                    "max_user_id": from_id,
                    "max_user_name": from_user.username,
                },
            )

        if event.message.body is not None and event.message.body.text:
            await ctl.store_user_message(
                event_id=uuid_from_str(event.message.body.mid),
                content=event.message.body.text,
            )

        await self._run_pipeline(ctl, chat_id, self.app.pipeline)

    async def handle_callback(self, event: MessageCallback) -> None:
        from_id = event.callback.user.user_id

        ctl = await ThreadController.latest_thread_from_contact_id(self.app.sessionmaker, f"max:{from_id}")
        message = event.message
        chat_id = message.recipient.chat_id if message is not None else None

        if ctl is None:
            ctl = await ThreadController.new_thread(
                self.app.sessionmaker,
                contact_id=f"max:{from_id}",
                thread_id=uuid7(),
                thread_config={
                    "interface": "max",
                    "max_chat_id": chat_id,
                    "max_user_id": from_id,
                    "max_user_name": event.callback.user.username,
                },
            )

        await ctl.store_event_dict(
            event_id=uuid7(),
            event_type="comm.user_button_click",
            event_data={
                "role": "user",
                "content": str(event.callback.payload or ""),
                "message_id": message.body.mid if message is not None and message.body is not None else None,
            },
        )

        with suppress(Exception):
            await event.answer()

        await self._run_pipeline(ctl, chat_id if chat_id is not None else from_id, self.app.pipeline)

    @staticmethod
    def _build_inline_keyboard(
        buttons: list[list[MaxButton]] | list[MaxButton] | list[Any],
    ) -> list[Any] | None:
        if not buttons:
            return None
        rows: list[list[CallbackButton]] = []
        source_rows = buttons if not all(isinstance(b, dict) for b in buttons) else [buttons]

        for row in source_rows:
            if not isinstance(row, list):
                continue
            kb_row: list[CallbackButton] = []
            for button in row:
                if not isinstance(button, dict):
                    continue
                text_v = button.get("text")
                if text_v is None:
                    continue
                text = str(text_v)
                raw_payload = str(
                    button.get("id")
                    or button.get("callback_data")
                    or text
                )
                payload = raw_payload if raw_payload.startswith("btn:") else f"btn:{raw_payload}"
                kb_row.append(CallbackButton(text=text, payload=payload))
            if kb_row:
                rows.append(kb_row)

        if not rows:
            return None
        return [ButtonsPayload(buttons=rows).pack()]

    async def run(self) -> None:
        await self.dispatcher.start_polling(self.bot)

