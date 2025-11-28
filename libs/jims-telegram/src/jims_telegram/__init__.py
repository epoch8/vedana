import asyncio
import uuid
from contextlib import suppress
from typing import Any, Awaitable, overload

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from jims_core.app import JimsApp
from jims_core.schema import Pipeline
from jims_core.thread.thread_context import StatusUpdater
from jims_core.thread.thread_controller import ThreadController
from jims_core.util import uuid7
from loguru import logger
from opentelemetry import trace
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import TypedDict

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


class TelegramButton(TypedDict):
    text: str
    id: str


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
        self.dispatcher.callback_query.register(self.handle_callback, F.data.startswith("btn:"))

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
            elif event.event_type == "comm.assistant_buttons":
                buttons = event.event_data.get("buttons", [])
                reply_markup = self._build_inline_keyboard(buttons)
                await self.bot.send_message(
                    chat_id=chat_id,
                    text=escape(event.event_data.get("content", "")),
                    parse_mode="MarkdownV2",
                    reply_markup=reply_markup,
                )

    async def command_start(self, message: Message) -> None:
        logger.debug(f"Received command start from {message.chat.id}")
        if not message.from_user:  # messages sent on behalf of chats, channels, or by tg
            logger.error("from_user.id not found in message")
            return

        from_id = message.from_user.id  # type: ignore[union-attr]

        thread_id = uuid7()
        ctl = await self.app.new_thread(
            contact_id=f"telegram:{from_id}",
            thread_id=thread_id,
            thread_config={
                "interface": "telegram",
                "telegram_chat_id": message.chat.id,
                "telegram_user_id": from_id,
                "telegram_user_name": message.from_user.username,  # type: ignore[union-attr]
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
            if not message.from_user:  # messages sent on behalf of chats, channels, or by tg
                logger.error("from_user.id not found in message")
                return

            from_id = message.from_user.id  # type: ignore[union-attr]

            ctl = await ThreadController.latest_thread_from_contact_id(self.app.sessionmaker, f"telegram:{from_id}")
            if ctl is None:
                logger.warning(f"Thread with id {message.chat.id} not found, recreating")
                thread_id = uuid7()
                ctl = await ThreadController.new_thread(
                    self.app.sessionmaker,
                    contact_id=f"telegram:{from_id}",
                    thread_id=thread_id,
                    thread_config={
                        "interface": "telegram",
                        "telegram_chat_id": message.chat.id,
                        "telegram_user_id": from_id,
                        "telegram_user_name": message.from_user.username,  # type: ignore[union-attr]
                    },
                )

            if message.text:
                await ctl.store_user_message(
                    event_id=uuid_from_int(message.message_id),
                    content=message.text,
                )

            await self._run_pipeline(ctl, message.chat.id, self.app.pipeline)

    async def handle_callback(self, callback: CallbackQuery) -> None:
        if not callback.from_user:
            return
        from_id = callback.from_user.id  # type: ignore[union-attr]

        ctl = await ThreadController.latest_thread_from_contact_id(self.app.sessionmaker, f"telegram:{from_id}")
        if ctl is None:
            thread_id = uuid7()
            ctl = await ThreadController.new_thread(
                self.app.sessionmaker,
                contact_id=f"telegram:{from_id}",
                thread_id=thread_id,
                thread_config={
                    "interface": "telegram",
                    "telegram_chat_id": callback.message.chat.id if callback.message else None,
                    "telegram_user_id": from_id,
                    "telegram_user_name": callback.message.from_user.username,  # type: ignore[union-attr]
                },
            )

        data = callback.data or ""
        try:
            await ctl.store_event_dict(
                event_id=uuid7(),
                event_type="comm.user_button_click",
                event_data={
                    "role": "user",
                    "content": data,
                    "message_id": callback.message.message_id if callback.message else None,
                },
            )
        except Exception as e:
            logger.error(f"Failed to store button click event: {e}")

        with suppress(Exception):
            await callback.answer()

        chat_id = callback.message.chat.id if callback.message else from_id
        await self._run_pipeline(ctl, chat_id, self.app.pipeline)

    @staticmethod
    def _build_inline_keyboard(
        buttons: list[list[TelegramButton]] | list[TelegramButton] | list[Any],
    ) -> InlineKeyboardMarkup | None:
        """
        Parse buttons dict into keyboard
        """
        rows: list[list[InlineKeyboardButton]] = []
        if buttons and isinstance(buttons, list):
            # normalize single row form
            if buttons and all(isinstance(b, dict) for b in buttons):
                buttons = [buttons]  # type: ignore[assignment]
            for row in buttons:
                if not isinstance(row, list):
                    continue
                kb_row: list[InlineKeyboardButton] = []
                for b in row:
                    if not isinstance(b, dict):
                        continue
                    label_v = b.get("text")
                    if label_v is None:
                        continue
                    label = str(label_v)
                    btn_id = str(b.get("id") or b.get("callback_data") or label)
                    kb_row.append(InlineKeyboardButton(text=label, callback_data=f"btn:{btn_id}"))
                if kb_row:
                    rows.append(kb_row)
        return InlineKeyboardMarkup(inline_keyboard=rows) if rows else None

    async def run(self):
        logger.debug("Starting Telegram bot")
        await self.dispatcher.start_polling(self.bot)
