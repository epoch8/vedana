import asyncio
import io
import logging
import os

import reflex as rx
import requests
from vedana_core.app import VedanaApp, make_vedana_app

vedana_app: VedanaApp | None = None
TELEGRAM_BOT_INFO_CACHE: dict[str, str] | None = None
TELEGRAM_BOT_INFO_REQUESTED: bool = False

EVAL_ENABLED = bool(os.environ.get("GRIST_TEST_SET_DOC_ID"))


async def get_vedana_app():
    global vedana_app
    if vedana_app is None:
        vedana_app = await make_vedana_app()
    return vedana_app


class MemLogger(logging.Logger):
    """Logger that captures logs to a string buffer for debugging purposes."""

    def __init__(self, name: str, level: int = 0) -> None:
        super().__init__(name, level)
        self.parent = logging.getLogger(__name__)
        self._buf = io.StringIO()
        handler = logging.StreamHandler(self._buf)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        self.addHandler(handler)

    def get_logs(self) -> str:
        return self._buf.getvalue()

    def clear(self) -> None:
        self._buf.truncate(0)
        self._buf.seek(0)


class AppVersionState(rx.State):
    version: str = f"`{os.environ.get('VERSION', 'unspecified_version')}`"  # md-formatted
    eval_enabled: bool = EVAL_ENABLED


class TelegramBotState(rx.State):
    """State for Telegram bot information."""

    bot_username: str = ""
    bot_url: str = ""
    has_bot: bool = False

    @rx.event(background=True)  # type: ignore[operator]
    async def load_bot_info(self) -> None:
        global TELEGRAM_BOT_INFO_CACHE, TELEGRAM_BOT_INFO_REQUESTED
        if TELEGRAM_BOT_INFO_CACHE:
            async with self:
                self.bot_username = TELEGRAM_BOT_INFO_CACHE["bot_username"]
                self.bot_url = TELEGRAM_BOT_INFO_CACHE["bot_url"]
                self.has_bot = True
            return
        if TELEGRAM_BOT_INFO_REQUESTED:
            return

        TELEGRAM_BOT_INFO_REQUESTED = True
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        if not token:
            return

        def _fetch():
            return requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=5)

        try:
            bot_status = await asyncio.to_thread(_fetch)
            if bot_status.status_code == 200:
                bot_status = bot_status.json()
                if bot_status.get("ok"):
                    bot_username = bot_status["result"]["username"]
                    bot_url = f"https://t.me/{bot_username}"
                    TELEGRAM_BOT_INFO_CACHE = {
                        "bot_username": bot_username,
                        "bot_url": bot_url,
                    }
                    async with self:
                        self.bot_username = bot_username
                        self.bot_url = bot_url
                        self.has_bot = True
        except Exception as e:
            logging.warning(f"Failed to load Telegram bot info: {e}")
