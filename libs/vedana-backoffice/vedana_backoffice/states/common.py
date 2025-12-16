import io
import logging
import os

import reflex as rx
import requests
from vedana_core.app import VedanaApp, make_vedana_app

vedana_app: VedanaApp | None = None


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


class TelegramBotState(rx.State):
    """State for Telegram bot information."""

    bot_username: str = ""
    bot_url: str = ""
    has_bot: bool = False

    def load_bot_info(self) -> None:
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        if not token:
            return

        try:
            bot_status = requests.get(f"https://api.telegram.org/bot{token}/getMe")
            if bot_status.status_code == 200:
                bot_status = bot_status.json()
                if bot_status["ok"]:
                    self.bot_username = bot_status["result"]["username"]
                    self.bot_url = f"https://t.me/{self.bot_username}"
                    self.has_bot = True
        except Exception as e:
            logging.warning(f"Failed to load Telegram bot info: {e}")
