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


class AppVersionState(rx.State):
    version: str = f"`{os.environ.get('VERSION', 'unspecified_version')}`"  # md-formatted
    eval_enabled: bool = bool(os.environ.get("GRIST_TEST_SET_DOC_ID"))


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
