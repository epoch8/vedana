import io
import logging
import os

import reflex as rx
import requests
from vedana_core.app import VedanaApp, make_vedana_app

vedana_app: VedanaApp | None = None

EVAL_ENABLED = bool(os.environ.get("GRIST_TEST_SET_DOC_ID"))
DEBUG_MODE = os.environ.get("DEBUG", "").lower() in ("true", "1")
HAS_OPENAI_KEY = bool(os.environ.get("OPENAI_API_KEY"))
HAS_OPENROUTER_KEY = bool(os.environ.get("OPENROUTER_API_KEY"))


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
    debug_mode: bool = DEBUG_MODE


class DebugState(rx.State):
    """State for debug mode API key setup."""

    debug_mode: bool = DEBUG_MODE
    needs_api_key: bool = DEBUG_MODE and not HAS_OPENAI_KEY and not HAS_OPENROUTER_KEY
    show_api_key_dialog: bool = False
    api_key: str = ""
    api_key_type: str = "openai"  # "openai" or "openrouter"
    api_key_saved: bool = False

    def check_and_show_dialog(self) -> None:
        """Called on app mount to show dialog if needed."""
        if self.needs_api_key and not self.api_key_saved:
            self.show_api_key_dialog = True

    def set_api_key(self, value: str) -> None:
        self.api_key = value

    def set_api_key_type(self, value: str) -> None:
        self.api_key_type = value

    def save_api_key(self) -> None:
        """Save the API key and close dialog."""
        if self.api_key.strip() and self.debug_mode:  # extra check for debug mode
            self.api_key_saved = True
            self.show_api_key_dialog = False
            # Set environment variable so it's available throughout the app
            if self.api_key_type == "openai":
                os.environ["OPENAI_API_KEY"] = self.api_key.strip()
            else:
                os.environ["OPENROUTER_API_KEY"] = self.api_key.strip()

    def close_dialog(self) -> None:
        """Close dialog without saving."""
        self.show_api_key_dialog = False

    def open_dialog(self) -> None:
        """Manually open the dialog to change API key."""
        self.show_api_key_dialog = True


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
