import asyncio
from async_lru import alru_cache
import io
import logging
import os
from typing import Iterable

import httpx
import reflex as rx
import requests
from jims_core.llms.llm_provider import env_settings as llm_settings
from vedana_core.app import VedanaApp, make_vedana_app

vedana_app: VedanaApp | None = None
TELEGRAM_BOT_INFO_CACHE: dict[str, str] | None = None
TELEGRAM_BOT_INFO_REQUESTED: bool = False

EVAL_ENABLED = bool(os.environ.get("GRIST_TEST_SET_DOC_ID"))
DEBUG_MODE = (os.environ.get("VEDANA_BACKOFFICE_DEBUG", "").lower() in ("true", "1")
              or os.environ.get("DEBUG", "").lower() in ("true", "1"))
HAS_OPENAI_KEY = bool(os.environ.get("OPENAI_API_KEY"))
HAS_OPENROUTER_KEY = bool(os.environ.get("OPENROUTER_API_KEY"))


def _filter_chat_capable_models(models: Iterable[dict]) -> list[str]:
    """Filter models that support text chat with tool calls."""
    result: list[str] = []
    for m in models:
        model_id = str(m.get("id", "")).strip()
        if not model_id:
            continue

        architecture = m.get("architecture", {})
        has_chat = bool(
            architecture
            and "text" in architecture.get("input_modalities", [])
            and "text" in architecture.get("output_modalities", [])
        )
        has_tools = "tools" in m.get("supported_parameters", [])

        if has_chat and has_tools:
            result.append(model_id)

    return result


@alru_cache
async def load_openrouter_models() -> list[str]:
    if not DEBUG_MODE:
        return []
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{llm_settings.openrouter_api_base_url}/models")
            resp.raise_for_status()
            models = resp.json().get("data", [])
            return sorted(_filter_chat_capable_models(models))
    except Exception as exc:
        logging.warning(f"Failed to fetch OpenRouter models: {exc}")
        return []


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
