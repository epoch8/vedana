import asyncio
import io
import logging
import os

import litellm
import reflex as rx
import requests
from jims_core.llms.llm_provider import LLMSettings

# Default LLM settings (env-driven). Subclasses can override defaults entirely.
_default_llm_settings = LLMSettings()  # type: ignore

TELEGRAM_BOT_INFO_CACHE: dict[str, str] | None = None
TELEGRAM_BOT_INFO_REQUESTED: bool = False

# Generic gate for the Eval page. Subclasses may flip this through their own env var.
EVAL_ENABLED = os.environ.get("JIMS_BACKOFFICE_EVAL_ENABLED", "1").lower() in ("true", "1", "yes")
DEBUG_MODE = (
    os.environ.get("JIMS_BACKOFFICE_DEBUG", "").lower() in ("true", "1")
    or os.environ.get("DEBUG", "").lower() in ("true", "1")
)

# Registry of the active concrete Chat/Eval state classes, used by DebugState
# to dispatch refresh events when the available model list changes.
_chat_state_cls: type | None = None
_eval_state_cls: type | None = None


def register_chat_state(cls: type) -> None:
    """Register the concrete chat state class used by the running app."""
    global _chat_state_cls
    _chat_state_cls = cls


def register_eval_state(cls: type) -> None:
    """Register the concrete eval state class used by the running app."""
    global _eval_state_cls
    _eval_state_cls = cls


async def load_litellm_models(
    *,
    provider: str | None = None,
    check_provider_endpoint: bool = False,
) -> list[str]:
    def _fetch() -> list[str]:
        raw = litellm.get_valid_models(
            custom_llm_provider=provider,
            check_provider_endpoint=check_provider_endpoint,
        )
        result: list[str] = [
            model if (provider is None or model.startswith(provider)) else f"{provider}/{model}" for model in raw
        ]
        return sorted(set(result))

    return await asyncio.to_thread(_fetch)  # type: ignore[return-value]


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
    show_api_key_dialog: bool = False
    runtime_model_api_key: str = ""
    runtime_model_provider: str | None = None
    default_embeddings_model: str = _default_llm_settings.embeddings_model
    api_key_saved: bool = False
    available_models: list[str] = []

    @rx.var
    def provider_options(self) -> list[str]:
        return ["openai", "openrouter", "anthropic", "cohere", "xai"]

    @rx.var
    def embeddings_model(self) -> str | None:
        """embeddings model is fixed so its availability/correct name can be resolved here."""
        models = self.available_models
        if not models:
            return _default_llm_settings.embeddings_model
        for m in models:
            if m.rsplit("/", 1)[-1] == _default_llm_settings.embeddings_model:
                return m
        else:
            _model, embeddings_model_provider, _1, _2 = litellm.get_llm_provider(_default_llm_settings.embeddings_model)

            if self.runtime_model_provider == "openrouter":
                if embeddings_model_provider == "openrouter":
                    return _default_llm_settings.embeddings_model
                elif _default_llm_settings.embeddings_model.startswith(embeddings_model_provider):
                    return f"openrouter/{_default_llm_settings.embeddings_model}"
                elif embeddings_model_provider:
                    return f"openrouter/{embeddings_model_provider}/{_default_llm_settings.embeddings_model}"
        return None

    @rx.var
    def embeddings_model_available(self) -> bool:
        return self.embeddings_model is not None

    @rx.event(background=True)  # type: ignore[operator]
    async def load_available_models(self):
        if not self.debug_mode:
            return
        models = await load_litellm_models(
            provider=self.runtime_model_provider,
        )
        async with self:
            self.available_models = models
            if not models and not self.api_key_saved:
                self.show_api_key_dialog = True

        # Notify the registered concrete Chat/Eval state classes that the model
        # list changed. The registry is populated by the app entrypoint.
        if _chat_state_cls is not None and hasattr(_chat_state_cls, "refresh_model_list"):
            yield _chat_state_cls.refresh_model_list()
        if _eval_state_cls is not None and hasattr(_eval_state_cls, "refresh_model_list"):
            yield _eval_state_cls.refresh_model_list()

    def set_model_api_key(self, value: str) -> None:
        self.runtime_model_api_key = value

    def set_model_provider(self, value: str) -> None:
        self.runtime_model_provider = value

    def save_api_key(self):
        if not self.debug_mode:
            return
        key = self.runtime_model_api_key.strip()
        if key:
            litellm.api_key = key
            self.api_key_saved = True
            self.show_api_key_dialog = False
        else:
            litellm.api_key = None
            self.api_key_saved = False
            self.show_api_key_dialog = False
            self.available_models = []
            self.runtime_model_provider = None
        yield DebugState.load_available_models()

    def close_dialog(self) -> None:
        self.show_api_key_dialog = False

    def open_dialog(self) -> None:
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
