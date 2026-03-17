import asyncio
import io
import logging
import os
from contextlib import contextmanager

import litellm
import reflex as rx
import requests
from vedana_core.app import VedanaApp, make_vedana_app
from vedana_core.settings import settings as core_settings

vedana_app: VedanaApp | None = None

TELEGRAM_BOT_INFO_CACHE: dict[str, str] | None = None
TELEGRAM_BOT_INFO_REQUESTED: bool = False

EVAL_ENABLED = bool(os.environ.get("GRIST_TEST_SET_DOC_ID"))
DEBUG_MODE = (os.environ.get("VEDANA_BACKOFFICE_DEBUG", "").lower() in ("true", "1")
              or os.environ.get("DEBUG", "").lower() in ("true", "1"))


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


async def get_vedana_app():
    global vedana_app
    if vedana_app is None:
        vedana_app = await make_vedana_app()
    return vedana_app


class DatapipeStepError(RuntimeError):
    """Raised when a datapipe step fails without propagating the exception."""

    pass


@contextmanager
def datapipe_log_capture():
    """Detect datapipe step failures that are logged but not raised.

    datapipe catches and logs exceptions in some step types (e.g. batch_generate)
    without re-raising them. This captures ERROR-level log messages from the
    ``datapipe`` logger hierarchy and raises :class:`DatapipeStepError` if any
    errors were recorded once the guarded block completes.
    """
    errors: list[str] = []

    class _ErrorCapture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                errors.append(record.getMessage())
            except Exception:
                pass

    handler = _ErrorCapture()
    handler.setLevel(logging.ERROR)

    dp_logger = logging.getLogger("datapipe")
    dp_logger.addHandler(handler)
    try:
        yield errors
    finally:
        dp_logger.removeHandler(handler)
    if errors:
        raise DatapipeStepError(errors[0])


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
    default_embeddings_model: str = core_settings.embeddings_model
    api_key_saved: bool = False
    available_models: list[str] = []

    @rx.var
    def provider_options(self) -> list[str]:
        return ["openai", "openrouter", "anthropic", "cohere", "xai"]

    @rx.var
    def embeddings_model(self) -> bool:
        """embeddings model is fixed so its availability/correct name can be resolved here."""
        models = self.available_models
        if not models:
            return core_settings.embeddings_model
        for m in models:
            if m.rsplit("/", 1)[-1] == core_settings.embeddings_model:
                return m
        else:
            _model, embeddings_model_provider, _1, _2 = litellm.get_llm_provider(core_settings.embeddings_model)

            if self.runtime_model_provider == "openrouter":  # openrouter does not list embeddings models in its model list
                if embeddings_model_provider == "openrouter":
                    return core_settings.embeddings_model
                elif core_settings.embeddings_model.startswith(embeddings_model_provider):
                    return f"openrouter/{core_settings.embeddings_model}"
                elif embeddings_model_provider:
                    return f"openrouter/{embeddings_model_provider}/{core_settings.embeddings_model}"
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
            # check_provider is not necessary in most cases 
            # check_provider_endpoint=True if self.runtime_model_provider and self.runtime_model_provider == "openrouter" else False,
        )
        async with self:
            self.available_models = models
            if not models and not self.api_key_saved:
                self.show_api_key_dialog = True

        from vedana_backoffice.states.chat import ChatState
        from vedana_backoffice.states.eval import EvalState

        yield ChatState.refresh_model_list()
        yield EvalState.refresh_model_list()

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
        # Background refresh will repopulate available_models and notify Chat/Eval.
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
