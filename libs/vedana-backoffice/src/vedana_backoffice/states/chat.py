import asyncio
import logging
import os
import threading
import traceback
from datetime import datetime
from typing import Any, Dict, Iterable, Tuple
from uuid import UUID, uuid4

import orjson as json
import reflex as rx
import requests
from jims_core.llms.llm_provider import env_settings as llm_settings
from jims_core.thread.thread_controller import ThreadController
from jims_core.util import uuid7
from vedana_core.settings import settings as core_settings

from vedana_backoffice.states.common import MemLogger, get_vedana_app
from vedana_backoffice.states.jims import ThreadViewState

# Global cache for OpenRouter models, populated at startup
OPENROUTER_MODELS_CACHE: list[str] = []
_openrouter_models_lock = threading.Lock()
_openrouter_models_loaded = threading.Event()


def _filter_chat_capable_models(models: Iterable[dict]) -> list[str]:
    """Filter models that support text chat with tool calls."""
    result: list[str] = []
    for m in models:
        model_id = str(m.get("id", "")).strip()
        if not model_id:
            continue

        has_chat = False
        architecture = m.get("architecture", {})
        if architecture:
            if "text" in architecture.get("input_modalities", []) and "text" in architecture.get(
                "output_modalities", []
            ):
                has_chat = True

        has_tools = False  # only accept models with tool calls
        if "tools" in m.get("supported_parameters", []):
            has_tools = True

        if has_chat and has_tools:
            result.append(model_id)

    return result


def _fetch_openrouter_models_sync() -> list[str]:
    """Fetch OpenRouter models synchronously. Used for startup loading."""
    try:
        resp = requests.get(
            f"{llm_settings.openrouter_api_base_url}/models",
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json()
        models = payload.get("data", [])
        parsed = _filter_chat_capable_models(models)
    except Exception as exc:
        logging.warning(f"Failed to fetch OpenRouter models at startup: {exc}")
        parsed = []
    return sorted(list(parsed))


def _load_openrouter_models_background():
    """Background thread function to load OpenRouter models at startup."""
    global OPENROUTER_MODELS_CACHE
    models = _fetch_openrouter_models_sync()
    with _openrouter_models_lock:
        if models:
            OPENROUTER_MODELS_CACHE = models
    _openrouter_models_loaded.set()
    logging.info(f"OpenRouter models loaded at startup: {len(OPENROUTER_MODELS_CACHE)} models")


# Start loading OpenRouter models in background immediately on module import
_startup_thread = threading.Thread(target=_load_openrouter_models_background, daemon=True)
_startup_thread.start()


class ChatState(rx.State):
    """Minimal chatbot with per-answer technical details."""

    input_text: str = ""
    is_running: bool = False
    messages: list[dict[str, Any]] = []
    chat_thread_id: str = ""
    data_model_text: str = ""
    data_model_snapshot_id: str = ""
    is_refreshing_dm: bool = False
    provider: str = "openai"  # default llm provider
    model: str = core_settings.model
    _default_models: tuple[str, ...] = (
        "gpt-5.1-chat-latest",
        "gpt-5.1",
        "gpt-5-chat-latest",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-4o",
        "gpt-4o-mini",
        "o4-mini",
    )
    custom_openrouter_key: str = ""
    default_openrouter_key_present: bool = bool(os.environ.get("OPENROUTER_API_KEY"))  # to require api_key input
    openai_models: list[str] = list(set(list(_default_models) + [core_settings.model]))
    openrouter_models: list[str] = []
    openrouter_models_loaded: bool = False
    available_models: list[str] = list(set(list(_default_models) + [core_settings.model]))
    enable_dm_filtering: bool = bool(os.environ.get("ENABLE_DM_FILTERING", False))
    data_model_branch: str = core_settings.config_plane_dev_branch
    data_model_snapshot_input: str = ""

    def mount(self):
        """Load models from startup cache."""
        with _openrouter_models_lock:
            self.openrouter_models = list(OPENROUTER_MODELS_CACHE)
        self.openrouter_models_loaded = _openrouter_models_loaded.is_set()
        self._sync_available_models()

    def set_input(self, value: str) -> None:
        self.input_text = value

    def set_model(self, value: str) -> None:
        if value in self.available_models:
            self.model = value

    def set_custom_openrouter_key(self, value: str) -> None:
        self.custom_openrouter_key = value  # need validating?

    def set_enable_dm_filtering(self, value: bool) -> None:
        self.enable_dm_filtering = value

    def set_data_model_branch(self, value: str) -> None:
        self.data_model_branch = value

    def set_data_model_snapshot_input(self, value: str) -> None:
        self.data_model_snapshot_input = value

    def _resolve_data_model_snapshot(self) -> int | None:
        raw = (self.data_model_snapshot_input or "").strip()
        if not raw:
            return None
        try:
            return int(raw)
        except Exception:
            return None

    async def _apply_data_model_selection(self) -> None:
        va = await get_vedana_app()
        va.data_model.set_branch(self.data_model_branch)
        va.data_model.set_snapshot_override(self._resolve_data_model_snapshot())

    def set_provider(self, value: str) -> None:
        self.provider = value
        self._sync_available_models()

    def _sync_available_models(self) -> None:
        """
        Recompute available_models based on selected provider, and realign
        the selected model if it is no longer valid.
        """

        if self.provider == "openrouter":
            models = self.openrouter_models
            if not models:
                if self.openrouter_models_loaded:
                    self.provider = "openai"
                    models = self.openai_models
                else:
                    models = self.available_models or self.openai_models
        else:
            models = self.openai_models

        self.available_models = list(models)

        if self.model not in self.available_models and self.available_models:
            self.model = self.available_models[0]

    def toggle_details_by_id(self, message_id: str) -> None:
        for idx, m in enumerate(self.messages):
            if str(m.get("id")) == str(message_id):
                msg = dict(m)
                msg["show_details"] = not bool(msg.get("show_details"))
                self.messages[idx] = msg
                break

    def reset_session(self) -> None:
        """Clear current chat history and start a fresh session (new thread on next send)."""
        self.messages = []
        self.input_text = ""
        self.chat_thread_id = ""

    @rx.var
    def total_conversation_cost(self) -> float:
        """Calculate total cost by summing requests_cost from all assistant messages."""
        total = 0.0
        for msg in self.messages:
            if msg.get("is_assistant") and msg.get("model_stats"):
                model_stats = msg.get("model_stats", {})
                if isinstance(model_stats, dict):
                    # model_stats can be {model_name: {requests_cost: value, ...}}
                    for model_name, stats in model_stats.items():
                        if isinstance(stats, dict):
                            cost = stats.get("requests_cost")
                            if cost is not None:
                                try:
                                    total += float(cost)
                                except (ValueError, TypeError):
                                    pass
        return total

    @rx.var
    def total_conversation_cost_str(self) -> str:
        """Formatted string representation of total conversation cost."""
        cost = self.total_conversation_cost
        if cost > 0:
            return f"${cost:.6f}"
        return ""

    def _append_message(
        self,
        role: str,
        content: str,
        technical_info: dict[str, Any] | None = None,
        debug_logs: str | None = None,
    ) -> None:
        message: dict[str, Any] = {
            "id": str(uuid4()),
            "role": role,
            "content": content,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "is_assistant": role == "assistant",
            "has_tech": False,
            "show_details": False,
            "tags": [],
            "comments": [],
        }

        if technical_info:
            vts_list = [str(x) for x in technical_info.get("vts_queries") or []]
            cypher_list = [str(x) for x in technical_info.get("cypher_queries") or []]
            models_raw = technical_info.get("model_stats", {})
            message["dm_snapshot_id"] = str(technical_info.get("dm_snapshot_id", "") or "")
            model_pairs: list[tuple[str, str]] = []
            if isinstance(models_raw, dict):
                for mk, mv in models_raw.items():
                    try:
                        model_pairs.append((str(mk), json.dumps(mv).decode()))
                    except Exception:
                        model_pairs.append((str(mk), str(mv)))
            models_list = [f"{k}: {v}" for k, v in model_pairs]

            message["vts_str"] = "\n".join(vts_list)
            message["cypher_str"] = "\n".join(cypher_list)
            message["models_str"] = "\n".join(models_list)
            message["has_vts"] = bool(vts_list)
            message["has_cypher"] = bool(cypher_list)
            message["has_models"] = bool(models_list)
            message["has_tech"] = bool(vts_list or cypher_list or models_list or message.get("dm_snapshot_id"))
            message["model_stats"] = models_raw

        logs = (debug_logs or "").replace("\r\n", "\n").rstrip()
        message["logs_str"] = logs
        message["has_logs"] = bool(logs)

        self.messages.append(message)

    async def _ensure_thread(self) -> str:
        vedana_app = await get_vedana_app()
        try:
            existing = await ThreadController.from_thread_id(vedana_app.sessionmaker, UUID(self.chat_thread_id))
        except Exception:
            existing = None

        if existing is None:
            thread_id = uuid7()
            ctl = await ThreadController.new_thread(
                vedana_app.sessionmaker,
                contact_id=f"reflex:{thread_id}",
                thread_id=thread_id,
                thread_config={"interface": "reflex"},
            )
            return str(ctl.thread.thread_id)

        return self.chat_thread_id

    async def _run_message(self, thread_id: str, user_text: str) -> Tuple[str, Dict[str, Any], str]:
        vedana_app = await get_vedana_app()
        await self._apply_data_model_selection()
        try:
            tid = UUID(thread_id)
        except Exception:
            tid = uuid7()

        ctl = await ThreadController.from_thread_id(vedana_app.sessionmaker, tid)
        if ctl is None:
            thr_id = uuid7()
            ctl = await ThreadController.new_thread(
                vedana_app.sessionmaker,
                contact_id=f"reflex:{thr_id}",
                thread_id=thr_id,
                thread_config={"interface": "reflex"},
            )

        mem_logger = MemLogger("rag_debug", level=logging.DEBUG)

        await ctl.store_user_message(uuid7(), user_text)

        pipeline = vedana_app.pipeline
        pipeline.logger = mem_logger
        pipeline.model = f"{self.provider}/{self.model}"
        pipeline.enable_filtering = self.enable_dm_filtering

        ctx = await ctl.make_context()

        # override model api_key if custom api_key is provided
        if self.custom_openrouter_key and self.provider == "openrouter":
            ctx.llm.model_api_key = self.custom_openrouter_key
            # embeddings model is not customisable in chat, it's configured on project level.

        events = await ctl.run_pipeline_with_context(pipeline, ctx)

        answer: str = ""
        tech: dict[str, Any] = {}
        for ev in events:
            if ev.event_type == "comm.assistant_message":
                answer = str(ev.event_data.get("content", ""))
            elif ev.event_type == "rag.query_processed":
                tech = dict(ev.event_data.get("technical_info", {}))

        logs = mem_logger.get_logs()
        return answer, tech, logs

    # mypy raises error: "EventNamespace" not callable [operator], though event definition is according to reflex docs
    @rx.event(background=True)  # type: ignore[operator]
    async def send_background(self, user_text: str):
        """This runs in background (non-blocking), after send() submission."""
        # Need context to modify state safely
        async with self:
            try:
                thread_id = await self._ensure_thread()
                answer, tech, logs = await self._run_message(thread_id, user_text)
                # update shared session thread id
                self.chat_thread_id = thread_id
            except Exception as e:
                answer, tech, logs = (f"Error: {e}", {}, traceback.format_exc())

            self._append_message("assistant", answer, technical_info=tech, debug_logs=logs)
            self.is_running = False

    def send(self):
        """form submit / button click"""
        if self.is_running:
            return

        user_text = (self.input_text or "").strip()
        if not user_text:
            return

        self._append_message("user", user_text)
        self.input_text = ""
        self.is_running = True
        yield  # trigger UI update

        yield ChatState.send_background(user_text)

    def open_jims_thread(self):
        """Open the JIMS page and preselect the current chat thread."""
        if not (self.chat_thread_id or "").strip():
            return

        # First set selection on JIMS state, then navigate to /jims
        yield ThreadViewState.select_thread(thread_id=self.chat_thread_id)  # type: ignore[operator]
        yield rx.redirect("/jims")  # type: ignore[operator]

    @rx.event(background=True)  # type: ignore[operator]
    async def load_data_model_text(self):
        async with self:
            va = await get_vedana_app()
            try:
                await self._apply_data_model_selection()
                snapshot_id = va.data_model.get_snapshot_id()
                self.data_model_snapshot_id = str(snapshot_id) if snapshot_id is not None else ""
                self.data_model_text = await va.data_model.to_text_descr()
            except Exception:
                self.data_model_text = "(failed to load data model text)"
                self.data_model_snapshot_id = ""

            self.is_refreshing_dm = False  # make the update button available
            yield

    def reload_data_model(self):
        """Connecting button with a background task. Used to trigger animations properly."""
        if self.is_refreshing_dm:
            return
        self.is_refreshing_dm = True
        yield
        yield ChatState.reload_data_model_background()

    @rx.event(background=True)  # type: ignore[operator]
    async def reload_data_model_background(self):
        try:
            """Reload the data model from Grist into config-plane dev branch."""
            async with self:
                va = await get_vedana_app()
                snapshot_id = await asyncio.to_thread(
                    va.data_model.update_from_grist,
                    core_settings.config_plane_dev_branch,
                )
                await self._apply_data_model_selection()
                snapshot_id = va.data_model.get_snapshot_id()
                self.data_model_snapshot_id = str(snapshot_id) if snapshot_id is not None else ""
                self.data_model_text = await va.data_model.to_text_descr()
            msg = (
                f"Data model updated in dev (snapshot {snapshot_id})"
                if snapshot_id is not None
                else "Data model unchanged"
            )
            yield rx.toast.success(msg)
        except Exception as e:
            async with self:
                error_msg = str(e)
                self.data_model_text = f"(error reloading data model: {error_msg})"
            yield rx.toast.error(f"Failed to reload data model\n{error_msg}")
        finally:
            async with self:
                self.is_refreshing_dm = False
            yield
