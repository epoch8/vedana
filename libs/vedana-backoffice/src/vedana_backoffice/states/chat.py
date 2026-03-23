import asyncio
import logging
import traceback
from datetime import datetime
from typing import Any, Dict, Tuple
from uuid import UUID, uuid4

import orjson as json
import reflex as rx
from datapipe.compute import Catalog, run_pipeline
from jims_core.llms.llm_provider import LLMSettings
from jims_core.thread.thread_controller import ThreadController
from jims_core.util import uuid7
from vedana_core.settings import settings as core_settings
from vedana_etl.app import app as etl_app
from vedana_etl.pipeline import get_data_model_pipeline

from vedana_backoffice.states.common import (
    DEBUG_MODE,
    DebugState,
    MemLogger,
    datapipe_log_capture,
    get_vedana_app,
)
from jims_backoffice.states.jims import ThreadViewState


class ChatState(rx.State):
    """Minimal chatbot with per-answer technical details."""

    input_text: str = ""
    is_running: bool = False
    messages: list[dict[str, Any]] = []
    chat_thread_id: str = ""
    data_model_text: str = ""
    is_refreshing_dm: bool = False
    model: str = core_settings.model
    default_models: list[str] = list({core_settings.model, core_settings.filter_model})
    available_models: list[str] = default_models

    model_selection_allowed: bool = DEBUG_MODE
    enable_dm_filtering: bool = core_settings.enable_dm_filtering
    dm_filter_model: str = core_settings.filter_model

    async def mount(self):
        if DEBUG_MODE:
            yield DebugState.load_available_models()

    @rx.event(background=True)  # type: ignore[operator]
    async def refresh_model_list(self) -> None:
        async with self:
            self.available_models = await self.get_var_value(DebugState.available_models)  # type: ignore[arg-type]
            self._sync_model()
            self._sync_dm_filter_model()

    def set_input(self, value: str) -> None:
        self.input_text = value

    def set_model(self, value: str) -> None:
        self.model = value

    def set_enable_dm_filtering(self, value: bool) -> None:
        self.enable_dm_filtering = value

    def set_dm_filter_model(self, value: str) -> None:
        self.dm_filter_model = value

    def _sync_model(self) -> None:
        """Realign selected model when model list changes."""
        if self.model not in self.available_models and self.available_models:
            for model in self.available_models:
                if model.endswith(core_settings.model):  # or model.rsplit("/", 1)[-1] == core_settings.model
                    self.model = model
                    break
            else:
                if "openrouter/openrouter/free" in self.available_models:
                    self.model = "openrouter/openrouter/free"  # Openrouter has an endpoint with all free models, set it as default
                else:
                    self.model = self.available_models[0]

    def _sync_dm_filter_model(self) -> None:
        """Realign selected filter model when model list changes."""
        if self.dm_filter_model not in self.available_models and self.available_models:
            for model in self.available_models:
                if model.endswith(core_settings.filter_model):
                    self.dm_filter_model = model
                    break
            else:
                if "openrouter/openrouter/free" in self.available_models:  # openrouter provider
                    self.dm_filter_model = "openrouter/openrouter/free"
                else:
                    self.dm_filter_model = self.available_models[0]

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
            message["has_tech"] = bool(vts_list or cypher_list or models_list)
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
        pipeline.model = self.model
        pipeline.enable_filtering = self.enable_dm_filtering
        pipeline.filter_model = self.dm_filter_model

        ctx = await ctl.make_context(
            llm_settings=LLMSettings(
                model=self.model, 
                embeddings_model=core_settings.embeddings_model if not DEBUG_MODE else await self.get_var_value(DebugState.embeddings_model)  # type: ignore[arg-type]
            )
        )

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
                self.data_model_text = await va.data_model.to_text_descr()
            except Exception:
                self.data_model_text = "(failed to load data model text)"

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

            def _run_dm_pipeline():
                with datapipe_log_capture():
                    run_pipeline(etl_app.ds, Catalog({}), get_data_model_pipeline())

            await asyncio.to_thread(_run_dm_pipeline)
            async with self:
                va = await get_vedana_app()
                self.data_model_text = await va.data_model.to_text_descr()
            yield rx.toast.success("Data model reloaded")
        except Exception as e:
            async with self:
                error_msg = str(e)
                # self.data_model_text = f"(error reloading data model: {error_msg})"
            yield rx.toast.error(f"Failed to reload data model\n{error_msg}")
        finally:
            async with self:
                self.is_refreshing_dm = False
            yield
