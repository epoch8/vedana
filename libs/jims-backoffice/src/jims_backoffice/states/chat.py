import traceback
from datetime import datetime
from typing import Any, Dict, Tuple
from uuid import UUID, uuid4

import orjson as json
import reflex as rx
from jims_core.app import JimsApp
from jims_core.llms.llm_provider import LLMSettings
from jims_core.thread.thread_controller import ThreadController
from jims_core.util import uuid7

from jims_backoffice.app_loader import get_jims_app
from jims_backoffice.states.common import DEBUG_MODE, DebugState
from jims_backoffice.states.jims import ThreadViewState

_default_llm_settings = LLMSettings()  # type: ignore


class ChatState(rx.State, mixin=True):
    """Minimal chatbot with per-answer technical details.

    This is a state mixin. The running app must declare a concrete subclass
    (``class AppChatState(ChatState, rx.State): ...``) so it becomes a real
    Reflex substate, then register it via
    ``jims_backoffice.register_chat_state(AppChatState)``.

    Subclasses can extend ``_apply_pipeline_settings``, ``_make_llm_settings``,
    or ``_run_message`` to wire pipeline-specific knobs (e.g. data-model
    filtering, log capture) without touching the base flow.
    """

    input_text: str = ""
    is_running: bool = False
    messages: list[dict[str, Any]] = []
    chat_thread_id: str = ""
    model: str = _default_llm_settings.model
    default_models: list[str] = [_default_llm_settings.model]
    available_models: list[str] = [_default_llm_settings.model]

    model_selection_allowed: bool = DEBUG_MODE

    async def mount(self):
        if DEBUG_MODE:
            yield DebugState.load_available_models()

    @rx.event(background=True)  # type: ignore[operator]
    async def refresh_model_list(self) -> None:
        async with self:
            self.available_models = await self.get_var_value(DebugState.available_models)  # type: ignore[arg-type]
            self._sync_model()

    def set_input(self, value: str) -> None:
        self.input_text = value

    def set_model(self, value: str) -> None:
        self.model = value

    def _sync_model(self) -> None:
        """Realign selected model when model list changes."""
        if self.model not in self.available_models and self.available_models:
            for model in self.available_models:
                if model.endswith(_default_llm_settings.model):
                    self.model = model
                    break
            else:
                if "openrouter/openrouter/free" in self.available_models:
                    self.model = "openrouter/openrouter/free"
                else:
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

    async def _ensure_thread(self, jims_app: JimsApp) -> str:
        try:
            existing = await ThreadController.from_thread_id(jims_app.sessionmaker, UUID(self.chat_thread_id))
        except Exception:
            existing = None

        if existing is None:
            thread_id = uuid7()
            ctl = await ThreadController.new_thread(
                jims_app.sessionmaker,
                contact_id=f"reflex:{thread_id}",
                thread_id=thread_id,
                thread_config={"interface": "reflex"},
            )
            return str(ctl.thread.thread_id)

        return self.chat_thread_id

    async def _apply_pipeline_settings(self, pipeline: Any) -> str | None:
        """Configure the pipeline before running it.

        Subclasses can override this hook to wire pipeline-specific knobs (DM
        filtering, log capture, etc.). May return a ``debug_logs`` string that
        will be attached to the assistant message after the run.
        """
        if hasattr(pipeline, "model"):
            try:
                pipeline.model = self.model
            except Exception:
                pass
        return None

    async def _make_llm_settings(self) -> LLMSettings:
        """Build LLMSettings used to run the pipeline.

        Subclasses may override to inject embeddings model overrides etc.
        """
        return LLMSettings(model=self.model)  # type: ignore

    async def _run_message(self, thread_id: str, user_text: str) -> Tuple[str, Dict[str, Any], str]:
        jims_app = await get_jims_app()
        try:
            tid = UUID(thread_id)
        except Exception:
            tid = uuid7()

        ctl = await ThreadController.from_thread_id(jims_app.sessionmaker, tid)
        if ctl is None:
            thr_id = uuid7()
            ctl = await ThreadController.new_thread(
                jims_app.sessionmaker,
                contact_id=f"reflex:{thr_id}",
                thread_id=thr_id,
                thread_config={"interface": "reflex"},
            )

        await ctl.store_user_message(uuid7(), user_text)

        pipeline = jims_app.pipeline
        debug_logs = await self._apply_pipeline_settings(pipeline)

        ctx = await ctl.make_context(llm_settings=await self._make_llm_settings())
        events = await ctl.run_pipeline_with_context(pipeline, ctx)

        answer: str = ""
        tech: dict[str, Any] = {}
        for ev in events:
            if ev.event_type == "comm.assistant_message":
                answer = str(ev.event_data.get("content", ""))
            elif ev.event_type == "rag.query_processed":
                tech = dict(ev.event_data.get("technical_info", {}))

        return answer, tech, debug_logs or ""

    @rx.event(background=True)  # type: ignore[operator]
    async def send_background(self, user_text: str):
        """This runs in background (non-blocking), after send() submission."""
        async with self:
            try:
                jims_app = await get_jims_app()
                thread_id = await self._ensure_thread(jims_app)
                answer, tech, logs = await self._run_message(thread_id, user_text)
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
        yield

        # Use type(self) so subclasses dispatch their own send_background
        # (mixins are not addressable as event targets).
        yield type(self).send_background(user_text)

    def open_jims_thread(self):
        """Open the JIMS page and preselect the current chat thread."""
        if not (self.chat_thread_id or "").strip():
            return

        yield ThreadViewState.select_thread(thread_id=self.chat_thread_id)  # type: ignore[operator]
        yield rx.redirect("/jims")  # type: ignore[operator]
