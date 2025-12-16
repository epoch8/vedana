import logging
import traceback
from datetime import datetime
from typing import Any, Dict, Tuple
from uuid import UUID, uuid4

import orjson as json
import reflex as rx
from jims_core.thread.thread_controller import ThreadController
from jims_core.util import uuid7
from vedana_core.app import RagPipeline
from vedana_core.data_model import DataModel
from vedana_core.settings import settings as core_settings

from vedana_backoffice.states.common import MemLogger, get_vedana_app
from vedana_backoffice.states.jims import ThreadViewState


class ChatState(rx.State):
    """Minimal chatbot with per-answer technical details."""

    input_text: str = ""
    is_running: bool = False
    messages: list[dict[str, Any]] = []
    chat_thread_id: str = ""
    data_model_text: str = ""
    data_model_last_sync: str = ""
    is_refreshing_dm: bool = True
    model = core_settings.model

    async def mount(self) -> None:
        self.data_model_last_sync: str = datetime.now().strftime("%Y-%m-%d %H:%M")

    def set_input(self, value: str) -> None:
        self.input_text = value

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
        # Create a per-request pipeline to avoid mutating the globally cached pipeline/logger
        req_pipeline = RagPipeline(
            graph=vedana_app.graph,
            data_model=vedana_app.data_model,
            logger=mem_logger,
            threshold=getattr(vedana_app.pipeline, "threshold", 0.8),
            top_n=getattr(vedana_app.pipeline, "top_n", 5),
            model=getattr(vedana_app.pipeline, "model", None),
        )

        answer: str = ""
        tech: dict[str, Any] = {}
        try:
            await ctl.store_user_message(uuid7(), user_text)
            events = await ctl.run_pipeline_with_context(req_pipeline)

            for ev in events:
                if ev.event_type == "comm.assistant_message":
                    answer = str(ev.event_data.get("content", ""))
                elif ev.event_type == "rag.query_processed":
                    tech = dict(ev.event_data.get("technical_info", {}))
        except Exception as e:
            mem_logger.exception(f"Error processing query: {e}")
            answer, tech = (f"Error: {e}", {})

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
                self.data_model_text = va.data_model.to_text_descr()
            except Exception:
                self.data_model_text = "(failed to load data model text)"

            self.is_refreshing_dm = False  # make the update button available
            yield

    @rx.event(background=True)  # type: ignore[operator]
    async def refresh_data_model(self):
        async with self:
            try:
                # todo: UI updates are not triggered from async with self, need to fix
                # https://reflex.dev/blog/2023-09-28-unlocking-new-workflows-with-background-tasks/
                self.is_refreshing_dm = True
                yield  # trigger UI update

                # get App
                va = await get_vedana_app()

                # Get new DM
                new_dm = DataModel.load_grist_online(
                    core_settings.grist_data_model_doc_id,
                    grist_server=core_settings.grist_server_url,
                    api_key=core_settings.grist_api_key,
                )
                await new_dm.update_data_model_node(va.graph)

                # update App
                va.data_model = new_dm
                try:
                    va.pipeline.data_model = new_dm
                except Exception:
                    pass

                # update UI
                self.data_model_text = new_dm.to_text_descr()
                self.data_model_last_sync = datetime.now().strftime("%Y-%m-%d %H:%M")
                self.is_refreshing_dm = False
                try:
                    yield rx.toast.success("Data model refreshed")  # type: ignore[attr-defined]
                except Exception:
                    pass
            except Exception as e:
                self.data_model_text = f"Error refreshing DataModel: {e}"
                self.is_refreshing_dm = False
                try:
                    yield rx.toast.error(f"Failed to refresh data model: {e}")  # type: ignore[attr-defined]
                except Exception:
                    pass
