import json
import logging
import threading
import time
from datetime import datetime
from queue import Empty, Queue
from typing import Any, Dict, Iterable, Tuple
from uuid import UUID, uuid4

import pandas as pd
import reflex as rx
from datapipe.compute import run_steps
from jims_core.thread.thread_controller import ThreadController
from jims_core.util import uuid7
from vedana_core.app import VedanaApp, make_vedana_app
from vedana_core.data_model import DataModel
from vedana_core.settings import settings as core_settings
from vedana_etl.app import app as etl_app
from vedana_etl.app import pipeline
from vedana_etl.config import DBCONN_DATAPIPE

vedana_app: VedanaApp | None = None


async def get_vedana_app():
    global vedana_app
    if vedana_app is None:
        vedana_app = await make_vedana_app()
    return vedana_app


class EtlState(rx.State):
    """ETL control state and actions."""

    # Selections
    selected_flow: str = ""
    selected_stage: str = ""

    # Derived from pipeline
    all_steps: list[dict[str, Any]] = []  # [{name, type, inputs, outputs, labels}]
    filtered_steps: list[dict[str, Any]] = []
    available_tables: list[str] = []

    # Run status
    is_running: bool = False
    last_run_started_at: float | None = None
    last_run_finished_at: float | None = None
    logs: list[str] = []

    # Table preview
    preview_table_name: str | None = None
    preview_rows: list[dict[str, Any]] = []
    preview_columns: list[str] = []
    has_preview: bool = False

    def _start_log_capture(self) -> tuple[Queue[str], logging.Handler, logging.Logger]:
        q: Queue[str] = Queue()

        class _QueueHandler(logging.Handler):
            def __init__(self, queue: Queue[str]):
                super().__init__()
                self._q = queue

            def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
                try:
                    msg = self.format(record)
                    self._q.put(msg)
                except Exception:
                    pass

        handler = _QueueHandler(q)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))

        logger = logging.getLogger("datapipe")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        return q, handler, logger

    def _stop_log_capture(self, handler: logging.Handler, logger: logging.Logger) -> None:
        try:
            logger.removeHandler(handler)
        except Exception:
            pass

    def _drain_queue_into_logs(self, q: Queue[str]) -> None:
        while True:
            try:
                msg = q.get_nowait()
            except Empty:
                break
            else:
                self.logs.append(msg)

    def _append_log(self, msg: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {msg}")

    def load_pipeline_metadata(self) -> None:
        """Populate all_steps and available_tables by introspecting vedana_etl pipeline and catalog."""

        steps_meta: list[dict[str, Any]] = []
        for idx, step in enumerate(pipeline.steps):  # type: ignore[attr-defined]
            inputs = [el.name for el in getattr(step, "inputs", [])]
            outputs = [el.name for el in getattr(step, "outputs", [])]
            labels = getattr(step, "labels", []) or []
            steps_meta.append(
                {
                    "index": idx,
                    "name": step.func.__name__,
                    "inputs": list(inputs),
                    "outputs": list(outputs),
                    "labels": list(labels),
                    "inputs_str": ", ".join([str(x) for x in list(inputs)]),
                    "outputs_str": ", ".join([str(x) for x in list(outputs)]),
                    "labels_str": ", ".join([f"{k}:{v}" for k, v in list(labels)]),
                }
            )

        self.all_steps = steps_meta
        self._update_filtered_steps()

        tables: set[str] = set()
        for step in pipeline.steps:  # get tables from pipeline
            if hasattr(step, "inputs"):
                for input in step.inputs:
                    tables.add(input.name)
            if hasattr(step, "outputs"):
                for output in step.outputs:
                    tables.add(output.name)

        self.available_tables = sorted(tables)

    def set_flow(self, flow: str) -> None:
        self.selected_flow = "" if str(flow).lower() == "all" else flow
        self._update_filtered_steps()

    def set_stage(self, stage: str) -> None:
        self.selected_stage = "" if str(stage).lower() == "all" else stage
        self._update_filtered_steps()

    def _filter_steps_by_labels(self, steps: Iterable[Any]) -> list[Any]:
        if not self.selected_flow and not self.selected_stage:
            return list(steps)

        flow_val = self.selected_flow
        stage_val = self.selected_stage

        def matches(step: Any) -> bool:
            labels = getattr(step, "labels", []) or []
            label_map: dict[str, set[str]] = {}
            for key, value in labels:
                label_map.setdefault(str(key), set()).add(str(value))

            if flow_val and flow_val not in label_map.get("flow", set()):
                return False
            if stage_val and stage_val not in label_map.get("stage", set()):
                return False
            return True

        return [s for s in steps if matches(s)]

    def _update_filtered_steps(self) -> None:
        """Update filtered_steps used for UI from all_steps based on current filters."""
        flow_val = self.selected_flow
        stage_val = self.selected_stage

        def matches_meta(meta: dict[str, Any]) -> bool:
            labels = meta.get("labels", []) or []
            label_map: dict[str, set[str]] = {}
            for key, value in labels:
                label_map.setdefault(str(key), set()).add(str(value))
            if flow_val and flow_val not in label_map.get("flow", set()):
                return False
            if stage_val and stage_val not in label_map.get("stage", set()):
                return False
            return True

        if not flow_val and not stage_val:
            self.filtered_steps = list(self.all_steps)
        else:
            self.filtered_steps = [m for m in self.all_steps if matches_meta(m)]

    def _run_steps_sync(self, steps_to_run: list[Any]) -> None:
        # Run each step sequentially to provide granular logs
        for step in steps_to_run:
            step_name = getattr(step, "name", type(step).__name__)
            self._append_log(f"Running step: {step_name}")
            run_steps(etl_app.ds, [step])  # type: ignore[arg-type]
            self._append_log(f"Completed step: {step_name}")

    def run_selected(self):  # type: ignore[override]
        """Run the ETL for selected labels in background, streaming logs."""
        if self.is_running:
            return None

        self.is_running = True
        self.last_run_started_at = time.time()
        self.logs = []
        self._append_log("Starting ETL run …")
        yield

        try:
            steps_to_run = self._filter_steps_by_labels(etl_app.steps)
            if not steps_to_run:
                self._append_log("No steps match selected filters")
                return

            # stream datapipe logs into UI while each step runs
            q, handler, logger = self._start_log_capture()
            try:
                for step in steps_to_run:
                    step_name = getattr(step, "name", type(step).__name__)
                    self._append_log(f"Running step: {step_name}")

                    def _runner(s=step):
                        run_steps(etl_app.ds, [s])  # type: ignore[arg-type]

                    t = threading.Thread(target=_runner, daemon=True)
                    t.start()
                    while t.is_alive():
                        self._drain_queue_into_logs(q)
                        yield
                        time.sleep(0.1)
                    t.join(timeout=0)
                    self._drain_queue_into_logs(q)
                    self._append_log(f"Completed step: {step_name}")
                    yield
            finally:
                self._stop_log_capture(handler, logger)
        finally:
            self.is_running = False
            self.last_run_finished_at = time.time()
            self._append_log("ETL run finished")

    # removed async runner; running synchronously with yields

    def run_one_step(self, index: int | None = None):  # type: ignore[override]
        if self.is_running:
            return None

        if index is None:
            return None

        if index < 0 or index >= len(etl_app.steps):
            self._append_log(f"Invalid step index: {index}")
            return None

        step = etl_app.steps[index]
        self.is_running = True
        self.logs = []
        self.last_run_started_at = time.time()
        self._append_log(f"Starting single step {step.name}")
        yield

        try:
            q, handler, logger = self._start_log_capture()
            try:

                def _runner():
                    run_steps(etl_app.ds, [step])  # type: ignore[arg-type]

                t = threading.Thread(target=_runner, daemon=True)
                t.start()
                while t.is_alive():
                    self._drain_queue_into_logs(q)
                    yield
                    time.sleep(0.1)
                t.join(timeout=0)
                self._drain_queue_into_logs(q)
            finally:
                self._stop_log_capture(handler, logger)
        finally:
            self.is_running = False
            self.last_run_finished_at = time.time()
            self._append_log("Single step finished")
        return None

    def preview_table(self, table_name: str) -> None:
        """Load a small preview from the datapipe DB for a selected table."""

        self.preview_table_name = table_name
        self.has_preview = False
        try:
            engine = DBCONN_DATAPIPE.con  # type: ignore[attr-defined]
        except Exception:
            self._append_log("DB engine not available")
            return

        try:
            df = pd.read_sql(f'SELECT * FROM "{table_name}" LIMIT 50', con=engine)
        except Exception as e:
            self._append_log(f"Failed to load table {table_name}: {e}")
            return

        self.preview_columns = [str(c) for c in df.columns]
        records_any: list[dict[Any, Any]] = df.astype(object).where(pd.notna(df), None).to_dict(orient="records")
        coerced: list[dict[str, Any]] = []
        for r in records_any:
            try:
                coerced.append({str(k): v for k, v in dict(r).items()})
            except Exception:
                coerced.append({})
        self.preview_rows = coerced
        self.has_preview = len(self.preview_rows) > 0


class ChatState(rx.State):
    """Minimal chatbot with per-answer technical details."""

    input_text: str = ""
    is_running: bool = False
    messages: list[dict[str, Any]] = []
    chat_thread_id: str = ""
    data_model_text: str = ""
    data_model_last_sync: str = ""
    is_refreshing_dm: bool = True

    async def mount(self) -> None:
        global vedana_app
        if vedana_app is None:
            vedana_app = await make_vedana_app()
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

    def _append_message(
        self,
        role: str,
        content: str,
        technical_info: dict[str, Any] | None = None,
    ) -> None:
        message: dict[str, Any] = {
            "id": str(uuid4()),
            "role": role,
            "content": content,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "is_assistant": role == "assistant",
            "has_tech": False,
            "show_details": False,
        }

        if technical_info:
            vts_list = [str(x) for x in technical_info.get("vts_queries") or []]
            cypher_list = [str(x) for x in technical_info.get("cypher_queries") or []]
            models_raw = technical_info.get("model_stats", {})
            model_pairs: list[tuple[str, str]] = []
            if isinstance(models_raw, dict):
                for mk, mv in models_raw.items():
                    try:
                        model_pairs.append((str(mk), json.dumps(mv)))
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

        self.messages.append(message)

    async def _ensure_thread(self) -> str:
        vedana_app = await get_vedana_app()
        try:
            existing = await ThreadController.from_thread_id(vedana_app.sessionmaker, UUID(self.chat_thread_id))
        except Exception:
            existing = None

        if existing is None:
            ctl = await ThreadController.new_thread(
                vedana_app.sessionmaker,
                uuid7(),
                {"interface": "reflex-chat", "created_at": time.time()},
            )
            return str(ctl.thread.thread_id)

        return self.chat_thread_id

    async def _run_message(self, thread_id: str, user_text: str) -> Tuple[str, Dict[str, Any]]:
        vedana_app = await get_vedana_app()
        try:
            tid = UUID(thread_id)
        except Exception:
            tid = uuid7()

        ctl = await ThreadController.from_thread_id(vedana_app.sessionmaker, tid)
        if ctl is None:
            ctl = await ThreadController.new_thread(vedana_app.sessionmaker, uuid7(), {"interface": "reflex"})

        await ctl.store_user_message(uuid7(), user_text)
        events = await ctl.run_pipeline_with_context(vedana_app.pipeline)

        answer: str = ""
        tech: dict[str, Any] = {}
        for ev in events:
            if ev.event_type == "comm.assistant_message":
                answer = str(ev.event_data.get("content", ""))
            elif ev.event_type == "rag.query_processed":
                tech = dict(ev.event_data.get("technical_info", {}))

        return answer, tech

    @rx.event(background=True)
    async def send_background(self, user_text: str):
        """This runs in background (non-blocking), after send() submission."""
        # Need context to modify state safely
        async with self:
            try:
                thread_id = await self._ensure_thread()
                answer, tech = await self._run_message(thread_id, user_text)
                # update shared session thread id
                self.chat_thread_id = thread_id
            except Exception as e:
                answer, tech = (f"Error: {e}", {})

            self._append_message("assistant", answer, technical_info=tech)
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

    @rx.event(background=True)
    async def load_data_model_text(self):
        async with self:
            va = await get_vedana_app()
            try:
                self.data_model_text = va.data_model.to_text_descr()
            except Exception:
                self.data_model_text = "(failed to load data model text)"

            self.is_refreshing_dm = False  # make the update button available
            yield

    @rx.event(background=True)
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
