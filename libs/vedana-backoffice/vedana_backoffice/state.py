import logging
import threading
import time
from queue import Empty, Queue
from typing import Any, Iterable

import pandas as pd
import reflex as rx
from datapipe.compute import run_steps
from vedana_etl.app import app as etl_app
from vedana_etl.app import pipeline


class AppState(rx.State):
    """Shared application state placeholder."""

    toast: str | None = None

    def notify(self, message: str) -> None:
        self.toast = message


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
        self._append_log(f"Starting single step #{index}")
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
        from vedana_etl.config import DBCONN_DATAPIPE

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
        records_any: list[dict[Any, Any]] = df.astype(object).where(pd.notna(df), None).to_dict(orient="records")  # type: ignore[assignment]
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
    messages: list[dict[str, Any]] = []  # {id, role, content, created_at, is_assistant, has_tech?, show_details?, ...}

    def set_input(self, value: str) -> None:
        self.input_text = value

    def clear(self) -> None:
        self.input_text = ""
        self.messages = []

    def toggle_details_by_id(self, message_id: str) -> None:
        for idx, m in enumerate(self.messages):
            if str(m.get("id")) == str(message_id):
                msg = dict(m)
                msg["show_details"] = not bool(msg.get("show_details"))
                self.messages[idx] = msg
                break

    def _append_message(self, role: str, content: str, technical_info: dict[str, Any] | None = None) -> None:
        from datetime import datetime
        from uuid import uuid4

        message: dict[str, Any] = {
            "id": str(uuid4()),
            "role": role,
            "content": content,
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "is_assistant": role == "assistant",
            "has_tech": False,
            "show_details": False,
        }
        if technical_info:
            import json as _json

            try:
                vts_list = [str(x) for x in (technical_info.get("vts_queries") or [])]
            except Exception:
                vts_list = []
            try:
                cypher_list = [str(x) for x in (technical_info.get("cypher_queries") or [])]
            except Exception:
                cypher_list = []
            models_raw = technical_info.get("model_stats", {}) or technical_info.get("model_used", {}) or {}
            model_pairs: list[tuple[str, str]] = []
            if isinstance(models_raw, dict):
                for mk, mv in models_raw.items():
                    try:
                        model_pairs.append((str(mk), _json.dumps(mv)))
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

    def send(self):  # type: ignore[override]
        if self.is_running:
            return None

        user_text = (self.input_text or "").strip()
        if not user_text:
            return None

        self._append_message("user", user_text)
        self.input_text = ""
        self.is_running = True
        yield

        result: dict[str, Any] = {"answer": "", "tech": {}}

        def _runner():
            try:
                import asyncio as _aio

                from jims_core.llms.llm_provider import LLMProvider
                from jims_core.thread.schema import CommunicationEvent
                from jims_core.thread.thread_context import ThreadContext
                from jims_core.util import uuid7
                from vedana_core.app import make_vedana_app

                async def _do():
                    vedana_app = await make_vedana_app()
                    history = [CommunicationEvent(role="user", content=user_text)]
                    ctx = ThreadContext(thread_id=uuid7(), history=history, events=[], llm=LLMProvider())
                    await vedana_app.pipeline(ctx)

                    answer = ""
                    tech: dict[str, Any] = {}
                    for ev in ctx.outgoing_events:
                        if ev.event_type == "comm.assistant_message":
                            answer = str(ev.event_data.get("content", ""))
                        elif ev.event_type == "rag.query_processed":
                            tech = dict(ev.event_data.get("technical_info", {}))
                    result["answer"] = answer
                    result["tech"] = tech

                _aio.run(_do())
            except Exception as e:
                result["answer"] = f"Error: {e}"
                result["tech"] = {}

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        while t.is_alive():
            yield
            time.sleep(0.1)
        t.join(timeout=0)

        self._append_message("assistant", str(result.get("answer", "")), technical_info=result.get("tech", {}))
        self.is_running = False
        yield
