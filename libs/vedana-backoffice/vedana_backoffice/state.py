import asyncio
import time
from typing import Any, Iterable

import pandas as pd
import reflex as rx


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

    def _append_log(self, msg: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {msg}")

    def load_pipeline_metadata(self) -> None:
        """Populate all_steps and available_tables by introspecting vedana_etl pipeline and catalog."""
        from vedana_etl.app import catalog, pipeline  # lazy import

        steps_meta: list[dict[str, Any]] = []
        for idx, step in enumerate(pipeline.steps):  # type: ignore[attr-defined]
            step_type = type(step).__name__
            inputs = getattr(step, "inputs", []) or []
            outputs = getattr(step, "outputs", []) or []
            labels = getattr(step, "labels", []) or []
            steps_meta.append(
                {
                    "index": idx,
                    "name": getattr(step, "name", f"{step_type}_{idx}"),
                    "type": step_type,
                    "inputs": list(inputs),
                    "outputs": list(outputs),
                    "labels": list(labels),
                    "inputs_str": ", ".join([str(x) for x in list(inputs)]),
                    "outputs_str": ", ".join([str(x) for x in list(outputs)]),
                    "labels_str": ", ".join([f"{k}:{v}" for k, v in list(labels)]),
                }
            )

        self.all_steps = steps_meta

        tables: list[str] = []
        for table_name in catalog.catalog.keys():  # type: ignore[attr-defined]
            tables.append(str(table_name))
        self.available_tables = sorted(tables)

    def set_flow(self, flow: str) -> None:
        self.selected_flow = flow

    def set_stage(self, stage: str) -> None:
        self.selected_stage = stage

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

    def _run_steps_sync(self, steps_to_run: list[Any]) -> None:
        from vedana_etl.app import app as etl_app  # DatapipeAPI with ds and steps
        from datapipe.compute import run_steps

        # Run each step sequentially to provide granular logs
        for step in steps_to_run:
            step_name = getattr(step, "name", type(step).__name__)
            self._append_log(f"Running step: {step_name}")
            run_steps(etl_app.ds, [step])  # type: ignore[arg-type]
            self._append_log(f"Completed step: {step_name}")

    def run_selected(self) -> rx.event.EventSpec | None:  # type: ignore[override]
        """Run the ETL for selected labels in background, streaming logs."""
        if self.is_running:
            return None

        from vedana_etl.app import pipeline  # type: ignore

        self.is_running = True
        self.last_run_started_at = time.time()
        self.logs = []
        self._append_log("Starting ETL run …")
        yield

        try:
            steps_to_run = self._filter_steps_by_labels(pipeline.steps)  # type: ignore[attr-defined]
            if not steps_to_run:
                self._append_log("No steps match selected filters")
                return

            # Run synchronously, yielding between steps
            for step in steps_to_run:
                self._run_steps_sync([step])
                yield
        finally:
            self.is_running = False
            self.last_run_finished_at = time.time()
            self._append_log("ETL run finished")

    # removed async runner; running synchronously with yields

    def run_one_step(self, index: int) -> rx.event.EventSpec | None:  # type: ignore[override]
        if self.is_running:
            return None

        from vedana_etl.app import pipeline  # type: ignore

        steps = list(pipeline.steps)  # type: ignore[attr-defined]
        if index < 0 or index >= len(steps):
            self._append_log(f"Invalid step index: {index}")
            return None

        step = steps[index]
        self.is_running = True
        self.logs = []
        self.last_run_started_at = time.time()
        self._append_log(f"Starting single step #{index}")
        yield

        try:
            self._run_steps_sync([step])
            yield
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
        self.preview_rows = df.astype(object).where(pd.notna(df), None).to_dict(orient="records")
        self.has_preview = len(self.preview_rows) > 0



