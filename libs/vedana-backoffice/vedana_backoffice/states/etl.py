import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from queue import Empty, Queue
from typing import Any, Iterable

import pandas as pd
import reflex as rx
import sqlalchemy as sa
from datapipe.compute import run_steps
from datapipe.step.batch_transform import BaseBatchTransformStep
from vedana_etl.app import app as etl_app
from vedana_etl.app import pipeline
from vedana_etl.config import DBCONN_DATAPIPE

from vedana_backoffice.graph.build import build_canonical, derive_step_edges, derive_table_edges, refine_layer_orders
from vedana_backoffice.util import safe_render_value


@dataclass
class EtlTableStats:
    table_name: str
    process_ts: float  # last process_ts
    row_count: int  # excluding those with delete_ts != NULL

    last_update_ts: float


@dataclass
class EtlDataTableStats(EtlTableStats):
    last_update_rows: int  # update_ts = max(update_ts)
    last_added_rows: int  # create_ts = last_update_ts
    last_deleted_rows: int  # delete_ts = last_update_ts


@dataclass
class EtlStepRunStats:
    """Stats for a pipeline step's last run (grouped batch execution)."""

    step_name: str
    meta_table_name: str  # The transform's meta table name
    last_run_start: float  # Start of the last run window (earliest process_ts in the run)
    last_run_end: float  # End of the last run window (latest process_ts in the run)
    rows_processed: int  # Total rows processed in the last run
    rows_success: int  # Rows with is_success=True in last run
    rows_failed: int  # Rows with is_success=False in last run
    total_success: int  # All rows with is_success=True (all time)
    total_failed: int  # All rows with is_success=False (all time)


# Threshold in seconds to group consecutive process_ts values into a single "run"
# If gap between consecutive process_ts > threshold, it's a new run
RUN_GAP_THRESHOLD_SECONDS = 300  # 5 minutes


class EtlState(rx.State):
    """ETL control state and actions."""

    default_pipeline_name = "main"

    # Selections
    selected_flow: str = "all"
    selected_stage: str = "all"
    selected_pipeline: str = default_pipeline_name

    # Derived from pipeline
    all_steps: list[dict[str, Any]] = []  # [{name, type, inputs, outputs, labels}]
    filtered_steps: list[dict[str, Any]] = []
    available_tables: list[str] = []
    available_flows: list[str] = []
    available_stages: list[str] = []
    available_pipelines: list[str] = [default_pipeline_name]
    pipeline_steps: dict[str, list[dict[str, Any]]] = {}
    pipeline_flows: dict[str, list[str]] = {}
    pipeline_stages: dict[str, list[str]] = {}

    # Graph view state
    graph_nodes: list[dict[str, Any]] = []  # [{index, name, x, y, w, h, labels_str}]
    graph_edges: list[dict[str, Any]] = []  # [{source, target, label, path, label_x, label_y}]
    graph_width_px: int = 1200
    graph_height_px: int = 600
    graph_svg: str = ""
    graph_width_css: str = "1200px"
    graph_height_css: str = "600px"

    # Run status
    is_running: bool = False
    logs: list[str] = []
    max_log_lines: int = 2000

    # UI toggles
    sidebar_open: bool = True
    logs_open: bool = True

    # Multi-select of nodes (by step index)
    selected_node_ids: list[int] = []
    selection_source: str = "filter"

    # View mode: False = step-centric, True = data(table)-centric
    data_view: bool = False

    # Table metadata for data-centric view
    table_meta: dict[str, EtlDataTableStats] = {}
    step_meta: dict[int, EtlStepRunStats] = {}  # for step-centric view as well (keys=index)

    # Table preview panel state
    preview_open: bool = False
    preview_anchor_left: str = "0px"
    preview_anchor_top: str = "0px"

    # Step status cache: name -> {total_idx_count, changed_idx_count}
    step_status_by_name: dict[str, dict[str, int]] = {}
    step_status_loading: bool = False

    # Table preview
    preview_table_name: str | None = None
    preview_display_name: str = ""
    preview_rows: list[dict[str, Any]] = []
    preview_columns: list[str] = []
    has_preview: bool = False

    # Server-side pagination for preview
    preview_page: int = 0  # 0-indexed current page
    preview_page_size: int = 100  # rows per page
    preview_total_rows: int = 0  # total count
    preview_is_meta_table: bool = False  # whether we're viewing _meta table

    # Show only changes from last run (with styling)
    preview_changes_only: bool = False

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
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))

        logger = logging.getLogger("datapipe")
        logger.setLevel(logging.INFO)
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
        self.logs = self.logs[-self.max_log_lines :]

    def load_pipeline_metadata(self) -> None:
        """Populate metadata by introspecting the ETL pipeline definition."""

        pipeline_buckets: dict[str, dict[str, Any]] = {}
        pipeline_order: list[str] = []
        tables: set[str] = set()

        def get_bucket(name: str) -> dict[str, Any]:
            key = str(name).strip() or self.default_pipeline_name
            if key not in pipeline_buckets:
                pipeline_buckets[key] = {"steps": [], "flows": set(), "stages": set()}
                pipeline_order.append(key)
            return pipeline_buckets[key]

        for idx, step in enumerate(pipeline.steps):
            inputs = [el.name for el in getattr(step, "inputs", [])]
            outputs = [el.name for el in getattr(step, "outputs", [])]
            labels = getattr(step, "labels", []) or []

            meta = {
                "index": idx,
                "name": step.func.__name__,  # type: ignore[attr-defined]
                "step_type": type(step).__name__,
                "inputs": list(inputs),
                "outputs": list(outputs),
                "labels": list(labels),
                "inputs_str": ", ".join([str(x) for x in list(inputs)]),
                "outputs_str": ", ".join([str(x) for x in list(outputs)]),
                "labels_str": ", ".join([f"{k}:{v}" for k, v in list(labels)]),
            }

            for input_name in inputs:
                tables.add(input_name)
            for output_name in outputs:
                tables.add(output_name)

            normalized_labels: list[tuple[str, str]] = []
            for key, value in labels:
                k = str(key)
                v = str(value)
                if v == "":
                    continue
                normalized_labels.append((k, v))

            pipeline_names = {v for k, v in normalized_labels if k == "pipeline"}
            flow_labels = {v for k, v in normalized_labels if k == "flow"}
            stage_labels = {v for k, v in normalized_labels if k == "stage"}

            if not pipeline_names:
                pipeline_names = {self.default_pipeline_name}

            for pipeline_name in pipeline_names:
                bucket = get_bucket(pipeline_name)
                bucket["steps"].append(meta)
                bucket["flows"].update(flow_labels)
                bucket["stages"].update(stage_labels)

        # Ensure default pipeline is always available even if no explicit labels exist.
        get_bucket(self.default_pipeline_name)

        self.pipeline_steps = {name: list(bucket["steps"]) for name, bucket in pipeline_buckets.items()}
        self.pipeline_flows = {name: sorted(bucket["flows"]) for name, bucket in pipeline_buckets.items()}
        self.pipeline_stages = {name: sorted(bucket["stages"]) for name, bucket in pipeline_buckets.items()}

        ordered_pipelines: list[str] = []
        seen: set[str] = set()
        default_name = self.default_pipeline_name

        if default_name in pipeline_buckets:
            ordered_pipelines.append(default_name)
            seen.add(default_name)

        for name in pipeline_order:
            if name in seen:
                continue
            ordered_pipelines.append(name)
            seen.add(name)

        for name in pipeline_buckets.keys():
            if name in seen:
                continue
            ordered_pipelines.append(name)
            seen.add(name)

        if not ordered_pipelines:
            ordered_pipelines = [default_name]

        self.available_pipelines = ordered_pipelines
        self.available_tables = sorted(tables)

        self._load_table_stats()
        self._load_step_stats()
        self._apply_pipeline_selection(
            target_pipeline=self.selected_pipeline,
            preserve_filters=True,
            preserve_selection=True,
        )

    def set_pipeline(self, pipeline_name: str) -> None:
        """Switch active pipeline tab."""
        desired = str(pipeline_name).strip() or self.default_pipeline_name
        if desired not in self.pipeline_steps:
            desired = self.default_pipeline_name
        if desired == self.selected_pipeline:
            return
        self._apply_pipeline_selection(target_pipeline=desired)

    def _apply_pipeline_selection(
        self,
        target_pipeline: str | None = None,
        preserve_filters: bool = False,
        preserve_selection: bool = False,
    ) -> None:
        previous_pipeline = getattr(self, "selected_pipeline", self.default_pipeline_name)
        prev_flow = self.selected_flow
        prev_stage = self.selected_stage
        prev_selection = list(self.selected_node_ids)
        prev_selection_source = self.selection_source

        active_name = target_pipeline or previous_pipeline or self.default_pipeline_name
        if active_name not in self.pipeline_steps:
            active_name = self.default_pipeline_name
        if active_name not in self.pipeline_steps and self.pipeline_steps:
            active_name = next(iter(self.pipeline_steps.keys()))

        self.selected_pipeline = active_name

        current_steps = self.pipeline_steps.get(active_name, [])
        self.all_steps = list(current_steps)

        flow_values = self.pipeline_flows.get(active_name, [])
        stage_values = self.pipeline_stages.get(active_name, [])
        self.available_flows = ["all", *flow_values]
        self.available_stages = ["all", *stage_values]

        if preserve_filters and (not prev_flow or prev_flow in flow_values):
            self.selected_flow = prev_flow
        else:
            self.selected_flow = "all"

        if preserve_filters and (not prev_stage or prev_stage in stage_values):
            self.selected_stage = prev_stage
        else:
            self.selected_stage = "all"

        if preserve_selection and previous_pipeline == active_name:
            self.selected_node_ids = prev_selection
            self.selection_source = prev_selection_source
        else:
            self.selected_node_ids = []
            self.selection_source = "filter"

        self._update_filtered_steps()

    def toggle_sidebar(self) -> None:
        self.sidebar_open = not self.sidebar_open

    def toggle_logs(self) -> None:
        self.logs_open = not self.logs_open

    def set_flow(self, flow: str) -> None:
        self.selected_flow = flow
        self.selection_source = "filter"
        self._update_filtered_steps()

    def set_stage(self, stage: str) -> None:
        self.selected_stage = stage
        self.selection_source = "filter"
        self._update_filtered_steps()

    def reset_filters(self) -> None:
        """Reset flow and stage selections and rebuild the graph."""
        self.selected_flow = "all"
        self.selected_stage = "all"
        self.selected_node_ids = []
        self.selection_source = "filter"
        self._update_filtered_steps()

    def set_data_view(self, checked: bool) -> None:
        """Toggle between step-centric and data-centric graph."""
        try:
            self.data_view = bool(checked)
        except Exception:
            self.data_view = False
        # Do not alter filters or explicit selections automatically
        self._rebuild_graph()

    def toggle_node_selection(self, index: int) -> None:
        """Toggle selection; manual interactions become authoritative unless "all selected" case."""
        try:
            sid = int(index)
        except Exception:
            return

        # Compute current filter-driven set
        filter_ids: set[int] = set()
        for m in self.filtered_steps or []:
            try:
                midx = int(m.get("index", -1))
                if midx >= 0:
                    filter_ids.add(midx)
            except Exception:
                continue

        manual_set: set[int] = set(self.selected_node_ids or [])

        # Special case: when current selection is filter-driven, on click switch to manual with a single selection
        if self.selection_source == "filter":
            # If current filter selects all or many, treat as filter-driven
            # On click, switch to manual with single selection
            self.selected_node_ids = [sid]
            self.selection_source = "manual"
            self._rebuild_graph()
            return

        # Otherwise we are in manual mode: toggle within manual set
        if sid in manual_set:
            manual_set.remove(sid)
        else:
            manual_set.add(sid)
        # If manual set becomes empty, fall back to filter selection
        if not manual_set:
            self.selection_source = "filter"
            self.selected_node_ids = []
        else:
            self.selection_source = "manual"
            self.selected_node_ids = sorted(list(manual_set))
        self._rebuild_graph()

    def clear_node_selection(self) -> None:
        self.selected_node_ids = []
        self._rebuild_graph()

    def _get_current_pipeline_step_indices(self) -> set[int]:
        """Get the indices of steps that belong to the currently selected pipeline."""
        return {int(m.get("index", -1)) for m in self.all_steps if m.get("index") is not None}

    def _filter_steps_by_labels(self, steps: Iterable[Any], restrict_to_pipeline: bool = True) -> list[Any]:
        """Filter steps by flow/stage labels.

        Args:
            steps: The steps to filter (from etl_app.steps)
            restrict_to_pipeline: If True, only include steps from the current pipeline
        """
        # Get valid step indices for the current pipeline
        if restrict_to_pipeline:
            valid_indices = self._get_current_pipeline_step_indices()
        else:
            valid_indices = None

        def matches(step: Any, idx: int) -> bool:
            # Check if step is in the current pipeline
            if valid_indices is not None and idx not in valid_indices:
                return False

            # Check flow/stage labels
            if self.selected_flow == "all" and self.selected_stage == "all":
                return True

            labels = getattr(step, "labels", []) or []
            label_map: dict[str, set[str]] = {}
            for key, value in labels:
                label_map.setdefault(str(key), set()).add(str(value))

            if self.selected_flow != "all" and self.selected_flow not in label_map.get("flow", set()):
                return False
            if self.selected_stage != "all" and self.selected_stage not in label_map.get("stage", set()):
                return False
            return True

        return [s for idx, s in enumerate(steps) if matches(s, idx)]

    def _update_filtered_steps(self) -> None:
        """Update filtered_steps used for UI from all_steps based on current filters."""

        def matches_meta(meta: dict[str, Any]) -> bool:
            labels = meta.get("labels", []) or []
            label_map: dict[str, set[str]] = {}
            for key, value in labels:
                label_map.setdefault(str(key), set()).add(str(value))
            # Both conditions must match (additive filtering)
            flow_match = self.selected_flow == "all" or self.selected_flow in label_map.get("flow", set())
            stage_match = self.selected_stage == "all" or self.selected_stage in label_map.get("stage", set())
            return flow_match and stage_match

        if self.selected_flow == "all" and self.selected_stage == "all":
            self.filtered_steps = list(self.all_steps)
        else:
            self.filtered_steps = [m for m in self.all_steps if matches_meta(m)]
        # Keep graph in sync with filter changes
        self._rebuild_graph()

    # --- Graph building and layout ---

    def _rebuild_graph(self) -> None:
        """Build graph nodes/edges and compute a simple layered layout.

        Nodes are steps; edges are derived when an output table of one step
        appears as an input table of another step. A basic DAG layering is
        computed using indegrees (Kahn) and used to place nodes left-to-right.
        """
        try:
            metas = self.all_steps  # Build graph based on current view mode
            cg = build_canonical(metas)  # Build canonical graph indexes once

            # Styling: basic node size and spacing constants (px)
            MIN_NODE_W = 220
            MAX_NODE_W = 420
            NODE_H = 90
            H_SPACING = 120
            V_SPACING = 40
            MARGIN = 24

            if not self.data_view:  # -------- STEP-CENTRIC VIEW --------
                step_ids: list[int] = sorted([s.index for s in cg.steps])
                name_by: dict[int, str] = dict(cg.step_name_by_index)
                step_type_by: dict[int, str] = dict(cg.step_type_by_index)
                # Optional labels_str for node text
                labels_str_by: dict[int, str] = {}
                for m in metas:
                    try:
                        idx = int(m.get("index", -1))
                        labels_str_by[idx] = str(m.get("labels_str", ""))
                    except Exception:
                        pass

                unique_ids = list(step_ids)

                # Derive edges by table indexes (linear)
                edges = derive_step_edges(cg)

                # Compute indegrees for Kahn layering
                indeg: dict[int, int] = {sid: 0 for sid in unique_ids}
                children: dict[int, list[int]] = {sid: [] for sid in unique_ids}
                parents: dict[int, list[int]] = {sid: [] for sid in unique_ids}
                for s, t, _ in edges:
                    indeg[t] = indeg.get(t, 0) + 1
                    children.setdefault(s, []).append(t)
                    parents.setdefault(t, []).append(s)

                # Initialize layers
                layer_by: dict[int, int] = {sid: 0 for sid in unique_ids}

                q: deque[int] = deque([sid for sid in unique_ids if indeg.get(sid, 0) == 0])
                visited: set[int] = set()
                while q:
                    sid = q.popleft()
                    visited.add(sid)
                    for ch in children.get(sid, []):
                        # longest path layering
                        layer_by[ch] = max(layer_by.get(ch, 0), layer_by.get(sid, 0) + 1)
                        indeg[ch] -= 1
                        if indeg[ch] == 0:
                            q.append(ch)

                # Any nodes not visited (cycle/isolated) keep default layer 0
                # Group nodes by layer
                layers: dict[int, list[int]] = {}
                max_layer = 0
                for sid in unique_ids:
                    layer = layer_by.get(sid, 0)
                    max_layer = max(max_layer, layer)
                    layers.setdefault(layer, []).append(sid)

                # Stable order in each layer by barycenter of incoming neighbors (reduces edge length)
                for layer, arr in list(layers.items()):

                    def _barycenter(node_id: int) -> float:
                        parent_ids = [s for s, t, _ in edges if t == node_id]
                        if not parent_ids:
                            return float(layer)
                        return sum([layer_by.get(p, 0) for p in parent_ids]) / float(len(parent_ids))

                    layers[layer] = sorted(arr, key=lambda i: (_barycenter(i), name_by.get(i, "")))

                refine_layer_orders(layers, parents, children, max_layer)

                # Pre-compute content-based widths/heights per node
                w_by: dict[int, int] = {}
                h_by: dict[int, int] = {}
                for sid in unique_ids:
                    nlen = len(name_by.get(sid, "")) + 15  # + 15 from the step type label (BatchTransform etc.)
                    llen = len(labels_str_by.get(sid, ""))
                    w = min(max(nlen * 8 + 40, llen * 6 + 10, MIN_NODE_W), MAX_NODE_W)
                    w_by[sid] = w
                    chars_per_line = max(10, int((w - 40) / 7))
                    lines = 1
                    try:
                        lines = max(1, -(-llen // chars_per_line))
                    except Exception:
                        lines = 1
                    base_h = NODE_H
                    extra_lines = max(0, lines - 2)
                    h_by[sid] = base_h + extra_lines * 14

            else:  # -------- DATA-CENTRIC VIEW --------
                # Build table-centric graph inputs from canonical
                table_list = sorted(list(set([t for s in cg.steps for t in s.inputs + s.outputs])))
                table_id_by_name: dict[str, int] = {name: i for i, name in enumerate(table_list)}
                # Create pseudo ids for layout
                unique_ids = list(range(len(table_list)))
                name_by = {i: n for i, n in enumerate(table_list)}
                labels_str_by = {i: "" for i in unique_ids}
                # Edges between tables from canonical
                table_edges = derive_table_edges(cg)
                # Build adjacency for layering based on table graph (DV-specific names)
                edges_dv: list[tuple[int, int, list[str]]] = []
                for s_id, t_id, _ in table_edges:
                    if s_id < 0:
                        continue
                    edges_dv.append((s_id, t_id, []))
                indeg2: dict[int, int] = {sid: 0 for sid in unique_ids}
                children2: dict[int, list[int]] = {sid: [] for sid in unique_ids}
                parents2: dict[int, list[int]] = {sid: [] for sid in unique_ids}
                for s2, t2, _ in edges_dv:
                    indeg2[t2] = indeg2.get(t2, 0) + 1
                    children2.setdefault(s2, []).append(t2)
                    parents2.setdefault(t2, []).append(s2)
                layer_by2: dict[int, int] = {sid: 0 for sid in unique_ids}

                q2: deque[int] = deque([sid for sid in unique_ids if indeg2.get(sid, 0) == 0])
                visited2: set[int] = set()
                while q2:
                    sid = q2.popleft()
                    visited2.add(sid)
                    for ch in children2.get(sid, []):
                        layer_by2[ch] = max(layer_by2.get(ch, 0), layer_by2.get(sid, 0) + 1)
                        indeg2[ch] -= 1
                        if indeg2[ch] == 0:
                            q2.append(ch)
                layers2: dict[int, list[int]] = {}
                max_layer = 0
                for sid2 in unique_ids:
                    layer = layer_by2.get(sid2, 0)
                    max_layer = max(max_layer, layer)
                    layers2.setdefault(layer, []).append(sid2)
                # Data view: order by barycenter of incoming neighbors to reduce crossings/length
                for layer, arr in list(layers2.items()):

                    def _barycenter(node_id: int) -> float:
                        parent_ids = [s for s, t, _ in edges_dv if t == node_id]
                        if not parent_ids:
                            return float(layer)
                        return sum([layer_by2.get(p, 0) for p in parent_ids]) / float(len(parent_ids))

                    layers2[layer] = sorted(arr, key=lambda i: (_barycenter(i), name_by.get(i, "")))

                refine_layer_orders(layers2, parents2, children2, max_layer)
                # Sizes for tables
                w_by = {}
                h_by = {}
                for sid in unique_ids:
                    nlen = len(name_by.get(sid, ""))
                    est = max(nlen * 8 + 60, MIN_NODE_W)
                    w = min(max(int(est), MIN_NODE_W), MAX_NODE_W)
                    w_by[sid] = w
                    h_by[sid] = NODE_H

            # Selected node ids based on selection source
            selected_ids: set[int] = set()
            try:
                if not self.data_view:
                    if self.selection_source == "manual" and self.selected_node_ids:
                        for midx in self.selected_node_ids:
                            selected_ids.add(int(midx))
                    else:
                        for m in self.filtered_steps or []:
                            midx = int(m.get("index", -1))
                            if midx >= 0:
                                selected_ids.add(midx)
            except Exception:
                selected_ids = set()

            # Column widths (max of nodes in that layer)
            col_w: dict[int, int] = (
                {
                    _l: (max([w_by[sid] for sid in layers.get(_l, [])]) if layers.get(_l) else MIN_NODE_W)
                    for _l in layers
                }
                if not self.data_view
                else {
                    _l: (max([w_by[sid] for sid in layers2.get(_l, [])]) if layers2.get(_l) else MIN_NODE_W)
                    for _l in (layers2 if "layers2" in locals() else {})
                }
            )

            # Prefix sums for x offsets per layer
            def layer_x(layer: int) -> int:
                x = MARGIN
                for i in range(0, layer):
                    x += col_w.get(i, MIN_NODE_W) + H_SPACING
                return x

            # Compute positions
            nodes: list[dict[str, Any]] = []
            pos_by: dict[int, tuple[int, int]] = {}
            max_rows = max(
                [len((layers2 if self.data_view else layers).get(layer, [])) for layer in range(0, max_layer + 1)]
                or [1]
            )
            # compute row heights as max node height across layers for each row index
            row_heights: list[int] = [NODE_H for _ in range(max_rows)]
            for layer in range(0, max_layer + 1):
                cols = (layers2 if self.data_view else layers).get(layer, [])
                for r_idx, sid in enumerate(cols):
                    row_heights[r_idx] = max(row_heights[r_idx], h_by.get(sid, NODE_H))

            # precompute y offsets
            y_offsets: list[int] = []
            acc = MARGIN
            for r in range(max_rows):
                y_offsets.append(acc)
                acc += row_heights[r] + V_SPACING

            for _l in range(0, max_layer + 1):
                cols = (layers2 if self.data_view else layers).get(_l, [])
                for r_idx, sid in enumerate(cols):
                    x = layer_x(_l)
                    y = y_offsets[r_idx]
                    pos_by[sid] = (x, y)
                    if not self.data_view:  # transformation view
                        step_name = name_by.get(sid, f"step_{sid}")

                        # Get step stats from step_meta
                        step_stats = self.step_meta.get(sid)
                        if step_stats and step_stats.last_run_end > 0:
                            dt = datetime.fromtimestamp(step_stats.last_run_end)
                            last_run_str = dt.strftime("%Y-%m-%d %H:%M")
                            rows_processed = step_stats.rows_processed
                            rows_success = step_stats.rows_success
                            rows_failed = step_stats.rows_failed
                            total_success = step_stats.total_success
                            total_failed = step_stats.total_failed
                        else:
                            last_run_str = "—"
                            rows_processed = 0
                            rows_success = 0
                            rows_failed = 0
                            total_success = 0
                            total_failed = 0

                        nodes.append(
                            {
                                "index": sid,
                                "index_value": str(sid),
                                "name": step_name,
                                "labels_str": labels_str_by.get(sid, ""),
                                "step_type": step_type_by.get(sid, ""),
                                "node_type": "step",
                                # numeric position/size (might be useful elsewhere)
                                "x": x,
                                "y": y,
                                "w": w_by.get(sid, MIN_NODE_W),
                                "h": h_by.get(sid, NODE_H),
                                # css strings for Reflex styles (avoid Python ops on Vars)
                                "left": f"{x}px",
                                "top": f"{y}px",
                                "width": f"{w_by.get(sid, MIN_NODE_W)}px",
                                "height": f"{h_by.get(sid, NODE_H)}px",
                                # Step run stats
                                "last_run": last_run_str,
                                "rows_processed": rows_processed,
                                "rows_success": rows_success,
                                "rows_failed": rows_failed,
                                "has_failed": rows_failed > 0,  # Pre-computed for Reflex rx.cond
                                "rows_failed_str": f"({rows_failed} ✗)" if rows_failed > 0 else "",
                                # All-time run stats
                                "total_success": total_success,
                                "total_failed": total_failed,
                                "has_total_failed": total_failed > 0,  # Pre-computed for Reflex rx.cond
                                "total_failed_str": f"({total_failed} ✗)" if total_failed > 0 else "",
                                "selected": sid in selected_ids,
                                "border_css": "2px solid #3b82f6" if sid in selected_ids else "1px solid #e5e7eb",
                            }
                        )
                    else:  # data view: nodes are tables
                        table_name = name_by.get(sid, f"table_{sid}")
                        rc = self.table_meta[table_name]
                        if rc.process_ts:
                            dt = datetime.fromtimestamp(rc.process_ts)
                            last_run_str = dt.strftime("%Y-%m-%d %H:%M")
                        else:
                            last_run_str = "—"

                        nodes.append(
                            {
                                "index": sid,
                                "index_value": str(sid),
                                "name": table_name,
                                "labels_str": "",
                                "node_type": "table",
                                "x": x,
                                "y": y,
                                "w": w_by.get(sid, MIN_NODE_W),
                                "h": h_by.get(sid, NODE_H),
                                "left": f"{x}px",
                                "top": f"{y}px",
                                "width": f"{w_by.get(sid, MIN_NODE_W)}px",
                                "height": f"{h_by.get(sid, NODE_H)}px",
                                "last_run": last_run_str,
                                "row_count": rc.row_count,
                                "last_add": f"+{rc.last_added_rows}",
                                "last_upd": f"{rc.last_update_rows}",
                                "last_rm": f"-{rc.last_deleted_rows}",
                                "selected": (self.preview_table_name == table_name),
                                "border_css": (
                                    "2px solid #3b82f6"
                                    if (self.preview_table_name == table_name)
                                    else "1px solid #e5e7eb"
                                ),
                            }
                        )

            # Compute canvas size
            layers_count = max_layer + 1 if unique_ids else 1
            width_px = (
                MARGIN * 2
                + sum([col_w.get(i, MIN_NODE_W) for i in range(0, layers_count)])
                + max(0, layers_count - 1) * H_SPACING
            )
            height_px = MARGIN * 2 + sum(row_heights) + max(0, max_rows - 1) * V_SPACING

            # Build edges visuals
            edge_objs: list[dict[str, Any]] = []
            if not self.data_view:
                for s, t, shared in edges:
                    sx, sy = pos_by.get(s, (MARGIN, MARGIN))
                    tx, ty = pos_by.get(t, (MARGIN, MARGIN))
                    x1 = sx + w_by.get(s, MIN_NODE_W)
                    y1 = sy + h_by.get(s, NODE_H) // 2
                    x2 = tx
                    y2 = ty + h_by.get(t, NODE_H) // 2
                    cx1 = x1 + 40
                    cx2 = x2 - 40
                    path = f"M{x1},{y1} C{cx1},{y1} {cx2},{y2} {x2},{y2}"
                    label = ", ".join(shared)
                    label_x = (x1 + x2) / 2
                    label_y = (y1 + y2) / 2 - 6
                    edge_objs.append(
                        {
                            "source": s,
                            "target": t,
                            "label": label,
                            "path": path,
                            "label_x": label_x,
                            "label_y": label_y,
                            # highlight only if both endpoints are selected
                            "selected": (s in selected_ids) and (t in selected_ids),
                        }
                    )
            else:
                for s_id, t_id, step_label in table_edges:
                    tx, ty = pos_by.get(t_id, (MARGIN, MARGIN))
                    if s_id < 0:
                        x1 = max(0, MARGIN - 120)
                        y1 = ty + h_by.get(t_id, NODE_H) // 2
                    else:
                        sx, sy = pos_by.get(s_id, (MARGIN, MARGIN))
                        x1 = sx + w_by.get(s_id, MIN_NODE_W)
                        y1 = sy + h_by.get(s_id, NODE_H) // 2
                    x2 = tx
                    y2 = ty + h_by.get(t_id, NODE_H) // 2
                    cx1 = x1 + 40
                    cx2 = x2 - 40
                    path = f"M{x1},{y1} C{cx1},{y1} {cx2},{y2} {x2},{y2}"
                    label = step_label
                    label_x = (x1 + x2) / 2
                    label_y = (y1 + y2) / 2 - 6
                    # Highlight edges incident to the previewed table
                    sel = False
                    try:
                        if s_id < 0:  # For generator edges, highlight when the target table is highlighted
                            sel = self.preview_table_name == name_by.get(t_id, "")
                        else:  # For regular edges, require both endpoints highlighted (not applicable for now)
                            sel = False
                    except Exception:
                        sel = False
                    edge_objs.append(
                        {
                            "source": s_id,
                            "target": t_id,
                            "label": label,
                            "path": path,
                            "label_x": label_x,
                            "label_y": label_y,
                            "selected": sel,
                        }
                    )

            # Build SVG string for edges
            svg_parts: list[str] = []
            svg_parts.append(
                f'<svg width="{width_px}" height="{height_px}" viewBox="0 0 {width_px} {height_px}" preserveAspectRatio="xMinYMin meet" xmlns="http://www.w3.org/2000/svg">'
            )
            for e in edge_objs:
                stroke = "#3b82f6" if e.get("selected") else "#9ca3af"
                width = "2.5" if e.get("selected") else "2"
                opacity = "1.0" if e.get("selected") else "0.6"
                svg_parts.append(
                    f'<path d="{e["path"]}" stroke="{stroke}" stroke-width="{width}" opacity="{opacity}" fill="none" />'
                )
                if e.get("label"):
                    svg_parts.append(
                        f'<text x="{e["label_x"]}" y="{e["label_y"]}" font-size="10" fill="#6b7280" text-anchor="middle">{e["label"]}</text>'
                    )
            svg_parts.append("</svg>")

            self.graph_nodes = nodes
            self.graph_edges = edge_objs
            self.graph_width_px = int(width_px)
            self.graph_height_px = int(height_px)
            self.graph_svg = "".join(svg_parts)
            self.graph_width_css = f"{int(width_px)}px"
            self.graph_height_css = f"{int(height_px)}px"

            # Position the popover anchor centrally in the viewport to avoid overflow
            # Using fixed positioning (set in UI), so we use viewport-relative values
            # Center both horizontally and vertically to give popover room to expand in any direction
            self.preview_anchor_left = "50vw"
            self.preview_anchor_top = "50vh"

        except Exception:
            self.graph_nodes = []
            self.graph_edges = []
            self.graph_width_px = 800
            self.graph_height_px = 400
            self.graph_svg = ""
            self.graph_width_css = "800px"
            self.graph_height_css = "400px"

    def _load_table_stats(self):
        """
        Load stats for each data table using run-based grouping.

        A "run" is a group of consecutive timestamps where gaps are < RUN_GAP_THRESHOLD_SECONDS.
        This accounts for batch processing where each batch has different timestamps.
        """
        con = DBCONN_DATAPIPE.con  # type: ignore[attr-defined]
        self.table_meta = {}

        for tname in list(self.available_tables):
            try:
                meta_table_name = f"{tname}_meta"

                # Get all distinct update_ts values to detect runs
                ts_query = sa.text(f"""
                    SELECT DISTINCT update_ts FROM "{meta_table_name}" 
                    WHERE update_ts IS NOT NULL 
                    ORDER BY update_ts DESC 
                    LIMIT 1000
                """)

                with con.begin() as conn:
                    result = conn.execute(ts_query)
                    timestamps = [float(row[0]) for row in result.fetchall() if row[0] is not None]

                run_start, run_end = self._detect_last_run_window(timestamps)

                # Now query stats using the run window
                if run_start > 0.0:
                    query = sa.text(f"""
                        SELECT 
                            MAX(process_ts) AS process_ts,
                            COUNT(*) FILTER (WHERE delete_ts IS NULL) AS row_count,
                            {run_end} AS last_update_ts,
                            COUNT(*) FILTER (WHERE update_ts IS NOT NULL AND update_ts != create_ts AND update_ts >= {run_start} AND update_ts <= {run_end}) AS last_update_rows,
                            COUNT(*) FILTER (WHERE create_ts IS NOT NULL AND create_ts >= {run_start} AND create_ts <= {run_end}) AS last_added_rows,
                            COUNT(*) FILTER (WHERE delete_ts IS NOT NULL AND delete_ts >= {run_start} AND delete_ts <= {run_end}) AS last_deleted_rows
                        FROM "{meta_table_name}";
                    """)
                else:
                    # Fallback to original query if no timestamps found
                    query = sa.text(f"""
                        WITH max_update_ts AS (
                            SELECT MAX(update_ts) AS max_ts FROM "{meta_table_name}"
                        )
                        SELECT 
                            MAX(process_ts) AS process_ts,
                            COUNT(*) FILTER (WHERE delete_ts IS NULL) AS row_count,
                            (SELECT max_ts FROM max_update_ts) AS last_update_ts,
                            COUNT(*) FILTER (WHERE update_ts IS NOT NULL AND update_ts != create_ts AND update_ts = (SELECT max_ts FROM max_update_ts)) AS last_update_rows,
                            COUNT(*) FILTER (WHERE create_ts IS NOT NULL AND create_ts = (SELECT max_ts FROM max_update_ts)) AS last_added_rows,
                            COUNT(*) FILTER (WHERE delete_ts IS NOT NULL AND delete_ts = (SELECT max_ts FROM max_update_ts)) AS last_deleted_rows
                        FROM "{meta_table_name}", max_update_ts;
                    """)

                with con.begin() as conn:
                    result = conn.execute(query)
                    row = result.fetchone()

                self.table_meta[tname] = EtlDataTableStats(
                    table_name=tname,
                    process_ts=float(row[0]) if row[0] is not None else 0.0,
                    row_count=int(row[1]) if row[1] is not None else 0,
                    last_update_ts=float(row[2]) if row[2] is not None else 0.0,
                    last_update_rows=int(row[3]) if row[3] is not None else 0,
                    last_added_rows=int(row[4]) if row[4] is not None else 0,
                    last_deleted_rows=int(row[5]) if row[5] is not None else 0,
                )

            except Exception as e:
                logging.warning(f"Failed to load stats for table {tname}: {e}", exc_info=True)
                self.table_meta[tname] = EtlDataTableStats(
                    table_name=tname,
                    process_ts=0.0,
                    row_count=0,
                    last_update_ts=0.0,
                    last_update_rows=0,
                    last_added_rows=0,
                    last_deleted_rows=0,
                )

    def _detect_last_run_window(self, timestamps: list[float]) -> tuple[float, float]:
        """Detect the last "run" window from a list of timestamps.

        A run is a group of consecutive timestamps where gaps are < RUN_GAP_THRESHOLD_SECONDS.
        Returns (run_start, run_end) for the most recent run, or (0.0, 0.0) if no timestamps.
        """
        if not timestamps:
            return 0.0, 0.0

        # Sort descending (most recent first)
        sorted_ts = sorted(timestamps, reverse=True)

        # Find the last run by walking backwards until gap exceeds threshold
        run_end = sorted_ts[0]
        run_start = sorted_ts[0]

        for i in range(1, len(sorted_ts)):
            gap = run_start - sorted_ts[i]
            if gap > RUN_GAP_THRESHOLD_SECONDS:
                # Gap too large - we've found the boundary of the last run
                break
            run_start = sorted_ts[i]

        return run_start, run_end

    def _load_step_stats(self) -> None:
        """Load stats for each pipeline step by querying their transform meta tables."""
        con = DBCONN_DATAPIPE.con  # type: ignore[attr-defined]
        self.step_meta = {}

        for idx, step in enumerate(etl_app.steps):
            try:
                # Only BaseBatchTransformStep has meta_table
                if not isinstance(step, BaseBatchTransformStep):
                    continue

                # Get the meta table name from the step
                meta_table = step.meta_table
                meta_table_name = str(meta_table.sql_table.name)

                # Query all distinct process_ts values to detect runs
                ts_query = sa.text(
                    f'SELECT DISTINCT process_ts FROM "{meta_table_name}" WHERE process_ts IS NOT NULL ORDER BY process_ts DESC LIMIT 1000'
                )

                with con.begin() as conn:
                    result = conn.execute(ts_query)
                    timestamps = [float(row[0]) for row in result.fetchall() if row[0] is not None]

                run_start, run_end = self._detect_last_run_window(timestamps)

                # Query all-time totals first
                totals_query = sa.text(f"""
                    SELECT 
                        COUNT(*) FILTER (WHERE is_success = TRUE) AS total_success,
                        COUNT(*) FILTER (WHERE is_success = FALSE) AS total_failed
                    FROM "{meta_table_name}"
                """)

                with con.begin() as conn:
                    totals_result = conn.execute(totals_query)
                    totals_row = totals_result.fetchone()

                total_success = int(totals_row[0]) if totals_row and totals_row[0] else 0
                total_failed = int(totals_row[1]) if totals_row and totals_row[1] else 0

                if run_start == 0.0 and run_end == 0.0:
                    # No data
                    self.step_meta[idx] = EtlStepRunStats(
                        step_name=step.name,
                        meta_table_name=meta_table_name,
                        last_run_start=0.0,
                        last_run_end=0.0,
                        rows_processed=0,
                        rows_success=0,
                        rows_failed=0,
                        total_success=total_success,
                        total_failed=total_failed,
                    )
                    continue

                # Query rows in the last run window
                stats_query = sa.text(f"""
                    SELECT 
                        COUNT(*) AS total,
                        COUNT(*) FILTER (WHERE is_success = TRUE) AS success,
                        COUNT(*) FILTER (WHERE is_success = FALSE) AS failed
                    FROM "{meta_table_name}"
                    WHERE process_ts >= {run_start} AND process_ts <= {run_end}
                """)

                with con.begin() as conn:
                    result = conn.execute(stats_query)
                    row = result.fetchone()

                self.step_meta[idx] = EtlStepRunStats(
                    step_name=step.name,
                    meta_table_name=meta_table_name,
                    last_run_start=run_start,
                    last_run_end=run_end,
                    rows_processed=int(row[0]) if row and row[0] else 0,
                    rows_success=int(row[1]) if row and row[1] else 0,
                    rows_failed=int(row[2]) if row and row[2] else 0,
                    total_success=total_success,
                    total_failed=total_failed,
                )

            except Exception as e:
                logging.warning(f"Failed to load step stats for step {idx}: {e}", exc_info=True)
                step_name = getattr(step, "name", f"step_{idx}")
                self.step_meta[idx] = EtlStepRunStats(
                    step_name=step_name,
                    meta_table_name="",
                    last_run_start=0.0,
                    last_run_end=0.0,
                    rows_processed=0,
                    rows_success=0,
                    rows_failed=0,
                    total_success=0,
                    total_failed=0,
                )

    def run_selected(self):  # type: ignore[override]
        """Run the ETL for selected labels in background, streaming logs."""
        if self.is_running:
            return None

        self.is_running = True
        self.logs = []
        self._append_log("Starting ETL run …")
        yield

        try:
            valid_pipeline_indices = self._get_current_pipeline_step_indices()

            if self.selection_source == "manual" and self.selected_node_ids:
                selected = [
                    i for i in self.selected_node_ids or [] if isinstance(i, int) and i in valid_pipeline_indices
                ]
                steps_to_run = [etl_app.steps[i] for i in sorted(selected) if 0 <= i < len(etl_app.steps)]
            else:
                steps_to_run = self._filter_steps_by_labels(etl_app.steps)
            if not steps_to_run:
                self._append_log("No steps match selected filters")
                return

            self._append_log(f"Steps to execute: {[getattr(s, 'name', type(s).__name__) for s in steps_to_run]}")

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
            self._append_log("ETL run finished")

    # removed async runner; running synchronously with yields

    def run_one_step(self, index: int | None = None):  # type: ignore[override]
        if self.is_running:
            return None

        if index is None:
            return None

        try:
            idx = int(index)
        except Exception:
            return None

        if idx < 0 or idx >= len(etl_app.steps):
            self._append_log(f"Invalid step index: {index}")
            return None

        step = etl_app.steps[idx]
        self.is_running = True
        self.logs = []
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
            self._append_log("Single step finished")
        return None

    def preview_table(self, table_name: str) -> None:
        """Load a paginated preview from the datapipe DB for a selected table."""

        # Toggle preview: if same table clicked, close; otherwise open new table
        if self.preview_table_name == table_name and self.preview_open:
            self.close_preview()
            return

        self.preview_table_name = table_name
        self.preview_display_name = ""
        self.preview_open = True
        self.has_preview = False
        self.preview_page = 0  # Reset to first page
        self.preview_is_meta_table = False
        # Rebuild immediately to reflect selection highlight in data view
        self._rebuild_graph()

        self._load_preview_page()

    def _load_preview_page(self) -> None:
        """Load the current page of preview data from the database."""
        if not self.preview_table_name:
            return

        if self.preview_changes_only:
            self._load_preview_changes_page()
        else:
            self._load_preview_all_page()

    def _load_preview_all_page(self) -> None:
        """Load all records (standard preview mode)."""
        engine = DBCONN_DATAPIPE.con  # type: ignore[attr-defined]
        table_name = self.preview_table_name
        if not table_name:
            return

        actual_table = table_name if not self.preview_is_meta_table else f"{table_name}_meta"
        offset = self.preview_page * self.preview_page_size

        try:
            # Try main table first (only on first load)
            if self.preview_page == 0 and not self.preview_is_meta_table:
                try:
                    # Get total count
                    count_result = pd.read_sql(f'SELECT COUNT(*) as cnt FROM "{table_name}"', con=engine)
                    self.preview_total_rows = int(count_result["cnt"].iloc[0])
                    actual_table = table_name
                except Exception:
                    # Fall back to _meta table
                    count_result = pd.read_sql(f'SELECT COUNT(*) as cnt FROM "{table_name}_meta"', con=engine)
                    self.preview_total_rows = int(count_result["cnt"].iloc[0])
                    self.preview_display_name = f"{table_name}_meta"
                    self.preview_is_meta_table = True
                    actual_table = f"{table_name}_meta"
            elif self.preview_is_meta_table:
                actual_table = f"{table_name}_meta"

            # Load page data
            df = pd.read_sql(
                f'SELECT * FROM "{actual_table}" LIMIT {self.preview_page_size} OFFSET {offset}',
                con=engine,
            )

        except Exception as e:
            self._append_log(f"Failed to load table {table_name}: {e}")
            return

        self.preview_columns = [str(c) for c in df.columns]
        records_any: list[dict[Any, Any]] = df.astype(object).where(pd.notna(df), None).to_dict(orient="records")

        coerced: list[dict[str, Any]] = []
        for r in records_any:
            try:
                coerced.append({str(k): safe_render_value(v) for k, v in dict(r).items()})
            except Exception:
                coerced.append({})

        self.preview_rows = coerced
        self.has_preview = len(self.preview_rows) > 0
        try:
            self._rebuild_graph()
        except Exception:
            pass

    def _load_preview_changes_page(self) -> None:
        """Load only records changed in the last run (with change type styling)."""
        table_name = self.preview_table_name
        if not table_name:
            return

        engine = DBCONN_DATAPIPE.con  # type: ignore[attr-defined]
        meta_table = f"{table_name}_meta"
        offset = self.preview_page * self.preview_page_size

        # Get the last run window using the same logic as _load_table_stats
        try:
            ts_query = sa.text(f"""
                SELECT DISTINCT update_ts FROM "{meta_table}" 
                WHERE update_ts IS NOT NULL 
                ORDER BY update_ts DESC 
                LIMIT 1000
            """)
            with engine.begin() as conn:
                result = conn.execute(ts_query)
                timestamps = [float(row[0]) for row in result.fetchall() if row[0] is not None]
            run_start, run_end = self._detect_last_run_window(timestamps)
        except Exception as e:
            self._append_log(f"Failed to detect last run window for {table_name}: {e}")
            run_start, run_end = 0.0, 0.0

        if run_start == 0.0:
            # No run detected, show empty
            self.preview_columns = []
            self.preview_rows = []
            self.preview_total_rows = 0
            self.has_preview = False
            return

        # Build query for changed records in the last run window
        meta_exclude = {"hash", "create_ts", "update_ts", "process_ts", "delete_ts"}

        try:
            inspector = sa.inspect(engine)
            try:
                base_cols = [c.get("name", "") for c in inspector.get_columns(table_name)]
                base_cols = [str(c) for c in base_cols if c]
            except Exception:
                base_cols = []
            meta_cols = [c.get("name", "") for c in inspector.get_columns(meta_table)]
            meta_cols = [str(c) for c in meta_cols if c]
        except Exception as e:
            self._append_log(f"Failed to inspect columns for {table_name}: {e}")
            return

        data_cols: list[str] = [c for c in meta_cols if c not in meta_exclude]
        display_cols: list[str] = [c for c in base_cols if c] if base_cols else list(data_cols)

        # Get total count of changed records on first page
        if self.preview_page == 0:
            q_count = sa.text(f"""
                SELECT COUNT(*)
                FROM "{meta_table}" AS m
                WHERE 
                    (m.delete_ts IS NOT NULL AND m.delete_ts >= {run_start} AND m.delete_ts <= {run_end})
                    OR
                    (m.update_ts IS NOT NULL AND m.update_ts >= {run_start} AND m.update_ts <= {run_end}
                        AND m.update_ts > m.create_ts)
                    OR
                    (m.create_ts IS NOT NULL AND m.create_ts >= {run_start} AND m.create_ts <= {run_end} 
                        AND m.delete_ts IS NULL)
            """)
            try:
                with engine.begin() as conn:
                    self.preview_total_rows = int(conn.execute(q_count).scalar() or 0)
            except Exception:
                self.preview_total_rows = 0

        # Build SELECT with join to base table (if exists) to get full data
        if base_cols:
            select_exprs: list[str] = []
            for c in display_cols:
                if c in meta_cols:
                    select_exprs.append(f'COALESCE(b."{c}", m."{c}") AS "{c}"')
                else:
                    select_exprs.append(f'b."{c}" AS "{c}"')
            select_cols = ", ".join(select_exprs)
            on_cond = " AND ".join([f'b."{c}" = m."{c}"' for c in data_cols])

            q_data = sa.text(f"""
                SELECT
                    {select_cols},
                    CASE 
                        WHEN m.delete_ts IS NOT NULL AND m.delete_ts >= {run_start} AND m.delete_ts <= {run_end} THEN 'deleted'
                        WHEN m.update_ts IS NOT NULL AND m.update_ts >= {run_start} AND m.update_ts <= {run_end}
                             AND m.update_ts > m.create_ts THEN 'updated'
                        WHEN m.create_ts IS NOT NULL AND m.create_ts >= {run_start} AND m.create_ts <= {run_end}
                             AND m.delete_ts IS NULL THEN 'added'
                        ELSE NULL
                    END AS change_type
                FROM "{meta_table}" AS m
                LEFT JOIN "{table_name}" AS b ON {on_cond}
                WHERE 
                    (m.delete_ts IS NOT NULL AND m.delete_ts >= {run_start} AND m.delete_ts <= {run_end})
                    OR
                    (m.update_ts IS NOT NULL AND m.update_ts >= {run_start} AND m.update_ts <= {run_end}
                        AND m.update_ts > m.create_ts)
                    OR
                    (m.create_ts IS NOT NULL AND m.create_ts >= {run_start} AND m.create_ts <= {run_end}
                        AND m.delete_ts IS NULL)
                ORDER BY COALESCE(m.update_ts, m.create_ts, m.delete_ts) DESC
                LIMIT {self.preview_page_size} OFFSET {offset}
            """)
        else:
            # No base table, query meta only
            select_cols = ", ".join([f'm."{c}"' for c in display_cols])
            q_data = sa.text(f"""
                SELECT
                    {select_cols},
                    CASE 
                        WHEN m.delete_ts IS NOT NULL AND m.delete_ts >= {run_start} AND m.delete_ts <= {run_end} THEN 'deleted'
                        WHEN m.update_ts IS NOT NULL AND m.update_ts >= {run_start} AND m.update_ts <= {run_end}
                             AND m.update_ts > m.create_ts THEN 'updated'
                        WHEN m.create_ts IS NOT NULL AND m.create_ts >= {run_start} AND m.create_ts <= {run_end}
                             AND m.delete_ts IS NULL THEN 'added'
                        ELSE NULL
                    END AS change_type
                FROM "{meta_table}" AS m
                WHERE 
                    (m.delete_ts IS NOT NULL AND m.delete_ts >= {run_start} AND m.delete_ts <= {run_end})
                    OR
                    (m.update_ts IS NOT NULL AND m.update_ts >= {run_start} AND m.update_ts <= {run_end}
                        AND m.update_ts > m.create_ts)
                    OR
                    (m.create_ts IS NOT NULL AND m.create_ts >= {run_start} AND m.create_ts <= {run_end}
                        AND m.delete_ts IS NULL)
                ORDER BY COALESCE(m.update_ts, m.create_ts, m.delete_ts) DESC
                LIMIT {self.preview_page_size} OFFSET {offset}
            """)

        try:
            df = pd.read_sql(q_data, con=engine)
        except Exception as e:
            self._append_log(f"Failed to load changes for {table_name}: {e}")
            return

        # Set columns (exclude change_type from display columns)
        self.preview_columns = [str(c) for c in display_cols]
        records_any: list[dict[Any, Any]] = df.astype(object).where(pd.notna(df), None).to_dict(orient="records")

        # Apply row styling based on change_type
        row_styling = {
            "added": {"backgroundColor": "rgba(34,197,94,0.12)"},
            "updated": {"backgroundColor": "rgba(245,158,11,0.12)"},
            "deleted": {"backgroundColor": "rgba(239,68,68,0.12)"},
        }

        styled: list[dict[str, Any]] = []
        for r in records_any:
            try:
                row_disp: dict[str, Any] = {str(k): safe_render_value(r.get(k)) for k in self.preview_columns}
                row_disp["row_style"] = row_styling.get(r.get("change_type", ""), {})
                styled.append(row_disp)
            except Exception:
                styled.append({})

        self.preview_rows = styled
        self.has_preview = len(self.preview_rows) > 0
        try:
            self._rebuild_graph()
        except Exception:
            pass

    def toggle_preview_changes_only(self, checked: bool) -> None:
        """Toggle between showing all records or only changes from last run."""
        self.preview_changes_only = bool(checked)
        self.preview_page = 0  # Reset to first page
        self._load_preview_page()

    def preview_next_page(self) -> None:
        """Load the next page of preview data."""
        max_page = (self.preview_total_rows - 1) // self.preview_page_size if self.preview_total_rows > 0 else 0
        if self.preview_page < max_page:
            self.preview_page += 1
            self._load_preview_page()

    def preview_prev_page(self) -> None:
        """Load the previous page of preview data."""
        if self.preview_page > 0:
            self.preview_page -= 1
            self._load_preview_page()

    def preview_first_page(self) -> None:
        """Jump to the first page."""
        if self.preview_page != 0:
            self.preview_page = 0
            self._load_preview_page()

    def preview_last_page(self) -> None:
        """Jump to the last page."""
        max_page = (self.preview_total_rows - 1) // self.preview_page_size if self.preview_total_rows > 0 else 0
        if self.preview_page != max_page:
            self.preview_page = max_page
            self._load_preview_page()

    @rx.var
    def preview_page_display(self) -> str:
        """Current page display (1-indexed for users)."""
        total_pages = (self.preview_total_rows - 1) // self.preview_page_size + 1 if self.preview_total_rows > 0 else 1
        return f"Page {self.preview_page + 1} of {total_pages}"

    @rx.var
    def preview_rows_display(self) -> str:
        """Display range of rows being shown."""
        if self.preview_total_rows == 0:
            return "No rows"
        start = self.preview_page * self.preview_page_size + 1
        end = min(start + self.preview_page_size - 1, self.preview_total_rows)
        return f"Rows {start}-{end} of {self.preview_total_rows}"

    @rx.var
    def preview_has_next(self) -> bool:
        """Whether there's a next page."""
        max_page = (self.preview_total_rows - 1) // self.preview_page_size if self.preview_total_rows > 0 else 0
        return self.preview_page < max_page

    @rx.var
    def preview_has_prev(self) -> bool:
        """Whether there's a previous page."""
        return self.preview_page > 0

    def close_preview(self) -> None:
        self.preview_open = False
        self.preview_table_name = None
        self.preview_display_name = ""
        self.preview_page = 0
        self.preview_total_rows = 0
        self.preview_is_meta_table = False
        self.preview_changes_only = False
        self.preview_rows = []
        self.preview_columns = []
        self.has_preview = False
        try:
            self._rebuild_graph()
        except Exception:
            pass

    def set_preview_open(self, open: bool) -> None:
        """Handle popover open/close state changes."""
        if not open:
            self.close_preview()
