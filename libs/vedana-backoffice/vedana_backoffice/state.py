import logging
import threading
import time
from datetime import datetime
from queue import Empty, Queue
from typing import Any, Dict, Iterable, Tuple
from uuid import UUID, uuid4

import orjson as json
import pandas as pd
import reflex as rx
from datapipe.compute import run_steps
from jims_core.thread.thread_controller import ThreadController
from jims_core.util import uuid7
from jims_core.db import ThreadEventDB
from vedana_core.app import VedanaApp, make_vedana_app
from vedana_core.data_model import DataModel
from vedana_core.settings import settings as core_settings
from vedana_etl.app import app as etl_app
from vedana_etl.app import pipeline
from vedana_etl.config import DBCONN_DATAPIPE

from vedana_backoffice.graph.build import build_canonical, derive_step_edges, derive_table_edges
from vedana_backoffice.util import datetime_to_age

from dataclasses import dataclass
import sqlalchemy as sa

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
    available_flows: list[str] = []
    available_stages: list[str] = []

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
    last_run_started_at: float | None = None
    last_run_finished_at: float | None = None
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
    table_row_counts: dict[str, int] = {}

    # Table preview panel state
    preview_open: bool = False
    preview_anchor_left: str = "0px"
    preview_anchor_top: str = "0px"

    # Step last-run timestamps (loaded from meta table)
    steps_last_run: dict[str, str] = {}
    # Step status cache: name -> {total_idx_count, changed_idx_count}
    step_status_by_name: dict[str, dict[str, int]] = {}
    step_status_loading: bool = False

    # Table preview
    preview_table_name: str | None = None
    preview_display_name: str = ""
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
        self.logs = self.logs[-self.max_log_lines :]

    def load_pipeline_metadata(self) -> None:
        """Populate all_steps and available_tables by introspecting vedana_etl pipeline and catalog."""

        steps_meta: list[dict[str, Any]] = []
        for idx, step in enumerate(pipeline.steps):
            inputs = [el.name for el in getattr(step, "inputs", [])]
            outputs = [el.name for el in getattr(step, "outputs", [])]
            labels = getattr(step, "labels", []) or []
            steps_meta.append(
                {
                    "index": idx,
                    "name": step.func.__name__,
                    "step_type": type(step).__name__,
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
        flows: set[str] = set()
        stages: set[str] = set()
        for step in pipeline.steps:  # get tables from pipeline
            if hasattr(step, "inputs"):
                for input in step.inputs:
                    tables.add(input.name)
            if hasattr(step, "outputs"):
                for output in step.outputs:
                    tables.add(output.name)
            if hasattr(step, "labels"):
                for key, value in step.labels:
                    k = str(key)
                    v = str(value)
                    if k == "flow" and v:
                        flows.add(v)
                    if k == "stage" and v:
                        stages.add(v)

        self.available_tables = sorted(tables)
        self.available_flows = ["all", *sorted(flows)]  # add "all"
        self.available_stages = ["all", *sorted(stages)]

        # Build initial graph
        try:
            self._load_last_run_timestamps()
        except Exception:
            self.steps_last_run = {}
        try:
            self._load_table_row_counts()
        except Exception:
            self.table_row_counts = {}
        self._rebuild_graph()

    def toggle_sidebar(self) -> None:
        self.sidebar_open = not self.sidebar_open

    def toggle_logs(self) -> None:
        self.logs_open = not self.logs_open

    def set_flow(self, flow: str) -> None:
        self.selected_flow = "" if str(flow).lower() == "all" else flow
        self.selection_source = "filter"
        self._update_filtered_steps()

    def set_stage(self, stage: str) -> None:
        self.selected_stage = "" if str(stage).lower() == "all" else stage
        self.selection_source = "filter"
        self._update_filtered_steps()

    def reset_filters(self) -> None:
        """Reset flow and stage selections and rebuild the graph."""
        self.selected_flow = ""
        self.selected_stage = ""
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
            # Both conditions must match (additive filtering)
            flow_match = not flow_val or flow_val in label_map.get("flow", set())
            stage_match = not stage_val or stage_val in label_map.get("stage", set())
            return flow_match and stage_match

        if not flow_val and not stage_val:
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
                for s, t, _ in edges:
                    indeg[t] = indeg.get(t, 0) + 1
                    children.setdefault(s, []).append(t)

                # Initialize layers
                layer_by: dict[int, int] = {sid: 0 for sid in unique_ids}
                from collections import deque

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
                        parents = [s for s, t, _ in edges if t == node_id]
                        if not parents:
                            return float(layer)
                        return sum([layer_by.get(p, 0) for p in parents]) / float(len(parents))

                    layers[layer] = sorted(arr, key=lambda i: (_barycenter(i), name_by.get(i, "")))

                # Pre-compute content-based widths/heights per node
                w_by: dict[int, int] = {}
                h_by: dict[int, int] = {}
                for sid in unique_ids:
                    nlen = len(name_by.get(sid, ""))
                    llen = len(labels_str_by.get(sid, ""))
                    est = max(nlen * 8 + 60, llen * 6 + 40, MIN_NODE_W)
                    w = min(max(int(est), MIN_NODE_W), MAX_NODE_W)
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
                for s2, t2, _ in edges_dv:
                    indeg2[t2] = indeg2.get(t2, 0) + 1
                    children2.setdefault(s2, []).append(t2)
                layer_by2: dict[int, int] = {sid: 0 for sid in unique_ids}
                from collections import deque

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
                        parents = [s for s, t, _ in edges_dv if t == node_id]
                        if not parents:
                            return float(layer)
                        return sum([layer_by2.get(p, 0) for p in parents]) / float(len(parents))

                    layers2[layer] = sorted(arr, key=lambda i: (_barycenter(i), name_by.get(i, "")))
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
                    if not self.data_view:
                        step_name = name_by.get(sid, f"step_{sid}")
                        last_run_str = self.steps_last_run.get(step_name, "—")
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
                                "last_run": last_run_str,
                                "selected": sid in selected_ids,
                                "border_css": "2px solid #3b82f6" if sid in selected_ids else "1px solid #e5e7eb",
                            }
                        )
                    else:
                        # data view: nodes are tables
                        table_name = name_by.get(sid, f"table_{sid}")
                        rc = self.table_row_counts.get(table_name, None)
                        rc_text = f"rows: {rc}" if rc is not None else "rows: —"
                        # In data view, use selection to highlight clicked tables
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
                                "last_run": rc_text,
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
            svg_parts.append(
                '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="10" refY="5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#9ca3af" /></marker></defs>'
            )
            for e in edge_objs:
                stroke = "#3b82f6" if e.get("selected") else "#9ca3af"
                width = "2.5" if e.get("selected") else "2"
                opacity = "1.0" if e.get("selected") else "0.6"
                svg_parts.append(
                    f'<path d="{e["path"]}" stroke="{stroke}" stroke-width="{width}" opacity="{opacity}" fill="none" marker-end="url(#arrow)" />'
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

            # Update preview anchor position (place trigger near the selected table in data view)
            try:
                if self.data_view and self.preview_table_name:
                    anchor_left = "0px"
                    anchor_top = "0px"
                    for n in nodes:
                        try:
                            if n.get("node_type") == "table" and str(n.get("name")) == str(self.preview_table_name):
                                ax = int(n.get("x", 0)) + int(n.get("w", 0)) + 12
                                ay = int(n.get("y", 0)) + int(int(n.get("h", 0)) / 2)
                                anchor_left = f"{ax}px"
                                anchor_top = f"{ay}px"
                                break
                        except Exception:
                            pass
                    self.preview_anchor_left = anchor_left
                    self.preview_anchor_top = anchor_top
                else:
                    # Default somewhere inside the card; not used if preview is closed
                    self.preview_anchor_left = "24px"
                    self.preview_anchor_top = "24px"
            except Exception:
                self.preview_anchor_left = "24px"
                self.preview_anchor_top = "24px"
        except Exception:
            # On any error, fall back to empty visuals
            self.graph_nodes = []
            self.graph_edges = []
            self.graph_width_px = 800
            self.graph_height_px = 400
            self.graph_svg = ""
            self.graph_width_css = "800px"
            self.graph_height_css = "400px"

    # --- Meta: last-run timestamps ---

    def _load_last_run_timestamps(self) -> None:
        """Load last run timestamps per step from the meta table.

        Tries to query a table named "meta" with columns (step_name, process_ts).
        If unavailable, silently falls back to empty mapping.
        """
        try:
            engine = DBCONN_DATAPIPE.con  # type: ignore[attr-defined]
        except Exception:
            self.steps_last_run = {}
            return

        try:
            import pandas as pd  # local import to keep typing hints clean

            df = pd.read_sql(
                "SELECT step_name, MAX(process_ts) AS last_ts FROM meta GROUP BY step_name",
                con=engine,
            )
            mapping: dict[str, str] = {}
            for _, row in df.iterrows():
                name = str(row.get("step_name", ""))
                ts = row.get("last_ts")
                try:
                    if ts is not None:
                        mapping[name] = str(pd.to_datetime(ts)).replace("T", " ")[:19]
                except Exception:
                    mapping[name] = str(ts)
            self.steps_last_run = mapping
        except Exception:
            self.steps_last_run = {}

    def _load_table_row_counts(self) -> None:
        """Load row counts for available tables for data view."""
        counts: dict[str, int] = {}
        try:
            engine = DBCONN_DATAPIPE.con  # type: ignore[attr-defined]
        except Exception:
            self.table_row_counts = {}
            return

        for name in list(self.available_tables or []):
            tname = str(name)
            try:
                import pandas as pd  # local import

                df = pd.read_sql(f'SELECT COUNT(*) AS cnt FROM "{tname}"', con=engine)
                try:
                    cnt_val = int(df.iloc[0]["cnt"]) if not df.empty else 0
                except Exception:
                    cnt_val = 0
                counts[tname] = cnt_val
            except Exception:
                counts[tname] = 0
        self.table_row_counts = counts
        return

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
            if self.selection_source == "manual" and self.selected_node_ids:
                selected = [i for i in self.selected_node_ids or [] if isinstance(i, int)]
                steps_to_run = [etl_app.steps[i] for i in sorted(selected) if 0 <= i < len(etl_app.steps)]
            else:
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

        # Toggle preview: if same table clicked, close; otherwise open new table
        if self.preview_table_name == table_name and self.preview_open:
            self.close_preview()
            return

        self.preview_table_name = table_name
        self.preview_open = True
        self.has_preview = False
        # Rebuild immediately to reflect selection highlight in data view
        try:
            self._rebuild_graph()
        except Exception:
            pass
        try:
            engine = DBCONN_DATAPIPE.con  # type: ignore[attr-defined]
        except Exception:
            self._append_log("DB engine not available")
            return

        # Resolve actual SQL table to preview (fallback to meta for non-SQL stores)
        actual_table = self._resolve_preview_table_name(table_name)
        try:
            self.preview_display_name = str(actual_table)
        except Exception:
            self.preview_display_name = str(table_name)

        try:
            df = pd.read_sql(f'SELECT * FROM "{actual_table}" LIMIT 50', con=engine)
        except Exception as e:
            self._append_log(f"Failed to load table {actual_table}: {e}")
            return

        self.preview_columns = [str(c) for c in df.columns]
        records_any: list[dict[Any, Any]] = df.astype(object).where(pd.notna(df), None).to_dict(orient="records")

        def _safe_render_value(v: Any) -> str:
            # if v is None:
            #     return "—"
            try:
                s = json.dumps(v).decode() if isinstance(v, (dict, list)) else str(v)
            except Exception:
                try:
                    s = repr(v)
                except Exception:
                    s = "<error rendering value>"
            return s

        coerced: list[dict[str, Any]] = []
        for r in records_any:
            try:
                coerced.append({str(k): _safe_render_value(v) for k, v in dict(r).items()})
            except Exception:
                coerced.append({})

        self.preview_rows = coerced
        self.has_preview = len(self.preview_rows) > 0
        # Update graph again now that preview is ready (keeps highlight)
        try:
            self._rebuild_graph()
        except Exception:
            pass

    def close_preview(self) -> None:
        self.preview_open = False
        self.preview_table_name = None
        self.preview_display_name = ""
        try:
            self._rebuild_graph()
        except Exception:
            pass

    def _resolve_preview_table_name(self, table_name: str) -> str:
        """Return SQL table to use for preview. If the catalog table is backed by
        a non-SQL store, return '<name>_meta'. If resolution fails, return the
        original name.
        """
        try:
            import vedana_etl.catalog as catalog  # local import to avoid cycles
            from datapipe.store.database import TableStoreDB  # type: ignore

            def _iter_tables():
                for attr in dir(catalog):
                    try:
                        obj = getattr(catalog, attr)
                        if hasattr(obj, "name") and hasattr(obj, "store"):
                            yield obj
                    except Exception:
                        continue

            for tbl in _iter_tables():
                try:
                    if str(getattr(tbl, "name", "")) == str(table_name):
                        store = getattr(tbl, "store", None)
                        if not isinstance(store, TableStoreDB):
                            return f"{table_name}_meta"
                        break
                except Exception:
                    continue
            return str(table_name)
        except Exception:
            return str(table_name)

    def set_preview_open(self, open: bool) -> None:
        """Handle popover open/close state changes."""
        if not open:
            self.close_preview()


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

    # mypy raises error: "EventNamespace" not callable [operator], though event definition is according to reflex docs
    @rx.event(background=True)  # type: ignore[operator]
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

@dataclass
class ThreadEventVis:
    event_id: str
    created_at: datetime
    created_at_str: str
    event_type: str
    role: str
    content: str
    tags: list[str]
    event_age: str
    event_data_list: list[tuple[str, str]]
    technical_vts_queries: list[str]
    technical_cypher_queries: list[str]
    technical_models: list[tuple[str, str]]
    vts_str: str
    cypher_str: str
    models_str: str
    has_technical_info: bool
    has_vts: bool
    has_cypher: bool
    has_models: bool

    @classmethod
    def create(cls, event_id: Any, created_at: datetime, event_type: str, event_data: dict) -> "ThreadEventVis":
        import json

        # Parse technical_info if present
        tech: dict = event_data.get("technical_info", {})
        has_technical_info = bool(tech)

        # Extract message-like fields
        # Role: Only comm.user_message is user; all others assistant.
        role = "user" if event_type == "comm.user_message" else "assistant"
        content: str = event_data.get("content", "")
        # tags may be stored in event_data["tags"] as list[str]
        tags_value = event_data.get("tags")  # todo check fmt
        tags: list[str] = list(tags_value or []) if isinstance(tags_value, (list, tuple)) else []

        vts_queries: list[str] = list(tech.get("vts_queries", []) or [])
        cypher_queries: list[str] = list(tech.get("cypher_queries", []) or [])

        model_stats = tech.get("model_stats", {}) or tech.get("model_used", {}) or {}
        models_list: list[tuple[str, str]] = []
        try:
            # If nested dict like {model: {...}} flatten to stringified value
            for mk, mv in model_stats.items() if isinstance(model_stats, dict) else []:
                try:
                    models_list.append((str(mk), json.dumps(mv)))
                except Exception:
                    models_list.append((str(mk), str(mv)))
        except Exception:
            pass

        vts_str = "\n".join(vts_queries)
        cypher_str = "\n".join([str(x) for x in cypher_queries])
        models_str = "\n".join([f"{k}: {v}" for k, v in models_list])

        return cls(
            event_id=str(event_id),
            created_at=created_at.replace(microsecond=0),
            created_at_str=datetime.strftime(created_at, "%Y-%m-%d %H:%M:%S"),
            event_type=event_type,
            role=role,
            content=content,
            tags=tags,
            event_age=datetime_to_age(created_at),
            event_data_list=[(str(k), str(v)) for k, v in event_data.items()],
            technical_vts_queries=vts_queries,
            technical_cypher_queries=cypher_queries,
            technical_models=models_list,
            vts_str=vts_str,
            cypher_str=cypher_str,
            models_str=models_str,
            has_technical_info=has_technical_info,
            has_vts=bool(vts_queries),
            has_cypher=bool(cypher_queries),
            has_models=bool(models_list),
        )


class ThreadViewState(rx.State):
    loading: bool = True
    events: list[ThreadEventVis] = []
    new_tag_text: str = ""
    note_text: str = ""
    note_severity: str = "Low"
    note_text_by_event: dict[str, str] = {}
    note_severity_by_event: dict[str, str] = {}
    selected_thread_id: str = ""
    expanded_event_id: str = ""

    async def _reload(self) -> None:
        vedana_app = await make_vedana_app()

        async with vedana_app.sessionmaker() as session:
            stmt = sa.select(ThreadEventDB).where(
                ThreadEventDB.thread_id == self.selected_thread_id,
                sa.not_(ThreadEventDB.event_type.like("jims.%")),
            )
            events = (await session.execute(stmt)).scalars().all()

        self.events = [
            ThreadEventVis.create(
                event_id=event.event_id,
                created_at=event.created_at,
                event_type=event.event_type,
                event_data=event.event_data,
            )
            for event in events
        ]
        self.loading = False

    @rx.event
    async def get_data(self):
        await self._reload()

    @rx.event
    async def select_thread(self, thread_id: str) -> None:
        self.selected_thread_id = thread_id
        await self._reload()

    # UI field updates
    @rx.event
    def set_new_tag_text(self, value: str) -> None:
        self.new_tag_text = value

    @rx.event
    def set_note_text(self, value: str) -> None:
        self.note_text = value

    @rx.event
    def set_note_severity(self, value: str) -> None:
        self.note_severity = value

    @rx.event
    def toggle_details(self, event_id: str) -> None:
        self.expanded_event_id = "" if self.expanded_event_id == event_id else event_id

    # Per-message note editing
    # todo check if necessary or just keep jims sending only
    @rx.event
    def set_note_text_for(self, value: str, event_id: str) -> None:
        self.note_text_by_event[event_id] = value

    @rx.event
    def set_note_severity_for(self, value: str, event_id: str) -> None:
        self.note_severity_by_event[event_id] = value

    # Persistence events: we encode admin actions as jims.backoffice.* events for now
    @rx.event
    async def add_tag(self, event_id: str) -> None:
        if not self.new_tag_text.strip():
            return
        vedana_app = await make_vedana_app()
        async with vedana_app.sessionmaker() as session:
            # Store an auxiliary event carrying tagging info
            from jims_core.util import uuid7

            tag_event = ThreadEventDB(
                thread_id=self.selected_thread_id,
                event_id=uuid7(),
                event_type="jims.backoffice.tag_added",
                event_data={"target_event_id": event_id, "tag": self.new_tag_text.strip()},
            )
            session.add(tag_event)
            await session.commit()
        self.new_tag_text = ""
        await self._reload()

    @rx.event
    async def remove_tag(self, event_id: str, tag: str) -> None:
        vedana_app = await make_vedana_app()
        async with vedana_app.sessionmaker() as session:
            from jims_core.util import uuid7

            tag_event = ThreadEventDB(
                thread_id=self.selected_thread_id,
                event_id=uuid7(),
                event_type="jims.backoffice.tag_removed",
                event_data={"target_event_id": event_id, "tag": tag},
            )
            session.add(tag_event)
            await session.commit()
        await self._reload()

    @rx.event
    async def submit_note_for(self, event_id: str) -> None:
        text = (self.note_text_by_event.get(event_id) or "").strip()
        if not text:
            return
        # Collect current tags from the target event if present
        try:
            target = next((e for e in self.events if e.event_id == event_id), None)
            tags_list = list(getattr(target, "tags", []) or []) if target is not None else []
        except Exception:
            tags_list = []
        vedana_app = await make_vedana_app()
        async with vedana_app.sessionmaker() as session:
            from jims_core.util import uuid7

            note_event = ThreadEventDB(
                thread_id=self.selected_thread_id,
                event_id=uuid7(),
                event_type="jims.backoffice.feedback",
                event_data={
                    "event_id": event_id,
                    "tags": tags_list,
                    "note": text,
                },
            )
            session.add(note_event)
            await session.commit()
        try:
            del self.note_text_by_event[event_id]
        except Exception:
            pass
        try:
            del self.note_severity_by_event[event_id]
        except Exception:
            pass
        await self._reload()
