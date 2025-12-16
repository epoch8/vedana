import io
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from queue import Empty, Queue
import traceback
from typing import Any, Dict, Iterable, Tuple
from uuid import UUID, uuid4

import orjson as json
import pandas as pd
import reflex as rx
import requests
import sqlalchemy as sa
from datapipe.compute import run_steps
from jims_core.db import ThreadEventDB
from jims_core.thread.thread_controller import ThreadController
from jims_core.util import uuid7
from vedana_core.app import VedanaApp, make_vedana_app, RagPipeline
from vedana_core.data_model import DataModel
from vedana_core.settings import settings as core_settings
from vedana_etl.app import app as etl_app
from vedana_etl.app import pipeline
from vedana_etl.config import DBCONN_DATAPIPE

from vedana_backoffice.graph.build import build_canonical, derive_step_edges, derive_table_edges
from vedana_backoffice.util import safe_render_value

vedana_app: VedanaApp | None = None


async def get_vedana_app():
    global vedana_app
    if vedana_app is None:
        vedana_app = await make_vedana_app()
    return vedana_app


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


class TelegramBotState(rx.State):
    """State for Telegram bot information."""

    bot_username: str = ""
    bot_url: str = ""
    has_bot: bool = False

    def load_bot_info(self) -> None:
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        if not token:
            return

        try:
            bot_status = requests.get(f"https://api.telegram.org/bot{token}/getMe")
            if bot_status.status_code == 200:
                bot_status = bot_status.json()
                if bot_status["ok"]:
                    self.bot_username = bot_status["result"]["username"]
                    self.bot_url = f"https://t.me/{self.bot_username}"
                    self.has_bot = True
        except Exception as e:
            logging.warning(f"Failed to load Telegram bot info: {e}")


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
        """Populate all_steps and available_tables by introspecting vedana_etl pipeline and catalog."""

        steps_meta: list[dict[str, Any]] = []
        for idx, step in enumerate(pipeline.steps):
            inputs = [el.name for el in getattr(step, "inputs", [])]
            outputs = [el.name for el in getattr(step, "outputs", [])]
            labels = getattr(step, "labels", []) or []
            steps_meta.append(
                {
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
            )

        self.all_steps = steps_meta
        self._update_filtered_steps()

        tables: set[str] = set()
        flows: set[str] = set()
        stages: set[str] = set()

        # get tables from pipeline
        for step in pipeline.steps:
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

        self._load_table_stats()
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
                    if not self.data_view:  # transformation view
                        step_name = name_by.get(sid, f"step_{sid}")
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
                                # "last_run": ,
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
                            last_run_str = "None"

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
                        if n.get("node_type") == "table" and str(n.get("name")) == str(self.preview_table_name):
                            ax = int(n.get("x", 0)) + int(n.get("w", 0)) + 12
                            ay = int(n.get("y", 0)) + int(int(n.get("h", 0)) / 2)
                            anchor_left = f"{ax}px"
                            anchor_top = f"{ay}px"
                            break
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
            self.graph_nodes = []
            self.graph_edges = []
            self.graph_width_px = 800
            self.graph_height_px = 400
            self.graph_svg = ""
            self.graph_width_css = "800px"
            self.graph_height_css = "400px"

    def _load_table_stats(self):
        con = DBCONN_DATAPIPE.con  # type: ignore[attr-defined]
        self.table_meta = {}

        for tname in list(self.available_tables):
            try:
                meta_table_name = f"{tname}_meta"
                query = sa.text(f"""
                    WITH max_update_ts AS (
                        SELECT MAX(update_ts) AS max_ts FROM "{meta_table_name}"
                    )
                    SELECT 
                        MAX(process_ts) AS process_ts,
                        COUNT(*) FILTER (WHERE delete_ts IS NULL) AS row_count,
                        (SELECT max_ts FROM max_update_ts) AS last_update_ts,
                        COUNT(*) FILTER (WHERE update_ts IS NOT NULL AND update_ts = (SELECT max_ts FROM max_update_ts)) AS last_update_rows,
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
        """Load a small preview from the datapipe DB for a selected table."""

        # Toggle preview: if same table clicked, close; otherwise open new table
        if self.preview_table_name == table_name and self.preview_open:
            self.close_preview()
            return

        self.preview_table_name = table_name
        self.preview_open = True
        self.has_preview = False
        # Rebuild immediately to reflect selection highlight in data view
        self._rebuild_graph()

        engine = DBCONN_DATAPIPE.con  # type: ignore[attr-defined]

        try:
            df = pd.read_sql(f'SELECT * FROM "{table_name}" LIMIT 50', con=engine)
        except Exception:
            try:  # For tables which are stored not in TableStoreDB (not in database) - load _meta table for preview
                df = pd.read_sql(f'SELECT * FROM "{table_name}_meta" LIMIT 50', con=engine)
                self.preview_display_name = f"{table_name}_meta"
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


@dataclass
class ThreadEventVis:
    event_id: str
    created_at: datetime
    created_at_str: str
    event_type: str
    role: str
    content: str
    tags: list[str]
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
    # Aggregated annotations from jims.backoffice.* events
    visible_tags: list[str] = field(default_factory=list)
    feedback_comments: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(cls, event_id: Any, created_at: datetime, event_type: str, event_data: dict) -> "ThreadEventVis":
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
                    models_list.append((str(mk), json.dumps(mv).decode()))
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
            visible_tags=list(tags),
            feedback_comments=[],
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
    tag_dialog_open_for_event: str = ""
    selected_tags_for_event: dict[str, list[str]] = {}
    new_tag_text_for_event: dict[str, str] = {}
    available_tags: list[str] = []

    async def _reload(self) -> None:
        vedana_app = await get_vedana_app()

        async with vedana_app.sessionmaker() as session:
            stmt = (
                sa.select(ThreadEventDB)
                .where(
                    ThreadEventDB.thread_id == self.selected_thread_id,
                )
                .order_by(ThreadEventDB.created_at.asc())
            )
            all_events = (await session.execute(stmt)).scalars().all()

        # Split base convo events vs backoffice annotations
        base_events: list[Any] = []
        backoffice_events: list[Any] = []
        for ev in all_events:
            etype = str(getattr(ev, "event_type", ""))
            if etype.startswith("jims.backoffice."):
                backoffice_events.append(ev)
            elif etype.startswith("jims."):
                # ignore other jims.* noise
                continue
            else:
                base_events.append(ev)

        # Prepare aggregations
        # 1) Tags per original event
        tags_by_event: dict[str, set[str]] = {}
        for ev in base_events:
            eid = str(getattr(ev, "event_id", ""))
            try:
                base_tags = getattr(ev, "event_data", {}).get("tags") or []
                tags_by_event[eid] = set([str(t) for t in base_tags])
            except Exception:
                tags_by_event[eid] = set()

        # Apply tag add/remove in chronological order
        for ev in backoffice_events:
            etype = str(getattr(ev, "event_type", ""))
            edata = dict(getattr(ev, "event_data", {}) or {})
            if etype == "jims.backoffice.tag_added":
                tid = str(edata.get("target_event_id", ""))
                tag = str(edata.get("tag", "")).strip()
                if tid:
                    tags_by_event.setdefault(tid, set()).add(tag)
            elif etype == "jims.backoffice.tag_removed":
                tid = str(edata.get("target_event_id", ""))
                tag = str(edata.get("tag", "")).strip()
                if tid and tag:
                    try:
                        tags_by_event.setdefault(tid, set()).discard(tag)
                    except Exception:
                        pass

        # 2) Comments per original event + status mapping
        comments_by_event: dict[str, list[dict[str, Any]]] = {}
        # status by comment id (event_id of feedback)
        comment_status: dict[str, str] = {}
        for ev in backoffice_events:
            etype = str(getattr(ev, "event_type", ""))
            if etype == "jims.backoffice.feedback":
                edata = dict(getattr(ev, "event_data", {}) or {})
                target = str(edata.get("event_id", ""))
                if not target:
                    continue
                note_text = str(edata.get("note", ""))
                severity = str(edata.get("severity", "Low"))
                created_at = getattr(ev, "created_at", datetime.utcnow()).replace(microsecond=0)
                comments_by_event.setdefault(target, []).append(
                    {
                        "id": str(getattr(ev, "event_id", "")),
                        "note": note_text,
                        "severity": severity,
                        "created_at": datetime.strftime(created_at, "%Y-%m-%d %H:%M:%S"),
                        "status": "open",
                    }
                )
            elif etype in ("jims.backoffice.comment_resolved", "jims.backoffice.comment_closed"):
                ed = dict(getattr(ev, "event_data", {}) or {})
                cid = str(ed.get("comment_id", ""))
                if not cid:
                    continue
                comment_status[cid] = "resolved" if etype.endswith("comment_resolved") else "closed"

        # Convert base events into visual items and attach aggregations
        ev_items: list[ThreadEventVis] = []
        for bev in base_events:
            item = ThreadEventVis.create(
                event_id=bev.event_id,
                created_at=bev.created_at,
                event_type=bev.event_type,
                event_data=bev.event_data,
            )
            eid = item.event_id
            try:
                item.visible_tags = sorted(list(tags_by_event.get(eid, set())))
            except Exception:
                item.visible_tags = list(item.tags or [])
            try:
                cmts = []
                for c in comments_by_event.get(eid, []) or []:
                    c = dict(c)
                    cid = str(c.get("id", ""))
                    if cid in comment_status:
                        c["status"] = comment_status[cid]
                    cmts.append(c)
                item.feedback_comments = cmts
            except Exception:
                item.feedback_comments = []
            ev_items.append(item)

        # Present in chronological order as originally shown (created_at asc)
        self.events = ev_items

        # Collect all available tags from all threads
        all_tags: set[str] = set()
        # From current thread
        for tags_set in tags_by_event.values():
            all_tags.update(tags_set)

        # From all threads in the database
        try:
            async with vedana_app.sessionmaker() as session:
                # Query all tag_added events to get all tags ever used
                tag_stmt = sa.select(ThreadEventDB.event_data).where(
                    ThreadEventDB.event_type == "jims.backoffice.tag_added"
                )
                tag_results = (await session.execute(tag_stmt)).scalars().all()
                for edata in tag_results:
                    try:
                        ed = dict(edata or {})
                        tag = str(ed.get("tag", "")).strip()
                        if tag:
                            all_tags.add(tag)
                    except Exception:
                        pass
        except Exception:
            pass

        self.available_tags = sorted(list(all_tags))
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

    # Tag dialog management
    @rx.event
    def open_tag_dialog(self, event_id: str) -> None:
        """Open tag dialog for a specific event and initialize selected tags with current tags."""
        self.tag_dialog_open_for_event = event_id
        # Initialize selected tags with current visible tags for this event
        current_event = next((e for e in self.events if e.event_id == event_id), None)
        if current_event:
            self.selected_tags_for_event[event_id] = list(current_event.visible_tags or [])
        else:
            self.selected_tags_for_event[event_id] = []

    @rx.event
    def close_tag_dialog(self) -> None:
        """Close tag dialog and clear temporary state."""
        event_id = self.tag_dialog_open_for_event
        self.tag_dialog_open_for_event = ""
        # Optionally clear temporary state
        if event_id in self.new_tag_text_for_event:
            del self.new_tag_text_for_event[event_id]

    @rx.event
    def handle_tag_dialog_open_change(self, is_open: bool) -> None:
        """Handle dialog open/close state changes."""
        if not is_open:
            self.close_tag_dialog()  # type: ignore[operator]

    @rx.event
    def set_new_tag_text_for_event(self, value: str, event_id: str) -> None:
        """Set new tag text for a specific event."""
        self.new_tag_text_for_event[event_id] = value

    @rx.event
    def toggle_tag_selection_for_event(self, tag: str, event_id: str, checked: bool) -> None:
        """Toggle tag selection for a specific event."""
        selected = self.selected_tags_for_event.get(event_id, [])
        if checked:
            if tag not in selected:
                self.selected_tags_for_event[event_id] = [*selected, tag]
        else:
            self.selected_tags_for_event[event_id] = [t for t in selected if t != tag]

    @rx.event
    async def add_new_tag_to_available(self, event_id: str) -> None:
        """Add a new tag to available tags list."""
        new_tag = (self.new_tag_text_for_event.get(event_id) or "").strip()
        if new_tag and new_tag not in self.available_tags:
            self.available_tags = sorted([*self.available_tags, new_tag])
            # Also add to selected tags for this event
            selected = self.selected_tags_for_event.get(event_id, [])
            if new_tag not in selected:
                self.selected_tags_for_event[event_id] = [*selected, new_tag]
            # Clear the input
            self.new_tag_text_for_event[event_id] = ""

    @rx.event
    async def apply_tags_to_event(self, event_id: str):
        """Apply selected tags to an event by adding/removing tags as needed."""
        current_event = next((e for e in self.events if e.event_id == event_id), None)
        if not current_event:
            return

        current_tags = set(current_event.visible_tags or [])
        selected_tags = set(self.selected_tags_for_event.get(event_id, []))

        tags_to_add = selected_tags - current_tags
        tags_to_remove = current_tags - selected_tags

        vedana_app = await get_vedana_app()
        async with vedana_app.sessionmaker() as session:
            thread_uuid = (
                UUID(self.selected_thread_id) if isinstance(self.selected_thread_id, str) else self.selected_thread_id
            )

            for tag in tags_to_add:
                tag_event = ThreadEventDB(
                    thread_id=thread_uuid,
                    event_id=uuid7(),
                    event_type="jims.backoffice.tag_added",
                    event_data={"target_event_id": event_id, "tag": tag},
                )
                session.add(tag_event)
                await session.flush()

            for tag in tags_to_remove:
                tag_event = ThreadEventDB(
                    thread_id=thread_uuid,
                    event_id=uuid7(),
                    event_type="jims.backoffice.tag_removed",
                    event_data={"target_event_id": event_id, "tag": tag},
                )
                session.add(tag_event)
                await session.flush()

            await session.commit()

        # Close dialog and reload
        self.tag_dialog_open_for_event = ""
        await self._reload()
        try:
            # local import to avoid cycles
            from vedana_backoffice.pages.jims_thread_list_page import ThreadListState

            yield ThreadListState.get_data()  # type: ignore[operator]
        except Exception:
            pass

    @rx.event
    async def remove_tag(self, event_id: str, tag: str):
        vedana_app = await get_vedana_app()
        async with vedana_app.sessionmaker() as session:
            tag_event = ThreadEventDB(
                thread_id=self.selected_thread_id,
                event_id=uuid7(),
                event_type="jims.backoffice.tag_removed",
                event_data={"target_event_id": event_id, "tag": tag},
            )
            session.add(tag_event)
            await session.commit()
        await self._reload()
        try:
            from vedana_backoffice.pages.jims_thread_list_page import ThreadListState  # local import to avoid cycles

            yield ThreadListState.get_data()  # type: ignore[operator]
        except Exception:
            pass

    @rx.event
    async def submit_note_for(self, event_id: str):
        text = (self.note_text_by_event.get(event_id) or "").strip()
        if not text:
            return
        # Collect current tags from the target event if present
        try:
            target = next((e for e in self.events if e.event_id == event_id), None)
            tags_list = list(getattr(target, "tags", []) or []) if target is not None else []
        except Exception:
            tags_list = []
        vedana_app = await get_vedana_app()
        async with vedana_app.sessionmaker() as session:
            severity_val = self.note_severity_by_event.get(event_id, self.note_severity or "Low")  # todo check
            note_event = ThreadEventDB(
                thread_id=self.selected_thread_id,
                event_id=uuid7(),
                event_type="jims.backoffice.feedback",
                event_data={
                    "event_id": event_id,
                    "tags": tags_list,
                    "note": text,
                    "severity": severity_val,
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
        try:
            # todo check
            from vedana_backoffice.pages.jims_thread_list_page import ThreadListState  # local import

            yield ThreadListState.get_data()  # type: ignore[operator]
        except Exception:
            pass

    # --- Comment status actions ---
    @rx.event
    async def mark_comment_resolved(self, comment_id: str):
        vedana_app = await get_vedana_app()
        async with vedana_app.sessionmaker() as session:
            ev = ThreadEventDB(
                thread_id=self.selected_thread_id,
                event_id=uuid7(),
                event_type="jims.backoffice.comment_resolved",
                event_data={"comment_id": comment_id},
            )
            session.add(ev)
            await session.commit()
        await self._reload()
        try:
            from vedana_backoffice.pages.jims_thread_list_page import ThreadListState

            yield ThreadListState.get_data()  # type: ignore[operator]
        except Exception:
            pass

    @rx.event
    async def mark_comment_closed(self, comment_id: str):
        vedana_app = await get_vedana_app()
        async with vedana_app.sessionmaker() as session:
            ev = ThreadEventDB(
                thread_id=self.selected_thread_id,
                event_id=uuid7(),
                event_type="jims.backoffice.comment_closed",
                event_data={"comment_id": comment_id},
            )
            session.add(ev)
            await session.commit()
        await self._reload()
        try:
            from vedana_backoffice.pages.jims_thread_list_page import ThreadListState

            yield ThreadListState.get_data()  # type: ignore[operator]
        except Exception:
            pass


# --- Dashboard ---


class DashboardState(rx.State):
    """Business-oriented dashboard state for ETL status and data changes."""

    # Loading / errors
    loading: bool = False
    error_message: str = ""

    # Time window selector (in days)
    time_window_days: int = 1
    time_window_options: list[str] = ["1", "3", "7"]

    # Graph (Memgraph) totals
    graph_total_nodes: int = 0
    graph_total_edges: int = 0
    graph_nodes_by_label: list[dict[str, Any]] = []
    graph_edges_by_type: list[dict[str, Any]] = []

    # Datapipe (staging before upload to graph) totals
    dp_nodes_total: int = 0
    dp_edges_total: int = 0
    dp_nodes_by_type: list[dict[str, Any]] = []
    dp_edges_by_type: list[dict[str, Any]] = []

    # Consistency (graph vs datapipe) high-level diffs
    nodes_total_diff: int = 0
    edges_total_diff: int = 0

    # Changes in the selected time window (seconds resolution)
    new_nodes: int = 0
    updated_nodes: int = 0
    deleted_nodes: int = 0

    new_edges: int = 0
    updated_edges: int = 0
    deleted_edges: int = 0

    # Ingest-side new data entries (aggregated across snapshot-style generator tables)
    ingest_new_total: int = 0
    ingest_updated_total: int = 0
    ingest_deleted_total: int = 0
    ingest_breakdown: list[dict[str, Any]] = []  # [{table, added, updated, deleted}]

    # Change preview (dialog/popover) state
    changes_preview_open: bool = False
    changes_preview_table_name: str = ""
    changes_preview_columns: list[str] = []
    changes_preview_rows: list[dict[str, Any]] = []
    changes_has_preview: bool = False
    changes_preview_anchor_left: str = "48px"
    changes_preview_anchor_top: str = "48px"

    def set_time_window_days(self, value: str) -> None:
        try:
            d = int(value)
            if d <= 0:
                d = 1
        except Exception:
            d = 1
        self.time_window_days = d

    @rx.event(background=True)  # type: ignore[operator]
    async def load_dashboard(self):
        """Load all dashboard data in background."""
        async with self:
            self.loading = True
            self.error_message = ""
            try:
                await self._load_graph_counters()
                self._load_datapipe_counters()
                self._compute_consistency()
                self._load_change_metrics_window()
                self._load_ingest_metrics_window()
            except Exception as e:
                self.error_message = f"Failed to load dashboard: {e}"
            finally:
                self.loading = False
                yield

    async def _load_graph_counters(self) -> None:
        """Query Memgraph for totals and per-label counts."""
        va = await get_vedana_app()
        self.graph_total_nodes = int(await va.graph.number_of_nodes())
        self.graph_total_edges = int(await va.graph.number_of_edges())
        # Per-label nodes/edges from graph
        try:
            node_label_rows = await va.graph.execute_ro_cypher_query(
                "MATCH (n) UNWIND labels(n) as label RETURN label, count(*) as cnt ORDER BY cnt DESC"
            )
            graph_nodes_count_by_label = {str(r["label"]): int(r["cnt"]) for r in node_label_rows}
        except Exception:
            graph_nodes_count_by_label = {}

        try:
            edge_type_rows = await va.graph.execute_ro_cypher_query(
                "MATCH ()-[r]->() RETURN type(r) as label, count(*) as cnt ORDER BY cnt DESC"
            )
            graph_edges_count_by_label = {str(r["label"]): int(r["cnt"]) for r in edge_type_rows}
        except Exception:
            graph_edges_count_by_label = {}

        since_ts = float(time.time() - self.time_window_days * 86400)
        con = DBCONN_DATAPIPE.con  # type: ignore[attr-defined]

        # Nodes
        db_nodes_by_label: dict[str, int] = {}
        nodes_added_by_label: dict[str, int] = {}
        nodes_updated_by_label: dict[str, int] = {}
        nodes_deleted_by_label: dict[str, int] = {}
        # Edges
        db_edges_by_label: dict[str, int] = {}
        edges_added_by_label: dict[str, int] = {}
        edges_updated_by_label: dict[str, int] = {}
        edges_deleted_by_label: dict[str, int] = {}

        try:
            with con.begin() as conn:
                for row in conn.execute(
                    sa.text("SELECT COALESCE(node_type, '') AS label, COUNT(*) AS cnt FROM \"nodes\" GROUP BY label")
                ).fetchall():
                    db_nodes_by_label[str(row[0])] = int(row[1] or 0)
        except Exception:
            db_nodes_by_label = {}
        try:
            with con.begin() as conn:
                for row in conn.execute(
                    sa.text("SELECT COALESCE(edge_label, '') AS label, COUNT(*) AS cnt FROM \"edges\" GROUP BY label")
                ).fetchall():
                    db_edges_by_label[str(row[0])] = int(row[1] or 0)
        except Exception:
            db_edges_by_label = {}

        # Per-label change counts in window using *_meta
        try:
            with con.begin() as conn:
                # Added nodes
                q = sa.text(
                    f"SELECT COALESCE(node_type, '') AS label, COUNT(*) "
                    f'FROM "nodes_meta" WHERE delete_ts IS NULL AND create_ts >= {since_ts} GROUP BY label'
                )
                nodes_added_by_label = {str(i): int(c or 0) for i, c in conn.execute(q).fetchall()}
                # Updated nodes
                q = sa.text(
                    f"SELECT COALESCE(node_type, '') AS label, COUNT(*) "
                    f'FROM "nodes_meta" WHERE delete_ts IS NULL AND update_ts >= {since_ts} '
                    f"AND update_ts > create_ts AND create_ts < {since_ts} GROUP BY label"
                )
                nodes_updated_by_label = {str(i): int(c or 0) for i, c in conn.execute(q).fetchall()}
                # Deleted nodes
                q = sa.text(
                    f"SELECT COALESCE(node_type, '') AS label, COUNT(*) "
                    f'FROM "nodes_meta" WHERE delete_ts IS NOT NULL AND delete_ts >= {since_ts} GROUP BY label'
                )
                nodes_deleted_by_label = {str(i): int(c or 0) for i, c in conn.execute(q).fetchall()}
        except Exception:
            nodes_added_by_label = {}
            nodes_updated_by_label = {}
            nodes_deleted_by_label = {}

        try:
            with con.begin() as conn:
                # Added edges
                q = sa.text(
                    f"SELECT COALESCE(edge_label, '') AS label, COUNT(*) "
                    f'FROM "edges_meta" WHERE delete_ts IS NULL AND create_ts >= {since_ts} GROUP BY label'
                )
                edges_added_by_label = {str(i): int(c or 0) for i, c in conn.execute(q).fetchall()}
                # Updated edges
                q = sa.text(
                    f"SELECT COALESCE(edge_label, '') AS label, COUNT(*) "
                    f'FROM "edges_meta" WHERE delete_ts IS NULL AND update_ts >= {since_ts} '
                    f"AND update_ts > create_ts AND create_ts < {since_ts} GROUP BY label"
                )
                edges_updated_by_label = {str(i): int(c or 0) for i, c in conn.execute(q).fetchall()}
                # Deleted edges
                q = sa.text(
                    f"SELECT COALESCE(edge_label, '') AS label, COUNT(*) "
                    f'FROM "edges_meta" WHERE delete_ts IS NOT NULL AND delete_ts >= {since_ts} GROUP BY label'
                )
                edges_deleted_by_label = {str(i): int(c or 0) for i, c in conn.execute(q).fetchall()}
        except Exception:
            edges_added_by_label = {}
            edges_updated_by_label = {}
            edges_deleted_by_label = {}

        # Merge into per-label stats
        node_labels = set(graph_nodes_count_by_label.keys()) | set(db_nodes_by_label.keys())
        edge_labels = set(graph_edges_count_by_label.keys()) | set(db_edges_by_label.keys())

        node_stats: list[dict[str, Any]] = []
        for lab in sorted(node_labels):
            g = int(graph_nodes_count_by_label.get(lab, 0))
            d = int(db_nodes_by_label.get(lab, 0))

            node_stats.append(
                {
                    "label": lab,
                    "graph_count": g,
                    "etl_count": d,
                    "added": nodes_added_by_label.get(lab),
                    "updated": nodes_updated_by_label.get(lab),
                    "deleted": nodes_deleted_by_label.get(lab),
                }
            )
        edge_stats: list[dict[str, Any]] = []

        for lab in sorted(edge_labels):
            g = int(graph_edges_count_by_label.get(lab, 0))
            d = int(db_edges_by_label.get(lab, 0))

            edge_stats.append(
                {
                    "label": lab,
                    "graph_count": g,
                    "etl_count": d,
                    "added": edges_added_by_label.get(lab),
                    "updated": edges_updated_by_label.get(lab),
                    "deleted": edges_deleted_by_label.get(lab),
                }
            )

        self.graph_nodes_by_label = node_stats
        self.graph_edges_by_type = edge_stats

    def _load_datapipe_counters(self) -> None:
        """Query Datapipe DB for totals and per-type counts for staging tables."""
        con = DBCONN_DATAPIPE.con  # type: ignore[attr-defined]
        try:
            with con.begin() as conn:
                total_nodes = conn.execute(sa.text('SELECT COUNT(*) FROM "nodes"')).scalar()
                total_edges = conn.execute(sa.text('SELECT COUNT(*) FROM "edges"')).scalar()
            self.dp_nodes_total = int(total_nodes or 0)
            self.dp_edges_total = int(total_edges or 0)
        except Exception:
            self.dp_nodes_total = 0
            self.dp_edges_total = 0

        try:
            with con.begin() as conn:
                rows = conn.execute(
                    sa.text(
                        "SELECT COALESCE(node_type, '') AS label, COUNT(*) AS cnt "
                        'FROM "nodes" GROUP BY label ORDER BY cnt DESC'
                    )
                )
                self.dp_nodes_by_type = [{"label": str(r[0]), "count": int(r[1])} for r in rows.fetchall()]
        except Exception:
            self.dp_nodes_by_type = []

        try:
            with con.begin() as conn:
                rows = conn.execute(
                    sa.text(
                        "SELECT COALESCE(edge_label, '') AS label, COUNT(*) AS cnt "
                        'FROM "edges" GROUP BY label ORDER BY cnt DESC'
                    )
                )
                self.dp_edges_by_type = [{"label": str(r[0]), "count": int(r[1])} for r in rows.fetchall()]
        except Exception:
            self.dp_edges_by_type = []

    def _compute_consistency(self) -> None:
        self.nodes_total_diff = self.graph_total_nodes - self.dp_nodes_total
        self.edges_total_diff = self.graph_total_edges - self.dp_edges_total

    def _load_change_metrics_window(self) -> None:
        """Compute adds/edits/deletes for nodes and edges within selected window based on *_meta tables."""
        con = DBCONN_DATAPIPE.con  # type: ignore[attr-defined]

        since_ts = float(time.time() - self.time_window_days * 86400)

        def _counts_for(table_base: str) -> tuple[int, int, int]:
            meta = f"{table_base}_meta"
            q_added = sa.text(f'SELECT COUNT(*) FROM "{meta}" WHERE delete_ts IS NULL AND create_ts >= {since_ts}')
            q_updated = sa.text(
                f'SELECT COUNT(*) FROM "{meta}" '
                f"WHERE delete_ts IS NULL "
                f"AND update_ts >= {since_ts} "
                f"AND update_ts > create_ts "
                f"AND create_ts < {since_ts}"
            )
            q_deleted = sa.text(
                f'SELECT COUNT(*) FROM "{meta}" WHERE delete_ts IS NOT NULL AND delete_ts >= {since_ts}'
            )
            try:
                with con.begin() as conn:
                    added = conn.execute(q_added).scalar() or 0
                    updated = conn.execute(q_updated).scalar() or 0
                    deleted = conn.execute(q_deleted).scalar() or 0
                return int(added), int(updated), int(deleted)
            except Exception:
                return 0, 0, 0

        a, u, d = _counts_for("nodes")
        self.new_nodes, self.updated_nodes, self.deleted_nodes = a, u, d
        a, u, d = _counts_for("edges")
        self.new_edges, self.updated_edges, self.deleted_edges = a, u, d

    def _load_ingest_metrics_window(self) -> None:
        """Aggregate new/updated/deleted entries for ingest-side generator outputs."""
        con = DBCONN_DATAPIPE.con  # type: ignore[attr-defined]

        since_ts = float(time.time() - self.time_window_days * 86400)

        tables = [tt.name for t in etl_app.steps if ("stage", "extract") in t.labels for tt in t.output_dts]

        breakdown: list[dict[str, Any]] = []
        total_a = total_u = total_d = 0

        for t in tables:
            meta = f"{t}_meta"
            q_total = sa.text(f'SELECT COUNT(*) FROM "{meta}" WHERE delete_ts IS NULL')
            q_added = sa.text(f'SELECT COUNT(*) FROM "{meta}" WHERE delete_ts IS NULL AND create_ts >= {since_ts}')
            q_updated = sa.text(
                f'SELECT COUNT(*) FROM "{meta}" '
                f"WHERE delete_ts IS NULL "
                f"AND update_ts >= {since_ts} "
                f"AND update_ts > create_ts "
                f"AND create_ts < {since_ts}"
            )
            q_deleted = sa.text(
                f'SELECT COUNT(*) FROM "{meta}" WHERE delete_ts IS NOT NULL AND delete_ts >= {since_ts}'
            )
            try:
                with con.begin() as conn:
                    c = conn.execute(q_total).scalar_one()
                    a = conn.execute(q_added).scalar_one()
                    u = conn.execute(q_updated).scalar_one()
                    d = conn.execute(q_deleted).scalar_one()
            except Exception as e:
                logging.error(f"error collecting counters: {e}")
                c = a = u = d = 0

            breakdown.append({"table": t, "total": c, "added": a, "updated": u, "deleted": d})
            total_a += a
            total_u += u
            total_d += d

        self.ingest_breakdown = breakdown
        self.ingest_new_total = total_a
        self.ingest_updated_total = total_u
        self.ingest_deleted_total = total_d

    # --- Ingest change preview ---
    def set_changes_preview_open(self, open: bool) -> None:
        if not open:
            self.close_changes_preview()

    def close_changes_preview(self) -> None:
        self.changes_preview_open = False
        self.changes_preview_table_name = ""
        self.changes_preview_columns = []
        self.changes_preview_rows = []
        self.changes_has_preview = False

    def open_changes_preview(self, table_name: str) -> None:
        """Load changed rows for the given ingest table using *_meta within the selected time window."""
        self.changes_preview_table_name = table_name
        self.changes_preview_open = True
        self.changes_has_preview = False

        con = DBCONN_DATAPIPE.con  # type: ignore[attr-defined]

        since_ts = float(time.time() - self.time_window_days * 86400)
        meta = f"{table_name}_meta"

        LIMIT_ROWS = 100

        # Build a join between base table and its _meta on data columns (exclude meta columns).
        meta_exclude = {"hash", "create_ts", "update_ts", "process_ts", "delete_ts"}

        # Determine data columns using reflection
        inspector = sa.inspect(con)
        base_cols = [c.get("name", "") for c in inspector.get_columns(table_name)]
        meta_cols = [c.get("name", "") for c in inspector.get_columns(meta)]
        base_cols = [str(c) for c in base_cols if c]
        meta_cols = [str(c) for c in meta_cols if c]

        data_cols: list[str] = [c for c in (meta_cols or []) if c not in meta_exclude]

        # Columns to display: prefer all base table columns
        display_cols: list[str] = [c for c in (base_cols or []) if c]
        if not display_cols:
            display_cols = list(data_cols)

        # Build SELECT list coalescing base and meta to display base values when present
        select_exprs: list[str] = []
        for c in display_cols:
            if c in meta_cols:
                select_exprs.append(f'COALESCE(b."{c}", m."{c}") AS "{c}"')
            else:
                select_exprs.append(f'b."{c}" AS "{c}"')
        select_cols = ", ".join(select_exprs)
        # Join condition across all data columns
        on_cond = " AND ".join([f'b."{c}" = m."{c}"' for c in data_cols])

        q_join = sa.text(
            f"""
            SELECT
                {select_cols},
                CASE 
                    WHEN m.delete_ts IS NOT NULL AND m.delete_ts >= {since_ts} THEN 'deleted'
                    WHEN m.update_ts IS NOT NULL AND m.update_ts >= {since_ts} 
                         AND (m.create_ts IS NULL OR m.create_ts < m.update_ts)
                         AND (m.delete_ts IS NULL OR m.delete_ts < m.update_ts) THEN 'updated'
                    WHEN m.create_ts IS NOT NULL AND m.create_ts >= {since_ts} AND m.delete_ts IS NULL THEN 'added'
                    ELSE NULL
                END AS change_type,
                m.create_ts, m.update_ts, m.delete_ts
            FROM "{meta}" AS m
            LEFT JOIN "{table_name}" AS b
                ON {on_cond}
            WHERE 
                (m.delete_ts IS NOT NULL AND m.delete_ts >= {since_ts})
                OR
                (m.update_ts IS NOT NULL AND m.update_ts >= {since_ts}
                    AND (m.create_ts IS NULL OR m.create_ts < m.update_ts)
                    AND (m.delete_ts IS NULL OR m.delete_ts < m.update_ts))
                OR
                (m.create_ts IS NOT NULL AND m.create_ts >= {since_ts} AND m.delete_ts IS NULL)
            ORDER BY COALESCE(m.update_ts, m.create_ts, m.delete_ts) DESC
            LIMIT {LIMIT_ROWS}
            """
        )

        try:
            df = pd.read_sql(q_join, con=con)
        except Exception as e:
            self._append_log(f"Failed to load joined changes for {table_name}: {e}")
            return

        # Only display data columns; hide change_type and timestamps
        self.changes_preview_columns = [str(c) for c in display_cols]
        records_any: list[dict] = df.astype(object).where(pd.notna(df), None).to_dict(orient="records")

        styled: list[dict[str, Any]] = []
        row_styling = {
            "added": {"backgroundColor": "rgba(34,197,94,0.08)"},
            "updated": {"backgroundColor": "rgba(245,158,11,0.08)"},
            "deleted": {"backgroundColor": "rgba(239,68,68,0.08)"},
        }
        for r in records_any:  # Build display row with only data columns, coercing values to safe strings
            row_disp: dict[str, Any] = {k: safe_render_value(r.get(k)) for k in self.changes_preview_columns}  # type: ignore[no-redef]
            row_disp["row_style"] = row_styling.get(r.get("change_type"), {})  # type: ignore[arg-type]
            styled.append(row_disp)

        self.changes_preview_rows = styled
        self.changes_has_preview = len(self.changes_preview_rows) > 0

    async def open_graph_per_label_changes_preview(self, kind: str, label: str) -> None:
        base_table = "nodes" if str(kind) == "nodes" else "edges"
        label_col = "node_type" if base_table == "nodes" else "edge_label"
        self.changes_preview_table_name = f"{base_table}:{label}"
        self.changes_preview_open = True
        self.changes_has_preview = False

        con = DBCONN_DATAPIPE.con  # type: ignore[attr-defined]

        since_ts = float(time.time() - self.time_window_days * 86400)
        meta = f"{base_table}_meta"

        LIMIT_ROWS = 100

        meta_exclude = {"hash", "create_ts", "update_ts", "process_ts", "delete_ts"}  # not displaying these

        inspector = sa.inspect(con)
        display_cols = [c["name"] for c in inspector.get_columns(base_table)]
        meta_cols = [c["name"] for c in inspector.get_columns(meta)]
        data_cols: list[str] = [c for c in meta_cols if c not in meta_exclude]

        select_exprs: list[str] = []
        for c in display_cols:
            if c in meta_cols:
                select_exprs.append(f'COALESCE(b."{c}", m."{c}") AS "{c}"')
            else:
                select_exprs.append(f'b."{c}" AS "{c}"')
        select_cols = ", ".join(select_exprs)
        on_cond = " AND ".join([f'b."{c}" = m."{c}"' for c in data_cols])

        label_escaped = label.replace("'", "''")
        q_join = sa.text(
            f"""
            SELECT
                {select_cols},
                CASE 
                    WHEN m.delete_ts IS NOT NULL AND m.delete_ts >= {since_ts} THEN 'deleted'
                    WHEN m.update_ts IS NOT NULL AND m.update_ts >= {since_ts} 
                         AND (m.create_ts IS NULL OR m.create_ts < m.update_ts)
                         AND (m.delete_ts IS NULL OR m.delete_ts < m.update_ts) THEN 'updated'
                    WHEN m.create_ts IS NOT NULL AND m.create_ts >= {since_ts} AND m.delete_ts IS NULL THEN 'added'
                    ELSE NULL
                END AS change_type,
                m.create_ts, m.update_ts, m.delete_ts
            FROM "{meta}" AS m
            LEFT JOIN "{base_table}" AS b
                ON {on_cond}
            WHERE 
                m."{label_col}" = '{label_escaped}' AND (
                    (m.delete_ts IS NOT NULL AND m.delete_ts >= {since_ts})
                    OR
                    (m.update_ts IS NOT NULL AND m.update_ts >= {since_ts}
                        AND (m.create_ts IS NULL OR m.create_ts < m.update_ts)
                        AND (m.delete_ts IS NULL OR m.delete_ts < m.update_ts))
                    OR
                    (m.create_ts IS NOT NULL AND m.create_ts >= {since_ts} AND m.delete_ts IS NULL)
                )
            ORDER BY COALESCE(m.update_ts, m.create_ts, m.delete_ts) DESC
            LIMIT {LIMIT_ROWS}
            """
        )

        try:
            df = pd.read_sql(q_join, con=con)
        except Exception as e:
            self._append_log(f"Failed to load joined label changes for {base_table}:{label}: {e}")
            return

        # Ensure key columns are present in columns list
        key_cols: list[str] = ["node_id"] if base_table == "nodes" else ["from_node_id", "to_node_id"]
        for kc in key_cols:
            if kc not in display_cols:
                display_cols = [kc, *display_cols]

        self.changes_preview_columns = [str(c) for c in display_cols]
        records_any: list[dict] = df.astype(object).where(pd.notna(df), None).to_dict(orient="records")

        # Build ETL rows map keyed by PKs
        styled: list[dict[str, Any]] = []
        row_styling = {
            "added": {"backgroundColor": "rgba(34,197,94,0.08)"},
            "updated": {"backgroundColor": "rgba(245,158,11,0.08)"},
            "deleted": {"backgroundColor": "rgba(239,68,68,0.08)"},
            "g_not_etl": {"backgroundColor": "rgba(59,130,246,0.08)"},  # record present in graph, not present in ETL
            "etl_not_g": {"backgroundColor": "rgba(139,92,246,0.10)"},  # record present in ETL, not present in graph
        }

        etl_rows_by_key: dict[Any, dict[str, Any]] = {}
        for r in records_any:
            row_disp: dict[str, Any] = {k: safe_render_value(r.get(k)) for k in self.changes_preview_columns}  # type: ignore[no-redef]
            # style by change_type
            row_disp["row_style"] = row_styling.get(r.get("change_type"), {})  # type: ignore[arg-type]
            # index by key
            if base_table == "nodes":
                k = str(r.get("node_id") or "")
            else:
                k = (str(r.get("from_node_id") or ""), str(r.get("to_node_id") or ""))  # type: ignore[assignment]
            etl_rows_by_key[k] = row_disp
            styled.append(row_disp)

        # FULL OUTER JOIN-like merge of keys: ETL-changed keys U graph keys (sampled)
        try:
            va = await get_vedana_app()
            lab_esc = label.replace("`", "``")
            if base_table == "nodes":
                # Collect graph node ids for the label
                q = f"MATCH (n:`{lab_esc}`) RETURN n.id AS node_id LIMIT 300"
                recs = await va.graph.execute_ro_cypher_query(q)
                graph_keys: set[str] = {str(r.get("node_id", "")) for r in recs if r.get("node_id")}
                # Union with ETL keys
                all_keys: set[str] = set(graph_keys) | {str(k) for k in etl_rows_by_key.keys() if isinstance(k, str)}
                # Update presence flags; add graph-only rows
                for gid in all_keys:
                    if gid in etl_rows_by_key:
                        pass
                    else:
                        # graph-only: create placeholder row with key and presence flags
                        row_disp: dict[str, Any] = {k: "" for k in self.changes_preview_columns}  # type: ignore[no-redef]
                        row_disp["node_id"] = gid
                        row_disp["row_style"] = row_styling["g_not_etl"]
                        styled.append(row_disp)

                # Also include ETL-only not changed within window: sample more ETL ids for this label
                try:
                    with con.begin() as conn:
                        etl_more = [
                            str(r[0])
                            for r in conn.execute(
                                sa.text('SELECT node_id FROM "nodes" WHERE node_type = :lab LIMIT 300'),
                                {"lab": label},
                            ).fetchall()
                            if r and r[0] is not None
                        ]
                    # Exclude keys already present in styled (changed keys)
                    known_keys = {str(k) for k in etl_rows_by_key.keys() if isinstance(k, str)}
                    candidates = [eid for eid in etl_more if eid not in known_keys]
                    if candidates:
                        # Check presence in graph for candidates
                        recs2 = await va.graph.execute_ro_cypher_query(
                            "UNWIND $ids AS id MATCH (n {id: id}) RETURN id(n) AS id",
                            {"ids": candidates},
                        )
                        present = {str(r.get("id", "")) for r in recs2 if r.get("id")}
                        missing = [eid for eid in candidates if eid not in present]
                        for eid in missing:
                            row_disp: dict[str, Any] = {k: "" for k in self.changes_preview_columns}  # type: ignore[no-redef]
                            row_disp["node_id"] = eid
                            row_disp["row_style"] = row_styling["etl_not_g"]
                            styled.append(row_disp)
                except Exception:
                    pass

            else:
                # Edges: collect from->to pairs for this label
                q = f"MATCH (f)-[r:`{lab_esc}`]->(t) RETURN f.id AS from_id, t.id AS to_id LIMIT 300"
                recs = await va.graph.execute_ro_cypher_query(q)
                graph_pairs: set[tuple[str, str]] = set()
                for r in recs:
                    fr = str(r.get("from_id", "") or "")
                    to = str(r.get("to_id", "") or "")
                    if fr and to:
                        graph_pairs.add((fr, to))
                etl_pairs: set[tuple[str, str]] = {
                    k for k in etl_rows_by_key.keys() if isinstance(k, tuple) and len(k) == 2
                }
                all_pairs = graph_pairs | etl_pairs
                for key in all_pairs:
                    if key in etl_rows_by_key:
                        pass
                    else:
                        row_disp: dict[str, Any] = {k: "" for k in self.changes_preview_columns}  # type: ignore[no-redef]
                        fr, to = key
                        row_disp["from_node_id"] = fr
                        row_disp["to_node_id"] = to
                        row_disp["row_style"] = {"backgroundColor": "rgba(59,130,246,0.08)"}
                        styled.append(row_disp)

                # Include ETL-only not changed within window for edges
                try:
                    with con.begin() as conn:
                        etl_more_pairs = [
                            (str(fr), str(to))
                            for fr, to in conn.execute(
                                sa.text(
                                    'SELECT from_node_id, to_node_id FROM "edges" WHERE edge_label = :lab LIMIT 300'
                                ),
                                {"lab": label},
                            ).fetchall()
                            if fr and to
                        ]
                    known_pairs = {k for k in etl_rows_by_key.keys() if isinstance(k, tuple) and len(k) == 2}
                    add_candidates = [p for p in etl_more_pairs if p not in known_pairs]
                    if add_candidates:
                        recs2 = await va.graph.execute_ro_cypher_query(
                            "UNWIND $pairs AS p "
                            "MATCH (f {id: p.from_id})-[r]->(t {id: p.to_id}) "
                            "RETURN p.from_id AS from_id, p.to_id AS to_id",
                            {"pairs": [{"from_id": fr, "to_id": to} for fr, to in add_candidates]},
                        )
                        present_pairs = {(str(r.get("from_id", "")), str(r.get("to_id", ""))) for r in recs2}
                        missing_pairs = [p for p in add_candidates if p not in present_pairs]
                        for fr, to in missing_pairs:
                            row_disp: dict[str, Any] = {k: "" for k in self.changes_preview_columns}  # type: ignore[no-redef]
                            row_disp["from_node_id"] = fr
                            row_disp["to_node_id"] = to
                            row_disp["row_style"] = {"backgroundColor": "rgba(139,92,246,0.10)"}  # violet-ish
                            styled.append(row_disp)
                except Exception:
                    pass
        except Exception:
            pass

        self.changes_preview_rows = styled
        self.changes_has_preview = len(self.changes_preview_rows) > 0
