import logging
import time
from typing import Any

import pandas as pd
import reflex as rx
import sqlalchemy as sa

from vedana_backoffice.states.common import get_vedana_app
from vedana_backoffice.util import safe_render_value
from vedana_etl.app import app as etl_app
from vedana_etl.config import DBCONN_DATAPIPE


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
