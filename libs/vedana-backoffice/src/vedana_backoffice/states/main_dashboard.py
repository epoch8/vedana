import asyncio
import logging
import time
from typing import Any

import pandas as pd
import reflex as rx
import sqlalchemy as sa
from vedana_core.settings import settings as core_settings
from vedana_etl.app import app as etl_app
from vedana_etl.config import DBCONN_DATAPIPE

from vedana_backoffice.states.common import get_vedana_app
from vedana_backoffice.util import safe_render_value


class DashboardState(rx.State):
    """Business-oriented dashboard state for ETL status and data changes."""

    # Loading / errors
    loading: bool = False
    error_message: str = ""

    # Time window selector (in days)
    time_window_days: int = 1
    time_window_options: list[str] = ["1", "3", "7"]
    ingest_source_tabs: list[str] = ["Grist"]
    selected_ingest_source_tab: str = "Grist"

    # Graph (Memgraph) totals
    graph_total_nodes: int = 0
    graph_total_edges: int = 0
    graph_nodes_by_label: list[dict[str, Any]] = []
    graph_edges_by_type: list[dict[str, Any]] = []

    # Embeddings (pgvector tables) totals and grouped stats
    emb_anchor_total: int = 0
    emb_edge_total: int = 0
    emb_anchor_by_node_type: list[dict[str, Any]] = []
    emb_edge_by_label: list[dict[str, Any]] = []
    emb_new_anchor: int = 0
    emb_updated_anchor: int = 0
    emb_deleted_anchor: int = 0
    emb_new_edge: int = 0
    emb_updated_edge: int = 0
    emb_deleted_edge: int = 0

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

    # Ingest-side entries grouped by source tab
    ingest_new_total: dict[str, int] = {}
    ingest_updated_total: dict[str, int] = {}
    ingest_deleted_total: dict[str, int] = {}
    ingest_breakdown: dict[str, list[dict[str, Any]]] = {}  # {source: [{table, added, updated, deleted}]}

    # Change preview (dialog) state
    changes_preview_open: bool = False
    changes_preview_table_name: str = ""
    changes_preview_columns: list[str] = []
    changes_preview_rows: list[dict[str, Any]] = []
    changes_has_preview: bool = False

    # Server-side pagination for changes preview
    changes_preview_page: int = 0  # 0-indexed current page
    changes_preview_page_size: int = 100  # rows per page
    changes_preview_total_rows: int = 0  # total count
    changes_preview_kind: str = ""  # "ingest", "nodes", or "edges"
    changes_preview_label: str = ""  # label for graph previews

    # Expandable row tracking for changes preview
    changes_preview_expanded_rows: list[str] = []

    def toggle_changes_preview_row_expand(self, row_id: str) -> None:
        """Toggle expansion state for a changes preview row."""
        row_id = str(row_id or "")
        if not row_id:
            return
        current = set(self.changes_preview_expanded_rows)
        if row_id in current:
            current.remove(row_id)
        else:
            current.add(row_id)
        self.changes_preview_expanded_rows = list(current)
        # Update expanded state in rows to trigger UI refresh
        updated_rows = []
        for row in self.changes_preview_rows:
            new_row = dict(row)
            new_row["expanded"] = row.get("row_id", "") in current
            updated_rows.append(new_row)
        self.changes_preview_rows = updated_rows

    @rx.var
    def time_window_days_str(self) -> str:
        return str(self.time_window_days)

    def set_time_window_days(self, value: str) -> None:
        try:
            d = int(value)
            if d <= 0:
                d = 1
        except Exception:
            d = 1
        self.time_window_days = d

    def set_selected_source_tab(self, value: str) -> None:
        self.selected_ingest_source_tab = value

    def _load_source_tabs(self) -> None:
        self.ingest_source_tabs = {str(v) for t in etl_app.steps for (k, v) in t.labels if k == "source"} | {"Grist"}
        if self.selected_ingest_source_tab not in self.ingest_source_tabs:
            self.selected_ingest_source_tab = self.ingest_source_tabs[0]

    def _append_log(self, msg: str) -> None:
        """Log a message (for consistency with EtlState pattern)."""
        logging.warning(msg)

    def load_dashboard(self):
        """Connecting with a background task. Used to trigger animations properly."""
        if self.loading:
            return
        self.loading = True
        self.error_message = ""
        yield
        yield DashboardState.load_dashboard_background()

    @rx.event(background=True)  # type: ignore[operator]
    async def load_dashboard_background(self):
        """Background task that loads all dashboard data."""
        try:
            async with self:
                self._load_source_tabs()
                await self._load_graph_counters()
            async with self:
                await asyncio.to_thread(self._load_datapipe_counters)
                await asyncio.to_thread(self._load_embeddings_counters)
                self._compute_consistency()
                self._load_change_metrics_window()
                self._load_ingest_metrics_window()
        except Exception as e:
            async with self:
                self.error_message = f"Failed to load dashboard: {e}"
        finally:
            async with self:
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

    def _load_embeddings_counters(self) -> None:
        """Query Datapipe DB for embedding table totals and grouped counts."""
        con = DBCONN_DATAPIPE.con  # type: ignore[attr-defined]
        since_ts = float(time.time() - self.time_window_days * 86400)

        try:
            with con.begin() as conn:
                self.emb_anchor_total = int(
                    conn.execute(sa.text('SELECT COUNT(*) FROM "rag_anchor_embeddings"')).scalar() or 0
                )
                self.emb_edge_total = int(
                    conn.execute(sa.text('SELECT COUNT(*) FROM "rag_edge_embeddings"')).scalar() or 0
                )
        except Exception:
            self.emb_anchor_total = 0
            self.emb_edge_total = 0

        try:
            with con.begin() as conn:
                rows = conn.execute(
                    sa.text(
                        "SELECT COALESCE(node_type, '') AS label, COUNT(*) AS cnt FROM \"rag_anchor_embeddings\" "
                        "GROUP BY label ORDER BY cnt DESC"
                    )
                ).fetchall()
                emb_anchor_count_by_label = {str(r[0]): int(r[1] or 0) for r in rows}
        except Exception:
            emb_anchor_count_by_label = {}

        try:
            with con.begin() as conn:
                rows = conn.execute(
                    sa.text(
                        "SELECT COALESCE(edge_label, '') AS label, COUNT(*) AS cnt FROM \"rag_edge_embeddings\" "
                        "GROUP BY label ORDER BY cnt DESC"
                    )
                ).fetchall()
                emb_edge_count_by_label = {str(r[0]): int(r[1] or 0) for r in rows}
        except Exception:
            emb_edge_count_by_label = {}

        def _load_change_counts(
            meta_table: str, label_col: str
        ) -> tuple[int, int, int, dict[str, int], dict[str, int], dict[str, int]]:
            try:
                with con.begin() as conn:
                    added_total = int(
                        conn.execute(
                            sa.text(
                                f'SELECT COUNT(*) FROM "{meta_table}" WHERE delete_ts IS NULL AND create_ts >= {since_ts}'
                            )
                        ).scalar()
                        or 0
                    )
                    updated_total = int(
                        conn.execute(
                            sa.text(
                                f'SELECT COUNT(*) FROM "{meta_table}" WHERE delete_ts IS NULL '
                                f"AND update_ts >= {since_ts} AND update_ts > create_ts AND create_ts < {since_ts}"
                            )
                        ).scalar()
                        or 0
                    )
                    deleted_total = int(
                        conn.execute(
                            sa.text(
                                f'SELECT COUNT(*) FROM "{meta_table}" WHERE delete_ts IS NOT NULL AND delete_ts >= {since_ts}'
                            )
                        ).scalar()
                        or 0
                    )

                    added_rows = conn.execute(
                        sa.text(
                            f"SELECT COALESCE({label_col}, '') AS label, COUNT(*) FROM \"{meta_table}\" "
                            f"WHERE delete_ts IS NULL AND create_ts >= {since_ts} GROUP BY label"
                        )
                    ).fetchall()
                    updated_rows = conn.execute(
                        sa.text(
                            f"SELECT COALESCE({label_col}, '') AS label, COUNT(*) FROM \"{meta_table}\" "
                            f"WHERE delete_ts IS NULL AND update_ts >= {since_ts} AND update_ts > create_ts "
                            f"AND create_ts < {since_ts} GROUP BY label"
                        )
                    ).fetchall()
                    deleted_rows = conn.execute(
                        sa.text(
                            f"SELECT COALESCE({label_col}, '') AS label, COUNT(*) FROM \"{meta_table}\" "
                            f"WHERE delete_ts IS NOT NULL AND delete_ts >= {since_ts} GROUP BY label"
                        )
                    ).fetchall()

                added_by = {str(label): int(cnt or 0) for label, cnt in added_rows}
                updated_by = {str(label): int(cnt or 0) for label, cnt in updated_rows}
                deleted_by = {str(label): int(cnt or 0) for label, cnt in deleted_rows}
                return added_total, updated_total, deleted_total, added_by, updated_by, deleted_by
            except Exception:
                return 0, 0, 0, {}, {}, {}

        (
            self.emb_new_anchor,
            self.emb_updated_anchor,
            self.emb_deleted_anchor,
            anchor_added_by,
            anchor_updated_by,
            anchor_deleted_by,
        ) = _load_change_counts("rag_anchor_embeddings_meta", "node_type")
        (
            self.emb_new_edge,
            self.emb_updated_edge,
            self.emb_deleted_edge,
            edge_added_by,
            edge_updated_by,
            edge_deleted_by,
        ) = _load_change_counts("rag_edge_embeddings_meta", "edge_label")

        anchor_labels = (
            set(emb_anchor_count_by_label.keys())
            | set(anchor_added_by.keys())
            | set(anchor_updated_by.keys())
            | set(anchor_deleted_by.keys())
        )
        edge_labels = (
            set(emb_edge_count_by_label.keys())
            | set(edge_added_by.keys())
            | set(edge_updated_by.keys())
            | set(edge_deleted_by.keys())
        )

        self.emb_anchor_by_node_type = [
            {
                "label": lab,
                "count": int(emb_anchor_count_by_label.get(lab, 0)),
                "added": int(anchor_added_by.get(lab, 0)),
                "updated": int(anchor_updated_by.get(lab, 0)),
                "deleted": int(anchor_deleted_by.get(lab, 0)),
            }
            for lab in sorted(anchor_labels)
        ]
        self.emb_edge_by_label = [
            {
                "label": lab,
                "count": int(emb_edge_count_by_label.get(lab, 0)),
                "added": int(edge_added_by.get(lab, 0)),
                "updated": int(edge_updated_by.get(lab, 0)),
                "deleted": int(edge_deleted_by.get(lab, 0)),
            }
            for lab in sorted(edge_labels)
        ]

    def _safe_preview_value(self, column_name: str, value: Any) -> str:
        is_embedding_preview_table = (
            self.changes_preview_kind == "ingest"
            and self.changes_preview_table_name in {"rag_anchor_embeddings", "rag_edge_embeddings"}
        ) or self.changes_preview_kind in {"emb_anchor", "emb_edge"}
        if is_embedding_preview_table and column_name == "embedding":
            return f"Vector({core_settings.embeddings_dim})"
        return safe_render_value(value)

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
        """Aggregate new/updated/deleted entries for ingest outputs grouped by source."""
        con = DBCONN_DATAPIPE.con  # type: ignore[attr-defined]

        since_ts = float(time.time() - self.time_window_days * 86400)
        sources = sorted({str(v) for t in etl_app.steps for (k, v) in t.labels if k == "source"} | {"grist"})

        breakdown_by_source: dict[str, list[dict[str, Any]]] = {}
        new_total_by_source: dict[str, int] = {}
        updated_total_by_source: dict[str, int] = {}
        deleted_total_by_source: dict[str, int] = {}

        for source in sources:
            tables = [
                tt.name
                for t in etl_app.steps
                if ("stage", "extract") in t.labels and ("source", source) in t.labels
                for tt in t.output_dts
            ]
            tables = sorted(set(tables))

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
                    logging.error(f"error collecting counters for {source}:{t}: {e}")
                    c = a = u = d = 0

                breakdown.append({"table": t, "total": c, "added": a, "updated": u, "deleted": d})
                total_a += a
                total_u += u
                total_d += d

            breakdown_by_source[source] = breakdown
            new_total_by_source[source] = int(total_a)
            updated_total_by_source[source] = int(total_u)
            deleted_total_by_source[source] = int(total_d)

        self.ingest_breakdown = breakdown_by_source
        self.ingest_new_total = new_total_by_source
        self.ingest_updated_total = updated_total_by_source
        self.ingest_deleted_total = deleted_total_by_source

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
        self.changes_preview_page = 0
        self.changes_preview_total_rows = 0
        self.changes_preview_kind = ""
        self.changes_preview_label = ""
        self.changes_preview_expanded_rows = []

    def open_changes_preview(self, table_name: str) -> None:
        """Load changed rows for the given ingest table using *_meta within the selected time window."""
        self.changes_preview_table_name = table_name
        self.changes_preview_open = True
        self.changes_has_preview = False
        self.changes_preview_page = 0
        self.changes_preview_kind = "ingest"
        self.changes_preview_label = ""
        self.changes_preview_expanded_rows = []  # Reset expanded rows

        self._load_ingest_changes_page()

    def _load_ingest_changes_page(self) -> None:
        """Load the current page of ingest changes from the database."""
        table_name = self.changes_preview_table_name
        if not table_name:
            return

        con = DBCONN_DATAPIPE.con  # type: ignore[attr-defined]

        since_ts = float(time.time() - self.time_window_days * 86400)
        meta = f"{table_name}_meta"
        offset = self.changes_preview_page * self.changes_preview_page_size

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

        # Get total count on first page load
        if self.changes_preview_page == 0:
            q_count = sa.text(
                f"""
                SELECT COUNT(*)
                FROM "{meta}" AS m
                WHERE
                    (m.delete_ts IS NOT NULL AND m.delete_ts >= {since_ts})
                    OR
                    (m.update_ts IS NOT NULL AND m.update_ts >= {since_ts}
                        AND (m.create_ts IS NULL OR m.create_ts < m.update_ts)
                        AND (m.delete_ts IS NULL OR m.delete_ts < m.update_ts))
                    OR
                    (m.create_ts IS NOT NULL AND m.create_ts >= {since_ts} AND m.delete_ts IS NULL)
                """
            )
            try:
                with con.begin() as conn:
                    self.changes_preview_total_rows = int(conn.execute(q_count).scalar() or 0)
            except Exception:
                self.changes_preview_total_rows = 0

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
            LIMIT {self.changes_preview_page_size} OFFSET {offset}
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

        expanded_set = set(self.changes_preview_expanded_rows)
        styled: list[dict[str, Any]] = []
        row_styling = {
            "added": {"backgroundColor": "rgba(34,197,94,0.08)"},
            "updated": {"backgroundColor": "rgba(245,158,11,0.08)"},
            "deleted": {"backgroundColor": "rgba(239,68,68,0.08)"},
        }
        for idx, r in enumerate(
            records_any
        ):  # Build display row with only data columns, coercing values to safe strings
            row_id = f"changes-{self.changes_preview_page}-{idx}"
            row_disp: dict[str, Any] = {k: self._safe_preview_value(k, r.get(k)) for k in self.changes_preview_columns}
            row_disp["row_style"] = row_styling.get(r.get("change_type", ""), {})
            row_disp["row_id"] = row_id
            row_disp["expanded"] = row_id in expanded_set
            styled.append(row_disp)

        self.changes_preview_rows = styled
        self.changes_has_preview = len(self.changes_preview_rows) > 0

    async def open_graph_per_label_changes_preview(self, kind: str, label: str) -> None:
        base_table = "nodes" if str(kind) == "nodes" else "edges"
        self.changes_preview_table_name = f"{base_table}:{label}"
        self.changes_preview_open = True
        self.changes_has_preview = False
        self.changes_preview_page = 0
        self.changes_preview_kind = kind
        self.changes_preview_label = label
        self.changes_preview_expanded_rows = []  # Reset expanded rows

        await self._load_labeled_changes_page()

    async def open_embedding_per_label_changes_preview(self, kind: str, label: str) -> None:
        base_table = "rag_anchor_embeddings" if str(kind) == "emb_anchor" else "rag_edge_embeddings"
        self.changes_preview_table_name = f"{base_table}:{label}"
        self.changes_preview_open = True
        self.changes_has_preview = False
        self.changes_preview_page = 0
        self.changes_preview_kind = kind
        self.changes_preview_label = label
        self.changes_preview_expanded_rows = []

        await self._load_labeled_changes_page()

    async def _load_labeled_changes_page(self) -> None:
        """Load filtered changes by label for graph/embedding tables."""
        kind = self.changes_preview_kind
        label = self.changes_preview_label
        if not kind or not label:
            return

        kind_map = {
            "nodes": ("nodes", "node_type", ["node_id"]),
            "edges": ("edges", "edge_label", ["from_node_id", "to_node_id"]),
            "emb_anchor": ("rag_anchor_embeddings", "node_type", ["node_id"]),
            "emb_edge": ("rag_edge_embeddings", "edge_label", ["from_node_id", "to_node_id"]),
        }
        mapped = kind_map.get(str(kind))
        if mapped is None:
            return
        base_table, label_col, key_cols = mapped

        con = DBCONN_DATAPIPE.con  # type: ignore[attr-defined]

        since_ts = float(time.time() - self.time_window_days * 86400)
        meta = f"{base_table}_meta"
        offset = self.changes_preview_page * self.changes_preview_page_size

        meta_exclude = {"hash", "create_ts", "update_ts", "process_ts", "delete_ts"}  # not displaying these

        inspector = sa.inspect(con)
        display_cols = [c["name"] for c in inspector.get_columns(base_table)]
        meta_cols = [c["name"] for c in inspector.get_columns(meta)]
        data_cols: list[str] = [c for c in meta_cols if c not in meta_exclude]

        label_escaped = label.replace("'", "''")

        # Get total count on first page load
        if self.changes_preview_page == 0:
            q_count = sa.text(
                f"""
                SELECT COUNT(*)
                FROM "{meta}" AS m
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
                """
            )
            try:
                with con.begin() as conn:
                    self.changes_preview_total_rows = int(conn.execute(q_count).scalar() or 0)
            except Exception:
                self.changes_preview_total_rows = 0

        select_exprs: list[str] = []
        for c in display_cols:
            if c in meta_cols:
                select_exprs.append(f'COALESCE(b."{c}", m."{c}") AS "{c}"')
            else:
                select_exprs.append(f'b."{c}" AS "{c}"')
        select_cols = ", ".join(select_exprs)
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
            LIMIT {self.changes_preview_page_size} OFFSET {offset}
            """
        )

        try:
            df = pd.read_sql(q_join, con=con)
        except Exception as e:
            self._append_log(f"Failed to load joined label changes for {base_table}:{label}: {e}")
            return

        # Ensure key columns are present in columns list
        for kc in key_cols:
            if kc not in display_cols:
                display_cols = [kc, *display_cols]

        self.changes_preview_columns = [str(c) for c in display_cols]
        records_any: list[dict] = df.astype(object).where(pd.notna(df), None).to_dict(orient="records")

        # Build styled rows
        expanded_set = set(self.changes_preview_expanded_rows)
        styled: list[dict[str, Any]] = []
        row_styling = {
            "added": {"backgroundColor": "rgba(34,197,94,0.08)"},
            "updated": {"backgroundColor": "rgba(245,158,11,0.08)"},
            "deleted": {"backgroundColor": "rgba(239,68,68,0.08)"},
        }

        for idx, r in enumerate(records_any):
            row_id = f"changes-{self.changes_preview_page}-{idx}"
            row_disp: dict[str, Any] = {k: self._safe_preview_value(k, r.get(k)) for k in self.changes_preview_columns}
            row_disp["row_style"] = row_styling.get(r.get("change_type", ""), {})
            row_disp["row_id"] = row_id
            row_disp["expanded"] = row_id in expanded_set
            styled.append(row_disp)

        self.changes_preview_rows = styled
        self.changes_has_preview = len(self.changes_preview_rows) > 0

    def changes_preview_next_page(self) -> None:
        """Load the next page of changes preview data."""
        max_page = (
            (self.changes_preview_total_rows - 1) // self.changes_preview_page_size
            if self.changes_preview_total_rows > 0
            else 0
        )
        if self.changes_preview_page < max_page:
            self.changes_preview_page += 1
            if self.changes_preview_kind == "ingest":
                self._load_ingest_changes_page()
            # Note: graph changes pagination would need async handling

    def changes_preview_prev_page(self) -> None:
        """Load the previous page of changes preview data."""
        if self.changes_preview_page > 0:
            self.changes_preview_page -= 1
            if self.changes_preview_kind == "ingest":
                self._load_ingest_changes_page()
            # Note: graph changes pagination would need async handling

    def changes_preview_first_page(self) -> None:
        """Jump to the first page."""
        if self.changes_preview_page != 0:
            self.changes_preview_page = 0
            if self.changes_preview_kind == "ingest":
                self._load_ingest_changes_page()

    def changes_preview_last_page(self) -> None:
        """Jump to the last page."""
        max_page = (
            (self.changes_preview_total_rows - 1) // self.changes_preview_page_size
            if self.changes_preview_total_rows > 0
            else 0
        )
        if self.changes_preview_page != max_page:
            self.changes_preview_page = max_page
            if self.changes_preview_kind == "ingest":
                self._load_ingest_changes_page()

    @rx.var
    def changes_preview_page_display(self) -> str:
        """Current page display (1-indexed for users)."""
        total_pages = (
            (self.changes_preview_total_rows - 1) // self.changes_preview_page_size + 1
            if self.changes_preview_total_rows > 0
            else 1
        )
        return f"Page {self.changes_preview_page + 1} of {total_pages}"

    @rx.var
    def changes_preview_rows_display(self) -> str:
        """Display range of rows being shown."""
        if self.changes_preview_total_rows == 0:
            return "No rows"
        start = self.changes_preview_page * self.changes_preview_page_size + 1
        end = min(start + self.changes_preview_page_size - 1, self.changes_preview_total_rows)
        return f"Rows {start}-{end} of {self.changes_preview_total_rows}"

    @rx.var
    def changes_preview_has_next(self) -> bool:
        """Whether there's a next page."""
        max_page = (
            (self.changes_preview_total_rows - 1) // self.changes_preview_page_size
            if self.changes_preview_total_rows > 0
            else 0
        )
        return self.changes_preview_page < max_page

    @rx.var
    def changes_preview_has_prev(self) -> bool:
        """Whether there's a previous page."""
        return self.changes_preview_page > 0

    @rx.var
    def ingest_new_total_selected(self) -> int:
        return int(self.ingest_new_total.get(self.selected_ingest_source_tab, 0))

    @rx.var
    def ingest_updated_total_selected(self) -> int:
        return int(self.ingest_updated_total.get(self.selected_ingest_source_tab, 0))

    @rx.var
    def ingest_deleted_total_selected(self) -> int:
        return int(self.ingest_deleted_total.get(self.selected_ingest_source_tab, 0))

    @rx.var
    def ingest_breakdown_selected(self) -> list[dict[str, Any]]:
        return list(self.ingest_breakdown.get(self.selected_ingest_source_tab, []))
