import logging
import time
from datetime import datetime
from typing import Any

import pandas as pd
import reflex as rx
import sqlalchemy as sa
from datapipe.compute import run_steps
from datapipe.run_config import RunConfig

from vedana_backoffice.states.common import get_vedana_app
from vedana_backoffice.util import safe_render_value
from vedana_core.data_provider import GristAPIDataProvider
from vedana_core.settings import settings as core_settings
from vedana_etl.app import app as etl_app
from vedana_etl.config import DBCONN_DATAPIPE
from vedana_etl.settings import settings as etl_settings


class EvalState(rx.State):
    """State holder for evaluation workflow."""

    loading: bool = False
    error_message: str = ""
    status_message: str = ""
    eval_gds_rows: list[dict[str, Any]] = []
    selected_question_ids: list[str] = []
    judge_configs: list[dict[str, Any]] = []
    selected_judge_model: str = ""
    judge_prompt_id: str = ""
    selected_judge_prompt: str = ""
    pipeline_model: str = ""
    embeddings_model: str = ""
    embeddings_dim: int = 0
    dm_id: str = ""
    dm_snapshot_updated: str = ""
    tests_rows: list[dict[str, Any]] = []
    tests_cost_total: float = 0.0
    is_running: bool = False
    run_progress: list[str] = []
    max_eval_rows: int = 500

    @rx.var
    def eval_gds_rows_with_selection(self) -> list[dict[str, Any]]:
        selected = set(self.selected_question_ids or [])
        rows: list[dict[str, Any]] = []
        for row in self.eval_gds_rows or []:
            enriched = dict(row)
            enriched["selected"] = row.get("id") in selected
            rows.append(enriched)
        return rows

    @rx.var
    def selected_count(self) -> int:
        return len(self.selected_question_ids or [])

    @rx.var
    def selection_label(self) -> str:
        total = len(self.eval_gds_rows or [])
        if total == 0:
            return "No questions available"
        return f"{self.selected_count} / {total} selected"

    @rx.var
    def all_selected(self) -> bool:
        rows = len(self.eval_gds_rows or [])
        return rows > 0 and self.selected_count == rows

    @rx.var
    def can_run(self) -> bool:
        return (self.selected_count > 0) and (not self.is_running)

    @rx.var
    def cost_label(self) -> str:
        if self.tests_cost_total > 0:
            return f"${self.tests_cost_total:.4f}"
        return "Cost data unavailable"

    @rx.var
    def tests_row_count(self) -> int:
        return len(self.tests_rows or [])

    @rx.var
    def tests_row_count_str(self) -> str:
        return f"{self.tests_row_count} rows" if self.tests_row_count else "No records"

    @rx.var
    def has_run_progress(self) -> bool:
        return len(self.run_progress or []) > 0

    @rx.var
    def embeddings_dim_label(self) -> str:
        return f"{self.embeddings_dim} dims" if self.embeddings_dim > 0 else ""

    @rx.var
    def judge_model_options(self) -> list[str]:
        return [str(cfg.get("judge_model", "")) for cfg in (self.judge_configs or []) if cfg.get("judge_model")]

    def toggle_question_selection(self, question: str, checked: bool) -> None:
        question = str(question or "").strip()
        if not question:
            return
        current = list(self.selected_question_ids or [])
        if checked:
            if question not in current:
                current.append(question)
        else:
            current = [q for q in current if q != question]
        self.selected_question_ids = current

    def toggle_select_all(self, checked: bool) -> None:
        if not checked:
            self.selected_question_ids = []
            return
        ids = [str(row.get("id", "")) for row in self.eval_gds_rows or [] if row.get("id")]
        self.selected_question_ids = ids

    def reset_selection(self) -> None:
        self.selected_question_ids = []
        self.status_message = ""

    def set_judge_model(self, value: str) -> None:
        value = str(value or "")
        for cfg in self.judge_configs or []:
            if cfg.get("judge_model") == value:
                self.selected_judge_model = value
                self.judge_prompt_id = cfg.get("judge_prompt_id", "")
                self.selected_judge_prompt = cfg.get("judge_prompt", "")
                break

    def _safe_value(self, value: Any) -> str:
        return str(safe_render_value(value) or "").strip()

    def _make_grist_provider(self) -> GristAPIDataProvider:
        return GristAPIDataProvider(
            doc_id=etl_settings.grist_test_set_doc_id,
            grist_server=core_settings.grist_server_url,
            api_key=core_settings.grist_api_key,
        )

    def _prune_selection(self) -> None:
        valid = {str(row.get("id")) for row in self.eval_gds_rows or [] if row.get("id")}
        self.selected_question_ids = [q for q in (self.selected_question_ids or []) if q in valid]

    def _load_eval_questions(self) -> None:
        con = DBCONN_DATAPIPE.con  # type: ignore[attr-defined]
        stmt = sa.text(
            f"""
            SELECT gds_question, gds_answer, question_context
            FROM "eval_gds"
            ORDER BY gds_question
            LIMIT {int(self.max_eval_rows)}
            """
        )
        with con.begin() as conn:
            df = pd.read_sql(stmt, conn)
        df = df.astype(object).where(pd.notna(df), None)
        rows: list[dict[str, Any]] = []
        for rec in df.to_dict(orient="records"):
            question = self._safe_value(rec.get("gds_question"))
            rows.append(
                {
                    "id": question,
                    "gds_question": question,
                    "gds_answer": self._safe_value(rec.get("gds_answer")),
                    "question_context": self._safe_value(rec.get("question_context")),
                }
            )
        self.eval_gds_rows = rows
        self._prune_selection()

    async def _load_judge_config(self) -> None:
        vedana_app = await get_vedana_app()
        query = sa.text(
            'SELECT judge_model, judge_prompt_id, judge_prompt FROM "eval_judge_config" ORDER BY judge_model'
        )
        try:
            async with vedana_app.sessionmaker() as session:
                result = await session.execute(query)
                rows = result.mappings().all()
        except Exception as exc:
            logging.warning("Failed to load eval_judge_config: %s", exc, exc_info=True)
            rows = []
        if not rows:
            self.judge_configs = []
            self.selected_judge_model = ""
            self.judge_prompt_id = ""
            self.selected_judge_prompt = ""
            return
        df = pd.DataFrame(rows)
        df = df.astype(object).where(pd.notna(df), None)
        configs: list[dict[str, Any]] = []
        for rec in df.to_dict(orient="records"):
            configs.append(
                {
                    "judge_model": self._safe_value(rec.get("judge_model")),
                    "judge_prompt_id": self._safe_value(rec.get("judge_prompt_id")),
                    "judge_prompt": rec.get("judge_prompt") or "",
                }
            )
        self.judge_configs = configs
        if configs:
            current = next(
                (
                    cfg
                    for cfg in configs
                    if cfg.get("judge_model") == self.selected_judge_model
                    and cfg.get("judge_prompt_id") == self.judge_prompt_id
                ),
                configs[0],
            )
            self.selected_judge_model = current.get("judge_model", "")
            self.judge_prompt_id = current.get("judge_prompt_id", "")
            self.selected_judge_prompt = current.get("judge_prompt", "")

    async def _load_pipeline_config(self) -> None:
        vedana_app = await get_vedana_app()
        pipeline_stmt = sa.text('SELECT pipeline_model FROM "llm_pipeline_config"')
        embeddings_stmt = sa.text('SELECT embeddings_model, embeddings_dim FROM "llm_embeddings_config"')
        dm_stmt = sa.text(
            """
            SELECT snap.dm_id, snap.dm_description, meta.process_ts
            FROM "dm_snapshot" AS snap
            JOIN "dm_snapshot_meta" AS meta USING (dm_id)
            ORDER BY meta.process_ts DESC
            LIMIT 1
            """
        )
        async with vedana_app.sessionmaker() as session:
            pipeline_rows = (await session.execute(pipeline_stmt)).scalars().all()
            embedding_rows = (await session.execute(embeddings_stmt)).mappings().all()
            dm_row = (await session.execute(dm_stmt)).mappings().first()

        if pipeline_rows:
            self.pipeline_model = self._safe_value(pipeline_rows[-1])
        if embedding_rows:
            last = embedding_rows[-1]
            self.embeddings_model = self._safe_value(last.get("embeddings_model"))
            try:
                self.embeddings_dim = int(last.get("embeddings_dim") or 0)
            except Exception:
                self.embeddings_dim = 0
        if dm_row:
            self.dm_id = self._safe_value(dm_row.get("dm_id"))
            ts = dm_row.get("process_ts")
            if ts:
                try:
                    dt = datetime.fromtimestamp(float(ts))
                    self.dm_snapshot_updated = dt.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    self.dm_snapshot_updated = datetime.now().strftime("%Y-%m-%d %H:%M")
            else:
                self.dm_snapshot_updated = datetime.now().strftime("%Y-%m-%d %H:%M")

    def _status_color(self, status: str) -> str:
        val = str(status or "").lower()
        if val == "pass":
            return "green"
        if val == "fail":
            return "red"
        return "gray"

    def _load_tests(self) -> None:
        provider = self._make_grist_provider()
        df = provider.get_table(etl_settings.tests_table_name)
        if df is None or df.empty:
            self.tests_rows = []
            self.tests_cost_total = 0.0
            return
        df = df.astype(object).where(pd.notna(df), None)
        if "test_date" in df.columns:
            df = df.sort_values(by="test_date", ascending=False)
        df = df.head(200)
        rows: list[dict[str, Any]] = []
        for rec in df.to_dict(orient="records"):
            status = self._safe_value(rec.get("test_status"))
            rows.append(
                {
                    "test_date": self._safe_value(rec.get("test_date")),
                    "gds_question": self._safe_value(rec.get("gds_question")),
                    "pipeline_model": self._safe_value(rec.get("pipeline_model")),
                    "test_status": status or "—",
                    "status_color": self._status_color(status),
                    "eval_judge_comment": rec.get("eval_judge_comment") or "",
                }
            )
        self.tests_rows = rows
        self.tests_cost_total = 0.0

    def _append_progress(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.run_progress = [*self.run_progress[-20:], f"[{stamp}] {message}"]

    def _resolve_eval_steps(self) -> list[Any]:
        target_order = ["run_tests", "judge_tests"]
        resolved: list[Any] = []
        for func_name in target_order:
            step = next(
                (
                    st
                    for st in getattr(etl_app, "steps", [])
                    if getattr(getattr(st, "func", None), "__name__", "") == func_name
                ),
                None,
            )
            if step is None:
                raise RuntimeError(f"Unable to locate compute step for {func_name}")
            resolved.append(step)
        return resolved

    def _run_eval_for_question(self, question: str, steps: list[Any]) -> None:
        rc = RunConfig(filters={"gds_question": question})
        for step in steps:
            meta_table = getattr(step, "meta_table", None)
            if meta_table is not None:
                try:
                    meta_table.mark_all_rows_unprocessed(run_config=rc)
                except Exception:
                    pass
            run_steps(etl_app.ds, [step], run_config=rc)
            self._append_progress(f"{question}: {getattr(step, '_name', getattr(step, 'name', ''))} done")

    def run_selected_tests(self):
        if self.is_running:
            return
        selection = [str(q) for q in (self.selected_question_ids or []) if str(q)]
        if not selection:
            self.error_message = "Select at least one question to run tests."
            return
        try:
            steps = self._resolve_eval_steps()
        except Exception as e:
            self.error_message = str(e)
            return
        self.is_running = True
        self.status_message = "Starting evaluation run…"
        self.run_progress = []
        yield
        try:
            for question in selection:
                self._append_progress(f"Running pipeline for '{question}'")
                self._run_eval_for_question(question, steps)
            self.status_message = f"Completed {len(selection)} question(s)"
            self.error_message = ""
            yield EvalState.load_eval_data()
        except Exception as e:
            self.error_message = f"Failed to run evaluation: {e}"
            self.status_message = ""
        finally:
            self.is_running = False
            yield

    def run_judge_refresh(self):
        if self.is_running:
            return
        step = next(
            (
                st
                for st in getattr(etl_app, "steps", [])
                if getattr(getattr(st, "func", None), "__name__", "") == "get_eval_judge_config"
            ),
            None,
        )
        if step is None:
            self.error_message = "Judge config step is not available."
            return
        self.status_message = "Refreshing judge config…"
        self.error_message = ""
        self.is_running = True
        yield
        try:
            run_steps(etl_app.ds, [step])
            self._append_progress("Judge config refreshed")
            yield EvalState.load_eval_data()
        except Exception as e:
            self.error_message = f"Failed to refresh judge config: {e}"
        finally:
            self.is_running = False
            yield

    @rx.event(background=True)  # type: ignore[operator]
    async def load_eval_data(self):
        async with self:
            self.loading = True
            self.error_message = ""
            self.status_message = ""
            yield
            try:
                self._load_eval_questions()
                await self._load_judge_config()
                await self._load_pipeline_config()
                self._load_tests()
            except Exception as e:
                self.error_message = f"Failed to load eval data: {e}"
            finally:
                self.loading = False
                yield

    @rx.event(background=True)  # type: ignore[operator]
    async def refresh_data_model(self):
        async with self:
            self.status_message = "Refreshing data model…"
            self.error_message = ""
            yield
            try:
                yield ChatState.refresh_data_model()
                yield EvalState.load_eval_data()
            except Exception as e:
                self.error_message = f"Data model refresh failed: {e}"
            finally:
                self.status_message = ""
                yield

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
