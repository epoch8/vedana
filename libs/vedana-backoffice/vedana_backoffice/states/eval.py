import logging
from datetime import datetime
from typing import Any

import pandas as pd
import reflex as rx
import sqlalchemy as sa
from datapipe.compute import run_steps
from datapipe.run_config import RunConfig
from vedana_etl.app import app as etl_app
from vedana_etl.config import DBCONN_DATAPIPE

from vedana_backoffice.states.common import get_vedana_app
from vedana_backoffice.util import safe_render_value


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
        return 0 < rows == self.selected_count

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

    def _prune_selection(self) -> None:
        valid = {str(row.get("id")) for row in self.eval_gds_rows or [] if row.get("id")}
        self.selected_question_ids = [q for q in (self.selected_question_ids or []) if q in valid]

    def _load_eval_questions(self) -> None:
        # Run datapipe step to refresh eval_gds from Grist first
        # todo check / simplify getting the step
        step = next(
            (
                st
                for st in getattr(etl_app, "steps", [])
                if getattr(getattr(st, "func", None), "__name__", "") == "get_eval_gds_from_grist"
            ),
            None,
        )
        if step is not None:
            try:
                run_steps(etl_app.ds, [step])
            except Exception as exc:
                logging.warning("Failed to run get_eval_gds_from_grist: %s", exc)

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
            question = safe_render_value(rec.get("gds_question"))
            rows.append(
                {
                    "id": question,
                    "gds_question": question,
                    "gds_answer": safe_render_value(rec.get("gds_answer")),
                    "question_context": safe_render_value(rec.get("question_context")),
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
                    "judge_model": safe_render_value(rec.get("judge_model")),
                    "judge_prompt_id": safe_render_value(rec.get("judge_prompt_id")),
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
            self.pipeline_model = safe_render_value(pipeline_rows[-1])
        if embedding_rows:
            last = embedding_rows[-1]
            self.embeddings_model = safe_render_value(last.get("embeddings_model"))
            try:
                self.embeddings_dim = int(last.get("embeddings_dim") or 0)
            except Exception:
                self.embeddings_dim = 0
        if dm_row:
            self.dm_id = safe_render_value(dm_row.get("dm_id"))
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
        con = DBCONN_DATAPIPE.con  # type: ignore[attr-defined]
        stmt = sa.text(
            """
            SELECT test_date, gds_question, gds_answer, llm_answer, pipeline_model, test_status, eval_judge_comment
            FROM "eval_tests"
            ORDER BY test_date DESC NULLS LAST
            LIMIT 500
            """
        )
        try:
            with con.begin() as conn:
                df = pd.read_sql(stmt, conn)
        except Exception as exc:
            logging.warning("Failed to load eval_tests: %s", exc)
            self.tests_rows = []
            self.tests_cost_total = 0.0
            return
        if df.empty:
            self.tests_rows = []
            self.tests_cost_total = 0.0
            return
        df = df.astype(object).where(pd.notna(df), None)
        rows: list[dict[str, Any]] = []
        for rec in df.to_dict(orient="records"):
            status = safe_render_value(rec.get("test_status"))
            rows.append(
                {
                    "test_date": safe_render_value(rec.get("test_date")),
                    "gds_question": safe_render_value(rec.get("gds_question")),
                    "llm_answer": safe_render_value(rec.get("llm_answer")),
                    "gds_answer": safe_render_value(rec.get("gds_answer")),
                    "pipeline_model": safe_render_value(rec.get("pipeline_model")),
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
            self.error_message = "Unable to locate get_eval_judge_config step"
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
                from vedana_backoffice.states.chat import ChatState  # todo temp, will be removed after PR#235 is done

                yield ChatState.refresh_data_model()
                yield EvalState.load_eval_data()
            except Exception as e:
                self.error_message = f"Data model refresh failed: {e}"
            finally:
                self.status_message = ""
                yield
