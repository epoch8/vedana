import asyncio
import hashlib
import json
import logging
from datetime import datetime
from typing import Any
from uuid import UUID

import reflex as rx
import sqlalchemy as sa
from datapipe.compute import run_steps
from jims_core.db import ThreadDB, ThreadEventDB
from jims_core.llms.llm_provider import LLMProvider
from jims_core.thread.thread_controller import ThreadController
from jims_core.util import uuid7
from pydantic import BaseModel, Field
from vedana_core.settings import settings as core_settings
from vedana_etl.app import app as etl_app

from vedana_backoffice.states.common import get_vedana_app
from vedana_backoffice.util import safe_render_value


class EvalState(rx.State):
    """State holder for evaluation workflow."""

    """
    TODO tasks
    - still no way to version the graph data properly. need to create some sort of metadata write (all node counts per-label, all edge counts per-label, plus SHOW VECTOR INDEX INFO and put all this into JSON and write as an eval.meta event in thread)
    - write data model JSON (+ its hash like in the page) in thread as a system event to version it (also eval.meta event)
    - same for judge prompt, also store it in eval.meta event
    - optimize runs if possible (parallel async execution? check if it is not done properly already)
    - make the 0-10 judge rating that is in judge's comment right now a separate column, update judge response schema.
    - add the ability to sort test results by fail / pass (and judge rating, combined in this sorting)
    """

    loading: bool = False
    error_message: str = ""
    status_message: str = ""
    eval_gds_rows: list[dict[str, Any]] = []
    gds_expanded_rows: list[str] = []
    selected_question_ids: list[str] = []
    selected_scenario: str = "all"  # Filter by scenario
    judge_model: str = ""
    judge_prompt_id: str = ""
    selected_judge_prompt: str = ""
    pipeline_model: str = core_settings.model
    embeddings_model: str = core_settings.embeddings_model
    embeddings_dim: int = core_settings.embeddings_dim
    dm_id: str = ""
    tests_rows: list[dict[str, Any]] = []
    tests_cost_total: float = 0.0
    run_passed: int = 0
    run_failed: int = 0
    selected_run_id: str = ""
    run_id_options: list[str] = []
    run_id_lookup: dict[str, str] = {}
    selected_tests_scenario: str = "All"
    tests_scenario_options: list[str] = ["All"]
    is_running: bool = False
    run_progress: list[str] = []
    max_eval_rows: int = 500
    current_question_index: int = -1  # Track which question is being processed
    total_questions_to_run: int = 0  # Total number of questions in current run
    judge_prompt_dialog_open: bool = False
    data_model_dialog_open: bool = False
    dm_description: str = ""
    # Server-side pagination for tests
    tests_page: int = 0  # 0-indexed current page
    tests_page_size: int = 100  # rows per page
    tests_total_rows: int = 0  # total count
    tests_sort_options: list[str] = ["Sort: Recent", "Sort: Rating"]
    selected_tests_sort: str = "Sort: Recent"
    max_parallel_tests: int = 4

    @rx.var
    def available_scenarios(self) -> list[str]:
        """Get unique scenarios from eval_gds_rows."""
        scenarios = set()
        for row in self.eval_gds_rows or []:
            scenario = row.get("question_scenario")
            if scenario:
                scenarios.add(str(scenario))
        return ["all"] + sorted(scenarios)

    @rx.var
    def eval_gds_rows_with_selection(self) -> list[dict[str, Any]]:
        selected = set(self.selected_question_ids or [])
        expanded = set(self.gds_expanded_rows or [])
        rows: list[dict[str, Any]] = []
        for row in self.eval_gds_rows or []:
            # Apply scenario filter
            if self.selected_scenario != "all":
                scenario = row.get("question_scenario")
                if str(scenario) != self.selected_scenario:
                    continue
            enriched = dict(row)
            enriched["selected"] = row.get("id") in selected
            enriched["expanded"] = row.get("id") in expanded
            # Add scenario color for badge display
            scenario_val = str(row.get("question_scenario", ""))
            enriched["scenario_color"] = self._scenario_color(scenario_val)
            rows.append(enriched)
        return rows

    @rx.var
    def selected_count(self) -> int:
        return len(self.selected_question_ids or [])

    @rx.var
    def selection_label(self) -> str:
        total = len(self.eval_gds_rows_with_selection)  # Use filtered count
        if total == 0:
            return "No questions available"
        if self.selected_count == 0:
            return f"Select tests to run ({self.selected_count}/{total})"
        return f"Run selected ({self.selected_count}/{total})"

    @rx.var
    def all_selected(self) -> bool:
        rows = len(self.eval_gds_rows_with_selection)  # Use filtered count
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
    def pass_fail_summary(self) -> str:
        return f"{self.run_passed} pass / {self.run_failed} fail"

    @rx.var
    def current_question_progress(self) -> str:
        """Display current question progress."""
        if self.total_questions_to_run == 0:
            return ""
        current = self.current_question_index + 1
        return f"Processing question {current} of {self.total_questions_to_run}"

    @rx.var
    def embeddings_dim_label(self) -> str:
        return f"{self.embeddings_dim} dims" if self.embeddings_dim > 0 else ""

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
        # Only select from filtered rows
        ids = [str(row.get("id", "")) for row in self.eval_gds_rows_with_selection if row.get("id")]
        self.selected_question_ids = ids

    def reset_selection(self) -> None:
        self.selected_question_ids = []
        self.status_message = ""

    def open_judge_prompt_dialog(self) -> None:
        self.judge_prompt_dialog_open = True

    def close_judge_prompt_dialog(self) -> None:
        self.judge_prompt_dialog_open = False

    def set_judge_prompt_dialog_open(self, open: bool) -> None:
        self.judge_prompt_dialog_open = open

    def open_data_model_dialog(self) -> None:
        self.data_model_dialog_open = True

    def close_data_model_dialog(self) -> None:
        self.data_model_dialog_open = False

    def set_data_model_dialog_open(self, open: bool) -> None:
        self.data_model_dialog_open = open

    def set_scenario(self, value: str) -> None:
        """Set the scenario filter and prune invalid selections."""
        self.selected_scenario = str(value or "all")
        if self.selected_scenario == "all":
            self._prune_selection()
            return

        # Drop selections that are not in the chosen scenario
        allowed_ids = {
            str(row.get("id"))
            for row in (self.eval_gds_rows or [])
            if str(row.get("question_scenario", "")) == self.selected_scenario
        }
        self.selected_question_ids = [q for q in (self.selected_question_ids or []) if q in allowed_ids]

    def _prune_selection(self) -> None:
        # Validate against all rows (not filtered) to keep selections valid across filter changes
        valid = {str(row.get("id")) for row in self.eval_gds_rows or [] if row.get("id")}
        self.selected_question_ids = [q for q in (self.selected_question_ids or []) if q in valid]

    def get_eval_gds_from_grist(self):
        # Run datapipe step to refresh eval_gds from Grist first
        step = next((s for s in etl_app.steps if s._name == "get_eval_gds_from_grist"), None)
        if step is not None:
            try:
                run_steps(etl_app.ds, [step])
            except Exception as exc:
                logging.exception(f"Failed to run get_eval_gds_from_grist: {exc}")

    async def _load_eval_questions(self) -> None:
        vedana_app = await get_vedana_app()

        stmt = sa.text(
            f"""
            SELECT gds_question, gds_answer, question_context, question_scenario
            FROM "eval_gds"
            ORDER BY gds_question
            LIMIT {int(self.max_eval_rows)}
            """
        )

        async with vedana_app.sessionmaker() as session:
            result = await session.execute(stmt)
            rs = result.mappings().all()

        rows: list[dict[str, Any]] = []
        for rec in rs:
            question = safe_render_value(rec.get("gds_question"))
            rows.append(
                {
                    "id": question,
                    "gds_question": question,
                    "gds_answer": safe_render_value(rec.get("gds_answer")),
                    "question_context": safe_render_value(rec.get("question_context")),
                    "question_scenario": safe_render_value(rec.get("question_scenario")),
                }
            )
        self.eval_gds_rows = rows
        self.gds_expanded_rows = []
        self._prune_selection()

    async def _load_judge_config(self) -> None:
        self.judge_model = core_settings.judge_model
        self.judge_prompt_id = ""
        self.selected_judge_prompt = ""

        vedana_app = await get_vedana_app()

        judge_prompt = vedana_app.data_model.prompt_templates().get("eval_judge_prompt")

        if judge_prompt:
            text_b = bytearray(judge_prompt, "utf-8")
            self.judge_prompt_id = hashlib.sha256(text_b).hexdigest()
            self.selected_judge_prompt = judge_prompt

    async def _load_pipeline_config(self) -> None:
        vedana_app = await get_vedana_app()
        dm = vedana_app.data_model
        self.dm_description = dm.to_json()
        dm_text_b = bytearray(self.dm_description, "utf-8")
        self.dm_id = hashlib.sha256(dm_text_b).hexdigest()

    def _status_color(self, status: str) -> str:
        if status == "pass":
            return "green"
        if status == "fail":
            return "red"
        return "gray"

    def _scenario_color(self, scenario: str) -> str:
        """Assign a consistent color to each unique scenario value."""
        if not scenario:
            return "gray"
        color_schemes = [
            "blue",
            "green",
            "purple",
            "pink",
            "indigo",
            "cyan",
            "amber",
            "lime",
            "emerald",
            "teal",
            "sky",
            "violet",
            "fuchsia",
            "rose",
            "orange",
            "slate",
        ]
        hash_val = hash(str(scenario))
        color_idx = abs(hash_val) % len(color_schemes)
        return color_schemes[color_idx]

    async def _load_tests(self) -> None:
        """Build test results table directly from JIMS threads (source=eval)."""
        vedana_app = await get_vedana_app()

        async with vedana_app.sessionmaker() as session:
            run_rows = (
                (
                    await session.execute(
                        sa.select(ThreadDB.contact_id)
                        .where(ThreadDB.thread_config.contains({"source": "eval"}))
                        .order_by(ThreadDB.created_at.desc())
                    )
                )
                .scalars()
                .all()
            )
            scenario_rows = (
                (
                    await session.execute(
                        sa.select(ThreadDB.thread_config).where(ThreadDB.thread_config.contains({"source": "eval"}))
                    )
                )
                .scalars()
                .all()
            )

            seen = set()
            ordered_run_ids = []
            for rid in run_rows:
                if rid not in seen:
                    ordered_run_ids.append(rid)
                    seen.add(rid)

            lookup: dict[str, str] = {}
            labels: list[str] = []
            for rid in ordered_run_ids:
                label = self._format_run_label(rid)
                lookup[label] = rid
                labels.append(label)

            self.run_id_lookup = lookup
            self.run_id_options = ["All", *labels]
            if not self.selected_run_id:
                self.selected_run_id = labels[0] if labels else "All"
            if self.selected_tests_sort not in self.tests_sort_options:
                self.selected_tests_sort = "Sort: Recent"

            # Scenario options
            scenarios = [
                str(cfg.get("question_scenario"))
                for cfg in scenario_rows
                if isinstance(cfg, dict) and cfg.get("question_scenario") not in (None, "", "None")
            ]
            scen_seen = set()
            scen_labels = []
            for sc in scenarios:
                if sc not in scen_seen:
                    scen_labels.append(sc)
                    scen_seen.add(sc)
            self.tests_scenario_options = ["All", *scen_labels]
            if self.selected_tests_scenario not in self.tests_scenario_options:
                self.selected_tests_scenario = "All"

            eval_result_subq = (
                sa.select(
                    ThreadEventDB.thread_id.label("thread_id"),
                    ThreadEventDB.event_data.label("eval_data"),
                    ThreadEventDB.created_at.label("eval_created_at"),
                    sa.func.row_number()
                    .over(partition_by=ThreadEventDB.thread_id, order_by=ThreadEventDB.created_at.desc())
                    .label("rn"),
                )
                .where(ThreadEventDB.event_type == "eval.result")
                .subquery()
            )

            # Base query for eval threads
            base_threads = sa.select(ThreadDB).where(ThreadDB.thread_config.contains({"source": "eval"}))
            if self.selected_run_id and self.selected_run_id != "All":
                selected_raw = self.run_id_lookup.get(self.selected_run_id)
                if selected_raw:
                    base_threads = base_threads.where(ThreadDB.contact_id == selected_raw)
            if self.selected_tests_scenario and self.selected_tests_scenario != "All":
                base_threads = base_threads.where(
                    ThreadDB.thread_config.contains({"question_scenario": self.selected_tests_scenario})
                )

            count_q = sa.select(sa.func.count()).select_from(base_threads.subquery())
            self.tests_total_rows = int((await session.execute(count_q)).scalar_one())

            offset = self.tests_page * self.tests_page_size
            threads_q = base_threads

            selected_sort = self.selected_tests_sort or "Sort: Recent"
            rating_expr = None
            if selected_sort in ("Sort: Rating"):
                threads_q = threads_q.join(
                    eval_result_subq,
                    sa.and_(eval_result_subq.c.thread_id == ThreadDB.thread_id, eval_result_subq.c.rn == 1),
                    isouter=True,
                )
                rating_expr = sa.cast(eval_result_subq.c.eval_data["eval_judge_rating"].astext, sa.Integer)

            if selected_sort == "Sort: Rating" and rating_expr is not None:
                threads_q = threads_q.order_by(sa.desc(rating_expr), ThreadDB.created_at.desc())
            else:
                threads_q = threads_q.order_by(ThreadDB.created_at.desc())

            threads_q = threads_q.limit(self.tests_page_size).offset(offset)
            page_threads = (await session.execute(threads_q)).scalars().all()

            if not page_threads:
                self.tests_rows = []
                self.tests_cost_total = 0.0
                self.run_passed = 0
                self.run_failed = 0
                return

            thread_ids = [t.thread_id for t in page_threads]
            ev_stmt = (
                sa.select(ThreadEventDB)
                .where(ThreadEventDB.thread_id.in_(thread_ids))
                .order_by(ThreadEventDB.created_at)
            )
            events_res = (await session.execute(ev_stmt)).scalars().all()

        events_by_thread: dict[UUID, list[ThreadEventDB]] = {}
        for ev in events_res:
            events_by_thread.setdefault(ev.thread_id, []).append(ev)

        rows: list[dict[str, Any]] = []
        passed = 0
        failed = 0
        cost_total = 0.0
        for thread in page_threads:
            evs = events_by_thread.get(thread.thread_id, [])
            answer = ""
            status = "—"
            judge_comment = ""
            rating_label = "—"
            run_label = self._format_run_label(thread.contact_id)
            test_date = run_label

            for ev in evs:
                if ev.event_type == "comm.assistant_message":
                    answer = str(ev.event_data.get("content", ""))
                elif ev.event_type == "rag.query_processed":
                    tech = ev.event_data.get("technical_info", {}) if isinstance(ev.event_data, dict) else {}
                    model_stats = tech.get("model_stats") if isinstance(tech, dict) else {}
                    if isinstance(model_stats, dict):
                        for stats in model_stats.values():
                            if isinstance(stats, dict):
                                cost_val = stats.get("requests_cost")
                                try:
                                    if cost_val is not None:
                                        cost_total += float(cost_val)
                                except (TypeError, ValueError):
                                    pass
                elif ev.event_type == "eval.result":
                    status = ev.event_data.get("test_status", status)
                    judge_comment = ev.event_data.get("eval_judge_comment", judge_comment)
                    rating_label = str(ev.event_data.get("eval_judge_rating", rating_label))
                    test_date = self._format_run_label(ev.event_data.get("test_date", test_date))

            if status == "pass":
                passed += 1
            elif status == "fail":
                failed += 1

            cfg = thread.thread_config or {}
            rows.append(
                {
                    "row_id": str(thread.thread_id),
                    "expanded": False,
                    "test_date": safe_render_value(test_date),
                    "gds_question": safe_render_value(cfg.get("gds_question")),
                    "llm_answer": safe_render_value(answer),
                    "gds_answer": safe_render_value(cfg.get("gds_answer")),
                    "pipeline_model": safe_render_value(cfg.get("pipeline_model")),
                    "test_status": status or "—",
                    "status_color": self._status_color(status),
                    "eval_judge_comment": safe_render_value(judge_comment),
                    "eval_judge_rating": rating_label,
                }
            )

        self.tests_rows = rows
        self.tests_cost_total = cost_total
        self.run_passed = passed
        self.run_failed = failed

    @rx.event(background=True)  # type: ignore[operator]
    async def tests_next_page(self):
        """Load the next page of tests."""
        async with self:
            max_page = (self.tests_total_rows - 1) // self.tests_page_size if self.tests_total_rows > 0 else 0
            if self.tests_page < max_page:
                self.tests_page += 1
                await self._load_tests()
                yield

    @rx.event(background=True)  # type: ignore[operator]
    async def tests_prev_page(self):
        """Load the previous page of tests."""
        async with self:
            if self.tests_page > 0:
                self.tests_page -= 1
                await self._load_tests()
                yield

    @rx.event(background=True)  # type: ignore[operator]
    async def tests_first_page(self):
        """Jump to the first page."""
        async with self:
            if self.tests_page != 0:
                self.tests_page = 0
                await self._load_tests()
                yield

    @rx.event(background=True)  # type: ignore[operator]
    async def tests_last_page(self):
        """Jump to the last page."""
        async with self:
            max_page = (self.tests_total_rows - 1) // self.tests_page_size if self.tests_total_rows > 0 else 0
            if self.tests_page != max_page:
                self.tests_page = max_page
                await self._load_tests()
                yield

    @rx.event(background=True)  # type: ignore[operator]
    async def select_run(self, value: str):
        """Update selected run id and reload tests."""
        async with self:
            self.selected_run_id = str(value or "")
            self.tests_page = 0
            await self._load_tests()
            yield

    @rx.event(background=True)  # type: ignore[operator]
    async def select_tests_scenario(self, value: str):
        """Update scenario filter for tests and reload."""
        async with self:
            self.selected_tests_scenario = str(value or "All")
            self.tests_page = 0
            await self._load_tests()
            yield

    @rx.event(background=True)  # type: ignore[operator]
    async def select_tests_sort(self, value: str):
        """Update sorting for tests and reload."""
        async with self:
            self.selected_tests_sort = str(value or "Sort: Recent")
            self.tests_page = 0
            await self._load_tests()
            yield

    def toggle_gds_row(self, row_id: str) -> None:
        """Toggle expansion for a golden dataset row."""
        row_id = str(row_id or "")
        if not row_id:
            return
        current = set(self.gds_expanded_rows or [])
        if row_id in current:
            current.remove(row_id)
        else:
            current.add(row_id)
        self.gds_expanded_rows = list(current)

    def toggle_row_expand(self, row_id: str) -> None:
        """Toggle expansion state for a result row."""
        row_id = str(row_id or "")
        if not row_id:
            return
        updated = []
        for row in self.tests_rows or []:
            if str(row.get("row_id")) == row_id:
                new_row = dict(row)
                new_row["expanded"] = not bool(row.get("expanded"))
                updated.append(new_row)
            else:
                updated.append(row)
        self.tests_rows = updated

    @rx.var
    def tests_page_display(self) -> str:
        """Current page display (1-indexed for users)."""
        total_pages = (self.tests_total_rows - 1) // self.tests_page_size + 1 if self.tests_total_rows > 0 else 1
        return f"Page {self.tests_page + 1} of {total_pages}"

    @rx.var
    def tests_rows_display(self) -> str:
        """Display range of rows being shown."""
        if self.tests_total_rows == 0:
            return "No rows"
        start = self.tests_page * self.tests_page_size + 1
        end = min(start + self.tests_page_size - 1, self.tests_total_rows)
        return f"Rows {start}-{end} of {self.tests_total_rows}"

    @rx.var
    def tests_has_next(self) -> bool:
        """Whether there's a next page."""
        max_page = (self.tests_total_rows - 1) // self.tests_page_size if self.tests_total_rows > 0 else 0
        return self.tests_page < max_page

    @rx.var
    def tests_has_prev(self) -> bool:
        """Whether there's a previous page."""
        return self.tests_page > 0

    def _append_progress(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.run_progress = [*self.run_progress[-20:], f"[{stamp}] {message}"]

    def _record_val(self, rec: Any, key: str) -> Any:
        """Best-effort extractor for neo4j / sqlalchemy records."""
        if isinstance(rec, dict):
            return rec.get(key)
        getter = getattr(rec, "get", None)
        if callable(getter):
            try:
                return getter(key)
            except Exception:
                pass
        try:
            return rec[key]  # type: ignore[index]
        except Exception:
            pass
        data_fn = getattr(rec, "data", None)
        if callable(data_fn):
            try:
                data = data_fn()
                if isinstance(data, dict):
                    return data.get(key)
            except Exception:
                return None
        return None

    async def _collect_graph_metadata(self, graph) -> dict[str, Any]:
        """Collect node/edge counts and vector index info from the graph."""
        meta: dict[str, Any] = {"nodes_by_label": {}, "edges_by_type": {}, "vector_indexes": []}

        try:
            node_res = await graph.execute_ro_cypher_query(
                "MATCH (n) UNWIND labels(n) AS lbl RETURN lbl, count(*) AS cnt"
            )
            for rec in node_res:
                lbl = str(self._record_val(rec, "lbl") or "")
                cnt_val = self._record_val(rec, "cnt")
                try:
                    cnt = int(cnt_val)
                except Exception:
                    cnt = None
                if lbl and cnt is not None:
                    meta["nodes_by_label"][lbl] = cnt
        except Exception as exc:
            meta["nodes_error"] = str(exc)

        try:
            edge_res = await graph.execute_ro_cypher_query(
                "MATCH ()-[r]->() RETURN type(r) AS rel_type, count(*) AS cnt"
            )
            for rec in edge_res:
                rel = str(self._record_val(rec, "rel_type") or "")
                cnt_val = self._record_val(rec, "cnt")
                try:
                    cnt = int(cnt_val)
                except Exception:
                    cnt = None
                if rel and cnt is not None:
                    meta["edges_by_type"][rel] = cnt
        except Exception as exc:
            meta["edges_error"] = str(exc)

        try:
            res = await graph.driver.execute_query("CALL vector_search.show_index_info() YIELD * RETURN *")
            for rec in res.records:
                row = {}
                try:
                    for key in rec.keys():
                        row[key] = rec.get(key)
                except Exception:
                    row = {}
                if row:
                    meta["vector_indexes"].append(row)
        except Exception as exc:
            meta["vector_indexes_error"] = str(exc)

        return meta

    def _build_data_model_meta(self, vedana_app) -> dict[str, Any]:
        """Serialize data model and attach hash for versioning."""
        dm_dict = vedana_app.data_model.to_dict()
        dm_json = json.dumps(dm_dict, ensure_ascii=False, sort_keys=True)
        dm_hash = hashlib.sha256(dm_json.encode("utf-8")).hexdigest()
        return {
            "data_model_json": dm_json,
            "data_model_hash": dm_hash,
            "dm_id": self.dm_id,
            "dm_description": self.dm_description,
        }

    async def _build_eval_meta_payload(self, vedana_app, test_run_name: str) -> dict[str, Any]:
        """Build a single eval.meta payload shared across threads for a run."""
        graph_meta = await self._collect_graph_metadata(vedana_app.graph)
        data_model_meta = self._build_data_model_meta(vedana_app)
        return {
            "meta_version": 1,
            "test_run": test_run_name,
            "pipeline_model": self.pipeline_model,
            "embeddings_model": self.embeddings_model,
            "embeddings_dim": self.embeddings_dim,
            "judge": {
                "judge_model": self.judge_model,
                "judge_prompt_id": self.judge_prompt_id,
                "judge_prompt": self.selected_judge_prompt,
            },
            "graph": graph_meta,
            "data_model": data_model_meta,
        }

    def _format_tool_calls(self, technical_info: dict[str, Any]) -> str:
        """Flatten VTS/Cypher info into a text blob for judge/storage."""
        if not isinstance(technical_info, dict):
            return ""
        vts = technical_info.get("vts_queries") or []
        cypher = technical_info.get("cypher_queries") or []
        vts_s = "\n".join([str(v) for v in vts]) if isinstance(vts, list) else ""
        cypher_s = "\n".join([str(c) for c in cypher]) if isinstance(cypher, list) else ""
        return "\n---\n".join(part for part in [vts_s, cypher_s] if part).strip()

    def _format_run_label(self, contact_id: str | None) -> str:
        """
        Convert run id like 'eval:20251208-214017' -> '2025-12-08 21:40:17'.
        Falls back to the raw value if parsing fails.
        """
        raw = str(contact_id or "").strip()
        if raw.startswith("eval:") and len(raw) >= 18:
            ts = raw.removeprefix("eval:")
            try:
                dt = datetime.strptime(ts, "%Y%m%d-%H%M%S")
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                return raw
        return raw

    def _build_thread_config(self, question_row: dict[str, Any], test_run_name: str) -> dict[str, Any]:
        """Pack metadata into thread_config so runs are traceable in JIMS."""
        return {
            "interface": "reflex-eval",
            "source": "eval",
            "test_run": test_run_name,
            "gds_question": question_row.get("gds_question"),
            "gds_answer": question_row.get("gds_answer"),
            "question_context": question_row.get("question_context"),
            "question_scenario": question_row.get("question_scenario"),
            "judge_model": self.judge_model,
            "judge_prompt_id": self.judge_prompt_id,
            "pipeline_model": self.pipeline_model,
            "embeddings_model": self.embeddings_model,
            "embeddings_dim": self.embeddings_dim,
            "dm_id": self.dm_id,
        }

    async def _run_question_thread(
        self,
        vedana_app,
        question_row: dict[str, Any],
        test_run_name: str,
        eval_meta_base: dict[str, Any],
    ) -> tuple[str, str, dict[str, Any]]:
        """Create a JIMS thread, post the question, run pipeline, return answer + tech info."""
        thread_id = uuid7()
        ctl = await ThreadController.new_thread(
            vedana_app.sessionmaker,
            contact_id=f"eval:{test_run_name}",
            thread_id=thread_id,
            thread_config=self._build_thread_config(question_row, test_run_name),
        )

        try:
            meta_payload = {
                **eval_meta_base,
                "question": {
                    "gds_question": question_row.get("gds_question"),
                    "gds_answer": question_row.get("gds_answer"),
                    "question_context": question_row.get("question_context"),
                    "question_scenario": question_row.get("question_scenario"),
                },
            }
            await ctl.store_event_dict(uuid7(), "eval.meta", meta_payload)
        except Exception as exc:
            logging.warning(f"Failed to store eval.meta for thread {thread_id}: {exc}")

        question_text = str(question_row.get("gds_question", "") or "")
        q_ctx = str(question_row.get("question_context", "") or "").strip()
        user_query = f"{question_text} {q_ctx}".strip()

        await ctl.store_user_message(uuid7(), user_query)
        events = await ctl.run_pipeline_with_context(vedana_app.pipeline)

        answer: str = ""
        technical_info: dict[str, Any] = {}
        for ev in events:
            if ev.event_type == "comm.assistant_message":
                answer = str(ev.event_data.get("content", ""))
            elif ev.event_type == "rag.query_processed":
                technical_info = dict(ev.event_data.get("technical_info", {}))

        return str(thread_id), answer, technical_info

    async def _judge_answer(self, question_row: dict[str, Any], answer: str, tool_calls: str) -> tuple[str, str, int]:
        """Judge model answer with current judge prompt/model and rating."""
        judge_prompt = self.selected_judge_prompt
        if not judge_prompt:
            return "fail", "Judge prompt not loaded", 0

        provider = LLMProvider()  # todo use single LLMProvider per-thread
        judge_model = self.judge_model
        if judge_model:
            try:
                provider.set_model(judge_model)
            except Exception:
                logging.warning(f"Failed to set judge model {judge_model}")

        class JudgeResult(BaseModel):
            test_status: str = Field(description="pass / fail")
            comment: str = Field(description="justification and hints")
            errors: str | list[str] | None = Field(default=None, description="Text description of errors found")
            rating: int = Field(description="Numeric rating between 0 (worst) and 10 (best)")

        user_msg = (
            f"Golden answer:\n{question_row.get('gds_answer', '')}\n\n"
            f"Expected context (if any):\n{question_row.get('question_context', '')}\n\n"
            f"Model answer:\n{answer}\n\n"
            f"Technical info (for reference):\n{tool_calls}\n\n"
            "Return test_status (pass/fail), a helpful comment, optional errors list/text, "
            "and rating as an integer number between 0 and 10 (10 = best possible answer)."
        )

        try:
            res = await provider.chat_completion_structured(
                [
                    {"role": "system", "content": judge_prompt},
                    {"role": "user", "content": user_msg},
                ],
                JudgeResult,
            )  # type: ignore[arg-type]
        except Exception as e:
            logging.exception(f"Judge failed for question '{question_row.get('gds_question')}': {e}")
            return "fail", f"Judge failed: {e}", 0

        if res is None:
            return "fail", "", 0

        try:
            rating = int(res.rating)
        except Exception:
            rating = 0

        return res.test_status or "fail", res.comment or "", rating

    async def _store_eval_result_event(self, thread_id: str, result_row: dict[str, Any]) -> None:
        """Persist eval result as a thread event for thread-based history. todo why is this needed?"""
        try:
            tid = UUID(str(thread_id))
        except Exception:
            return

        vedana_app = await get_vedana_app()
        ctl = await ThreadController.from_thread_id(vedana_app.sessionmaker, tid)
        if ctl is None:
            return

        data = dict(result_row)
        data.pop("thread_id", None)
        await ctl.store_event_dict(uuid7(), "eval.result", data)

    @rx.event(background=True)  # type: ignore[operator]
    async def run_selected_tests(self):
        """Run tests via JIMS sessions, one question at a time."""
        async with self:
            if self.is_running:
                return
            selection = [str(q) for q in (self.selected_question_ids or []) if str(q)]
            if not selection:
                self.error_message = "Select at least one question to run tests."
                return
            if not self.selected_judge_prompt:
                self.error_message = "Judge prompt not loaded. Refresh judge config first."
                return

            question_map = {str(row.get("id")): row for row in (self.eval_gds_rows or [])}
            test_run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")

            # Initialize run state
            self.is_running = True
            self.current_question_index = -1
            self.total_questions_to_run = len(selection)
            self.status_message = f"Starting evaluation run for {len(selection)} question(s)…"
            self.run_progress = []
            self.error_message = ""
            yield

            vedana_app = await get_vedana_app()
            eval_meta_base = await self._build_eval_meta_payload(vedana_app, test_run_ts)
            max_parallel = max(1, int(self.max_parallel_tests or 1))
            sem = asyncio.Semaphore(max_parallel)

            async def _run_one(idx: int, question: str) -> dict[str, Any]:
                async with sem:
                    row = question_map.get(question)
                    if row is None:
                        return {"question": question, "status": None, "error": "not found"}
                    try:
                        thread_id, answer, tech = await self._run_question_thread(
                            vedana_app, row, test_run_ts, eval_meta_base
                        )
                        tool_calls = self._format_tool_calls(tech)
                        status, comment, rating = await self._judge_answer(row, answer, tool_calls)

                        result_row = {
                            "judge_model": self.judge_model,
                            "judge_prompt_id": self.judge_prompt_id,
                            "dm_id": self.dm_id,
                            "pipeline_model": self.pipeline_model,
                            "embeddings_model": self.embeddings_model,
                            "embeddings_dim": self.embeddings_dim,
                            "gds_question": row.get("gds_question"),
                            "question_context": row.get("question_context"),
                            "gds_answer": row.get("gds_answer"),
                            "llm_answer": answer,
                            "tool_calls": tool_calls,
                            "test_status": status,
                            "eval_judge_comment": comment,
                            "eval_judge_rating": rating,
                            "test_date": test_run_ts,
                            "thread_id": thread_id,
                        }

                        await self._store_eval_result_event(thread_id, result_row)
                        return {"question": question, "status": status, "rating": rating, "error": None}
                    except Exception as exc:
                        logging.error(f"Failed for '{question}': {exc}", exc_info=True)
                        return {"question": question, "status": None, "error": str(exc)}

            self._append_progress(
                f"Queued {len(selection)} question(s) with up to {max_parallel} parallel worker(s)"
            )
            tasks = [asyncio.create_task(_run_one(idx, question)) for idx, question in enumerate(selection)]

            completed = 0
            for future in asyncio.as_completed(tasks):
                res = await future
                completed += 1
                self.current_question_index = completed - 1
                question = res.get("question", "")
                if res.get("error"):
                    msg = f"Failed for '{question}': {res.get('error')}"
                else:
                    status = res.get("status", "")
                    rating = res.get("rating", 0.0)
                    msg = f"Completed: '{question}' (status: {status}, rating: {rating})"
                self._append_progress(msg)
                self.status_message = f"Completed {completed} of {len(selection)} question(s)"
                yield  # Yield after each completion to update UI

            # Finalize
            self.status_message = f"Evaluation complete: {completed} of {len(selection)} question(s) processed"
            self.current_question_index = -1
            self.total_questions_to_run = 0

            # Reload data to show new test results
            try:
                yield EvalState.load_eval_data()
            except Exception as e:
                logging.warning(f"Failed to reload eval data after test run: {e}")

            self.is_running = False
            yield

    @rx.event(background=True)  # type: ignore[operator]
    async def refresh_golden_dataset(self):
        """Refresh the golden dataset from Grist."""
        async with self:
            if self.is_running:
                return
            self.status_message = "Refreshing golden dataset from Grist…"
            self.error_message = ""
            self.is_running = True
            yield
            try:
                self.get_eval_gds_from_grist()
                self._append_progress("Golden dataset refreshed from Grist")
                await self._load_eval_questions()
                self.status_message = "Golden dataset refreshed successfully"
            except Exception as e:
                self.error_message = f"Failed to refresh golden dataset: {e}"
                logging.error(f"Failed to refresh golden dataset: {e}", exc_info=True)
            finally:
                self.is_running = False
                yield

    def run_judge_refresh(self):
        if self.is_running:
            return
        step = next((s for s in etl_app.steps if s._name == "get_eval_judge_config"), None)
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
            self.tests_page = 0  # Reset to first page
            yield
            try:
                await self._load_eval_questions()
                self.tests_page_size = max(1, len(self.eval_gds_rows) * 2)
                await self._load_judge_config()
                await self._load_pipeline_config()
                await self._load_tests()
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
