import asyncio
import difflib
import hashlib
import json
import logging
import os
import statistics
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, TypedDict, cast
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

from vedana_backoffice.states.common import get_vedana_app, load_openrouter_models, HAS_OPENROUTER_KEY
from vedana_backoffice.util import safe_render_value


class QuestionResult(TypedDict):
    status: str
    rating: float | str | None
    comment: str
    answer: str
    tool_calls: str
    golden_answer: str
    thread_id: str


class RunSummary(TypedDict):
    run_id: str
    run_label: str
    tests_total: int
    passed: int
    failed: int
    pass_rate: float
    avg_rating: str  # rounded and converted to str
    cost_total: float
    test_run_name: str
    avg_answer_time_sec: float
    median_answer_time_sec: float


@dataclass
class GraphMeta:
    nodes_by_label: dict[str, int]
    edges_by_type: dict[str, int]
    vector_indexes: list[dict[str, object]]


@dataclass
class DmMeta:
    dm_id: str = ""
    dm_description: str = ""
    dm_snapshot_id: str = ""
    dm_branch: str = ""


@dataclass
class JudgeMeta:
    judge_model: str = ""
    judge_prompt_id: str = ""
    judge_prompt: str = ""


@dataclass
class RunConfig:
    pipeline_model: str = ""
    embeddings_model: str = ""
    embeddings_dim: int = 0


@dataclass
class RunMeta:
    graph: GraphMeta
    judge: JudgeMeta
    run_config: RunConfig
    dm: DmMeta


@dataclass
class RunData:
    summary: RunSummary
    meta: RunMeta
    config_summary: "RunConfigSummary"
    results: dict[str, QuestionResult]


@dataclass
class RunConfigSummary:
    test_run_id: str
    test_run_name: str
    pipeline_model: str
    embeddings_model: str
    embeddings_dim: int
    judge_model: str
    judge_prompt_id: str
    judge_prompt_hash: str
    dm_hash: str
    dm_id: str
    dm_snapshot_id: str
    dm_branch: str
    graph_nodes: dict[str, int]
    graph_edges: dict[str, int]
    vector_indexes: list[dict[str, object]]


@dataclass
class DiffLine:
    left: str
    right: str
    op: str
    left_color: str
    right_color: str
    strong: bool
    row_idx: int
    is_change: bool


@dataclass
class CompareRow:
    question: str
    golden_answer: str
    status_a: str
    rating_a: float | str | None
    comment_a: str
    answer_a: str
    tool_calls_a: str
    status_b: str
    rating_b: float | str | None
    comment_b: str
    answer_b: str
    tool_calls_b: str


DiffRow = dict[str, str | int | bool]


EMPTY_RESULT: QuestionResult = {
    "status": "—",
    "rating": "—",
    "comment": "",
    "answer": "",
    "tool_calls": "",
    "golden_answer": "",
    "thread_id": "",
}

EMPTY_SUMMARY: RunSummary = {
    "run_id": "",
    "run_label": "",
    "tests_total": 0,
    "passed": 0,
    "failed": 0,
    "pass_rate": 0.0,
    "avg_rating": "—",
    "cost_total": 0.0,
    "test_run_name": "",
    "avg_answer_time_sec": 0.0,
    "median_answer_time_sec": 0.0,
}


class EvalState(rx.State):
    """State holder for evaluation workflow."""

    loading: bool = False
    error_message: str = ""
    status_message: str = ""
    eval_gds_rows: list[dict[str, Any]] = []
    gds_expanded_rows: list[str] = []
    selected_question_ids: list[str] = []
    test_run_name: str = ""
    selected_scenario: str = "all"  # Filter by scenario
    judge_model: str = ""
    judge_prompt_id: str = ""
    judge_prompt: str = ""
    provider: str = "openai"
    pipeline_model: str = core_settings.model
    embeddings_model: str = core_settings.embeddings_model
    embeddings_dim: int = core_settings.embeddings_dim
    custom_openrouter_key: str = ""
    default_openrouter_key_present: bool = HAS_OPENROUTER_KEY
    enable_dm_filtering: bool = bool(os.environ.get("ENABLE_DM_FILTERING", False))
    _default_models: tuple[str, ...] = (
        "gpt-5.1-chat-latest",
        "gpt-5.1",
        "gpt-5-chat-latest",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-4o",
        "gpt-4o-mini",
        "o4-mini",
    )
    openai_models: list[str] = list(set(list(_default_models) + [core_settings.model]))
    openrouter_models: list[str] = []
    available_models: list[str] = list(set(list(_default_models) + [core_settings.model]))
    dm_id: str = ""
    dm_snapshot_id: str = ""
    dm_branch: str = core_settings.config_plane_dev_branch
    dm_snapshot_input: str = ""
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
    # Run comparison
    compare_run_a: str = ""
    compare_run_b: str = ""
    compare_dialog_open: bool = False
    compare_loading: bool = False
    compare_error: str = ""
    compare_rows: list[dict[str, Any]] = []
    compare_summary: dict[str, Any] = {}
    compare_summary_a: RunSummary = EMPTY_SUMMARY
    compare_summary_b: RunSummary = EMPTY_SUMMARY
    compare_config_a: dict[str, object] = {}
    compare_config_b: dict[str, object] = {}
    compare_config_a_rows: list[dict[str, object]] = []
    compare_config_b_rows: list[dict[str, object]] = []
    compare_configs: dict[str, Any] = {}
    compare_diff_keys: list[str] = []
    compare_prompt_diff: str = ""
    compare_dm_diff: str = ""
    compare_prompt_full_a: str = ""
    compare_prompt_full_b: str = ""
    compare_dm_full_a: str = ""
    compare_dm_full_b: str = ""
    compare_prompt_diff_rows: list[DiffRow] = []
    compare_dm_diff_rows: list[DiffRow] = []
    compare_judge_prompt_compact: bool = True
    compare_dm_compact: bool = True

    @rx.var
    def available_scenarios(self) -> list[str]:
        """Get unique scenarios from eval_gds_rows."""
        scenarios = set()
        for row in self.eval_gds_rows:
            scenario = row.get("question_scenario")
            if scenario:
                scenarios.add(str(scenario))
        return ["all"] + sorted(scenarios)

    @rx.var
    def eval_gds_rows_with_selection(self) -> list[dict[str, Any]]:
        selected = set(self.selected_question_ids)
        expanded = set(self.gds_expanded_rows)
        rows: list[dict[str, Any]] = []
        for row in self.eval_gds_rows:
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

    @rx.var
    def run_options_only(self) -> list[str]:
        """Run options excluding the 'All' placeholder."""
        return [opt for opt in (self.run_id_options or []) if opt != "All"]

    @rx.var
    def can_compare_runs(self) -> bool:
        """Enable compare when both runs are selected and distinct."""
        return (
            bool(self.compare_run_a)
            and bool(self.compare_run_b)
            and self.compare_run_a != self.compare_run_b
            and not self.compare_loading
        )

    @rx.var
    def available_models_view(self) -> list[str]:
        return self.available_models

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
        ids = [str(row.get("id", "") or "").strip() for row in self.eval_gds_rows_with_selection if row.get("id")]
        self.selected_question_ids = [qid for qid in ids if qid]  # Filter out empty IDs

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
            for row in self.eval_gds_rows
            if str(row.get("question_scenario", "")) == self.selected_scenario
        }
        self.selected_question_ids = [q for q in (self.selected_question_ids or []) if q in allowed_ids]

    def set_test_run_name(self, value: str) -> None:
        """Set user-provided test run name."""
        self.test_run_name = str(value or "").strip()

    def set_pipeline_model(self, value: str) -> None:
        if value in self.available_models:
            self.pipeline_model = value

    def set_custom_openrouter_key(self, value: str) -> None:
        self.custom_openrouter_key = str(value or "").strip()
        # optional: could refetch models with the override; keep static to avoid extra calls

    def set_enable_dm_filtering(self, value: bool) -> None:
        self.enable_dm_filtering = value

    def set_dm_branch(self, value: str) -> None:
        self.dm_branch = value

    def set_dm_snapshot_input(self, value: str) -> None:
        self.dm_snapshot_input = value

    def _resolve_dm_snapshot(self) -> int | None:
        raw = (self.dm_snapshot_input or "").strip()
        if not raw:
            return None
        try:
            return int(raw)
        except Exception:
            return None

    async def _apply_data_model_selection(self) -> None:
        vedana_app = await get_vedana_app()
        dm = vedana_app.data_model
        dm.set_branch(self.dm_branch)
        dm.set_snapshot_override(self._resolve_dm_snapshot())

    async def set_provider(self, value: str) -> None:
        self.provider = str(value or "openai")
        if self.provider == "openrouter" and not self.openrouter_models:
            self.openrouter_models = await load_openrouter_models()
        self._sync_available_models()

    def set_compare_run_a(self, value: str) -> None:
        self.compare_run_a = str(value or "").strip()

    def set_compare_run_b(self, value: str) -> None:
        self.compare_run_b = str(value or "").strip()

    def set_compare_dialog_open(self, open: bool) -> None:
        self.compare_dialog_open = open

    def _prune_selection(self) -> None:
        # Validate against all rows (not filtered) to keep selections valid across filter changes
        valid = {str(row.get("id")) for row in self.eval_gds_rows if row.get("id")}
        self.selected_question_ids = [q for q in (self.selected_question_ids or []) if q in valid]

    def _sync_available_models(self) -> None:
        if self.provider == "openrouter":
            models = self.openrouter_models
            if not models:
                self.provider = "openai"
                models = self.openai_models
        else:
            models = self.openai_models

        self.available_models = list(models)
        if self.pipeline_model not in self.available_models and self.available_models:
            self.pipeline_model = self.available_models[0]

    def _resolved_pipeline_model(self) -> str:
        provider = self.provider or "openai"
        return f"{provider}/{self.pipeline_model}"

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
            question = str(safe_render_value(rec.get("gds_question")) or "").strip()
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
        self.judge_prompt = ""

        vedana_app = await get_vedana_app()
        dm_pt = await vedana_app.data_model.prompt_templates()
        judge_prompt = dm_pt.get("eval_judge_prompt")

        if judge_prompt:
            text_b = bytearray(judge_prompt, "utf-8")
            self.judge_prompt_id = hashlib.sha256(text_b).hexdigest()
            self.judge_prompt = judge_prompt

    async def _load_pipeline_config(self) -> None:
        vedana_app = await get_vedana_app()
        await self._apply_data_model_selection()
        dm = vedana_app.data_model
        snapshot_id = dm.get_snapshot_id()
        self.dm_snapshot_id = str(snapshot_id) if snapshot_id is not None else ""
        self.dm_description = await dm.to_text_descr()
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
                await session.execute(
                    sa.select(ThreadDB.contact_id, ThreadDB.thread_config)
                    .where(ThreadDB.thread_config.contains({"source": "eval"}))
                    .order_by(ThreadDB.created_at.desc())
                )
            ).all()
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
            ordered_runs: list[tuple[str, dict[str, Any]]] = []
            for rid, cfg in run_rows:
                if rid not in seen:
                    ordered_runs.append((rid, cfg))
                    seen.add(rid)

            lookup: dict[str, str] = {}
            labels: list[str] = []
            for rid, cfg in ordered_runs:
                base_label = self._format_run_label_with_name(rid, cfg)
                label = base_label
                if label in lookup:
                    label = f"{base_label} ({rid})"
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
            cfg = thread.thread_config or {}
            evs = events_by_thread.get(thread.thread_id, [])
            answer = ""
            status = "—"
            judge_comment = ""
            rating_label = "—"
            run_label = self._format_run_label_with_name(thread.contact_id, cfg)
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

    def _resolve_run_contact(self, label: str) -> str:
        """Translate UI label to contact_id; fallback to provided label."""
        if not label:
            return ""
        return self.run_id_lookup.get(label, label)

    def set_compare_judge_prompt_compact(self, checked: bool) -> None:
        """Toggle compact diff view for prompt diff."""
        self.compare_judge_prompt_compact = bool(checked)

    def set_compare_dm_compact(self, checked: bool) -> None:
        """Toggle compact diff view for data model diff."""
        self.compare_dm_compact = bool(checked)

    def compare_runs(self):
        """Connecting button with a background task. Used to trigger animations properly."""
        if self.compare_loading:
            return
        if not self.compare_run_a or not self.compare_run_b or self.compare_run_a == self.compare_run_b:
            self.compare_error = "Select two different runs to compare."
            self.compare_dialog_open = True
            return

        run_a_id = self._resolve_run_contact(self.compare_run_a)
        run_b_id = self._resolve_run_contact(self.compare_run_b)
        if not run_a_id or not run_b_id:
            self.compare_error = "Unable to resolve selected runs."
            self.compare_dialog_open = True
            return

        self.compare_loading = True
        self.compare_error = ""
        self.compare_dialog_open = True
        yield
        yield EvalState.compare_runs_background(run_a_id, run_b_id)

    @rx.event(background=True)  # type: ignore[operator]
    async def compare_runs_background(self, run_a_id: str, run_b_id: str):
        try:
            vedana_app = await get_vedana_app()
            async with vedana_app.sessionmaker() as session:
                threads_res = (
                    (
                        await session.execute(
                            sa.select(ThreadDB)
                            .where(
                                ThreadDB.thread_config.contains({"source": "eval"}),
                                ThreadDB.contact_id.in_([run_a_id, run_b_id]),
                            )
                            .order_by(ThreadDB.created_at.desc())
                        )
                    )
                    .scalars()
                    .all()
                )

                if not threads_res:
                    async with self:
                        self.compare_error = "No threads found for selected runs."
                    return

                thread_ids = [t.thread_id for t in threads_res]
                events_res = (
                    (
                        await session.execute(
                            sa.select(ThreadEventDB)
                            .where(ThreadEventDB.thread_id.in_(thread_ids))
                            .order_by(ThreadEventDB.created_at)
                        )
                    )
                    .scalars()
                    .all()
                )

            events_by_thread: dict[UUID, list[ThreadEventDB]] = {}
            for ev in events_res:
                events_by_thread.setdefault(ev.thread_id, []).append(ev)

            threads_a = [t for t in threads_res if t.contact_id == run_a_id]
            threads_b = [t for t in threads_res if t.contact_id == run_b_id]

            async with self:
                run_a_data = self._collect_run_data(run_a_id, threads_a, events_by_thread)
                run_b_data = self._collect_run_data(run_b_id, threads_b, events_by_thread)

            # Align questions across both runs
            run_a_results = run_a_data.results
            run_b_results = run_b_data.results
            all_questions = set(run_a_results.keys()) | set(run_b_results.keys())

            aligned_rows: list[CompareRow] = []
            for q in sorted(all_questions):
                ra = run_a_results.get(q, EMPTY_RESULT)
                rb = run_b_results.get(q, EMPTY_RESULT)
                aligned_rows.append(
                    CompareRow(
                        question=q,
                        golden_answer=ra["golden_answer"] or rb["golden_answer"],
                        status_a=ra["status"],
                        rating_a=ra["rating"],
                        comment_a=ra["comment"],
                        answer_a=ra["answer"],
                        tool_calls_a=ra["tool_calls"],
                        status_b=rb["status"],
                        rating_b=rb["rating"],
                        comment_b=rb["comment"],
                        answer_b=rb["answer"],
                        tool_calls_b=rb["tool_calls"],
                    )
                )

            cfg_a = asdict(run_a_data.config_summary)
            cfg_b = asdict(run_b_data.config_summary)
            async with self:
                diff_keys = self._diff_config_keys(cfg_a, cfg_b)

            # Prompt and data model diffs
            meta_a = run_a_data.meta
            meta_b = run_b_data.meta
            prompt_a = meta_a.judge.judge_prompt
            prompt_b = meta_b.judge.judge_prompt

            dm_a_str = meta_a.dm.dm_description
            dm_b_str = meta_b.dm.dm_description

            prompt_diff = "\n".join(difflib.unified_diff(prompt_a.splitlines(), prompt_b.splitlines(), lineterm=""))
            dm_diff = "\n".join(difflib.unified_diff(dm_a_str.splitlines(), dm_b_str.splitlines(), lineterm=""))
            async with self:
                prompt_diff_rows = self._build_side_by_side_diff(prompt_a, prompt_b)
                dm_diff_rows = self._build_side_by_side_diff(dm_a_str, dm_b_str)

                self.compare_summary_a = run_a_data.summary
                self.compare_summary_b = run_b_data.summary
                self.compare_diff_keys = diff_keys
                self.compare_rows = [asdict(r) for r in aligned_rows]
                self.compare_prompt_diff = prompt_diff
                self.compare_dm_diff = dm_diff
                self.compare_prompt_full_a = prompt_a
                self.compare_prompt_full_b = prompt_b
                self.compare_dm_full_a = dm_a_str
                self.compare_dm_full_b = dm_b_str
                self.compare_prompt_diff_rows = prompt_diff_rows
                self.compare_dm_diff_rows = dm_diff_rows
                self.compare_config_a = cfg_a
                self.compare_config_b = cfg_b
                self.compare_config_a_rows = self._config_rows(cfg_a, diff_keys)
                self.compare_config_b_rows = self._config_rows(cfg_b, diff_keys)
        except Exception as e:
            async with self:
                self.compare_error = f"Failed to compare runs: {e}"
        finally:
            async with self:
                self.compare_loading = False
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

    def _build_data_model_meta(self) -> dict[str, Any]:
        # dm_json = vedana_app.data_model.to_json()
        return {
            # "dm_json": dm_json,  # may get used later
            "dm_id": self.dm_id,
            "dm_description": self.dm_description,
            "dm_snapshot_id": self.dm_snapshot_id,
            "dm_branch": self.dm_branch,
        }

    async def _build_eval_meta_payload(self, vedana_app, test_run_id: str, test_run_name: str) -> dict[str, Any]:
        """Build a single eval.meta payload shared across threads for a run."""
        graph_meta = await self._collect_graph_metadata(vedana_app.graph)
        data_model_meta = self._build_data_model_meta()
        judge_meta = JudgeMeta(
            judge_model=self.judge_model,
            judge_prompt_id=self.judge_prompt_id,
            judge_prompt=self.judge_prompt,
        )
        run_config = RunConfig(
            pipeline_model=self._resolved_pipeline_model(),
            embeddings_model=self.embeddings_model,
            embeddings_dim=self.embeddings_dim,
        )
        return {
            "meta_version": 1,
            "test_run": test_run_id,
            "test_run_name": test_run_name,
            "run_config": asdict(run_config),
            "judge": asdict(judge_meta),
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

    def _format_run_label_with_name(self, contact_id: str | None, cfg: dict[str, Any] | None) -> str:
        """
        Prefer user-provided test_run_name, fallback to formatted timestamp label.
        """
        name = cfg["test_run_name"] if isinstance(cfg, dict) and "test_run_name" in cfg else ""
        base = self._format_run_label(contact_id)
        if name:
            return f"{name} — {base}"
        return base

    def _normalize_diff_val(self, val: Any) -> Any:
        """Normalize values for diffing."""
        if isinstance(val, (dict, list)):
            try:
                return json.dumps(val, sort_keys=True)
            except Exception:
                return str(val)
        return val

    def _build_side_by_side_diff(self, a_text: str, b_text: str) -> list[DiffRow]:
        """Produce side-by-side diff rows with color hints (text color, no background)."""
        a_lines = a_text.splitlines()
        b_lines = b_text.splitlines()
        sm = difflib.SequenceMatcher(None, a_lines, b_lines)
        rows: list[DiffLine] = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for al, bl in zip(a_lines[i1:i2], b_lines[j1:j2]):
                    rows.append(
                        DiffLine(
                            left=al,
                            right=bl,
                            op=tag,
                            left_color="inherit",
                            right_color="inherit",
                            strong=False,
                            row_idx=0,
                            is_change=False,
                        )
                    )
            elif tag == "replace":
                max_len = max(i2 - i1, j2 - j1)
                for k in range(max_len):
                    al = a_lines[i1 + k] if i1 + k < i2 else ""
                    bl = b_lines[j1 + k] if j1 + k < j2 else ""
                    rows.append(
                        DiffLine(
                            left=al,
                            right=bl,
                            op=tag,
                            left_color="var(--indigo-11)",
                            right_color="var(--indigo-11)",
                            strong=True,
                            row_idx=0,
                            is_change=True,
                        )
                    )
            elif tag == "delete":
                for al in a_lines[i1:i2]:
                    rows.append(
                        DiffLine(
                            left=al,
                            right="",
                            op=tag,
                            left_color="var(--red-11)",
                            right_color="inherit",
                            strong=True,
                            row_idx=0,
                            is_change=True,
                        )
                    )
            elif tag == "insert":
                for bl in b_lines[j1:j2]:
                    rows.append(
                        DiffLine(
                            left="",
                            right=bl,
                            op=tag,
                            left_color="inherit",
                            right_color="var(--green-11)",
                            strong=True,
                            row_idx=0,
                            is_change=True,
                        )
                    )
        for idx, row in enumerate(rows):
            row.row_idx = idx
            row.is_change = row.op != "equal"
        return [cast(DiffRow, asdict(r)) for r in rows]

    def _diff_config_keys(self, cfg_a: dict[str, object], cfg_b: dict[str, object]) -> list[str]:
        """Return keys whose normalized values differ."""
        keys = set(cfg_a.keys()) | set(cfg_b.keys())
        diffs: list[str] = []
        for key in keys:
            left = cfg_a[key] if key in cfg_a else None
            right = cfg_b[key] if key in cfg_b else None
            if self._normalize_diff_val(left) != self._normalize_diff_val(right):
                diffs.append(key)
        return sorted(diffs)

    def _summarize_config_for_display(self, meta: RunMeta, cfg_fallback: dict[str, Any]) -> RunConfigSummary:
        judge = meta.judge
        dm = meta.dm
        graph = meta.graph
        run_cfg = meta.run_config

        pipeline_model_val = (
            run_cfg.pipeline_model if run_cfg.pipeline_model else cfg_fallback.get("pipeline_model", "")
        )
        embeddings_model_val = (
            run_cfg.embeddings_model if run_cfg.embeddings_model else cfg_fallback.get("embeddings_model", "")
        )
        embeddings_dim_val = run_cfg.embeddings_dim if run_cfg.embeddings_dim else cfg_fallback.get("embeddings_dim", 0)
        pipeline_model = str(pipeline_model_val)
        embeddings_model = str(embeddings_model_val)
        embeddings_dim = self._to_int(embeddings_dim_val)

        judge_model = judge.judge_model or cfg_fallback.get("judge_model", "")
        judge_prompt_id = judge.judge_prompt_id or cfg_fallback.get("judge_prompt_id", "")

        judge_prompt_hash = hashlib.sha256(judge.judge_prompt.encode("utf-8")).hexdigest() if judge.judge_prompt else ""

        test_run_id = cfg_fallback.get("test_run", "")
        test_run_name = cfg_fallback.get("test_run_name", "")

        return RunConfigSummary(
            test_run_id=test_run_id,
            test_run_name=test_run_name,
            pipeline_model=pipeline_model,
            embeddings_model=embeddings_model,
            embeddings_dim=embeddings_dim,
            judge_model=judge_model,
            judge_prompt_id=judge_prompt_id,
            judge_prompt_hash=judge_prompt_hash,
            dm_hash=str(cfg_fallback.get("dm_hash", "")),
            dm_id=dm.dm_id,
            dm_snapshot_id=dm.dm_snapshot_id,
            dm_branch=dm.dm_branch,
            graph_nodes=graph.nodes_by_label,
            graph_edges=graph.edges_by_type,
            vector_indexes=graph.vector_indexes,
        )

    @rx.var
    def compare_prompt_rows_view(self) -> list[DiffRow]:
        rows = self.compare_prompt_diff_rows or []
        if not self.compare_judge_prompt_compact:
            return rows
        change_idxs = [self._to_int(cast(int | float | str | None, r["row_idx"])) for r in rows if r.get("is_change")]
        window = 4
        return [
            r
            for r in rows
            if r.get("is_change")
            or any(abs(self._to_int(cast(int | float | str | None, r["row_idx"])) - ci) <= window for ci in change_idxs)
        ]

    @rx.var
    def compare_dm_rows_view(self) -> list[DiffRow]:
        rows = self.compare_dm_diff_rows or []
        if not self.compare_dm_compact:
            return rows
        change_idxs = [self._to_int(cast(int | float | str | None, r["row_idx"])) for r in rows if r.get("is_change")]
        window = 4
        return [
            r
            for r in rows
            if r.get("is_change")
            or any(abs(self._to_int(cast(int | float | str | None, r["row_idx"])) - ci) <= window for ci in change_idxs)
        ]

    @rx.var
    def compare_run_label_a(self) -> str:
        name = self.compare_summary_a.get("test_run_name", "")
        if isinstance(name, str) and name.strip():
            other = self.compare_summary_b.get("test_run_name", "")
            if isinstance(other, str) and other.strip() and other == name:
                return f"{self.compare_summary_a.get('run_label', '')}"
            return name
        label = self.compare_summary_a.get("run_label", "")
        return str(label) if label else "Run A"

    @rx.var
    def compare_run_label_b(self) -> str:
        name = self.compare_summary_b.get("test_run_name", "")
        if isinstance(name, str) and name.strip():
            other = self.compare_summary_a.get("test_run_name", "")
            if isinstance(other, str) and other.strip() and other == name:
                return f"{self.compare_summary_b.get('run_label', '')}"
            return name
        label = self.compare_summary_b.get("run_label", "")
        return str(label) if label else "Run B"

    def _extract_eval_meta(self, events: list[ThreadEventDB]) -> dict[str, Any]:
        """Return first eval.meta event data, if any."""
        for ev in events:
            if ev.event_type == "eval.meta" and isinstance(ev.event_data, dict):
                return ev.event_data
        return {}

    def _to_int(self, val: int | float | str | None, default: int = 0) -> int:
        if val is None:
            return default
        try:
            narrowed: int | float | str = cast(int | float | str, val)
            return int(narrowed)
        except Exception:
            return default

    def _parse_run_meta(self, meta: dict[str, Any], cfg_fallback: dict[str, Any]) -> RunMeta:
        graph_src = meta["graph"] if "graph" in meta and isinstance(meta["graph"], dict) else {}
        judge_src = meta["judge"] if "judge" in meta and isinstance(meta["judge"], dict) else {}
        dm_src = meta["data_model"] if "data_model" in meta and isinstance(meta["data_model"], dict) else {}
        run_cfg_src = meta["run_config"] if "run_config" in meta and isinstance(meta["run_config"], dict) else {}

        graph = GraphMeta(
            nodes_by_label=graph_src["nodes_by_label"] if "nodes_by_label" in graph_src else {},
            edges_by_type=graph_src["edges_by_type"] if "edges_by_type" in graph_src else {},
            vector_indexes=graph_src["vector_indexes"] if "vector_indexes" in graph_src else [],
        )
        judge = JudgeMeta(
            judge_model=str(judge_src["judge_model"])
            if "judge_model" in judge_src
            else str(cfg_fallback.get("judge_model", "")),
            judge_prompt_id=str(judge_src["judge_prompt_id"])
            if "judge_prompt_id" in judge_src
            else str(cfg_fallback.get("judge_prompt_id", "")),
            judge_prompt=str(judge_src["judge_prompt"]) if "judge_prompt" in judge_src else "",
        )
        embeddings_dim_src = (
            run_cfg_src["embeddings_dim"] if "embeddings_dim" in run_cfg_src else cfg_fallback.get("embeddings_dim", 0)
        )
        run_config = RunConfig(
            pipeline_model=str(run_cfg_src["pipeline_model"])
            if "pipeline_model" in run_cfg_src
            else str(cfg_fallback.get("pipeline_model", "")),
            embeddings_model=str(run_cfg_src["embeddings_model"])
            if "embeddings_model" in run_cfg_src
            else str(cfg_fallback.get("embeddings_model", "")),
            embeddings_dim=self._to_int(embeddings_dim_src),
        )
        dm = DmMeta(
            dm_id=str(dm_src["dm_id"]) if "dm_id" in dm_src else str(cfg_fallback.get("dm_id", "")),
            dm_description=str(dm_src["dm_description"]) if "dm_description" in dm_src else "",
            dm_snapshot_id=str(dm_src["dm_snapshot_id"]) if "dm_snapshot_id" in dm_src else str(cfg_fallback.get("dm_snapshot_id", "")),
            dm_branch=str(dm_src["dm_branch"]) if "dm_branch" in dm_src else str(cfg_fallback.get("dm_branch", "")),
        )
        return RunMeta(graph=graph, judge=judge, run_config=run_config, dm=dm)

    def _collect_run_data(
        self,
        run_id: str,
        threads: list[ThreadDB],
        events_by_thread: dict[UUID, list[ThreadEventDB]],
    ) -> RunData:
        """Aggregate results, meta, and stats for a single run."""
        results_by_question: dict[str, QuestionResult] = {}
        meta_sample: dict[str, Any] = {}
        cfg_sample: dict[str, Any] = {}
        total = 0
        passed = 0
        failed = 0
        cost_total = 0.0
        ratings: list[float] = []
        answer_times: list[float] = []

        for thread in threads:
            cfg = thread.thread_config or {}
            if not cfg_sample:
                cfg_sample = cfg
            evs = events_by_thread.get(thread.thread_id, [])
            if not meta_sample:
                meta_sample = self._extract_eval_meta(evs)

            answer = ""
            tool_calls = ""
            status = "—"
            judge_comment = ""
            rating_val: float | None = None
            question_text = str(cfg["gds_question"]) if "gds_question" in cfg else ""
            golden_answer = str(cfg["gds_answer"]) if "gds_answer" in cfg else ""
            user_ts = None
            answer_ts = None

            for ev in evs:
                if not isinstance(ev.event_data, dict):
                    continue
                if ev.event_type == "comm.user_message":
                    user_ts = ev.created_at
                if ev.event_type == "comm.assistant_message":
                    if "content" in ev.event_data:
                        answer = str(ev.event_data["content"])
                    answer_ts = ev.created_at
                elif ev.event_type == "rag.query_processed":
                    tech = (
                        cast(dict[str, Any], ev.event_data["technical_info"])
                        if "technical_info" in ev.event_data
                        else {}
                    )
                    model_stats = cast(dict[str, Any], tech["model_stats"]) if "model_stats" in tech else {}
                    for stats in model_stats.values():
                        if isinstance(stats, dict):
                            cost_val = stats["requests_cost"] if "requests_cost" in stats else None
                            try:
                                if cost_val is not None:
                                    cost_total += float(cost_val)
                            except (TypeError, ValueError):
                                pass
                elif ev.event_type == "eval.result":
                    status = str(ev.event_data["test_status"]) if "test_status" in ev.event_data else status
                    judge_comment = (
                        str(ev.event_data["eval_judge_comment"])
                        if "eval_judge_comment" in ev.event_data
                        else judge_comment
                    )
                    answer = str(ev.event_data["llm_answer"]) if "llm_answer" in ev.event_data else answer
                    tool_calls = str(ev.event_data["tool_calls"]) if "tool_calls" in ev.event_data else tool_calls
                    golden_answer = str(ev.event_data["gds_answer"]) if "gds_answer" in ev.event_data else golden_answer
                    rating_label = ev.event_data["eval_judge_rating"] if "eval_judge_rating" in ev.event_data else None
                    try:
                        rating_val = float(rating_label) if rating_label is not None else None
                    except Exception:
                        rating_val = None
                    if "gds_question" in ev.event_data:
                        question_text = str(ev.event_data["gds_question"])

            total += 1
            if status == "pass":
                passed += 1
            elif status == "fail":
                failed += 1

            if user_ts and answer_ts:
                try:
                    delta = (answer_ts - user_ts).total_seconds()
                    if delta >= 0:
                        answer_times.append(delta)
                except Exception:
                    pass

            if rating_val is not None:
                ratings.append(rating_val)

            key = str(question_text or f"question-{total}")
            results_by_question[key] = QuestionResult(
                status=status or "—",
                rating=rating_val if rating_val is not None else "—",
                comment=safe_render_value(judge_comment),
                answer=safe_render_value(answer),
                tool_calls=safe_render_value(tool_calls),
                golden_answer=safe_render_value(golden_answer),
                thread_id=str(thread.thread_id),
            )

        avg_rating = sum(ratings) / len(ratings) if ratings else 0.0
        summary: RunSummary = {
            "run_id": run_id,
            "run_label": self._format_run_label_with_name(run_id, meta_sample or cfg_sample),
            "tests_total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": (passed / total) if total else 0.0,
            "avg_rating": str(round(avg_rating, 2) if ratings else "—"),
            "cost_total": round(cost_total, 3),
            "test_run_name": meta_sample["test_run_name"]
            if "test_run_name" in meta_sample
            else (cfg_sample["test_run_name"] if "test_run_name" in cfg_sample else ""),
            "avg_answer_time_sec": round(sum(answer_times) / len(answer_times), 2) if answer_times else 0.0,
            "median_answer_time_sec": round(statistics.median(answer_times), 2) if answer_times else 0.0,
        }

        run_meta = self._parse_run_meta(meta_sample, cfg_sample)
        config_summary = self._summarize_config_for_display(run_meta, cfg_sample)

        return RunData(
            summary=summary,
            meta=run_meta,
            config_summary=config_summary,
            results=results_by_question,
        )

    def _config_rows(self, cfg: dict[str, object], diff_keys: list[str]) -> list[dict[str, object]]:
        def _as_text(val: object) -> str:
            if isinstance(val, (dict, list)):
                try:
                    return json.dumps(val, ensure_ascii=False)
                except Exception:
                    return str(val)
            return str(val)

        rows: list[dict[str, object]] = []
        for label, key in [
            ("Pipeline model", "pipeline_model"),
            ("Embeddings", "embeddings_model"),
            ("Embedding dims", "embeddings_dim"),
            ("Judge model", "judge_model"),
            ("Data model branch", "dm_branch"),
            ("Data model snapshot", "dm_snapshot_id"),
            ("Graph nodes", "graph_nodes"),
            ("Graph edges", "graph_edges"),
            # ("Vector indexes", "vector_indexes"),  # takes a lot of space
        ]:
            val = _as_text(cfg[key]) if key in cfg else "—"
            rows.append({"label": label, "value": val, "diff": key in diff_keys})
        return rows

    def _build_thread_config(
        self, question_row: dict[str, Any], test_run_id: str, test_run_name: str
    ) -> dict[str, Any]:
        """Pack metadata into thread_config so runs are traceable in JIMS."""
        resolved_model = self._resolved_pipeline_model()
        return {
            "interface": "eval",
            "source": "eval",
            "test_run": test_run_id,
            "test_run_name": test_run_name,
            "gds_question": question_row.get("gds_question"),
            "gds_answer": question_row.get("gds_answer"),
            "question_context": question_row.get("question_context"),
            "question_scenario": question_row.get("question_scenario"),
            "judge_model": self.judge_model,
            "judge_prompt_id": self.judge_prompt_id,
            "pipeline_model": resolved_model,
            "pipeline_provider": self.provider,
            "embeddings_model": self.embeddings_model,
            "embeddings_dim": self.embeddings_dim,
            "dm_id": self.dm_id,
            "dm_snapshot_id": self.dm_snapshot_id,
            "dm_branch": self.dm_branch,
        }

    async def _run_question_thread(
        self,
        vedana_app,
        question_row: dict[str, Any],
        test_run_name: str,
        test_run_id: str,
        eval_meta_base: dict[str, Any],
    ) -> tuple[str, str, dict[str, Any]]:
        """Create a JIMS thread, post the question, run pipeline, return answer + tech info."""
        thread_id = uuid7()
        await self._apply_data_model_selection()
        ctl = await ThreadController.new_thread(
            vedana_app.sessionmaker,
            contact_id=test_run_id,
            thread_id=thread_id,
            thread_config=self._build_thread_config(question_row, test_run_id, test_run_name),
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
        pipeline = vedana_app.pipeline
        resolved_model = self._resolved_pipeline_model()
        pipeline.model = resolved_model
        pipeline.enable_filtering = self.enable_dm_filtering

        ctx = await ctl.make_context()
        if self.provider == "openrouter" and self.custom_openrouter_key:
            ctx.llm.model_api_key = self.custom_openrouter_key

        events = await ctl.run_pipeline_with_context(pipeline, ctx)

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
        judge_prompt = self.judge_prompt
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

    def run_selected_tests(self):
        """Trigger test run - validates, sets loading state and starts background task."""
        if self.is_running:
            return

        # Validation
        selection = [str(q) for q in (self.selected_question_ids or []) if str(q)]
        if not selection:
            self.error_message = "Select at least one question to run tests."
            return
        if not self.judge_prompt:
            self.error_message = "Judge prompt not loaded. Refresh judge config first."
            return
        if self.provider == "openrouter":
            key = (self.custom_openrouter_key or os.environ.get("OPENROUTER_API_KEY") or "").strip()
            if not key:
                self.error_message = "OPENROUTER_API_KEY is required for OpenRouter provider."
                return

        test_run_name = self.test_run_name.strip() or ""

        # Initialize run state
        self.is_running = True
        self.current_question_index = -1
        self.total_questions_to_run = len(selection)
        self.status_message = f"Evaluation run '{test_run_name}' for {len(selection)} question(s)…"
        self.run_progress = []
        self.error_message = ""
        yield
        yield EvalState.run_selected_tests_background()

    @rx.event(background=True)  # type: ignore[operator]
    async def run_selected_tests_background(self):
        async with self:
            selection = [str(q) for q in (self.selected_question_ids or []) if str(q)]

            question_map = {}
            for row in self.eval_gds_rows:
                if row:
                    row_id = str(row.get("id", ""))
                    if row_id:
                        question_map[row_id] = row

            test_run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            test_run_id = f"eval:{test_run_ts}"
            test_run_name = self.test_run_name.strip() or ""
            resolved_pipeline_model = self._resolved_pipeline_model()

        vedana_app = await get_vedana_app()
        async with self:
            eval_meta_base = await self._build_eval_meta_payload(vedana_app, test_run_id, test_run_name)
            max_parallel = max(1, int(self.max_parallel_tests or 1))

        sem = asyncio.Semaphore(max_parallel)

        async def _run_one(question: str) -> dict[str, Any]:
            async with sem:
                row = question_map.get(str(question or "").strip())
                if row is None:
                    return {"question": question, "status": None, "error": "not found"}
                try:
                    thread_id, answer, tech = await self._run_question_thread(
                        vedana_app, row, test_run_name, test_run_id, eval_meta_base
                    )
                    tool_calls = self._format_tool_calls(tech)
                    async with self:
                        status, comment, rating = await self._judge_answer(row, answer, tool_calls)

                    result_row = {
                        "judge_model": self.judge_model,
                        "judge_prompt_id": self.judge_prompt_id,
                        "dm_id": self.dm_id,
                        "pipeline_model": resolved_pipeline_model,
                        "embeddings_model": self.embeddings_model,
                        "embeddings_dim": self.embeddings_dim,
                        "test_run_id": test_run_id,
                        "test_run_name": test_run_name,
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

        async with self:
            self._append_progress(f"Queued {len(selection)} question(s) with up to {max_parallel} parallel worker(s)")
        tasks = [asyncio.create_task(_run_one(question)) for question in selection]

        completed = 0
        for future in asyncio.as_completed(tasks):
            res = await future
            completed += 1
            async with self:
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
        async with self:
            self.status_message = f"Evaluation complete: {completed} of {len(selection)} question(s) processed"
            self.current_question_index = -1
            self.total_questions_to_run = 0

        # Reload data to show new test results
        try:
            yield EvalState.load_eval_data_background()
        except Exception as e:
            logging.warning(f"Failed to reload eval data after test run: {e}")

        async with self:
            self.is_running = False
        yield

    def refresh_golden_dataset(self):
        """Connecting button with a background task. Used to trigger animations properly."""
        if self.is_running:
            return
        self.status_message = "Refreshing golden dataset from Grist…"
        self.error_message = ""
        self.is_running = True
        yield
        yield EvalState.refresh_golden_dataset_background()

    @rx.event(background=True)  # type: ignore[operator]
    async def refresh_golden_dataset_background(self):
        try:
            await asyncio.to_thread(self.get_eval_gds_from_grist)
            async with self:
                self._append_progress("Golden dataset refreshed from Grist")
                await self._load_eval_questions()
                self.status_message = "Golden dataset refreshed successfully"
        except Exception as e:
            async with self:
                self.error_message = f"Failed to refresh golden dataset: {e}"
            logging.error(f"Failed to refresh golden dataset: {e}", exc_info=True)
        finally:
            async with self:
                self.is_running = False
            yield

    def load_eval_data(self):
        """Connecting button with a background task. Used to trigger animations properly."""
        if self.loading:
            return
        self.loading = True
        self.error_message = ""
        self.status_message = ""
        self.tests_page = 0  # Reset to first page
        yield
        yield EvalState.load_eval_data_background()

    @rx.event(background=True)  # type: ignore[operator]
    async def load_eval_data_background(self):
        try:
            async with self:
                self.openrouter_models = await load_openrouter_models()
                self._sync_available_models()
                await self._load_eval_questions()
                self.tests_page_size = max(1, len(self.eval_gds_rows) * 2)
                await self._load_judge_config()
                await self._load_pipeline_config()
                await self._load_tests()
        except Exception as e:
            async with self:
                self.error_message = f"Failed to load eval data: {e} {traceback.format_exc()}"
        finally:
            async with self:
                self.loading = False
            yield
