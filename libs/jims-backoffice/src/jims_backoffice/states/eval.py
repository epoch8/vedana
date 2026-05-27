import asyncio
import difflib
import hashlib
import json
import logging
import statistics
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, TypedDict, cast
from uuid import UUID

import reflex as rx
import sqlalchemy as sa
from jims_core.db import ThreadDB, ThreadEventDB
from jims_core.llms.llm_provider import LLMProvider, LLMSettings
from jims_core.thread.thread_controller import ThreadController
from jims_core.util import uuid7
from pydantic import BaseModel, Field

from jims_backoffice.app_loader import get_jims_app
from jims_backoffice.states.common import DEBUG_MODE, EVAL_ENABLED, DebugState
from jims_backoffice.states.jims import ThreadViewState
from jims_backoffice.util import safe_render_value, xlsx_bytes

_default_llm_settings = LLMSettings()  # type: ignore


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
    judge_cost_total: float
    test_run_name: str
    avg_answer_time_sec: float
    median_answer_time_sec: float


@dataclass
class GraphMeta:
    nodes_by_label: dict[str, int]
    edges_by_type: dict[str, int]


@dataclass
class DmMeta:
    dm_id: str = ""
    dm_description: str = ""


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
    graph_nodes: dict[str, int]
    graph_edges: dict[str, int]


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
    "judge_cost_total": 0.0,
    "test_run_name": "",
    "avg_answer_time_sec": 0.0,
    "median_answer_time_sec": 0.0,
}


eval_judge_prompt_template = """\
You are a strict evaluation judge. Compare the model's answer with the golden answer and the expected retrieval context. 
Consider whether the model's answer is factually aligned and sufficiently complete. 
Use the provided technical info (retrieval queries) only as hints for whether the context seems adequate. 
Return a JSON object with fields: test_status in {'pass','fail'}, comment, errors.

In comments return answer scoring from 1 to 10, where:
1 – totally wrong answer
10 – totally correct answer
"""


class EvalState(rx.State, mixin=True):
    """State holder for evaluation workflow.

    This is a state mixin. Concrete subclasses (e.g. ``VedanaEvalState``)
    override the data-loading hooks to supply golden datasets, judge prompts,
    and pipeline / graph metadata. The base class works against any pipeline
    that emits ``comm.assistant_message`` and (optionally) ``rag.query_processed``
    events.
    """

    loading: bool = False
    error_message: str = ""
    status_message: str = ""
    eval_gds_rows: list[dict[str, Any]] = []
    gds_expanded_rows: list[str] = []
    selected_question_ids: list[str] = []
    test_run_name: str = ""
    selected_scenario: str = "all"
    judge_prompt_id: str = ""
    judge_prompt: str = ""
    pipeline_model: str = _default_llm_settings.model
    default_embeddings_model: str = _default_llm_settings.embeddings_model
    embeddings_dim: int = _default_llm_settings.embeddings_dim
    available_models: list[str] = [_default_llm_settings.model]
    dm_id: str = ""
    judge_model: str = _default_llm_settings.model

    tests_rows: list[dict[str, Any]] = []
    tests_cost_total: float = 0.0
    tests_judge_cost_total: float = 0.0
    run_passed: int = 0
    run_failed: int = 0
    selected_run_id: str = ""
    run_id_options: list[str] = []
    run_id_lookup: dict[str, str] = {}
    selected_tests_scenario: str = "All"
    tests_scenario_options: list[str] = ["All"]
    selected_run_pipeline_model: str = ""
    is_running: bool = False
    run_progress: list[str] = []
    max_eval_rows: int = 500
    current_question_index: int = -1
    total_questions_to_run: int = 0
    judge_prompt_dialog_open: bool = False
    data_model_dialog_open: bool = False
    dm_description: str = ""
    tests_page: int = 0
    tests_page_size: int = 100
    tests_total_rows: int = 0
    tests_sort_options: list[str] = ["Recent ↓", "Recent ↑", "Rating ↓", "Rating ↑"]
    selected_tests_sort: str = "Recent ↓"
    max_parallel_tests: int = 4
    compare_run_a: str = ""
    compare_run_b: str = ""
    compare_dialog_open: bool = False
    compare_loading: bool = False
    compare_error: str = ""
    compare_rows: list[dict[str, Any]] = []
    compare_summary_a: RunSummary = EMPTY_SUMMARY
    compare_summary_b: RunSummary = EMPTY_SUMMARY
    compare_config_a_rows: list[dict[str, object]] = []
    compare_config_b_rows: list[dict[str, object]] = []
    compare_diff_keys: list[str] = []
    compare_prompt_diff_rows: list[DiffRow] = []
    compare_dm_diff_rows: list[DiffRow] = []
    compare_judge_prompt_compact: bool = True
    compare_dm_compact: bool = True

    @rx.var
    def available_scenarios(self) -> list[str]:
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
            if self.selected_scenario != "all":
                scenario = row.get("question_scenario")
                if str(scenario) != self.selected_scenario:
                    continue
            enriched = dict(row)
            enriched["selected"] = row.get("id") in selected
            enriched["expanded"] = row.get("id") in expanded
            scenario_val = str(row.get("question_scenario", ""))
            enriched["scenario_color"] = self._scenario_color(scenario_val)
            rows.append(enriched)
        return rows

    @rx.var
    def selected_count(self) -> int:
        return len(self.selected_question_ids or [])

    @rx.var
    def selection_label(self) -> str:
        total = len(self.eval_gds_rows_with_selection)
        if total == 0:
            return "No questions available"
        if self.selected_count == 0:
            return f"Select tests to run ({self.selected_count}/{total})"
        return f"Run selected ({self.selected_count}/{total})"

    @rx.var
    def all_selected(self) -> bool:
        rows = len(self.eval_gds_rows_with_selection)
        return 0 < rows == self.selected_count

    @rx.var
    def cost_label(self) -> str:
        if self.tests_cost_total > 0:
            return f"${self.tests_cost_total:.4f}"
        return "Cost data unavailable"

    @rx.var
    def judge_cost_label(self) -> str:
        if self.tests_judge_cost_total > 0:
            return f"${self.tests_judge_cost_total:.4f}"
        return "—"

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
        if self.total_questions_to_run == 0:
            return ""
        current = self.current_question_index + 1
        return f"Processing question {current} of {self.total_questions_to_run}"

    @rx.var
    def embeddings_dim_label(self) -> str:
        return f"{self.embeddings_dim} dims" if self.embeddings_dim > 0 else ""

    @rx.var
    def run_options_only(self) -> list[str]:
        return [opt for opt in (self.run_id_options or []) if opt != "All"]

    @rx.var
    def can_compare_runs(self) -> bool:
        return (
            bool(self.compare_run_a)
            and bool(self.compare_run_b)
            and self.compare_run_a != self.compare_run_b
            and not self.compare_loading
        )

    @rx.var
    def tests_selected_run_model(self) -> str:
        if self.selected_run_id and self.selected_run_id != "All":
            return self.selected_run_pipeline_model
        return ""

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
        ids = [str(row.get("id", "") or "").strip() for row in self.eval_gds_rows_with_selection if row.get("id")]
        self.selected_question_ids = [qid for qid in ids if qid]

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

        allowed_ids = {
            str(row.get("id"))
            for row in self.eval_gds_rows
            if str(row.get("question_scenario", "")) == self.selected_scenario
        }
        self.selected_question_ids = [q for q in (self.selected_question_ids or []) if q in allowed_ids]

    def set_test_run_name(self, value: str) -> None:
        self.test_run_name = str(value or "").strip()

    def set_pipeline_model(self, value: str) -> None:
        if value in self.available_models:
            self.pipeline_model = value

    def set_judge_model(self, value: str) -> None:
        if value in self.available_models:
            self.judge_model = value

    def set_compare_run_a(self, value: str) -> None:
        self.compare_run_a = str(value or "").strip()

    def set_compare_run_b(self, value: str) -> None:
        self.compare_run_b = str(value or "").strip()

    def set_compare_dialog_open(self, open: bool) -> None:
        self.compare_dialog_open = open

    def _prune_selection(self) -> None:
        valid = {str(row.get("id")) for row in self.eval_gds_rows if row.get("id")}
        self.selected_question_ids = [q for q in (self.selected_question_ids or []) if q in valid]

    def _sync_available_models(self) -> None:
        if self.pipeline_model not in self.available_models and self.available_models:
            for model in self.available_models:
                if model.endswith(_default_llm_settings.model):
                    self.pipeline_model = model
                    break
            else:
                if "openrouter/openrouter/free" in self.available_models:
                    self.pipeline_model = "openrouter/openrouter/free"
                else:
                    self.pipeline_model = self.available_models[0]

    def _sync_judge_model(self) -> None:
        if self.judge_model not in self.available_models and self.available_models:
            for model in self.available_models:
                if model.endswith(_default_llm_settings.model):
                    self.judge_model = model
                    break
            else:
                if "openrouter/openrouter/free" in self.available_models:
                    self.judge_model = "openrouter/openrouter/free"
                else:
                    self.judge_model = self.available_models[0]

    @rx.event(background=True)  # type: ignore[operator]
    async def refresh_model_list(self) -> None:
        async with self:
            self.available_models = await self.get_var_value(DebugState.available_models)  # type: ignore[arg-type]
            self._sync_available_models()
            self._sync_judge_model()

    # --- Overridable hooks ----------------------------------------------------

    async def _load_eval_questions(self) -> None:
        """Load the golden dataset into ``self.eval_gds_rows``.

        Base implementation leaves the list empty. Subclasses populate it from
        their own source (Grist, CSV, etc.).
        """
        self.eval_gds_rows = []
        self.gds_expanded_rows = []
        self._prune_selection()

    async def _load_judge_config(self) -> None:
        """Load the judge prompt and judge model id."""
        if not self.judge_model:
            self.judge_model = _default_llm_settings.model
        self.judge_prompt = eval_judge_prompt_template
        text_b = bytearray(self.judge_prompt, "utf-8")
        self.judge_prompt_id = hashlib.sha256(text_b).hexdigest()

    async def _load_pipeline_config(self) -> None:
        """Populate pipeline/data-model metadata (e.g. ``dm_id``, ``dm_description``).

        Base implementation is a no-op; subclasses with a data model fill these.
        """
        return None

    async def _collect_eval_meta_sections(self) -> dict[str, Any]:
        """Return extra ``eval.meta`` sections to merge into the payload.

        The base payload built by :meth:`_build_eval_meta_payload` always carries
        ``run_config`` and ``judge``. Subclasses use this hook to attach
        backend-specific sections (e.g. ``graph`` for Memgraph, ``data_model``
        for projects with a DM). The default exposes ``data_model`` populated
        from the state vars set in :meth:`_load_pipeline_config`; subclasses
        should call ``super()._collect_eval_meta_sections()`` and extend the
        returned dict.

        Sections are stored verbatim under their key in the eval.meta event and
        rendered by the comparison UI when keys match well-known names
        (``data_model``, ``graph``).
        """
        return {"data_model": self._build_data_model_meta()}

    @staticmethod
    def _eval_threads_clause() -> Any:
        return ThreadDB.thread_config.contains({"interface": "eval"})

    def _build_thread_config(
        self, question_row: dict[str, Any], test_run_id: str, test_run_name: str
    ) -> dict[str, Any]:
        """Minimal thread_config for JIMS list/filter; run id lives on contact_id."""
        scenario = question_row.get("question_scenario")
        return {
            "interface": "eval",
            "test_run_name": test_run_name,
            "question_scenario": "" if not scenario else str(scenario),
        }

    async def _apply_pipeline_settings(self, pipeline: Any) -> None:
        """Configure pipeline before running a question.

        Subclasses can extend this to set DM filters, attach loggers etc.
        """
        if hasattr(pipeline, "model"):
            try:
                pipeline.model = self.pipeline_model
            except Exception:
                pass

    async def _make_pipeline_llm_settings(self) -> LLMSettings:
        """Build LLMSettings used to run pipelines for eval questions."""
        return LLMSettings(model=self.pipeline_model)  # type: ignore

    async def _make_judge_llm_settings(self) -> LLMSettings:
        """Build LLMSettings used by the judge."""
        return LLMSettings(model=self.judge_model)  # type: ignore

    # --- Color/format helpers -------------------------------------------------

    def _status_color(self, status: str) -> str:
        if status == "pass":
            return "green"
        if status == "fail":
            return "red"
        return "gray"

    def _scenario_color(self, scenario: str) -> str:
        if not scenario:
            return "gray"
        color_schemes = [
            "blue", "green", "purple", "pink", "indigo", "cyan",
            "amber", "lime", "emerald", "teal", "sky", "violet",
            "fuchsia", "rose", "orange", "slate",
        ]
        hash_val = hash(str(scenario))
        color_idx = abs(hash_val) % len(color_schemes)
        return color_schemes[color_idx]

    # --- Tests / runs loading -------------------------------------------------

    async def _load_tests(self, *, select_contact_id: str | None = None) -> None:
        """Build test results table directly from JIMS eval threads.

        When ``select_contact_id`` is set (e.g. after a run finishes), refresh the run
        dropdown and select that run's results.
        """
        jims_app = await get_jims_app()

        async with jims_app.sessionmaker() as session:
            run_rows = (
                await session.execute(
                    sa.select(ThreadDB.contact_id, ThreadDB.thread_config)
                    .where(self._eval_threads_clause())
                    .order_by(ThreadDB.created_at.desc())
                )
            ).all()
            scenario_rows = (
                (
                    await session.execute(
                        sa.select(ThreadDB.thread_config).where(self._eval_threads_clause())
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

            if select_contact_id:
                new_label = next(
                    (lbl for lbl, rid in lookup.items() if rid == select_contact_id),
                    None,
                )
                if new_label:
                    self.selected_run_id = new_label
                elif labels:
                    self.selected_run_id = labels[0]
                self.selected_tests_sort = "Rating ↓"
            elif not self.selected_run_id:
                self.selected_run_id = labels[0] if labels else "All"
                if self.selected_run_id and self.selected_run_id != "All":
                    self.selected_tests_sort = "Rating ↓"
            elif self.selected_run_id not in self.run_id_options:
                self.selected_run_id = labels[0] if labels else "All"

            if self.selected_tests_sort not in self.tests_sort_options:
                self.selected_tests_sort = "Recent ↓"

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

            base_threads = sa.select(ThreadDB).where(self._eval_threads_clause())
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

            selected_sort = self.selected_tests_sort or "Recent ↓"
            rating_expr = None
            if selected_sort.startswith("Rating"):
                threads_q = threads_q.join(
                    eval_result_subq,
                    sa.and_(eval_result_subq.c.thread_id == ThreadDB.thread_id, eval_result_subq.c.rn == 1),
                    isouter=True,
                )
                rating_expr = sa.cast(eval_result_subq.c.eval_data["eval_judge_rating"].astext, sa.Integer)

            if selected_sort == "Rating ↓" and rating_expr is not None:
                threads_q = threads_q.order_by(sa.desc(rating_expr).nullslast(), ThreadDB.created_at.desc())
            elif selected_sort == "Rating ↑" and rating_expr is not None:
                threads_q = threads_q.order_by(sa.asc(rating_expr).nullslast(), ThreadDB.created_at.desc())
            elif selected_sort == "Recent ↑":
                threads_q = threads_q.order_by(ThreadDB.created_at.asc())
            else:
                threads_q = threads_q.order_by(ThreadDB.created_at.desc())

            threads_q = threads_q.limit(self.tests_page_size).offset(offset)
            page_threads = (await session.execute(threads_q)).scalars().all()

            if not page_threads:
                self.tests_rows = []
                self.tests_cost_total = 0.0
                self.tests_judge_cost_total = 0.0
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
        judge_cost_total = 0.0
        for thread in page_threads:
            cfg = thread.thread_config or {}
            evs = events_by_thread.get(thread.thread_id, [])
            answer = ""
            status = "—"
            judge_comment = ""
            rating_label = "—"
            gds_question = ""
            gds_answer = ""
            pipeline_model = ""
            run_label = self._format_run_label_with_name(thread.contact_id, cfg)
            test_date = run_label
            row_pipeline_cost: float = 0.0

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
                                        v = float(cost_val)
                                        cost_total += v
                                        row_pipeline_cost += v
                                except (TypeError, ValueError):
                                    pass
                elif ev.event_type == "eval.result":
                    ed = ev.event_data
                    status = ed.get("test_status", status)
                    judge_comment = ed.get("eval_judge_comment", judge_comment)
                    rating_label = str(ed.get("eval_judge_rating", rating_label))
                    test_date = self._format_run_label(ed.get("test_date", test_date))
                    gds_question = str(ed.get("gds_question") or "")
                    gds_answer = str(ed.get("gds_answer") or "")
                    pipeline_model = str(ed.get("pipeline_model") or "")
                    try:
                        jc = ed.get("judge_cost")
                        if jc is not None:
                            v = float(jc)
                            judge_cost_total += v
                    except (TypeError, ValueError):
                        pass

            if status == "pass":
                passed += 1
            elif status == "fail":
                failed += 1

            rows.append(
                {
                    "row_id": str(thread.thread_id),
                    "expanded": False,
                    "test_date": safe_render_value(test_date),
                    "gds_question": safe_render_value(gds_question),
                    "llm_answer": safe_render_value(answer),
                    "gds_answer": safe_render_value(gds_answer),
                    "pipeline_model": safe_render_value(pipeline_model),
                    "test_status": status or "—",
                    "status_color": self._status_color(status),
                    "eval_judge_comment": safe_render_value(judge_comment),
                    "eval_judge_rating": rating_label,
                    "pipeline_cost": f"${row_pipeline_cost:.6f}" if row_pipeline_cost else "—",
                }
            )

        self.tests_rows = rows
        self.tests_cost_total = cost_total
        self.tests_judge_cost_total = judge_cost_total
        self.run_passed = passed
        self.run_failed = failed
        self.selected_run_pipeline_model = "—" if not rows else str(rows[0].get("pipeline_model"))

    @rx.event(background=True)  # type: ignore[operator]
    async def tests_next_page(self):
        async with self:
            max_page = (self.tests_total_rows - 1) // self.tests_page_size if self.tests_total_rows > 0 else 0
            if self.tests_page < max_page:
                self.tests_page += 1
                await self._load_tests()
                yield

    @rx.event(background=True)  # type: ignore[operator]
    async def tests_prev_page(self):
        async with self:
            if self.tests_page > 0:
                self.tests_page -= 1
                await self._load_tests()
                yield

    @rx.event(background=True)  # type: ignore[operator]
    async def tests_first_page(self):
        async with self:
            if self.tests_page != 0:
                self.tests_page = 0
                await self._load_tests()
                yield

    @rx.event(background=True)  # type: ignore[operator]
    async def tests_last_page(self):
        async with self:
            max_page = (self.tests_total_rows - 1) // self.tests_page_size if self.tests_total_rows > 0 else 0
            if self.tests_page != max_page:
                self.tests_page = max_page
                await self._load_tests()
                yield

    @rx.event(background=True)  # type: ignore[operator]
    async def select_run(self, value: str):
        async with self:
            new_run = str(value or "")
            self.selected_run_id = new_run
            self.tests_page = 0
            # Default sort to Rating ↓ when viewing a specific run; back to Recent ↓ for All.
            self.selected_tests_sort = "Rating ↓" if (new_run and new_run != "All") else "Recent ↓"
            await self._load_tests()
            yield

    @rx.event(background=True)  # type: ignore[operator]
    async def select_tests_scenario(self, value: str):
        async with self:
            self.selected_tests_scenario = str(value or "All")
            self.tests_page = 0
            await self._load_tests()
            yield

    @rx.event(background=True)  # type: ignore[operator]
    async def select_tests_sort(self, value: str):
        async with self:
            self.selected_tests_sort = str(value or "Recent ↓")
            self.tests_page = 0
            await self._load_tests()
            yield

    _EXPORT_COLUMNS: list[str] = [
        "thread_id",
        "test_date",
        "run_id",
        "run_label",
        "test_run_name",
        "question_scenario",
        "gds_question",
        "gds_answer",
        "llm_answer",
        "test_status",
        "eval_judge_rating",
        "eval_judge_comment",
        "judge_cost",
        "pipeline_cost",
        "pipeline_model",
        "embeddings_model",
        "embeddings_dim",
        "judge_model",
        "judge_prompt_id",
        "graph_nodes_by_label",
        "graph_edges_by_type",
        "dm_id",
        "dm_description",
        "rag_query_processed_trace",
    ]

    async def _build_export_rows(
        self, contact_id_filter: str | None = None
    ) -> tuple[list[dict[str, Any]], str]:
        """Fetch threads and events, build export rows.

        Returns (rows, run_part_for_filename).
        contact_id_filter: if set, export only threads with this contact_id.
        """
        jims_app = await get_jims_app()
        async with jims_app.sessionmaker() as session:
            base_threads = sa.select(ThreadDB).where(self._eval_threads_clause())
            if contact_id_filter:
                base_threads = base_threads.where(ThreadDB.contact_id == contact_id_filter)
            elif self.selected_run_id and self.selected_run_id != "All":
                selected_raw = self.run_id_lookup.get(self.selected_run_id)
                if selected_raw:
                    base_threads = base_threads.where(ThreadDB.contact_id == selected_raw)
            if self.selected_tests_scenario and self.selected_tests_scenario != "All":
                base_threads = base_threads.where(
                    ThreadDB.thread_config.contains({"question_scenario": self.selected_tests_scenario})
                )
            base_threads = base_threads.order_by(ThreadDB.created_at.desc())

            threads = (await session.execute(base_threads)).scalars().all()
            if not threads:
                return [], "all"

            thread_ids = [t.thread_id for t in threads]
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
        for thread in threads:
            cfg = thread.thread_config or {}
            evs = events_by_thread.get(thread.thread_id, [])

            answer = ""
            status = "—"
            judge_comment = ""
            rating_label = ""
            gds_question = ""
            gds_answer = ""
            pipeline_model = ""
            judge_cost: float | str = ""
            row_pipeline_cost: float = 0.0
            test_date = self._format_run_label_with_name(thread.contact_id, cfg)
            meta_payload: dict[str, Any] = {}
            trace_events: list[dict[str, Any]] = []

            for ev in evs:
                trace_events.append(
                    {
                        "created_at": ev.created_at.isoformat() if ev.created_at else None,
                        "event_type": ev.event_type,
                        "event_data": ev.event_data,
                    }
                )
                if ev.event_type == "comm.assistant_message":
                    answer = str(ev.event_data.get("content", ""))
                elif ev.event_type == "rag.query_processed":
                    tech = ev.event_data.get("technical_info", {}) if isinstance(ev.event_data, dict) else {}
                    if isinstance(tech, dict):
                        model_stats = tech.get("model_stats") or {}
                        if isinstance(model_stats, dict):
                            for stats in model_stats.values():
                                if isinstance(stats, dict):
                                    cost_val = stats.get("requests_cost")
                                    try:
                                        if cost_val is not None:
                                            row_pipeline_cost += float(cost_val)
                                    except (TypeError, ValueError):
                                        pass
                elif ev.event_type == "eval.result":
                    ed = ev.event_data
                    status = ed.get("test_status", status)
                    judge_comment = ed.get("eval_judge_comment", judge_comment)
                    rating_label = str(ed.get("eval_judge_rating", rating_label))
                    test_date = self._format_run_label(ed.get("test_date", test_date))
                    gds_question = str(ed.get("gds_question") or "")
                    gds_answer = str(ed.get("gds_answer") or "")
                    pipeline_model = str(ed.get("pipeline_model") or "")
                    try:
                        jc = ed.get("judge_cost")
                        if jc is not None:
                            judge_cost = float(jc)
                    except (TypeError, ValueError):
                        pass
                elif ev.event_type == "eval.meta" and isinstance(ev.event_data, dict):
                    meta_payload = ev.event_data

            run_config = meta_payload.get("run_config", {}) if isinstance(meta_payload, dict) else {}
            judge_meta = meta_payload.get("judge", {}) if isinstance(meta_payload, dict) else {}
            graph_meta = meta_payload.get("graph", {}) if isinstance(meta_payload, dict) else {}
            dm_meta = meta_payload.get("data_model", {}) if isinstance(meta_payload, dict) else {}

            rows.append(
                {
                    "thread_id": str(thread.thread_id),
                    "test_date": test_date,
                    "run_id": str(thread.contact_id or ""),
                    "run_label": self._format_run_label_with_name(thread.contact_id, cfg),
                    "test_run_name": cfg.get("test_run_name") or meta_payload.get("test_run_name") or "",
                    "question_scenario": cfg.get("question_scenario") or "",
                    "gds_question": gds_question,
                    "gds_answer": gds_answer,
                    "llm_answer": answer,
                    "test_status": status or "",
                    "eval_judge_rating": rating_label,
                    "eval_judge_comment": judge_comment,
                    "judge_cost": judge_cost,
                    "pipeline_cost": row_pipeline_cost,
                    "pipeline_model": pipeline_model or run_config.get("pipeline_model") or "",
                    "embeddings_model": run_config.get("embeddings_model") or "",
                    "embeddings_dim": run_config.get("embeddings_dim") or "",
                    "judge_model": judge_meta.get("judge_model") or "",
                    "judge_prompt_id": judge_meta.get("judge_prompt_id") or "",
                    "graph_nodes_by_label": graph_meta.get("nodes_by_label") or {},
                    "graph_edges_by_type": graph_meta.get("edges_by_type") or {},
                    "dm_id": dm_meta.get("dm_id") or "",
                    "dm_description": dm_meta.get("dm_description") or "",
                    "rag_query_processed_trace": (
                        json.dumps(trace_events, ensure_ascii=False, default=str) if trace_events else ""
                    ),
                }
            )

        run_part = "all"
        if contact_id_filter:
            run_part = contact_id_filter.replace(":", "-").replace("/", "-")
        elif self.selected_run_id and self.selected_run_id != "All":
            raw = self.run_id_lookup.get(self.selected_run_id, self.selected_run_id)
            run_part = str(raw).replace(":", "-").replace("/", "-") or "all"
        return rows, run_part

    async def export_tests_xlsx(self):
        """Export test results matching active filters to .xlsx."""
        rows, run_part = await self._build_export_rows()
        if not rows:
            return rx.toast.info("No test results to export.")
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"eval_results_{run_part}_{ts}.xlsx"
        return rx.download(data=xlsx_bytes(rows, list(self._EXPORT_COLUMNS)), filename=filename)

    async def export_compare_a_xlsx(self):
        """Export run A from the compare dialog to .xlsx."""
        contact_id = self._resolve_run_contact(self.compare_run_a)
        if not contact_id:
            return rx.toast.error("No run selected for A.")
        rows, run_part = await self._build_export_rows(contact_id_filter=contact_id)
        if not rows:
            return rx.toast.info("No test results for run A.")
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        return rx.download(data=xlsx_bytes(rows, list(self._EXPORT_COLUMNS)), filename=f"eval_{run_part}_{ts}.xlsx")

    async def export_compare_b_xlsx(self):
        """Export run B from the compare dialog to .xlsx."""
        contact_id = self._resolve_run_contact(self.compare_run_b)
        if not contact_id:
            return rx.toast.error("No run selected for B.")
        rows, run_part = await self._build_export_rows(contact_id_filter=contact_id)
        if not rows:
            return rx.toast.info("No test results for run B.")
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        return rx.download(data=xlsx_bytes(rows, list(self._EXPORT_COLUMNS)), filename=f"eval_{run_part}_{ts}.xlsx")

    def _resolve_run_contact(self, label: str) -> str:
        if not label:
            return ""
        return self.run_id_lookup.get(label, label)

    def set_compare_judge_prompt_compact(self, checked: bool) -> None:
        self.compare_judge_prompt_compact = bool(checked)

    def set_compare_dm_compact(self, checked: bool) -> None:
        self.compare_dm_compact = bool(checked)

    def compare_runs(self):
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
        yield type(self).compare_runs_background(run_a_id, run_b_id)

    @rx.event(background=True)  # type: ignore[operator]
    async def compare_runs_background(self, run_a_id: str, run_b_id: str):
        try:
            jims_app = await get_jims_app()
            async with jims_app.sessionmaker() as session:
                threads_res = (
                    (
                        await session.execute(
                            sa.select(ThreadDB)
                            .where(
                                sa.and_(
                                    self._eval_threads_clause(),
                                    ThreadDB.contact_id.in_([run_a_id, run_b_id]),
                                )
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

            meta_a = run_a_data.meta
            meta_b = run_b_data.meta
            prompt_a = meta_a.judge.judge_prompt
            prompt_b = meta_b.judge.judge_prompt

            dm_a_str = meta_a.dm.dm_description
            dm_b_str = meta_b.dm.dm_description

            async with self:
                prompt_diff_rows = self._build_side_by_side_diff(prompt_a, prompt_b)
                dm_diff_rows = self._build_side_by_side_diff(dm_a_str, dm_b_str)

                self.compare_summary_a = run_a_data.summary
                self.compare_summary_b = run_b_data.summary
                self.compare_diff_keys = diff_keys
                self.compare_rows = [asdict(r) for r in aligned_rows]
                self.compare_prompt_diff_rows = prompt_diff_rows
                self.compare_dm_diff_rows = dm_diff_rows
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

    def open_jims_thread(self, thread_id: str):
        """Open the JIMS page and preselect the given eval thread."""
        tid = str(thread_id or "").strip()
        if not tid:
            return
        yield ThreadViewState.select_thread(thread_id=tid)  # type: ignore[operator]
        yield rx.redirect("/jims")  # type: ignore[operator]

    @rx.var
    def tests_page_display(self) -> str:
        total_pages = (self.tests_total_rows - 1) // self.tests_page_size + 1 if self.tests_total_rows > 0 else 1
        return f"Page {self.tests_page + 1} of {total_pages}"

    @rx.var
    def tests_rows_display(self) -> str:
        if self.tests_total_rows == 0:
            return "No rows"
        start = self.tests_page * self.tests_page_size + 1
        end = min(start + self.tests_page_size - 1, self.tests_total_rows)
        return f"Rows {start}-{end} of {self.tests_total_rows}"

    @rx.var
    def tests_has_next(self) -> bool:
        max_page = (self.tests_total_rows - 1) // self.tests_page_size if self.tests_total_rows > 0 else 0
        return self.tests_page < max_page

    @rx.var
    def tests_has_prev(self) -> bool:
        return self.tests_page > 0

    def _append_progress(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.run_progress = [*self.run_progress[-20:], f"[{stamp}] {message}"]

    def _record_val(self, rec: Any, key: str) -> Any:
        """Best-effort extractor for sqlalchemy / driver records."""
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

    def _build_data_model_meta(self) -> dict[str, Any]:
        return {
            "dm_id": self.dm_id,
            "dm_description": self.dm_description,
        }

    async def _build_eval_meta_payload(self, test_run_id: str, test_run_name: str) -> dict[str, Any]:
        """Build a single eval.meta payload shared across threads for a run."""
        judge_meta = JudgeMeta(
            judge_model=self.judge_model,
            judge_prompt_id=self.judge_prompt_id,
            judge_prompt=self.judge_prompt,
        )
        run_config = RunConfig(
            pipeline_model=self.pipeline_model,
            embeddings_model=self.default_embeddings_model,
            embeddings_dim=self.embeddings_dim,
        )
        payload: dict[str, Any] = {
            "meta_version": 1,
            "test_run": test_run_id,
            "test_run_name": test_run_name,
            "run_config": asdict(run_config),
            "judge": asdict(judge_meta),
        }
        payload.update(await self._collect_eval_meta_sections())
        return payload

    def _format_tool_calls(self, technical_info: dict[str, Any]) -> str:
        if not isinstance(technical_info, dict):
            return ""
        tool_calls = technical_info.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            parts: list[str] = []
            for idx, call in enumerate(tool_calls, start=1):
                if not isinstance(call, dict):
                    parts.append(str(call))
                    continue
                tool = str(call.get("tool") or "tool")
                args = call.get("args") or {}
                result = str(call.get("result") or "")
                args_text = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
                parts.append(f"[{idx}] {tool}\nArgs: {args_text}\nResult:\n{result}")
            return "\n\n".join(parts).strip()
        vts = technical_info.get("vts_queries") or []
        cypher = technical_info.get("cypher_queries") or []
        vts_s = "\n".join([str(v) for v in vts]) if isinstance(vts, list) else ""
        cypher_s = "\n".join([str(c) for c in cypher]) if isinstance(cypher, list) else ""
        return "\n---\n".join(part for part in [vts_s, cypher_s] if part).strip()

    def _format_run_label(self, contact_id: str | None) -> str:
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
        name = cfg["test_run_name"] if isinstance(cfg, dict) and "test_run_name" in cfg else ""
        base = self._format_run_label(contact_id)
        if name:
            return f"{name} — {base}"
        return base

    def _normalize_diff_val(self, val: Any) -> Any:
        if isinstance(val, (dict, list)):
            try:
                return json.dumps(val, sort_keys=True)
            except Exception:
                return str(val)
        return val

    def _build_side_by_side_diff(self, a_text: str, b_text: str) -> list[DiffRow]:
        a_lines = a_text.splitlines()
        b_lines = b_text.splitlines()
        sm = difflib.SequenceMatcher(None, a_lines, b_lines)
        rows: list[DiffLine] = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for al, bl in zip(a_lines[i1:i2], b_lines[j1:j2]):
                    rows.append(
                        DiffLine(
                            left=al, right=bl, op=tag,
                            left_color="inherit", right_color="inherit",
                            strong=False, row_idx=0, is_change=False,
                        )
                    )
            elif tag == "replace":
                max_len = max(i2 - i1, j2 - j1)
                for k in range(max_len):
                    al = a_lines[i1 + k] if i1 + k < i2 else ""
                    bl = b_lines[j1 + k] if j1 + k < j2 else ""
                    rows.append(
                        DiffLine(
                            left=al, right=bl, op=tag,
                            left_color="var(--indigo-11)", right_color="var(--indigo-11)",
                            strong=True, row_idx=0, is_change=True,
                        )
                    )
            elif tag == "delete":
                for al in a_lines[i1:i2]:
                    rows.append(
                        DiffLine(
                            left=al, right="", op=tag,
                            left_color="var(--red-11)", right_color="inherit",
                            strong=True, row_idx=0, is_change=True,
                        )
                    )
            elif tag == "insert":
                for bl in b_lines[j1:j2]:
                    rows.append(
                        DiffLine(
                            left="", right=bl, op=tag,
                            left_color="inherit", right_color="var(--green-11)",
                            strong=True, row_idx=0, is_change=True,
                        )
                    )
        for idx, row in enumerate(rows):
            row.row_idx = idx
            row.is_change = row.op != "equal"
        return [cast(DiffRow, asdict(r)) for r in rows]

    def _diff_config_keys(self, cfg_a: dict[str, object], cfg_b: dict[str, object]) -> list[str]:
        keys = set(cfg_a.keys()) | set(cfg_b.keys())
        diffs: list[str] = []
        for key in keys:
            left = cfg_a[key] if key in cfg_a else None
            right = cfg_b[key] if key in cfg_b else None
            if self._normalize_diff_val(left) != self._normalize_diff_val(right):
                diffs.append(key)
        return sorted(diffs)

    def _summarize_config_for_display(
        self, meta: RunMeta, cfg_fallback: dict[str, Any], *, run_contact_id: str = ""
    ) -> RunConfigSummary:
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

        test_run_name = str(cfg_fallback.get("test_run_name", "") or "")

        return RunConfigSummary(
            test_run_id=run_contact_id,
            test_run_name=test_run_name,
            pipeline_model=pipeline_model,
            embeddings_model=embeddings_model,
            embeddings_dim=embeddings_dim,
            judge_model=judge_model,
            judge_prompt_id=judge_prompt_id,
            judge_prompt_hash=judge_prompt_hash,
            dm_hash=str(cfg_fallback.get("dm_hash", "")),
            dm_id=dm.dm_id,
            graph_nodes=graph.nodes_by_label,
            graph_edges=graph.edges_by_type,
        )

    def _compact_diff_rows(self, rows: list[DiffRow], compact: bool) -> list[DiffRow]:
        if not compact:
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
    def compare_prompt_rows_view(self) -> list[DiffRow]:
        return self._compact_diff_rows(self.compare_prompt_diff_rows or [], self.compare_judge_prompt_compact)

    @rx.var
    def compare_dm_rows_view(self) -> list[DiffRow]:
        return self._compact_diff_rows(self.compare_dm_diff_rows or [], self.compare_dm_compact)

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
        )
        return RunMeta(graph=graph, judge=judge, run_config=run_config, dm=dm)

    def _collect_run_data(
        self,
        run_id: str,
        threads: list[ThreadDB],
        events_by_thread: dict[UUID, list[ThreadEventDB]],
    ) -> RunData:
        results_by_question: dict[str, QuestionResult] = {}
        meta_sample: dict[str, Any] = {}
        cfg_sample: dict[str, Any] = {}
        total = 0
        passed = 0
        failed = 0
        cost_total = 0.0
        judge_cost_total = 0.0
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
            question_text = ""
            golden_answer = ""
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
                    try:
                        jc = ev.event_data.get("judge_cost")
                        if jc is not None:
                            judge_cost_total += float(jc)
                    except (TypeError, ValueError):
                        pass

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
            "judge_cost_total": round(judge_cost_total, 6),
            "test_run_name": meta_sample["test_run_name"]
            if "test_run_name" in meta_sample
            else (cfg_sample["test_run_name"] if "test_run_name" in cfg_sample else ""),
            "avg_answer_time_sec": round(sum(answer_times) / len(answer_times), 2) if answer_times else 0.0,
            "median_answer_time_sec": round(statistics.median(answer_times), 2) if answer_times else 0.0,
        }

        run_meta = self._parse_run_meta(meta_sample, cfg_sample)
        config_summary = self._summarize_config_for_display(run_meta, cfg_sample, run_contact_id=run_id)

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
            ("Graph nodes", "graph_nodes"),
            ("Graph edges", "graph_edges"),
        ]:
            val = _as_text(cfg[key]) if key in cfg else "—"
            rows.append({"label": label, "value": val, "diff": key in diff_keys})
        return rows

    async def _run_question_thread(
        self,
        jims_app: Any,
        question_row: dict[str, Any],
        test_run_name: str,
        test_run_id: str,
        eval_meta_base: dict[str, Any],
    ) -> tuple[str, str, dict[str, Any]]:
        """Create a JIMS thread, post the question, run pipeline, return answer + tech info."""
        thread_id = uuid7()
        ctl = await ThreadController.new_thread(
            jims_app.sessionmaker,
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
        pipeline = jims_app.pipeline
        await self._apply_pipeline_settings(pipeline)

        ctx = await ctl.make_context(llm_settings=await self._make_pipeline_llm_settings())
        events = await ctl.run_pipeline_with_context(pipeline, ctx)

        answer: str = ""
        technical_info: dict[str, Any] = {}
        for ev in events:
            if ev.event_type == "comm.assistant_message":
                answer = str(ev.event_data.get("content", ""))
            elif ev.event_type == "rag.query_processed":
                technical_info = dict(ev.event_data.get("technical_info", {}))

        return str(thread_id), answer, technical_info

    async def _judge_answer(
        self, question_row: dict[str, Any], answer: str, tool_calls: str
    ) -> tuple[str, str, int, float]:
        """Judge model answer with current judge prompt/model and rating."""
        judge_prompt = self.judge_prompt
        if not judge_prompt:
            return "fail", "Judge prompt not loaded", 0, 0.0

        provider = LLMProvider(settings=await self._make_judge_llm_settings())

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
            return "fail", f"Judge failed: {e}", 0, 0.0

        if res is None:
            return "fail", "", 0, 0.0

        try:
            rating = int(res.rating)
        except Exception:
            rating = 0

        judge_cost = sum(u.requests_cost for u in provider.usage.values())
        return res.test_status or "fail", res.comment or "", rating, judge_cost

    async def _store_eval_result_event(self, thread_id: str, result_row: dict[str, Any]) -> None:
        try:
            tid = UUID(str(thread_id))
        except Exception:
            return

        jims_app = await get_jims_app()
        ctl = await ThreadController.from_thread_id(jims_app.sessionmaker, tid)
        if ctl is None:
            return

        data = dict(result_row)
        data.pop("thread_id", None)
        await ctl.store_event_dict(uuid7(), "eval.result", data)

    def run_selected_tests(self):
        """Trigger test run - validates, sets loading state and starts background task."""
        if self.is_running:
            return

        selection = [str(q) for q in (self.selected_question_ids or []) if str(q)]
        if not selection:
            self.error_message = "Select at least one question to run tests."
            return
        if not self.judge_prompt:
            self.error_message = "Judge prompt not loaded."
            return

        test_run_name = self.test_run_name.strip() or ""

        self.is_running = True
        self.current_question_index = -1
        self.total_questions_to_run = len(selection)
        self.status_message = f"Evaluation run '{test_run_name}' for {len(selection)} question(s)…"
        self.run_progress = []
        self.error_message = ""
        yield
        yield type(self).run_selected_tests_background()

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
            resolved_pipeline_model = self.pipeline_model

        jims_app = await get_jims_app()
        async with self:
            eval_meta_base = await self._build_eval_meta_payload(test_run_id, test_run_name)
            max_parallel = max(1, int(self.max_parallel_tests or 1))
            embeddings_model_for_result = self.default_embeddings_model
            embeddings_dim_for_result = self.embeddings_dim

        sem = asyncio.Semaphore(max_parallel)

        async def _run_one(question: str) -> dict[str, Any]:
            async with sem:
                row = question_map.get(str(question or "").strip())
                if row is None:
                    return {"question": question, "status": None, "error": "not found"}
                try:
                    thread_id, answer, tech = await self._run_question_thread(
                        jims_app, row, test_run_name, test_run_id, eval_meta_base
                    )
                    tool_calls = self._format_tool_calls(tech)
                    async with self:
                        status, comment, rating, judge_cost = await self._judge_answer(row, answer, tool_calls)

                    pipeline_cost = sum(
                        float(v)
                        for stats in (tech.get("model_stats") or {}).values()
                        if isinstance(stats, dict)
                        for k, v in stats.items()
                        if k == "requests_cost" and v is not None
                    )

                    result_row = {
                        "judge_model": self.judge_model,
                        "judge_prompt_id": self.judge_prompt_id,
                        "dm_id": self.dm_id,
                        "pipeline_model": resolved_pipeline_model,
                        "embeddings_model": embeddings_model_for_result,
                        "embeddings_dim": embeddings_dim_for_result,
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
                        "judge_cost": judge_cost,
                        "pipeline_cost": pipeline_cost,
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
            yield

        async with self:
            self.status_message = f"Evaluation complete: {completed} of {len(selection)} question(s) processed"
            self.current_question_index = -1
            self.total_questions_to_run = 0
            self.tests_page = 0
            try:
                await self._load_tests(select_contact_id=test_run_id)
            except Exception as e:
                logging.warning(f"Failed to reload test runs after evaluation: {e}", exc_info=True)
            self.is_running = False
        yield

    def load_eval_data(self):
        """Connecting button with a background task. Used to trigger animations properly."""
        if self.loading:
            return
        self.loading = True
        self.error_message = ""
        self.status_message = ""
        self.tests_page = 0
        yield
        yield type(self).load_eval_data_background()

    async def mount(self):
        if EVAL_ENABLED:
            if DEBUG_MODE:
                yield DebugState.load_available_models()

    @rx.event(background=True)  # type: ignore[operator]
    async def load_eval_data_background(self):
        try:
            async with self:
                self.available_models = await self.get_var_value(DebugState.available_models)
                self._sync_available_models()
                self._sync_judge_model()
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
