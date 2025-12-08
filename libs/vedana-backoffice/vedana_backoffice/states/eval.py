import logging
from datetime import datetime
from typing import Any
from uuid import UUID

import pandas as pd
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

    loading: bool = False
    error_message: str = ""
    status_message: str = ""
    eval_gds_rows: list[dict[str, Any]] = []
    selected_question_ids: list[str] = []
    selected_scenario: str = "all"  # Filter by scenario
    judge_configs: list[dict[str, Any]] = []
    selected_judge_model: str = ""
    judge_prompt_id: str = ""
    selected_judge_prompt: str = ""
    pipeline_model: str = core_settings.model
    embeddings_model: str = core_settings.embeddings_model
    embeddings_dim: int = core_settings.embeddings_dim
    dm_id: str = ""
    dm_snapshot_updated: str = ""
    tests_rows: list[dict[str, Any]] = []
    tests_cost_total: float = 0.0
    run_passed: int = 0
    run_failed: int = 0
    selected_run_id: str = ""
    run_id_options: list[str] = []
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
        rows: list[dict[str, Any]] = []
        for row in self.eval_gds_rows or []:
            # Apply scenario filter
            if self.selected_scenario != "all":
                scenario = row.get("question_scenario")
                if str(scenario) != self.selected_scenario:
                    continue
            enriched = dict(row)
            enriched["selected"] = row.get("id") in selected
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
            return f"Select tests to run"
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
        self._prune_selection()

    def set_judge_model(self, value: str) -> None:
        value = str(value or "")
        for cfg in self.judge_configs or []:
            if cfg.get("judge_model") == value:
                self.selected_judge_model = value
                self.judge_prompt_id = cfg.get("judge_prompt_id", "")
                self.selected_judge_prompt = cfg.get("judge_prompt", "")
                break

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
            logging.exception(f"Failed to load eval_judge_config: {exc}")
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

            dm_row = (await session.execute(dm_stmt)).mappings().first()

        if pipeline_rows:
            self.pipeline_model = safe_render_value(pipeline_rows[-1])

        if dm_row:
            self.dm_id = safe_render_value(dm_row.get("dm_id"))
            self.dm_description = safe_render_value(dm_row.get("dm_description"))
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
        """Build test results directly from JIMS threads (source=eval)."""
        vedana_app = await get_vedana_app()

        # Fetch eval threads (latest first)
        async with vedana_app.sessionmaker() as session:
            stmt_threads = sa.select(ThreadDB).order_by(ThreadDB.created_at.desc())
            threads_res = (await session.execute(stmt_threads)).scalars().all()

        eval_threads = [
            t for t in threads_res if isinstance(t.thread_config, dict) and t.thread_config.get("source") == "eval"
        ]

        # Run id: parse date from contact_id, keep unique order
        seen = set()
        ordered_run_ids = []
        for t in eval_threads:
            rid = t.contact_id
            if rid not in seen:
                ordered_run_ids.append(rid)
                seen.add(rid)

        self.run_id_options = ordered_run_ids
        if not self.selected_run_id and ordered_run_ids:
            self.selected_run_id = ordered_run_ids[0]

        if self.selected_run_id:
            eval_threads = [
                t
                for t in eval_threads
                if str(t.contact_id or t.thread_config.get("test_run") or "").strip() == self.selected_run_id
            ]

        self.tests_total_rows = len(eval_threads)
        offset = self.tests_page * self.tests_page_size
        end = offset + self.tests_page_size
        page_threads = eval_threads[offset:end]

        if not page_threads:
            self.tests_rows = []
            self.tests_cost_total = 0.0
            return

        thread_ids = [t.thread_id for t in page_threads]

        async with vedana_app.sessionmaker() as session:
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
            tool_calls = ""
            run_label = self._format_run_label(thread.contact_id)
            test_date = run_label

            for ev in evs:
                if ev.event_type == "comm.assistant_message":
                    answer = str(ev.event_data.get("content", ""))
                elif ev.event_type == "rag.query_processed":
                    tool_calls = self._format_tool_calls(ev.event_data.get("technical_info", {}))
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
                    test_date = self._format_run_label(ev.event_data.get("test_date", test_date))

            if str(status).lower() == "pass":
                passed += 1
            elif str(status).lower() == "fail":
                failed += 1

            cfg = thread.thread_config or {}
            rows.append(
                {
                    "test_date": safe_render_value(test_date),
                    "gds_question": safe_render_value(cfg.get("gds_question")),
                    "llm_answer": safe_render_value(answer),
                    "gds_answer": safe_render_value(cfg.get("gds_answer")),
                    "pipeline_model": safe_render_value(cfg.get("pipeline_model")),
                    "test_status": status or "—",
                    "status_color": self._status_color(status),
                    "eval_judge_comment": safe_render_value(judge_comment),
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
            "judge_model": self.selected_judge_model,
            "judge_prompt_id": self.judge_prompt_id,
            "pipeline_model": self.pipeline_model,
            "embeddings_model": self.embeddings_model,
            "embeddings_dim": self.embeddings_dim,
            "dm_id": self.dm_id,
        }

    async def _run_question_thread(
        self,
        question_row: dict[str, Any],
        test_run_name: str,
    ) -> tuple[str, str, dict[str, Any]]:
        """Create a JIMS thread, post the question, run pipeline, return answer + tech info."""
        vedana_app = await get_vedana_app()
        thread_id = uuid7()
        ctl = await ThreadController.new_thread(
            vedana_app.sessionmaker,
            contact_id=f"eval:{test_run_name}",
            thread_id=thread_id,
            thread_config=self._build_thread_config(question_row, test_run_name),
        )

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

    async def _judge_answer(self, question_row: dict[str, Any], answer: str, tool_calls: str) -> tuple[str, str]:
        """Judge model answer with current judge prompt/model."""
        judge_prompt = self.selected_judge_prompt
        if not judge_prompt:
            return "fail", "Judge prompt not loaded"

        provider = LLMProvider()  # todo use single LLMProvider per-thread
        judge_model = self.selected_judge_model
        if judge_model:
            try:
                provider.set_model(judge_model)
            except Exception:
                logging.warning(f"Failed to set judge model {judge_model}")

        class JudgeResult(BaseModel):
            test_status: str = Field(description="pass / fail")
            comment: str = Field(description="justification and hints")
            errors: str | list[str] | None = Field(default=None, description="Text description of errors found")

        user_msg = (
            f"Golden answer:\n{question_row.get('gds_answer', '')}\n\n"
            f"Expected context (if any):\n{question_row.get('question_context', '')}\n\n"
            f"Model answer:\n{answer}\n\n"
            f"Technical info (for reference):\n{tool_calls}"
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
            return "fail", f"Judge failed: {e}"

        if res is None:
            return "fail", ""
        return res.test_status or "fail", res.comment or ""

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

            # Process one question at a time
            completed = 0
            for idx, question in enumerate(selection):
                self.current_question_index = idx
                self.status_message = f"Processing question {idx + 1} of {len(selection)}: '{question[:50]}...'"
                self._append_progress(f"Starting: '{question}'")
                yield  # Yield before starting to update UI

                row = question_map.get(question)
                if row is None:
                    self._append_progress(f"Skipped: '{question}' not found in dataset")
                    yield
                    continue

                try:
                    thread_id, answer, tech = await self._run_question_thread(row, test_run_ts)
                    tool_calls = self._format_tool_calls(tech)
                    status, comment = await self._judge_answer(row, answer, tool_calls)

                    result_row = {
                        "judge_model": self.selected_judge_model,
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
                        "test_date": test_run_ts,
                        "thread_id": thread_id,
                    }

                    await self._store_eval_result_event(thread_id, result_row)

                    completed += 1
                    self._append_progress(f"Completed: '{question}' (status: {status})")
                    self.status_message = f"Completed {completed} of {len(selection)} question(s)"
                except Exception as e:
                    error_msg = f"Failed for '{question}': {e}"
                    self._append_progress(error_msg)
                    logging.error(error_msg, exc_info=True)
                    # Continue with next question instead of stopping

                yield  # Yield after each question to update UI

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
