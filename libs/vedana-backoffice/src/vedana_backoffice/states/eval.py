"""Vedana-specific eval state.

Subclasses :class:`jims_backoffice.EvalState`, replacing the generic data hooks
with Vedana-flavoured implementations:

  * Golden dataset is loaded from the ``eval_gds`` SQL table populated by
    Vedana's ETL.
  * Judge prompt is read from the data model's prompt templates (with a
    fallback to the JIMS default).
  * Pipeline config exposes the data-model hash + description.
  * Graph metadata is collected from Memgraph for the comparison dialog.
  * Adds Grist refresh button and DM filtering knobs (parallel to chat).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any

import jims_backoffice
import reflex as rx
import sqlalchemy as sa
from datapipe.compute import run_steps
from jims_backoffice.states.common import DEBUG_MODE
from jims_backoffice.states.eval import eval_judge_prompt_template
from jims_core.llms.llm_provider import LLMSettings
from vedana_core.settings import settings as core_settings

from vedana_backoffice.project_runtime import get_etl_bindings, get_vedana_app
from vedana_backoffice.states.common import datapipe_log_capture
from vedana_backoffice.util import safe_render_value


class EvalState(jims_backoffice.EvalState, rx.State):
    """Vedana eval state."""

    pipeline_model: str = core_settings.model
    default_embeddings_model: str = core_settings.embeddings_model
    embeddings_dim: int = core_settings.embeddings_dim
    enable_dm_filtering: bool = core_settings.enable_dm_filtering
    available_models: list[str] = list(
        {core_settings.model, core_settings.filter_model, core_settings.judge_model}
    )
    dm_filter_model: str = core_settings.filter_model
    judge_model: str = core_settings.judge_model

    def set_enable_dm_filtering(self, value: bool) -> None:
        self.enable_dm_filtering = value

    def set_dm_filter_model(self, value: str) -> None:
        if value in self.available_models:
            self.dm_filter_model = value

    def _sync_model_field(self, attr: str, preferred_suffix: str) -> None:
        if not self.available_models or getattr(self, attr) in self.available_models:
            return
        fallback = "openrouter/openrouter/free"
        selected = next((model for model in self.available_models if model.endswith(preferred_suffix)), None)
        setattr(self, attr, selected or (fallback if fallback in self.available_models else self.available_models[0]))

    def _sync_available_models(self) -> None:
        self._sync_model_field("pipeline_model", core_settings.model)
        self._sync_model_field("dm_filter_model", core_settings.filter_model)

    def _sync_judge_model(self) -> None:
        self._sync_model_field("judge_model", core_settings.judge_model)

    def get_eval_gds_from_grist(self):
        etl_app = get_etl_bindings().app
        step = next((s for s in etl_app.steps if s._name == "get_eval_gds_from_grist"), None)
        if step is not None:
            with datapipe_log_capture():
                run_steps(etl_app.ds, [step])

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
        if not self.judge_model:
            self.judge_model = core_settings.judge_model
        self.judge_prompt_id = ""
        self.judge_prompt = ""

        vedana_app = await get_vedana_app()
        dm_pt = await vedana_app.data_model.prompt_templates()
        judge_prompt = dm_pt.get("eval_judge_prompt", eval_judge_prompt_template)

        if judge_prompt:
            self.judge_prompt_id = hashlib.sha256(judge_prompt.encode("utf-8")).hexdigest()
            self.judge_prompt = judge_prompt

    async def _load_pipeline_config(self) -> None:
        vedana_app = await get_vedana_app()
        dm = vedana_app.data_model
        self.dm_description = await dm.to_text_descr()
        self.dm_id = hashlib.sha256(self.dm_description.encode("utf-8")).hexdigest()

    async def _collect_eval_meta_sections(self) -> dict[str, Any]:
        sections = await super()._collect_eval_meta_sections()
        sections["graph"] = await self._collect_graph_metadata()
        return sections

    async def _collect_graph_metadata(self) -> dict[str, Any]:
        """Collect node/edge counts from Memgraph."""
        vedana_app = await get_vedana_app()
        graph = vedana_app.graph
        meta: dict[str, Any] = {"nodes_by_label": {}, "edges_by_type": {}}

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

        return meta

    async def _apply_pipeline_settings(self, pipeline: Any) -> None:
        await super()._apply_pipeline_settings(pipeline)
        try:
            pipeline.enable_filtering = self.enable_dm_filtering
        except Exception:
            pass
        try:
            pipeline.filter_model = self.dm_filter_model
        except Exception:
            pass

    async def _make_pipeline_llm_settings(self) -> LLMSettings:
        embeddings_model = (
            core_settings.embeddings_model
            if not DEBUG_MODE
            else await self.get_var_value(jims_backoffice.DebugState.embeddings_model)  # type: ignore[arg-type]
        )
        return LLMSettings(model=self.pipeline_model, embeddings_model=embeddings_model)  # type: ignore

    async def _make_judge_llm_settings(self) -> LLMSettings:
        embeddings_model = (
            core_settings.embeddings_model
            if not DEBUG_MODE
            else await self.get_var_value(jims_backoffice.DebugState.embeddings_model)  # type: ignore[arg-type]
        )
        return LLMSettings(model=self.judge_model, embeddings_model=embeddings_model)  # type: ignore

    def refresh_golden_dataset(self):
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
        self.tests_page = 0
        yield
        yield EvalState.load_eval_data_background()
        yield EvalState.refresh_golden_dataset_background()
