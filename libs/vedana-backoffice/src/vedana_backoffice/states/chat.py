import asyncio
import logging
from typing import Any
from uuid import UUID

import jims_backoffice
import reflex as rx
from datapipe.compute import Catalog, run_pipeline
from jims_backoffice.states.common import DEBUG_MODE, MemLogger
from jims_core.llms.llm_provider import LLMSettings
from jims_core.thread.thread_controller import ThreadController
from jims_core.util import uuid7
from vedana_core.settings import settings as core_settings
from vedana_etl.pipeline import get_data_model_pipeline

from vedana_backoffice.project_runtime import get_etl_bindings, get_vedana_app
from vedana_backoffice.states.common import datapipe_log_capture


class ChatState(jims_backoffice.ChatState, rx.State):
    """Vedana chat state.

    Adds Vedana-specific knobs on top of the JIMS base:
      * Data-model filtering checkbox + filter-model select.
      * "View / Reload Data Model" controls.
      * Pipeline log capture via :class:`MemLogger` so each assistant message
        can show a "Logs" panel.
    """

    model: str = core_settings.model
    default_models: list[str] = list({core_settings.model, core_settings.filter_model})
    available_models: list[str] = default_models
    enable_dm_filtering: bool = core_settings.enable_dm_filtering
    dm_filter_model: str = core_settings.filter_model
    data_model_text: str = ""
    is_refreshing_dm: bool = False

    def set_enable_dm_filtering(self, value: bool) -> None:
        self.enable_dm_filtering = value

    def set_dm_filter_model(self, value: str) -> None:
        self.dm_filter_model = value

    def _sync_model(self) -> None:
        super()._sync_model()
        self._sync_dm_filter_model()

    def _sync_dm_filter_model(self) -> None:
        if self.dm_filter_model not in self.available_models and self.available_models:
            for model in self.available_models:
                if model.endswith(core_settings.filter_model):
                    self.dm_filter_model = model
                    break
            else:
                if "openrouter/openrouter/free" in self.available_models:
                    self.dm_filter_model = "openrouter/openrouter/free"
                else:
                    self.dm_filter_model = self.available_models[0]

    async def _run_message(self, thread_id: str, user_text: str):
        """Run the pipeline against the Vedana app, capturing logs for the UI."""
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

        await ctl.store_user_message(uuid7(), user_text)

        pipeline = vedana_app.pipeline
        pipeline.logger = mem_logger
        pipeline.model = self.model
        pipeline.enable_filtering = self.enable_dm_filtering
        pipeline.filter_model = self.dm_filter_model

        embeddings_model = (
            core_settings.embeddings_model
            if not DEBUG_MODE
            else await self.get_var_value(jims_backoffice.DebugState.embeddings_model)  # type: ignore[arg-type]
        )

        ctx = await ctl.make_context(
            llm_settings=LLMSettings(model=self.model, embeddings_model=embeddings_model)  # type: ignore
        )
        events = await ctl.run_pipeline_with_context(pipeline, ctx)

        answer: str = ""
        tech: dict[str, Any] = {}
        for ev in events:
            if ev.event_type == "comm.assistant_message":
                answer = str(ev.event_data.get("content", ""))
            elif ev.event_type == "rag.query_processed":
                tech = dict(ev.event_data.get("technical_info", {}))

        return answer, tech, mem_logger.get_logs()

    @rx.event(background=True)  # type: ignore[operator]
    async def load_data_model_text(self):
        async with self:
            va = await get_vedana_app()
            try:
                self.data_model_text = await va.data_model.to_text_descr()
            except Exception:
                self.data_model_text = "(failed to load data model text)"
            self.is_refreshing_dm = False
            yield

    def reload_data_model(self):
        if self.is_refreshing_dm:
            return
        self.is_refreshing_dm = True
        yield
        yield ChatState.reload_data_model_background()

    @rx.event(background=True)  # type: ignore[operator]
    async def reload_data_model_background(self):
        error_msg = ""
        try:

            def _run_dm_pipeline():
                etl_app = get_etl_bindings().app
                with datapipe_log_capture():
                    run_pipeline(etl_app.ds, Catalog({}), get_data_model_pipeline())

            await asyncio.to_thread(_run_dm_pipeline)
            async with self:
                va = await get_vedana_app()
                self.data_model_text = await va.data_model.to_text_descr()
            yield rx.toast.success("Data model reloaded")
        except Exception as e:
            error_msg = str(e)
            yield rx.toast.error(f"Failed to reload data model\n{error_msg}")
        finally:
            async with self:
                self.is_refreshing_dm = False
            yield
