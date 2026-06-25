"""Vedana backoffice Reflex app.

Extends ``jims_backoffice`` with ETL and main-dashboard pages. Wiring is fully
driven by two env vars resolved in :mod:`vedana_backoffice.project_runtime`:

* ``VEDANA_APP`` — factory returning :class:`vedana_core.app.VedanaApp` (default
  ``vedana_core.app:make_vedana_app``).
* ``DATAPIPE_PIPELINE`` — datapipe ETL app path (default ``vedana_etl.app``; same as
  ``datapipe run`` / ``datapipe api``).

The JimsApp consumed by ``jims_backoffice`` is :attr:`VedanaApp.jims_app`, so
no separate ``JIMS_APP`` is needed.
"""

from __future__ import annotations

import jims_backoffice
import reflex as rx

from vedana_backoffice.pages.chat import page as chat_page
from vedana_backoffice.pages.etl import page as etl_page
from vedana_backoffice.pages.eval import page as eval_page
from vedana_backoffice.pages.jims_thread_list_page import ThreadListState, jims_thread_list_page
from vedana_backoffice.pages.main_dashboard import page as main_dashboard_page
from vedana_backoffice.project_runtime import get_vedana_app
from vedana_backoffice.states.chat import ChatState
from vedana_backoffice.states.etl import EtlState, warm_static_pipeline_metadata
from vedana_backoffice.states.eval import EvalState
from vedana_backoffice.states.main_dashboard import DashboardState
from vedana_mcp.auth import McpAuthSettings


async def _resolve_jims_app():
    return (await get_vedana_app()).jims_app


jims_backoffice.register_app(_resolve_jims_app())
jims_backoffice.register_chat_state(ChatState)
jims_backoffice.register_eval_state(EvalState)


warm_static_pipeline_metadata()

app = rx.App(stylesheets=["/styles.css"])

if McpAuthSettings().enable_mcp:
    from vedana_mcp.mcp import mcp as _vedana_mcp
    import vedana_mcp.tools_rag  # noqa: F401
    import vedana_mcp.tools_graph  # noqa: F401

    mcp_app = _vedana_mcp.http_app(path="/")
    app.register_lifespan_task(mcp_app.lifespan)
    assert app._api is not None, "Reflex app._api is not initialized"
    app._api.mount("/mcp", mcp_app)

app.add_page(main_dashboard_page, route="/", title="Vedana Backoffice", on_load=DashboardState.load_dashboard)
app.add_page(etl_page, route="/etl", title="ETL", on_load=EtlState.load_pipeline_metadata)
app.add_page(chat_page, route="/chat", title="Chat", on_load=ChatState.reset_session)
app.add_page(jims_thread_list_page, route="/jims", title="JIMS", on_load=ThreadListState.get_data)
app.add_page(eval_page, route="/eval", title="Evaluation", on_load=EvalState.load_eval_data)
