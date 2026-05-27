"""Project-specific app + ETL bindings for ``vedana-backoffice``.

Configured by two env vars (same conventions as ``jims_api`` / ``datapipe`` CLI):

* ``VEDANA_APP`` — ``module[:attr]`` factory returning :class:`vedana_core.app.VedanaApp`.
  Default ``vedana_core.app:make_vedana_app``.

* ``DATAPIPE_PIPELINE`` — datapipe app import path (``module[:attr]``, default attr
  ``app``). Same variable used by ``datapipe run`` / ``datapipe api``. Examples:
  ``vedana_etl.app``, ``maytoni_etl.app``. The ETL config module is derived as
  ``<package>.config`` (e.g. ``maytoni_etl.config``).
"""

from __future__ import annotations

import asyncio
import functools
import importlib
import os
from dataclasses import dataclass
from typing import Any

from datapipe.cli import load_pipeline
from vedana_core.app import VedanaApp


@dataclass(frozen=True, slots=True)
class EtlBindings:
    """Datapipe ETL bindings resolved from ``DATAPIPE_PIPELINE``."""

    app: Any  # datapipe_app.DatapipeAPI
    pipeline: Any  # datapipe.compute.Pipeline
    dbconn: Any  # datapipe.store.database.DBConn (.con is the SQLAlchemy engine)


@functools.cache
def _vedana_app_factory():
    vedana_app_ref = os.environ.get("VEDANA_APP", "vedana_core.app:make_vedana_app")
    module_name, _, attr = vedana_app_ref.partition(":")
    module = importlib.import_module(module_name)
    fn = getattr(module, attr or "make_vedana_app")
    if not callable(fn):
        raise TypeError(f"VEDANA_APP {fn!r} is not callable")
    return fn


async def get_vedana_app() -> VedanaApp:
    """Return the configured :class:`VedanaApp`. Cached by the factory itself."""
    res = _vedana_app_factory()()
    app = await res if asyncio.iscoroutine(res) else res
    if not isinstance(app, VedanaApp):
        raise TypeError(f"VEDANA_APP factory must return a VedanaApp, got {type(app).__name__!r}")
    return app


@functools.cache
def get_etl_bindings() -> EtlBindings:
    """Resolve Datapipe ETL from ``DATAPIPE_PIPELINE`` (shared with the datapipe CLI)."""
    ref = os.environ.get("DATAPIPE_PIPELINE", "vedana_etl.app")
    etl_app = load_pipeline(ref)
    cfg_mod = importlib.import_module(f"{ref.split('.')[0]}.config")
    return EtlBindings(app=etl_app, pipeline=etl_app.pipeline, dbconn=cfg_mod.DBCONN_DATAPIPE)
