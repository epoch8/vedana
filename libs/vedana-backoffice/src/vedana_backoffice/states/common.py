"""Vedana-specific shared state utilities.

The generic ``AppVersionState``, ``DebugState``, ``TelegramBotState`` and
``DEBUG_MODE`` / ``EVAL_ENABLED`` flags live in
:mod:`jims_backoffice.states.common`. Project-specific app + ETL loaders live
in :mod:`vedana_backoffice.project_runtime`. This module is just for the
datapipe error-bubbling helper that doesn't belong in either of those.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager


class DatapipeStepError(RuntimeError):
    """Raised when a datapipe step fails without propagating the exception."""


@contextmanager
def datapipe_log_capture():
    """Detect datapipe step failures that are logged but not raised.

    Datapipe catches and logs exceptions in some step types (e.g.
    ``batch_generate``) without re-raising them. This captures ERROR-level log
    messages from the ``datapipe`` logger hierarchy and raises
    :class:`DatapipeStepError` if any errors were recorded once the guarded
    block completes.
    """
    errors: list[str] = []

    class _ErrorCapture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                errors.append(record.getMessage())
            except Exception:
                pass

    handler = _ErrorCapture()
    handler.setLevel(logging.ERROR)

    dp_logger = logging.getLogger("datapipe")
    dp_logger.addHandler(handler)
    try:
        yield errors
    finally:
        dp_logger.removeHandler(handler)
    if errors:
        raise DatapipeStepError(errors[0])
