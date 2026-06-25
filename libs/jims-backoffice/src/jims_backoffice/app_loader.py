"""
JimsApp loader for jims-backoffice
"""

from __future__ import annotations

import asyncio
import os
from typing import Awaitable

from jims_core.app import JimsApp
from jims_core.util import load_jims_app

_registered: JimsApp | Awaitable[JimsApp] | None = None
_resolved: JimsApp | None = None
_lock = asyncio.Lock()


def register_app(app: JimsApp | Awaitable[JimsApp]) -> None:
    """Register the JIMS app used by jims-backoffice state classes."""
    global _registered, _resolved
    _registered = app
    _resolved = None


async def get_jims_app() -> JimsApp:
    """Return the active :class:`JimsApp`, resolving / caching on first call."""
    global _resolved
    if _resolved is not None:
        return _resolved

    async with _lock:
        if _resolved is not None:
            return _resolved

        src: JimsApp | Awaitable[JimsApp] | None = _registered
        if src is None:
            name = os.environ.get("JIMS_APP")
            if not name:
                raise RuntimeError(
                    "No JIMS app registered. Call jims_backoffice.register_app(...) "
                    "at startup or set the JIMS_APP environment variable."
                )
            src = load_jims_app(name)

        _resolved = src if isinstance(src, JimsApp) else await src
        return _resolved
