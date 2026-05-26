from typing import Any, Protocol

from jims_core.thread.thread_context import ThreadContext


class Pipeline(Protocol):
    async def __call__(self, ctx: ThreadContext) -> Any:
        pass
