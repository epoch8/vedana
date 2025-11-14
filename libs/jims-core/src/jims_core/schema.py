from typing import Protocol

from jims_core.thread.thread_context import ThreadContext


class Pipeline(Protocol):
    async def __call__(self, ctx: ThreadContext) -> None:
        pass
