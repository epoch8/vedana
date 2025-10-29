from typing import Protocol
from dataclasses import field, dataclass

from jims_core.thread.thread_context import ThreadContext


class Pipeline(Protocol):
    async def __call__(self, ctx: ThreadContext) -> None:
        pass


@dataclass
class Orchestrator:
    pipelines: dict[str, Pipeline] = field(default_factory=dict)

    def register_pipeline(self, name: str, pipeline: Pipeline) -> None:
        self.pipelines[name] = pipeline

    def route(self, ctx: ThreadContext) -> Pipeline:  # getter
        pass

    async def orchestrate(self, ctx: ThreadContext) -> None:  # entrypoint
        pass
