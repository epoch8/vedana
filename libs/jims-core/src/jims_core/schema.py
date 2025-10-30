from typing import Protocol, Generic
from dataclasses import field, dataclass

from jims_core.thread.thread_context import ThreadContext
from jims_core.thread.schema import PipelineState, TState


class Pipeline(Protocol):
    async def __call__(self, ctx: ThreadContext) -> None:
        pass


@dataclass
class Orchestrator(Generic[TState]):
    pipelines: dict[str, Pipeline] = field(default_factory=dict)
    state: TState = field(default_factory=PipelineState)  # used only for typing / passing model for validation

    def register_pipeline(self, name: str, pipeline: Pipeline) -> None:
        self.pipelines[name] = pipeline

    def route(self, ctx: ThreadContext) -> Pipeline:  # getter
        pass

    async def orchestrate(self, ctx: ThreadContext) -> None:  # entrypoint
        pass
