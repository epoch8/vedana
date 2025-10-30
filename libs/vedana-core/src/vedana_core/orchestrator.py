from dataclasses import dataclass

from jims_core.schema import Orchestrator, Pipeline
from jims_core.thread.thread_context import ThreadContext


@dataclass
class BasicOrchestrator(Orchestrator):
    default_route: str = "main"
    start_route: str | None = "start"

    def route(self, ctx: ThreadContext):
        current_pipeline = ctx.state.current_pipeline
        if current_pipeline in self.pipelines:
            return self.pipelines[current_pipeline]
        return self.pipelines[self.default_route]

    async def orchestrate(self, ctx: ThreadContext) -> None:
        pipeline = self.route(ctx)
        await pipeline(ctx)
