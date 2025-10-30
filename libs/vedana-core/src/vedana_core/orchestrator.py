from dataclasses import dataclass, field

from jims_core.schema import Orchestrator, Pipeline
from jims_core.thread.thread_context import ThreadContext


@dataclass
class BasicOrchestrator(Orchestrator):
    default_route: str = "main"
    start_route: str | None = "start"

    def route(self, ctx: ThreadContext) -> Pipeline:
        route_hint = ctx.state.current_pipeline
        if route_hint in self.pipelines.keys():  # "main" --> main, "start" --> start etc
            return self.pipelines.get(route_hint)
        else:
            raise RuntimeError(
                f"No pipeline registered for route '{route_hint}'. Available pipelines: {list(self.pipelines.keys())}"
            )

    async def orchestrate(self, ctx: ThreadContext) -> None:
        pipeline = self.route(ctx)
        await pipeline(ctx)
