from jims_core.thread.thread_context import ThreadContext
from vedana_core.data_model import DataModel


class StartPipeline:
    """
    Response for /start command
    """

    def __init__(self, data_model: DataModel) -> None:
        self.data_model = data_model

    async def __call__(self, ctx: ThreadContext) -> None:
        lifecycle_events = await self.data_model.conversation_lifecycle_events()
        start_response = lifecycle_events.get("/start")
        message = start_response or "Bot online. No response for /start command in LifecycleEvents"
        ctx.send_message(message)
