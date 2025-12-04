from jims_core.thread.thread_context import ThreadContext
from vedana_core.data_model import DataModel


class StartPipeline:
    """
    Response for /start command
    """

    def __init__(self, data_model: DataModel) -> None:
        self.lifecycle_events = data_model.conversation_lifecycle_events()
        self.start_response = self.lifecycle_events.get("/start")

    async def __call__(self, ctx: ThreadContext) -> None:
        message = self.start_response or "Bot online. No response for /start command in LifecycleEvents"
        ctx.send_message(message)
