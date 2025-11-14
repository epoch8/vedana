from typing import Awaitable
from uuid import UUID

from jims_core.app import JimsApp
from jims_core.schema import Pipeline
from jims_core.thread.thread_controller import ThreadController
from jims_core.util import uuid7
from opentelemetry import trace
from opentelemetry.trace import Span
from textual import events
from textual.app import App
from textual.widgets import Footer, Header, Input, RichLog

tracer = trace.get_tracer("jims_tui")


class ChatApp(App):
    CSS = """
    RichLog {
        height: 1fr;
        border: solid green;
        background: $surface;
    }
    Input {
        dock: bottom;
        border: solid blue;
    }
    """

    def __init__(
        self,
        app: JimsApp,
        thread_id: UUID,
        ctl: ThreadController,
    ) -> None:
        super().__init__()

        self.jims_app = app
        self.thread_id = thread_id
        self.ctl = ctl

    @classmethod
    async def create(cls, app: JimsApp | Awaitable[JimsApp]) -> "ChatApp":
        if isinstance(app, Awaitable):
            app = await app
        thread_id = uuid7()
        contact_id = f"tui:{thread_id}"
        ctl = await app.new_thread(contact_id, thread_id, {})

        return cls(app=app, thread_id=thread_id, ctl=ctl)

    def compose(self):
        yield Header(show_clock=True)
        yield RichLog(id="chat-history", wrap=True, highlight=True, markup=True)
        yield Input(placeholder="Enter message (Ctrl+C to exit)", id="message-input")
        yield Footer()

    async def _run_pipeline(self, ctl: ThreadController, span: Span, chat_log, pipeline: Pipeline) -> None:
        @self.call_later
        async def process():
            with trace.use_span(span):
                pipeline_events = await self.ctl.run_pipeline_with_context(pipeline)

            for pipeline_event in pipeline_events:
                if pipeline_event.event_type == "comm.assistant_message":
                    # Display assistant message
                    chat_log.write(f"[bold green]Assistant:[/] {pipeline_event.event_data.get('content')}")

            span.end()

    async def on_mount(self):
        span = tracer.start_span("jims_tui.on_start")
        with trace.use_span(span):
            # Display system message about thread creation
            chat_log = self.query_one("#chat-history")
            assert isinstance(chat_log, RichLog)
            chat_log.write(f"[bold green]System:[/] New thread created with ID: {self.ctl.thread.thread_id}")

            self.query_one("#message-input").focus()

            if self.jims_app.conversation_start_pipeline is not None:
                await self._run_pipeline(
                    self.ctl,
                    span,
                    chat_log,
                    self.jims_app.conversation_start_pipeline,
                )

    async def on_input_submitted(self, ui_event: Input.Submitted) -> None:
        span = tracer.start_span("jims_tui.on_input_submitted")
        with trace.use_span(span):
            text = ui_event.value
            if not text:
                return

            # Add message to chat history
            chat_log = self.query_one("#chat-history")
            assert isinstance(chat_log, RichLog)
            chat_log.write(f"[bold blue]You:[/] {text}")

            # Clear input
            chat_input = self.query_one("#message-input")
            assert isinstance(chat_input, Input)
            chat_input.value = ""

            # Store message using the controller
            await self.ctl.store_user_message(uuid7(), text)

            await self._run_pipeline(
                self.ctl,
                span,
                chat_log,
                self.jims_app.pipeline,
            )

    def on_key(self, event: events.Key) -> None:
        """Handle key press events."""
        if event.key == "ctrl+c":
            self.exit()
