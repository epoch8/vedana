import datetime
from dataclasses import dataclass, field
from typing import Any, Type, Generic
from uuid import UUID

from jims_core.llms.llm_provider import LLMProvider
from jims_core.thread.schema import CommunicationEvent, EventEnvelope, TState, PipelineState
from jims_core.util import uuid7
from pydantic import BaseModel


class StatusUpdater:
    async def update_status(self, status: str) -> None:
        pass


@dataclass
class ThreadContext(Generic[TState]):
    thread_id: UUID

    history: list[CommunicationEvent]

    events: list[EventEnvelope]

    llm: LLMProvider

    outgoing_events: list[EventEnvelope] = field(default_factory=list)

    status_updater: StatusUpdater | None = None

    state: TState = field(default_factory=PipelineState)

    def with_status_updater(self, status_updater: StatusUpdater) -> "ThreadContext":
        """Set the status updater for this thread context."""
        self.status_updater = status_updater
        return self

    def get_last_user_message(self) -> str | None:
        """Get the last user message from the history."""
        for event in reversed(self.history):
            if event["role"] == "user":
                return event["content"]
        return None

    def send_event(self, event_type: str, data: Any) -> None:
        """Send an event to the thread."""
        assert isinstance(data, dict), "Event data must be a dictionary"

        self.outgoing_events.append(
            EventEnvelope(
                thread_id=self.thread_id,
                event_id=uuid7(),
                created_at=datetime.datetime.now(),
                event_type=event_type,
                event_data=data,
            )
        )

    def send_message(self, message: str) -> None:
        """Send a message to the thread."""
        self.outgoing_events.append(
            EventEnvelope(
                thread_id=self.thread_id,
                event_id=uuid7(),
                created_at=datetime.datetime.now(),
                event_type="comm.assistant_message",
                event_data=CommunicationEvent(role="assistant", content=message),
            )
        )

    def set_state(self, state: dict | BaseModel, state_name: str = "") -> None:
        """Send an event to set the state of the thread."""
        if isinstance(state, BaseModel):
            state = state.model_dump()

        self.outgoing_events.append(
            EventEnvelope(
                thread_id=self.thread_id,
                event_id=uuid7(),
                created_at=datetime.datetime.now(),
                event_type=f"state.set.{state_name}" if state_name else "state.set",
                event_data=state,
            )
        )

    def get_state[T: BaseModel](self, state_type: Type[T], state_name: str = "") -> T | None:
        """Get the state of the thread."""
        target_state = f"state.set.{state_type}" if state_name else "state.set"
        for event in reversed(self.events):
            if event.event_type == target_state:
                return state_type.model_validate(event.event_data)
        return None

    def get_or_create_pipeline_state[T: BaseModel](self, state_type: Type[T]) -> T | None:
        """Get the state of the thread and load it as ctx.state, OR create a new one with provided model schema"""
        for event in reversed(self.events):
            if event.event_type == f"state.set":  # or f"state.set.{state_type.__class__.__name__}"
                state = state_type.model_validate(event.event_data)
                break
        else:
            print(f"passing state from orchestrator to ctx: {state_type}")
            state = state_type.model_construct()  # pass state from orchestrator to ctx
        self.state = state
        return state

    async def update_agent_status(self, status: str) -> None:
        """Update the agent status."""
        if self.status_updater:
            await self.status_updater.update_status(status)
