import datetime
from dataclasses import dataclass, field
from typing import Any, Type
from uuid import UUID

from jims_core.llms.llm_provider import LLMProvider
from jims_core.thread.schema import CommunicationEvent, EventEnvelope
from jims_core.util import uuid7
from pydantic import BaseModel


class StatusUpdater:
    async def update_status(self, status: str) -> None:
        pass


@dataclass
class ThreadContext:
    thread_id: UUID

    history: list[CommunicationEvent]

    events: list[EventEnvelope]

    llm: LLMProvider

    outgoing_events: list[EventEnvelope] = field(default_factory=list)

    status_updater: StatusUpdater | None = None

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

    def set_state(self, state: dict | BaseModel) -> None:
        """Send an event to set the state of the thread."""
        if isinstance(state, BaseModel):
            state = state.model_dump()

        self.outgoing_events.append(
            EventEnvelope(
                thread_id=self.thread_id,
                event_id=uuid7(),
                created_at=datetime.datetime.now(),
                event_type="state.set",
                event_data=state,
            )
        )

    def get_state[T: BaseModel](self, state_type: Type[T]) -> T | None:
        """Get the state of the thread."""
        for event in reversed(self.events):
            if event.event_type == "state.set":
                return state_type.model_validate(event.event_data)
        return None

    async def update_agent_status(self, status: str) -> None:
        """Update the agent status."""
        if self.status_updater:
            await self.status_updater.update_status(status)
