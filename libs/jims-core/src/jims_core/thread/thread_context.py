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

    thread_config: dict = field(default_factory=dict)

    status_updater: StatusUpdater | None = None

    def with_status_updater(self, status_updater: StatusUpdater) -> "ThreadContext":
        """Set the status updater for this thread context."""
        self.status_updater = status_updater
        return self

    def get_last_user_message(self) -> str | None:
        """Get the last user message from the history."""
        for event in reversed(self.history):
            if event["role"] == "user":  # type: ignore
                return event["content"]  # type: ignore
        return None

    def get_last_user_action(self):
        """Get the last user action (message / button click / command / other input in comm. domain) from ctx.events"""
        for event in reversed(self.events):
            if event.event_type.startswith("comm."):  # CommunicationEvent
                if event.event_data["role"] == "user":
                    event_name = event.event_type.removeprefix("comm.")
                    return event_name, event.event_data["content"]
        return None, None

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
            EventEnvelope[CommunicationEvent](
                thread_id=self.thread_id,
                event_id=uuid7(),
                created_at=datetime.datetime.now(),
                event_type="comm.assistant_message",
                event_data=CommunicationEvent(role="assistant", content=message),
            )
        )

    def set_state(self, state_name: str, state: dict | BaseModel) -> None:
        """Send an event to set the state of the thread."""
        state_data = state.model_dump() if isinstance(state, BaseModel) else state

        self.outgoing_events.append(
            EventEnvelope(
                thread_id=self.thread_id,
                event_id=uuid7(),
                created_at=datetime.datetime.now(),
                event_type=f"state.set.{state_name}",
                event_data=state_data,
            )
        )

    def get_state[T: BaseModel](self, state_name: str, state_type: Type[T]) -> T | None:
        """Get the state of the thread."""
        target_state = f"state.set.{state_name}"
        for event in reversed(self.events):
            if event.event_type == target_state:
                return state_type.model_validate(event.event_data)
        return None

    async def update_agent_status(self, status: str) -> None:
        """Update the agent status."""
        if self.status_updater:
            await self.status_updater.update_status(status)
