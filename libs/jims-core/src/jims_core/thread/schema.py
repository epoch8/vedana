import datetime
from uuid import UUID

from pydantic import BaseModel
from typing_extensions import TypedDict


class EventEnvelope[T: dict](BaseModel):
    thread_id: UUID
    event_id: UUID

    created_at: datetime.datetime

    event_type: str

    event_data: T


class CommunicationEvent(TypedDict, total=False):
    role: str  # "user" or "assistant"
    content: str
