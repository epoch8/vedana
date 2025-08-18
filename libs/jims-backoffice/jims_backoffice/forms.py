from typing import Annotated

from fastui.forms import Textarea
from pydantic import BaseModel, Field


class ThreadSearchForm(BaseModel):
    thread_id: str | None = Field(None, description="Thread ID")


class EventSearchForm(BaseModel):
    thread_id: str | None = Field(None, description="Thread ID")
    event_id: str | None = Field(None, description="Event ID")
    event_type: str | None = Field(None, description="Event Type")
    event_domain: str | None = Field(None, description="Event Domain")
    event_name: str | None = Field(None, description="Event Name")


class FeedbackForm(BaseModel):
    value: int = Field(description="Feedback Value", ge=1, le=5)
    comment: Annotated[str, Textarea(rows=5)] | None = Field(None, description="Feedback Comment")
