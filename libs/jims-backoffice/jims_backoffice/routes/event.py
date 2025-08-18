import datetime
from typing import Annotated
from uuid import UUID

import sqlalchemy as sa
from fastapi import Depends, HTTPException
from fastui import AnyComponent, FastUI
from fastui import components as c
from fastui.components.display import DisplayLookup, DisplayMode
from fastui.events import GoToEvent, PageEvent
from fastui.forms import fastui_form
from jims_backoffice.forms import EventSearchForm, FeedbackForm
from jims_backoffice.main_app import application
from jims_backoffice.utils import (
    PaginationModel,
    common_page,
    create_actions_buttons,
    create_table_and_search,
    generate_fields,
    get_async_sessionmaker,
    get_pagination,
    get_sessionmaker,
    get_thread_links,
)
from jims_core.db import ThreadEventDB
from jims_core.thread.thread_controller import ThreadController
from jims_core.util import uuid7
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Session, sessionmaker


@generate_fields
def generate_events() -> dict[str, DisplayLookup]:
    return {
        "thread_id": DisplayLookup(field="thread_id", on_click=GoToEvent(url="/thread/{thread_id}")),
        "event_id": DisplayLookup(field="event_id", on_click=GoToEvent(url="/thread/{thread_id}/event/{event_id}")),
        "created_at": DisplayLookup(field="created_at"),
        "event_type": DisplayLookup(field="event_type"),
        "event_domain": DisplayLookup(field="event_domain"),
        "event_name": DisplayLookup(field="event_name"),
        "event_data": DisplayLookup(field="event_data", mode=DisplayMode.json),
    }


def get_event_all_events(
    limit: int,
    offset: int,
    thread_id: str | None = None,
    event_id: str | None = None,
    event_type: str | None = None,
    event_domain: str | None = None,
    event_name: str | None = None,
    sessionmaker: sessionmaker[Session] = get_sessionmaker(),
) -> tuple[int, list[ThreadEventDB]]:
    stmt = sa.select(ThreadEventDB)
    if thread_id:
        try:
            tmp = UUID(thread_id)
            stmt = stmt.where(ThreadEventDB.thread_id == tmp)
        except ValueError:
            return 0, []
    if event_id:
        try:
            tmp = UUID(event_id)
            stmt = stmt.where(ThreadEventDB.event_id == tmp)
        except ValueError:
            return 0, []
    if event_type:
        stmt = stmt.where(ThreadEventDB.event_type.like(f"%{event_type}%"))
    if event_domain:
        stmt = stmt.where(ThreadEventDB.event_domain == event_domain)
    if event_name:
        stmt = stmt.where(ThreadEventDB.event_name == event_name)
    stmt = stmt.order_by(ThreadEventDB.created_at.desc())
    count_stmt = sa.select(sa.func.count("*")).select_from(stmt.subquery())
    stmt = stmt.limit(limit).offset(offset)
    with sessionmaker() as session:
        count_result = session.execute(count_stmt).scalar_one()
        result = session.execute(stmt).scalars().all()
    return count_result, list(result)


def get_event_by_thread_id_and_event_id(
    thread_id: UUID, event_id: UUID, sessionmaker: sessionmaker[Session] = get_sessionmaker()
) -> ThreadEventDB | None:
    stmt = sa.select(ThreadEventDB).where(ThreadEventDB.thread_id == thread_id, ThreadEventDB.event_id == event_id)
    with sessionmaker() as session:
        res = session.execute(stmt).scalar_one_or_none()
    return res


async def async_is_event_exists(
    thread_id: UUID,
    event_id: UUID,
    sessionmaker: async_sessionmaker[AsyncSession] = get_async_sessionmaker(),
) -> bool:
    stmt = sa.select(sa.exists().where(ThreadEventDB.thread_id == thread_id, ThreadEventDB.event_id == event_id))
    async with sessionmaker() as session:
        res = (await session.execute(stmt)).scalar_one()
    return res


class EventModel(BaseModel):
    thread_id: UUID
    event_id: UUID
    created_at: datetime.datetime
    event_type: str
    event_domain: str | None = None
    event_name: str | None = None
    event_data: dict | None = None


@application.get("/api/thread/{thread_id}/event", response_model=FastUI, response_model_exclude_none=True)
def get_event_page(
    thread_id: str,
    event_id: str | None = None,
    event_type: str | None = None,
    event_domain: str | None = None,
    event_name: str | None = None,
    pagination: PaginationModel = Depends(get_pagination),
) -> list[AnyComponent]:
    total, events = get_event_all_events(
        event_id=event_id,
        event_type=event_type,
        event_domain=event_domain,
        event_name=event_name,
        thread_id=thread_id,
        limit=pagination.items_per_page,
        offset=pagination.offset,
    )
    event_table = create_table_and_search(
        search_form=EventSearchForm,
        table=c.Table(
            data_model=EventModel,
            data=[
                EventModel(
                    thread_id=event.thread_id,
                    event_id=event.event_id,
                    created_at=event.created_at,
                    event_type=event.event_type,
                    event_domain=event.event_domain,
                    event_name=event.event_name,
                    event_data=event.event_data,
                )
                for event in events
            ],
            columns=generate_events(),
        ),
        page_event_name="search-event",
        page=pagination.page,
        items_per_page=pagination.items_per_page,
        total=total,
    )
    event_page = c.Div(
        components=[
            event_table,
        ]
    )
    return common_page(
        get_thread_links(thread_id),
        "Events",
        event_page,
    )


@application.get("/api/thread/{thread_id}/event/{event_id}", response_model=FastUI, response_model_exclude_none=True)
def get_event_detail_page(
    thread_id: UUID,
    event_id: UUID,
) -> list[AnyComponent]:
    event = get_event_by_thread_id_and_event_id(thread_id, event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    event_model = EventModel(
        thread_id=event.thread_id,
        event_id=event.event_id,
        created_at=event.created_at,
        event_type=event.event_type,
        event_domain=event.event_domain,
        event_name=event.event_name,
        event_data=event.event_data,
    )
    buttons = create_actions_buttons(
        c.Button(text="Add Feedback", on_click=PageEvent(name="add-feedback-modal")),
    )
    event_page = c.Div(
        components=[
            buttons,
            c.Details(data=event_model, fields=generate_events()),  # type: ignore
            c.Modal(
                title="Add Feedback",
                body=[
                    c.ModelForm(
                        model=FeedbackForm,
                        submit_url=f"/api/thread/{thread_id}/event/{event_id}/add-feedback",
                    )
                ],
                footer=[c.Button(text="Close", on_click=PageEvent(name="add-feedback-modal", clear=True))],
                open_trigger=PageEvent(name="add-feedback-modal", clear=True),
            ),
        ]  # type: ignore
    )
    return common_page(
        get_thread_links(thread_id),
        "Event",
        event_page,
    )


@application.post("/api/thread/{thread_id}/event/{event_id}/add-feedback")
async def add_feedback(
    thread_id: UUID,
    event_id: UUID,
    feedback: Annotated[FeedbackForm, fastui_form(FeedbackForm)],
) -> list[AnyComponent]:
    thread_controller = await ThreadController.from_thread_id(get_async_sessionmaker(), thread_id)
    if thread_controller is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    if not await async_is_event_exists(thread_id, event_id):
        raise HTTPException(status_code=404, detail="Event not found")

    data = {
        **feedback.model_dump(),
        "event_id": str(event_id),
    }
    await thread_controller.store_event_dict(
        event_id=uuid7(),
        event_type="jims.backoffice.feedback",
        event_data=data,
    )
    return [c.Text(text="Operation was a success. Refresh the page to see result!")]
