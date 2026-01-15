import datetime
from typing import cast
from uuid import UUID

import sqlalchemy as sa
from fastapi import Depends, HTTPException
from fastui import AnyComponent, FastUI
from fastui import components as c
from fastui.components.display import DisplayLookup, DisplayMode
from fastui.events import GoToEvent
from jims_backoffice.forms import ThreadSearchForm
from jims_backoffice.main_app import application
from jims_backoffice.utils import (
    FieldsType,
    PaginationModel,
    common_page,
    create_table_and_search,
    generate_fields,
    get_pagination,
    get_sessionmaker,
    get_thread_links,
)
from jims_core.db import ThreadDB
from pydantic import BaseModel
from sqlalchemy.orm import Session, sessionmaker


@generate_fields
def generate_threads() -> dict[str, DisplayLookup]:
    return {
        "thread_id": DisplayLookup(field="thread_id", on_click=GoToEvent(url="/thread/{thread_id}")),
        "created_at": DisplayLookup(field="created_at"),
        "thread_config": DisplayLookup(field="thread_config", mode=DisplayMode.json),
    }


def get_all_thread(
    limit: int, offset: int, thread_id: str | None = None, sessionmaker: sessionmaker[Session] = get_sessionmaker()
) -> tuple[int, list[ThreadDB]]:
    stmt = sa.select(ThreadDB)
    if thread_id:
        try:
            tmp = UUID(thread_id)
            stmt = stmt.where(ThreadDB.thread_id == tmp)
        except ValueError:
            return 0, []
    stmt = stmt.order_by(ThreadDB.created_at.desc())
    count_stmt = sa.select(sa.func.count("*")).select_from(stmt.subquery())
    stmt = stmt.limit(limit).offset(offset)
    with sessionmaker() as session:
        count_result = session.execute(count_stmt).scalar_one()
        result = session.execute(stmt).scalars().all()
    return count_result, list(result)


def get_thread_by_id(thread_id: str, sessionmaker: sessionmaker[Session] = get_sessionmaker()) -> ThreadDB | None:
    try:
        tmp = UUID(thread_id)
    except ValueError:
        return None

    stmt = sa.select(ThreadDB).where(ThreadDB.thread_id == tmp)
    with sessionmaker() as session:
        res = session.execute(stmt).scalar_one_or_none()
    return res


class ThreadModel(BaseModel):
    thread_id: UUID
    created_at: datetime.datetime
    thread_config: dict


@application.get("/api/", response_model=FastUI, response_model_exclude_none=True)
def get_thread_table_page(
    thread_id: str | None = None,
    pagination: PaginationModel = Depends(get_pagination),
) -> list[AnyComponent]:
    total, threads = get_all_thread(
        thread_id=thread_id,
        limit=pagination.items_per_page,
        offset=pagination.offset,
    )
    thread_table = create_table_and_search(
        search_form=ThreadSearchForm,
        table=c.Table(
            data_model=ThreadModel,
            data=[
                ThreadModel(
                    thread_id=thread.thread_id,
                    created_at=thread.created_at
                    if isinstance(thread.created_at, datetime.datetime)
                    else datetime.datetime.fromisoformat(str(thread.created_at)),
                    thread_config=thread.thread_config,
                )
                for thread in threads
            ],
            columns=generate_threads(),
        ),
        page_event_name="search-thread",
        page=pagination.page,
        items_per_page=pagination.items_per_page,
        total=total,
    )

    thread_page = c.Div(
        components=[
            thread_table,
        ]
    )
    return common_page(
        get_thread_links(),
        "Threads",
        thread_page,
    )


@application.get("/api/thread/{thread_id}", response_model=FastUI, response_model_exclude_none=True)
def get_thread_page(
    thread_id: str,
) -> list[AnyComponent]:
    thread = get_thread_by_id(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    thread_model = ThreadModel(
        thread_id=thread.thread_id,
        created_at=thread.created_at
        if isinstance(thread.created_at, datetime.datetime)
        else datetime.datetime.fromisoformat(str(thread.created_at)),
        thread_config=thread.thread_config,
    )
    data = c.Details(data=thread_model, fields=cast(FieldsType, generate_threads()))
    thread_page = c.Div(
        components=[
            data,
        ]
    )
    return common_page(
        get_thread_links(thread_id),
        "Thread",
        thread_page,
    )
