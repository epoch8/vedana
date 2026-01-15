import functools
from functools import cache
from typing import Callable, ParamSpec, Type, TypeVar
from uuid import UUID

from fastui import AnyComponent
from fastui import components as c
from fastui.components.display import Display, DisplayLookup
from fastui.events import GoToEvent, PageEvent
from pydantic import BaseModel
from sqlalchemy import Engine, create_engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

from jims_backoffice.settings import DatabaseSettings

T = TypeVar("T")


@cache
def get_engine() -> Engine:
    db_config = DatabaseSettings()  # type: ignore
    engine = create_engine(db_config.dsn, pool_size=10)
    return engine


@cache
def get_async_engine() -> AsyncEngine:
    db_config = DatabaseSettings()  # type: ignore
    engine = create_async_engine(db_config.async_dsn, pool_size=100)
    return engine


@cache
def get_sessionmaker() -> sessionmaker[Session]:
    engine = get_engine()
    return sessionmaker(bind=engine, autoflush=True, expire_on_commit=False)


@cache
def get_async_sessionmaker() -> async_sessionmaker[AsyncSession]:
    engine = get_async_engine()
    return async_sessionmaker(bind=engine, autoflush=True, expire_on_commit=False)


class PaginationModel(BaseModel):
    page: int
    items_per_page: int
    offset: int


def get_pagination(page: int = 1, items_per_page: int = 20) -> PaginationModel:
    if page <= 0:
        page = 1
    offset = (page - 1) * items_per_page
    return PaginationModel(page=page, items_per_page=items_per_page, offset=offset)


TITLE = "Backoffice"


def common_page(links: list | None = None, title: str | None = None, *components: AnyComponent) -> list[AnyComponent]:
    if links is None:
        links = []
    return [
        c.PageTitle(text=f"{TITLE} — {title}" if title else f"{TITLE}"),
        c.Navbar(
            title=TITLE,
            title_event=GoToEvent(url="/"),
            start_links=links,
        ),
        c.Page(
            components=[
                *((c.Heading(text=title),) if title else ()),
                *components,
            ],
        ),
        c.Footer(
            extra_text=TITLE,
            links=[],
        ),
    ]


def get_thread_links(thread_id: UUID | str | None = None) -> list[c.Link]:
    if thread_id is None:
        return []
    return [
        c.Link(
            components=[c.Text(text=f"Thread: {thread_id} >")],
            on_click=GoToEvent(url=f"/thread/{thread_id}"),
            active="startswith:/",
        ),
        c.Link(
            components=[c.Text(text="Events")],
            on_click=GoToEvent(url=f"/thread/{thread_id}/event"),
            active="startswith:/event",
        ),
    ]


P = ParamSpec("P")

FieldsType = list[DisplayLookup | Display] | None


def generate_fields(func: Callable[P, dict[str, DisplayLookup]]):
    @functools.wraps(func)
    def wrapper(exclude: set[str] | None = None, *args: P.args, **kwargs: P.kwargs) -> list[DisplayLookup]:
        if exclude is None:
            exclude = set()
        table = func(*args, **kwargs)
        return [value for key, value in table.items() if key not in exclude]

    return wrapper


def create_table_and_search(
    *, table: c.Table, search_form: Type[BaseModel], page_event_name: str, page: int, items_per_page: int, total: int
) -> c.Div:
    return c.Div(
        components=[
            c.Div(
                components=[
                    c.ModelForm(
                        model=search_form,
                        submit_url=".",
                        method="GOTO",
                        submit_on_change=True,
                        display_mode="inline",
                        submit_trigger=PageEvent(name=page_event_name),
                    ),
                    table,
                ],
                class_name="mt-1",
            ),
            c.Pagination(page=page, page_size=items_per_page, total=total),
        ],
    )


def create_actions_buttons(*buttons: c.Button | None) -> c.Div:
    return c.Div(
        components=[c.Div(components=[button], class_name="mt-1") for button in buttons if button is not None],
        class_name="col-md-6 mb-2",
    )
