from dataclasses import dataclass
from datetime import datetime
from typing import Any

import reflex as rx
import sqlalchemy as sa
from jims_core.db import ThreadEventDB
from vedana_core.app import make_vedana_app


def _datetime_to_age(created_at: datetime) -> str:
    from datetime import datetime as dt
    from datetime import timezone

    now = dt.now(timezone.utc)
    created_at_dt = created_at
    if created_at_dt.tzinfo is None:
        created_at_dt = created_at_dt.replace(tzinfo=timezone.utc)

    diff = now - created_at_dt
    if diff.days < 7:
        if diff.days > 0:
            hours = diff.seconds // 3600
            return f"{diff.days}d{hours}h" if hours > 0 else f"{diff.days}d"
        if diff.seconds >= 3600:
            return f"{diff.seconds // 3600}h"
        if diff.seconds >= 60:
            return f"{diff.seconds // 60}m"
        return "1m"
    return created_at_dt.strftime("%Y %b %d %H:%M")


@dataclass
class ThreadEventVis:
    event_id: str
    created_at: datetime
    event_type: str
    event_age: str
    event_data_list: list[tuple[str, str]]
    technical_vts_queries: list[str]
    technical_cypher_queries: list[str]
    technical_models: list[tuple[str, str]]

    @classmethod
    def create(cls, event_id: Any, created_at: datetime, event_type: str, event_data: dict) -> "ThreadEventVis":
        import json

        # Generic key-value list for display
        event_data_list = [(str(k), str(v)) for k, v in (event_data or {}).items()]

        # Parse technical_info if present
        tech = event_data.get("technical_info", {}) if isinstance(event_data, dict) else {}
        vts_queries: list[str] = list(tech.get("vts_queries", []) or [])
        cypher_queries: list[str] = list(tech.get("cypher_queries", []) or [])

        model_stats = tech.get("model_stats", {}) or tech.get("model_used", {}) or {}
        models_list: list[tuple[str, str]] = []
        try:
            # If nested dict like {model: {...}} flatten to stringified value
            for mk, mv in model_stats.items() if isinstance(model_stats, dict) else []:
                try:
                    models_list.append((str(mk), json.dumps(mv)))
                except Exception:
                    models_list.append((str(mk), str(mv)))
        except Exception:
            pass

        return cls(
            event_id=str(event_id),
            created_at=created_at,
            event_type=event_type,
            event_age=_datetime_to_age(created_at),
            event_data_list=event_data_list,
            technical_vts_queries=vts_queries,
            technical_cypher_queries=cypher_queries,
            technical_models=models_list,
        )


class ThreadViewState(rx.State):
    loading: bool = True
    events: list[ThreadEventVis] = []

    @rx.event
    async def get_data(self):
        vedana_app = await make_vedana_app()

        async with vedana_app.sessionmaker() as session:
            stmt = sa.select(ThreadEventDB).where(ThreadEventDB.thread_id == self.thread_id)
            events = (await session.execute(stmt)).scalars().all()

        self.events = [
            ThreadEventVis.create(
                event_id=event.event_id,
                created_at=event.created_at,
                event_type=event.event_type,
                event_data=event.event_data,
            )
            for event in events
        ]
        self.loading = False


def _event_row(event: ThreadEventVis) -> rx.Component:
    tech_section = rx.vstack(
        rx.heading("Technical Info", size="3"),
        rx.cond(
            event.technical_models,
            rx.data_list.root(
                rx.data_list.item(rx.data_list.label("Models"), rx.data_list.value("")),
                rx.foreach(
                    event.technical_models,
                    lambda kv: rx.data_list.item(
                        rx.data_list.label(kv[0]), rx.data_list.value(rx.code(kv[1], font_size="11px"))
                    ),
                ),
                size="1",
            ),
        ),
        rx.cond(
            event.technical_vts_queries,
            rx.vstack(
                rx.text("VTS Queries"),
                rx.foreach(event.technical_vts_queries, lambda q: rx.code(q, font_size="11px")),
            ),
        ),
        rx.cond(
            event.technical_cypher_queries,
            rx.vstack(
                rx.text("Cypher Queries"),
                rx.foreach(event.technical_cypher_queries, lambda q: rx.code(q, font_size="11px")),
            ),
        ),
        spacing="2",
    )

    generic_section = rx.data_list.root(
        rx.data_list.item(rx.data_list.label("event_id"), rx.data_list.value(event.event_id)),
        rx.data_list.item(rx.data_list.label("event_type"), rx.data_list.value(event.event_type)),
        rx.foreach(
            event.event_data_list,
            lambda item: rx.data_list.item(
                rx.data_list.label(item[0]), rx.data_list.value(rx.code(item[1], font_size="11px"))
            ),
        ),
        size="1",
    )

    return rx.table.row(
        rx.table.cell(event.event_age),
        rx.table.cell(rx.vstack(generic_section, tech_section, spacing="3")),
    )


@rx.page(route="/jims/thread/[thread_id]", on_load=ThreadViewState.get_data)
def jims_thread_page() -> rx.Component:
    return rx.container(
        rx.vstack(
            rx.heading(f"Thread Page {rx.State.thread_id}"),  # type: ignore
            rx.cond(
                ThreadViewState.loading,
                rx.text("Loading..."),
                rx.table.root(
                    rx.table.header(
                        rx.table.row(
                            rx.table.cell("Created"),
                            rx.table.cell("Event Data"),
                        )
                    ),
                    rx.table.body(
                        rx.foreach(
                            ThreadViewState.events,
                            _event_row,
                        )
                    ),
                ),
            ),
        )
    )
