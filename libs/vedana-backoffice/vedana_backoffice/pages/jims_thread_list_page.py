from dataclasses import dataclass
from datetime import datetime

import reflex as rx
import sqlalchemy as sa
from jims_core.db import ThreadDB
from vedana_core.app import make_vedana_app

from vedana_backoffice.ui import app_header, breadcrumbs


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
class ThreadVis:
    thread_id: str
    created_at: str
    thread_age: str
    interface: str

    @classmethod
    def create(cls, thread_id: str, created_at: datetime, thread_config: dict) -> "ThreadVis":
        cfg = thread_config or {}
        iface_val = cfg.get("interface") or cfg.get("channel") or cfg.get("source")
        if isinstance(iface_val, dict):
            iface_val = iface_val.get("name") or iface_val.get("type") or str(iface_val)
        return cls(
            thread_id=str(thread_id),
            created_at=str(created_at),
            thread_age=_datetime_to_age(created_at),
            interface=str(iface_val or ""),
        )


class ThreadListState(rx.State):
    threads_refreshing: bool = True
    threads: list[ThreadVis] = []

    @rx.event
    async def get_data(self) -> None:
        vedana_app = await make_vedana_app()

        async with vedana_app.sessionmaker() as session:
            stmt = sa.select(ThreadDB).order_by(ThreadDB.created_at.desc())
            threads = (await session.execute(stmt)).scalars().all()

        self.threads = [
            ThreadVis.create(
                thread_id=str(thread.thread_id),
                created_at=thread.created_at,
                thread_config=thread.thread_config,
            )
            for thread in threads
        ]
        self.threads_refreshing = False


@rx.page(route="/jims", on_load=ThreadListState.get_data)
def jims_thread_list_page() -> rx.Component:
    return rx.container(
        rx.vstack(
            app_header(),
            breadcrumbs([("Main", "/"), ("JIMS threads", "/jims")]),
            rx.heading("JIMS Threads"),
            rx.cond(
                ThreadListState.threads_refreshing,
                rx.text("Loading..."),
                rx.table.root(
                    rx.table.header(
                        rx.table.row(
                            rx.table.column_header_cell("Thread ID"),
                            rx.table.column_header_cell("Created"),
                            rx.table.column_header_cell("Interface"),
                        ),
                    ),
                    rx.table.body(
                        rx.foreach(
                            ThreadListState.threads,
                            lambda thread: rx.table.row(
                                rx.table.cell(
                                    rx.link(
                                        thread.thread_id,
                                        href=f"/jims/thread/{thread.thread_id}",
                                    ),
                                ),
                                rx.table.cell(thread.thread_age),
                                rx.table.cell(thread.interface),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
