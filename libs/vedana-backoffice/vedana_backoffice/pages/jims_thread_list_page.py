from dataclasses import dataclass
from datetime import datetime, timedelta

import reflex as rx
import sqlalchemy as sa
from jims_core.db import ThreadDB, ThreadEventDB
from vedana_core.app import make_vedana_app

from vedana_backoffice.components.ui_chat import render_message_bubble
from vedana_backoffice.state import ThreadViewState
from vedana_backoffice.ui import app_header
from vedana_backoffice.util import datetime_to_age


@dataclass
class ThreadVis:
    thread_id: str
    created_at: str
    thread_age: str
    interface: str
    # last_activity: str
    # last_activity_age: str

    @classmethod
    def create(
        cls,
        thread_id: str,
        created_at: datetime,
        # last_activity: datetime,
        thread_config: dict,
    ) -> "ThreadVis":
        cfg = thread_config or {}
        iface_val = cfg.get("interface") or cfg.get("channel") or cfg.get("source")
        if isinstance(iface_val, dict):
            iface_val = iface_val.get("name") or iface_val.get("type") or str(iface_val)
        return cls(
            thread_id=str(thread_id),
            created_at=datetime.strftime(created_at, "%Y-%m-%d %H:%M:%S"),
            thread_age=datetime_to_age(created_at),
            interface=str(iface_val or ""),
            # last_activity=datetime.strftime(last_activity, "%Y-%m-%d %H:%M:%S"),  # todo format tz
            # last_activity_age=datetime_to_age(last_activity),
        )


class ThreadListState(rx.State):
    threads_refreshing: bool = True
    threads: list[ThreadVis] = []

    # Filters
    from_date: str = ""
    to_date: str = ""
    search_text: str = ""

    @staticmethod
    def _parse_date(date_str: str | None) -> datetime | None:
        try:
            if date_str:
                return datetime.strptime(date_str, "%Y-%m-%d")
        except Exception:
            return None
        return None

    @rx.event
    def set_from_date(self, value: str) -> None:
        self.from_date = value

    @rx.event
    def set_to_date(self, value: str) -> None:
        self.to_date = value

    @rx.event
    def set_search_text(self, value: str) -> None:
        self.search_text = value

    @rx.event
    def clear_filters(self) -> None:
        self.from_date = ""
        self.to_date = ""
        self.search_text = ""
        return None

    @rx.event
    async def get_data(self) -> None:
        vedana_app = await make_vedana_app()

        from_dt = self._parse_date(self.from_date)
        to_dt = self._parse_date(self.to_date)
        if to_dt is not None:
            # Make end exclusive by adding one day
            to_dt = to_dt + timedelta(days=1)

        async with vedana_app.sessionmaker() as session:
            last_event_sq = (
                sa.select(
                    ThreadEventDB.thread_id.label("t_id"),
                    sa.func.max(ThreadEventDB.created_at).label("last_at"),
                )
                .group_by(ThreadEventDB.thread_id)
                .subquery()
            )

            stmt = (
                sa.select(ThreadDB, last_event_sq.c.last_at)
                .join(last_event_sq, last_event_sq.c.t_id == ThreadDB.thread_id)
                .order_by(last_event_sq.c.last_at.desc())
            )

            # Apply date filters at SQL level when possible
            if from_dt is not None:
                stmt = stmt.where(last_event_sq.c.last_at >= from_dt)
            if to_dt is not None:
                stmt = stmt.where(last_event_sq.c.last_at < to_dt)

            results = await session.execute(stmt)
            rows = results.all()

        items: list[ThreadVis] = []
        for thread_obj, last_at in rows:
            try:
                items.append(
                    ThreadVis.create(
                        thread_id=str(thread_obj.thread_id),
                        created_at=thread_obj.created_at,
                        # last_activity=last_at or thread_obj.created_at,
                        thread_config=thread_obj.thread_config,
                    )
                )
            except Exception:
                # Fall back if last_at is None or bad
                items.append(
                    ThreadVis.create(
                        thread_id=str(thread_obj.thread_id),
                        created_at=thread_obj.created_at,
                        # last_activity=thread_obj.created_at,
                        thread_config=thread_obj.thread_config,
                    )
                )

        # In-memory search (thread_id or interface)
        search = (self.search_text or "").strip().lower()
        if search:
            items = [it for it in items if search in it.thread_id.lower() or search in (it.interface or "").lower()]

        self.threads = items
        self.threads_refreshing = False


@rx.page(route="/jims", on_load=ThreadListState.get_data)
def jims_thread_list_page() -> rx.Component:
    filters = rx.hstack(
        rx.hstack(
            rx.text("From"),
            rx.input(value=ThreadListState.from_date, type="date", on_change=ThreadListState.set_from_date),
            align="center",
            spacing="2",
        ),
        rx.hstack(
            rx.text("To"),
            rx.input(value=ThreadListState.to_date, type="date", on_change=ThreadListState.set_to_date),
            align="center",
            spacing="2",
        ),
        rx.input(
            placeholder="Search by thread or interface",
            value=ThreadListState.search_text,
            on_change=ThreadListState.set_search_text,
            width="280px",
        ),
        rx.button("Apply", on_click=ThreadListState.get_data),
        rx.button("Clear", variant="soft", on_click=[ThreadListState.clear_filters, ThreadListState.get_data]),
        spacing="4",
        wrap="wrap",
    )

    table = rx.table.root(
        rx.table.header(
            rx.table.row(
                rx.table.column_header_cell("Thread ID"),
                rx.table.column_header_cell("Created"),
                rx.table.column_header_cell("Age"),
                # rx.table.column_header_cell("Last Activity"),
                # rx.table.column_header_cell("Since"),
                rx.table.column_header_cell("Interface"),
            ),
        ),
        rx.table.body(
            rx.foreach(
                ThreadListState.threads,
                lambda t: rx.table.row(
                    rx.table.cell(
                        rx.button(
                            t.thread_id,
                            variant="ghost",
                            color_scheme="gray",
                            size="1",
                            on_click=ThreadViewState.select_thread(thread_id=t.thread_id),  # type: ignore[operator]
                        )
                    ),  # type: ignore[call-arg,func-returns-value]
                    rx.table.cell(t.created_at),
                    rx.table.cell(t.thread_age),
                    # rx.table.cell(t.last_activity),
                    # rx.table.cell(t.last_activity_age),
                    rx.table.cell(t.interface),
                    style=rx.cond(
                        t.thread_id == ThreadViewState.selected_thread_id, {"backgroundColor": "var(--accent-3)"}, {}
                    ),
                ),
            ),
        ),
    )

    def _render_event_as_msg(ev):  # type: ignore[valid-type]
        msg = {
            "id": ev.event_id,
            "content": ev.content,
            "created_at": ev.created_at_str,
            "created_at_fmt": ev.event_age,
            "is_assistant": ev.role == "assistant",
            "tag_label": ev.event_type,
            "has_tech": ev.has_technical_info,
            "has_models": ev.has_models,
            "has_vts": ev.has_vts,
            "has_cypher": ev.has_cypher,
            "models_str": ev.models_str,
            "vts_str": ev.vts_str,
            "cypher_str": ev.cypher_str,
            "show_details": ThreadViewState.expanded_event_id == ev.event_id,
        }

        action_line = rx.cond(
            (ev.role == "assistant") & ev.content != "",
            rx.hstack(
                # Add Tag (only for messages with content)
                rx.input(
                    placeholder="Add tag",
                    value=ThreadViewState.new_tag_text,
                    on_change=ThreadViewState.set_new_tag_text,
                    width="20%",
                ),
                rx.button(
                    "Add tag",
                    size="1",
                    on_click=ThreadViewState.add_tag(event_id=ev.event_id),  # type: ignore[call-arg,func-returns-value]
                ),
                # Add Note (single-line input)
                rx.input(
                    placeholder="Add note...",
                    value=ThreadViewState.note_text_by_event.get(ev.event_id, ""),
                    on_change=lambda v: ThreadViewState.set_note_text_for(v, event_id=ev.event_id),  # type: ignore[call-arg,func-returns-value]
                    width="40%",
                ),
                rx.select(
                    items=["Low", "Medium", "High"],
                    value=ThreadViewState.note_severity_by_event.get(ev.event_id, "Low"),
                    on_change=lambda v: ThreadViewState.set_note_severity_for(v, event_id=ev.event_id),  # type: ignore[call-arg,func-returns-value]
                ),
                rx.button(
                    "Add note",
                    size="1",
                    on_click=ThreadViewState.submit_note_for(event_id=ev.event_id),  # type: ignore[call-arg,func-returns-value]
                ),
                spacing="2",
                wrap="wrap",
                width="100%",
            ),
            rx.box(),
        )

        extras = rx.vstack(
            rx.hstack(
                rx.foreach(
                    ev.tags,
                    lambda tag: rx.button(
                        tag,
                        variant="soft",
                        size="1",
                        color_scheme="gray",
                        on_click=ThreadViewState.remove_tag(event_id=ev.event_id, tag=tag),  # type: ignore[call-arg,func-returns-value]
                    ),
                ),
                spacing="1",
                wrap="wrap",
            ),
            action_line,
            spacing="2",
        )

        return render_message_bubble(
            msg,
            on_toggle_details=ThreadViewState.toggle_details(event_id=ev.event_id),  # type: ignore[call-arg,func-returns-value]
            extras=extras,
        )

    right_panel = rx.vstack(
        rx.cond(
            ThreadViewState.selected_thread_id == "",
            rx.text("Select a thread to view conversation"),
            rx.vstack(
                rx.heading("Conversation"),
                rx.scroll_area(
                    rx.vstack(rx.foreach(ThreadViewState.events, _render_event_as_msg), spacing="3", width="100%"),
                    type="always",
                    scrollbars="vertical",
                    style={"height": "60vh"},
                ),
                spacing="3",
                width="100%",
            ),
        ),
        width="100%",
    )

    return rx.vstack(
        app_header(),
        rx.grid(
            rx.vstack(
                rx.heading("Threads"),
                filters,
                rx.cond(ThreadListState.threads_refreshing, rx.text("Loading..."), table),
                spacing="3",
            ),
            right_panel,
            columns="2",
            spacing="4",
            sm_columns="1",
            width="100%",
        ),
        spacing="4",
    )
