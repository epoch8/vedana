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
    review_status: str
    priority: str

    @classmethod
    def create(
        cls,
        thread_id: str,
        created_at: datetime,
        thread_config: dict,
        review_status: str = "-",
        priority: str = "-",
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
            review_status=str(review_status),
            priority=str(priority),
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
    review_filter: str = "All"  # Default
    sort_by: str = "Date (desc)"

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
        self.review_filter = "All"
        self.sort_by = "Date (desc)"
        return None

    @rx.event
    def set_review_filter(self, value: str) -> None:
        self.review_filter = value

    @rx.event
    def set_sort_by(self, value: str) -> None:
        self.sort_by = value

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

            stmt = sa.select(ThreadDB, last_event_sq.c.last_at).join(
                last_event_sq, last_event_sq.c.t_id == ThreadDB.thread_id
            )

            # Apply date filters at SQL level when possible
            if from_dt is not None:
                stmt = stmt.where(last_event_sq.c.last_at >= from_dt)
            if to_dt is not None:
                stmt = stmt.where(last_event_sq.c.last_at < to_dt)

            results = await session.execute(stmt)
            rows = results.all()

            # Collect thread ids and last_at
            thread_ids: list[str] = []
            last_at_by_tid: dict[str, datetime] = {}
            for thread_obj, last_at in rows:
                tid = str(thread_obj.thread_id)
                thread_ids.append(tid)
                try:
                    last_at_by_tid[tid] = last_at or thread_obj.created_at
                except Exception:
                    last_at_by_tid[tid] = thread_obj.created_at

            # Load backoffice events for review/priority aggregation
            review_status_by_tid: dict[str, str] = {tid: "-" for tid in thread_ids}
            priority_by_tid: dict[str, str] = {tid: "-" for tid in thread_ids}
            priority_rank_by_tid: dict[str, int] = {tid: -1 for tid in thread_ids}

            if thread_ids:
                bo_stmt = sa.select(ThreadEventDB).where(
                    sa.and_(
                        ThreadEventDB.thread_id.in_(thread_ids),
                        ThreadEventDB.event_type.like("jims.backoffice.%"),  # todo split jims event_type in domains?
                    )
                )
                bo_rows = (await session.execute(bo_stmt)).scalars().all()

                has_feedback: dict[str, bool] = {tid: False for tid in thread_ids}
                has_resolved: dict[str, bool] = {tid: False for tid in thread_ids}
                for ev in bo_rows:
                    tid = str(ev.thread_id)
                    etype = str(getattr(ev, "event_type", ""))
                    if etype == "jims.backoffice.feedback":
                        has_feedback[tid] = True
                        try:
                            sev = str((getattr(ev, "event_data", {}) or {}).get("severity", "Low"))
                        except Exception:
                            sev = "Low"
                        rank = {"Low": 0, "Medium": 1, "High": 2}.get(sev, 0)
                        if rank > priority_rank_by_tid.get(tid, -1):
                            priority_rank_by_tid[tid] = rank
                            priority_by_tid[tid] = {0: "Low", 1: "Medium", 2: "High"}.get(rank, "Low")
                    elif etype == "jims.backoffice.review_resolved":
                        has_resolved[tid] = True

                for tid in thread_ids:
                    if has_resolved.get(tid):
                        review_status_by_tid[tid] = "Complete"
                    elif has_feedback.get(tid):
                        review_status_by_tid[tid] = "Pending"
                    else:
                        review_status_by_tid[tid] = "-"

        items: list[ThreadVis] = []
        for thread_obj, last_at in rows:
            try:
                items.append(
                    ThreadVis.create(
                        thread_id=str(thread_obj.thread_id),
                        created_at=thread_obj.created_at,
                        # last_activity=last_at or thread_obj.created_at,
                        thread_config=thread_obj.thread_config,
                        review_status=review_status_by_tid.get(str(thread_obj.thread_id), "-"),
                        priority=priority_by_tid.get(str(thread_obj.thread_id), "-"),
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
                        review_status=review_status_by_tid.get(str(thread_obj.thread_id), "-"),
                        priority=priority_by_tid.get(str(thread_obj.thread_id), "-"),
                    )
                )

        # In-memory search (thread_id or interface)
        search = (self.search_text or "").strip().lower()
        if search:
            items = [it for it in items if search in it.thread_id.lower() or search in (it.interface or "").lower()]

        # Filter by review status
        rf = (self.review_filter or "All").strip()
        if rf and rf != "All":
            items = [it for it in items if it.review_status == rf]

        # Sorting
        sort_val = (self.sort_by or "Date (desc)").strip()
        if sort_val == "Date (asc)":
            items = sorted(items, key=lambda it: last_at_by_tid.get(it.thread_id, datetime.min))
        elif sort_val == "Priority":
            rank_map = {"-": -1, "Low": 0, "Medium": 1, "High": 2}
            items = sorted(
                items,
                key=lambda it: (rank_map.get(it.priority, -1), last_at_by_tid.get(it.thread_id, datetime.min)),
                reverse=True,
            )
        else:  # Date (desc)
            items = sorted(items, key=lambda it: last_at_by_tid.get(it.thread_id, datetime.min), reverse=True)

        self.threads = items
        self.threads_refreshing = False


@rx.page(route="/jims", on_load=ThreadListState.get_data)
def jims_thread_list_page() -> rx.Component:
    def _badge_style(bg: str, fg: str) -> dict[str, str]:
        return {
            "backgroundColor": bg,
            "color": fg,
        }

    def review_badge(value: str) -> rx.Component:  # type: ignore[valid-type]
        return rx.cond(
            value == "Pending",
            rx.badge("Pending", variant="soft", size="1", style=_badge_style("var(--amber-4)", "var(--amber-11)")),
            rx.cond(
                value == "Complete",
                rx.badge("Complete", variant="soft", size="1", style=_badge_style("var(--green-4)", "var(--green-11)")),
                rx.badge(value, variant="soft", size="1", style=_badge_style("var(--gray-4)", "var(--gray-12)")),
            ),
        )

    def priority_badge(value: str) -> rx.Component:  # type: ignore[valid-type]
        return rx.cond(
            value == "High",
            rx.badge("High", variant="soft", size="1", style=_badge_style("var(--tomato-4)", "var(--tomato-11)")),
            rx.cond(
                value == "Medium",
                rx.badge("Medium", variant="soft", size="1", style=_badge_style("var(--amber-4)", "var(--amber-11)")),
                rx.badge(value, variant="soft", size="1", style=_badge_style("var(--gray-4)", "var(--gray-12)")),
            ),
        )

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
        rx.hstack(
            rx.text("Review Status"),
            rx.select(
                items=["All", "Pending", "Complete"],
                value=ThreadListState.review_filter,
                on_change=ThreadListState.set_review_filter,
                width="180px",
            ),
            align="center",
            spacing="2",
        ),
        rx.hstack(
            rx.text("Sort"),
            rx.select(
                items=["Date (desc)", "Date (asc)", "Priority"],
                value=ThreadListState.sort_by,
                on_change=ThreadListState.set_sort_by,
                width="160px",
            ),
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
                rx.table.column_header_cell("Interface"),
                rx.table.column_header_cell("Review"),
                rx.table.column_header_cell("Priority"),
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
                    rx.table.cell(t.interface),
                    rx.table.cell(review_badge(t.review_status)),
                    rx.table.cell(priority_badge(t.priority)),
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
            "tags": ev.visible_tags,  # tags/comments from backoffice annotations
            "comments": ev.feedback_comments,
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

        # Comments thread displayed under the message
        comments_component = rx.vstack(
            rx.foreach(
                ev.feedback_comments,
                lambda c: rx.card(
                    rx.vstack(
                        rx.hstack(
                            rx.cond(
                                c["severity"] == "High",
                                rx.badge("High", variant="soft", size="1", color_scheme="tomato"),
                                rx.cond(
                                    c["severity"] == "Medium",
                                    rx.badge("Medium", variant="soft", size="1", color_scheme="amber"),
                                    rx.badge("Low", variant="soft", size="1", color_scheme="gray"),
                                ),
                            ),
                            rx.text(c["created_at"], size="1", color="gray"),
                            spacing="2",
                        ),
                        rx.text(c["note"]),
                        spacing="1",
                        width="100%",
                    ),
                    padding="0.5em",
                    variant="surface",
                ),
            ),
            spacing="2",
            width="100%",
        )

        extras = rx.vstack(
            comments_component,
            action_line,
            spacing="2",
        )
        tags_component = rx.hstack(
            rx.foreach(
                ev.visible_tags,
                lambda t: rx.badge(t, variant="soft", size="1", color_scheme="gray"),
            ),
            spacing="1",
        )

        return render_message_bubble(
            msg,
            on_toggle_details=ThreadViewState.toggle_details(event_id=ev.event_id),  # type: ignore[call-arg,func-returns-value]
            extras=extras,
            corner_tags_component=tags_component,
        )

    # Left panel (thread list with its own scroll)
    left_panel = rx.vstack(
        rx.heading("Threads"),
        filters,
        rx.cond(
            ThreadListState.threads_refreshing,
            rx.center("Loading threads..."),
            rx.scroll_area(table, type="always", scrollbars="vertical", style={"flex": 1, "height": "100%"}),
        ),
        spacing="3",
        style={"height": "100%", "overflow": "hidden"},
    )

    # Right panel (conversation fills height with scroll)
    right_panel = rx.cond(
        ThreadViewState.selected_thread_id == "",
        rx.center(rx.text("Select a thread to view conversation"), style={"height": "100%"}),
        rx.vstack(
            rx.heading("Conversation"),
            rx.scroll_area(
                rx.vstack(rx.foreach(ThreadViewState.events, _render_event_as_msg), spacing="3", width="100%"),
                type="always",
                scrollbars="vertical",
                style={"flex": 1, "height": "100%"},
            ),
            spacing="3",
            style={"height": "100%", "overflow": "hidden"},
        ),
    )

    return rx.vstack(
        app_header(),
        rx.grid(
            left_panel,
            right_panel,
            columns="2",
            spacing="4",
            sm_columns="1",
            width="100%",
            style={"flex": 1, "height": "100%"},
        ),
        spacing="4",
        style={"height": "100vh", "overflow": "hidden"},
    )
