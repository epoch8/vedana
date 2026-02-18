from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import reflex as rx
import sqlalchemy as sa
from jims_core.db import ThreadDB, ThreadEventDB
from vedana_core.app import make_vedana_app

from vedana_backoffice.components.ui_chat import render_message_bubble
from vedana_backoffice.states.jims import ThreadViewState
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
    tag1: str = ""
    tag2: str = ""
    tag3: str = ""

    @classmethod
    def create(
        cls,
        thread_id: str,
        created_at: datetime,
        thread_config: dict,
        review_status: str = "-",
        priority: str = "-",
        tags_sample: list[str] | None = None,
    ) -> "ThreadVis":
        cfg = thread_config or {}
        iface_val = cfg.get("interface") or cfg.get("channel") or cfg.get("source")
        if isinstance(iface_val, dict):
            iface_val = iface_val.get("name") or iface_val.get("type") or str(iface_val)
        ts = list(tags_sample or [])
        while len(ts) < 3:
            ts.append("")
        return cls(
            thread_id=str(thread_id),
            created_at=datetime.strftime(created_at, "%Y-%m-%d %H:%M:%S"),
            thread_age=datetime_to_age(created_at),
            interface=str(iface_val or ""),
            review_status=str(review_status),
            priority=str(priority),
            tag1=str(ts[0] or ""),
            tag2=str(ts[1] or ""),
            tag3=str(ts[2] or ""),
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
    sort_reverse: bool = True
    review_filter: str = "Review: All"  # Default
    sort_by: str = "Sort by: Date"
    available_tags: list[str] = []
    selected_tags: list[str] = []
    page_size: int = 20
    current_page: int = 0
    total_threads: int = 0

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
        self.sort_reverse = True
        self.review_filter = "Review: All"
        self.sort_by = "Sort by: Date"
        self.selected_tags = []
        self.current_page = 0
        return None

    @rx.event
    async def toggle_sort(self) -> None:
        self.sort_reverse = not self.sort_reverse
        self.current_page = 0
        await self.get_data()  # type: ignore[operator]

    @rx.event
    async def set_review_filter(self, value: str) -> None:
        self.review_filter = value
        self.current_page = 0
        await self.get_data()  # type: ignore[operator]

    @rx.event
    async def set_sort_by(self, value: str) -> None:
        self.sort_by = value
        self.current_page = 0
        await self.get_data()  # type: ignore[operator]

    @rx.event
    def toggle_tag_filter(self, tag: str, value: bool) -> None:
        try:
            t = str(tag)
            if value:
                if t not in self.selected_tags:
                    self.selected_tags = [*self.selected_tags, t]
            else:
                self.selected_tags = [x for x in self.selected_tags if x != t]
        except Exception:
            pass

    @rx.event
    def clear_tag_filter(self) -> None:
        self.selected_tags = []
        self.current_page = 0

    @rx.event
    def reset_pagination(self) -> None:
        self.current_page = 0

    @rx.event
    async def next_page(self) -> None:
        if self.has_next_page:
            self.current_page += 1
            await self.get_data()  # type: ignore[operator]

    @rx.event
    async def prev_page(self) -> None:
        if self.current_page > 0:
            self.current_page -= 1
            await self.get_data()  # type: ignore[operator]

    @rx.event
    async def first_page(self) -> None:
        if self.current_page != 0:
            self.current_page = 0
            await self.get_data()  # type: ignore[operator]

    @rx.event
    async def last_page(self) -> None:
        max_page = (self.total_threads - 1) // self.page_size if self.total_threads > 0 else 0
        if self.current_page != max_page:
            self.current_page = max_page
            await self.get_data()  # type: ignore[operator]

    @rx.var
    def total_pages(self) -> int:
        return (self.total_threads - 1) // self.page_size + 1 if self.total_threads > 0 else 1

    @rx.var
    def page_display(self) -> str:
        return f"Page {self.current_page + 1} of {self.total_pages}"

    @rx.var
    def rows_display(self) -> str:
        if self.total_threads == 0:
            return "No rows"
        start = self.current_page * self.page_size + 1
        end = min(start + self.page_size - 1, self.total_threads)
        return f"Rows {start}-{end} of {self.total_threads}"

    @rx.var
    def has_next_page(self) -> bool:
        return self.current_page < (self.total_pages - 1)

    @rx.var
    def has_prev_page(self) -> bool:
        return self.current_page > 0

    @rx.event
    async def get_data(self) -> None:
        """compiles the entire thread list table"""
        self.threads_refreshing = True
        vedana_app = await make_vedana_app()

        from_dt = self._parse_date(self.from_date)
        to_dt = self._parse_date(self.to_date)
        if to_dt is not None:
            # Make end exclusive by adding one day
            to_dt = to_dt + timedelta(days=1)

        def _compute_backoffice_maps(
            thread_ids: list[str], bo_rows: list[ThreadEventDB]
        ) -> tuple[dict[str, str], dict[str, str], dict[str, set[str]]]:
            review_status_by_tid: dict[str, str] = {tid: "-" for tid in thread_ids}
            priority_by_tid: dict[str, str] = {tid: "-" for tid in thread_ids}
            priority_rank_by_tid: dict[str, int] = {tid: -1 for tid in thread_ids}

            has_feedback: dict[str, bool] = {tid: False for tid in thread_ids}
            has_resolved: dict[str, bool] = {tid: False for tid in thread_ids}
            unresolved_by_tid: dict[str, int] = {tid: 0 for tid in thread_ids}
            comment_tid_by_id: dict[str, str] = {}
            resolved_cids: set[str] = set()
            tags_by_event: dict[str, set[str]] = {}

            bo_rows_sorted = sorted(bo_rows, key=lambda r: getattr(r, "created_at", datetime.min))
            for ev in bo_rows_sorted:
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
                    try:
                        cid = str(getattr(ev, "event_id"))
                        if cid:
                            comment_tid_by_id[cid] = tid
                            unresolved_by_tid[tid] = unresolved_by_tid.get(tid, 0) + 1
                    except Exception:
                        pass
                elif etype == "jims.backoffice.review_resolved":
                    has_resolved[tid] = True
                elif etype in ("jims.backoffice.tag_added", "jims.backoffice.tag_removed"):
                    ed = dict(getattr(ev, "event_data", {}) or {})
                    tag = str(ed.get("tag", "")).strip()
                    target = str(ed.get("target_event_id", ""))
                    if not target or not tag:
                        continue
                    cur = tags_by_event.setdefault(target, set())
                    if etype == "jims.backoffice.tag_added":
                        cur.add(tag)
                    else:
                        cur.discard(tag)
                elif etype in ("jims.backoffice.comment_resolved", "jims.backoffice.comment_closed"):
                    ed = dict(getattr(ev, "event_data", {}) or {})
                    cid = str(ed.get("comment_id", ""))
                    if not cid or cid in resolved_cids:
                        continue
                    resolved_cids.add(cid)
                    rtid = comment_tid_by_id.get(cid, tid)
                    unresolved_by_tid[rtid] = max(0, unresolved_by_tid.get(rtid, 0) - 1)

            for tid in thread_ids:
                if has_resolved.get(tid) or (has_feedback.get(tid) and unresolved_by_tid.get(tid, 0) == 0):
                    review_status_by_tid[tid] = "Complete"
                elif has_feedback.get(tid):
                    review_status_by_tid[tid] = "Pending"
                else:
                    review_status_by_tid[tid] = "-"

            thread_tags_by_tid: dict[str, set[str]] = {tid: set() for tid in thread_ids}
            for ev in bo_rows_sorted:
                etype = str(getattr(ev, "event_type", ""))
                if etype not in ("jims.backoffice.tag_added", "jims.backoffice.tag_removed"):
                    continue
                ed = dict(getattr(ev, "event_data", {}) or {})
                target = str(ed.get("target_event_id", ""))
                if not target:
                    continue
                thread_tags_by_tid[str(ev.thread_id)].update(tags_by_event.get(target, set()))

            return review_status_by_tid, priority_by_tid, thread_tags_by_tid

        async with vedana_app.sessionmaker() as session:
            # Keep global tag values available across pages.
            tag_expr = ThreadEventDB.event_data["tag"].as_string()
            tag_stmt = sa.select(sa.distinct(tag_expr)).where(
                sa.and_(
                    ThreadEventDB.event_type.in_(["jims.backoffice.tag_added", "jims.backoffice.tag_removed"]),
                    tag_expr.is_not(None),
                )
            )
            raw_tags = (await session.execute(tag_stmt)).scalars().all()
            self.available_tags = sorted([str(t).strip() for t in raw_tags if str(t or "").strip()])

            last_event_sq = (
                sa.select(
                    ThreadEventDB.thread_id.label("t_id"),
                    sa.func.max(ThreadEventDB.created_at).label("last_at"),
                )
                .group_by(ThreadEventDB.thread_id)
                .subquery()
            )
            last_chat_sq = (
                sa.select(
                    ThreadEventDB.thread_id.label("t_id"),
                    sa.func.max(ThreadEventDB.created_at).label("last_chat_at"),
                )
                .where(sa.not_(ThreadEventDB.event_type.like("jims.backoffice.%")))
                .group_by(ThreadEventDB.thread_id)
                .subquery()
            )
            sev_expr = ThreadEventDB.event_data["severity"].as_string()
            priority_rank_expr = sa.case((sev_expr == "High", 2), (sev_expr == "Medium", 1), else_=0)
            priority_sq = (
                sa.select(
                    ThreadEventDB.thread_id.label("t_id"),
                    sa.func.max(priority_rank_expr).label("priority_rank"),
                )
                .where(ThreadEventDB.event_type == "jims.backoffice.feedback")
                .group_by(ThreadEventDB.thread_id)
                .subquery()
            )
            activity_expr = sa.func.coalesce(last_chat_sq.c.last_chat_at, last_event_sq.c.last_at, ThreadDB.created_at)

            base_stmt = (
                sa.select(
                    ThreadDB.thread_id.label("thread_id"),
                    ThreadDB.created_at.label("created_at"),
                    ThreadDB.thread_config.label("thread_config"),
                    last_event_sq.c.last_at.label("last_at"),
                    last_chat_sq.c.last_chat_at.label("last_chat_at"),
                    sa.func.coalesce(priority_sq.c.priority_rank, -1).label("priority_rank"),
                )
                .join(last_event_sq, last_event_sq.c.t_id == ThreadDB.thread_id)
                .outerjoin(last_chat_sq, last_chat_sq.c.t_id == ThreadDB.thread_id)
                .outerjoin(priority_sq, priority_sq.c.t_id == ThreadDB.thread_id)
            )

            if from_dt is not None:
                base_stmt = base_stmt.where(last_event_sq.c.last_at >= from_dt)
            if to_dt is not None:
                base_stmt = base_stmt.where(last_event_sq.c.last_at < to_dt)

            search = (self.search_text or "").strip().lower()
            if search:
                pattern = f"%{search}%"
                base_stmt = base_stmt.where(
                    sa.or_(
                        sa.func.lower(sa.cast(ThreadDB.thread_id, sa.String)).like(pattern),
                        sa.func.lower(sa.cast(ThreadDB.thread_config, sa.String)).like(pattern),
                    )
                )

            sort_val = self.sort_by.removeprefix("Sort by:").strip()
            if sort_val == "Priority":
                if self.sort_reverse:
                    base_stmt = base_stmt.order_by(sa.desc("priority_rank"), sa.desc(activity_expr))
                else:
                    base_stmt = base_stmt.order_by(sa.asc("priority_rank"), sa.asc(activity_expr))
            elif sort_val == "Date":
                base_stmt = base_stmt.order_by(sa.desc(activity_expr) if self.sort_reverse else sa.asc(activity_expr))
            else:
                base_stmt = base_stmt.order_by(sa.desc(activity_expr))

            rf = (self.review_filter.removeprefix("Review: ") or "All").strip()
            sel = set(self.selected_tags or [])
            needs_chunk_filtering = (rf and rf != "All") or bool(sel)

            rows_for_page: list[Any] = []
            review_status_by_tid: dict[str, str] = {}
            priority_by_tid: dict[str, str] = {}
            thread_tags_by_tid: dict[str, set[str]] = {}

            if not needs_chunk_filtering:
                total_stmt = sa.select(sa.func.count()).select_from(base_stmt.subquery())
                total_rows = int((await session.execute(total_stmt)).scalar() or 0)
                max_page = (total_rows - 1) // self.page_size if total_rows > 0 else 0
                if self.current_page > max_page:
                    self.current_page = max_page
                offset_rows = self.current_page * self.page_size
                page_stmt = base_stmt.limit(self.page_size).offset(offset_rows)
                rows_for_page = list((await session.execute(page_stmt)).all())

                page_tids = [str(r.thread_id) for r in rows_for_page]
                bo_rows: list[ThreadEventDB] = []
                if page_tids:
                    bo_stmt = (
                        sa.select(ThreadEventDB)
                        .where(
                            sa.and_(
                                ThreadEventDB.thread_id.in_(page_tids),
                                ThreadEventDB.event_type.like("jims.backoffice.%"),
                            )
                        )
                        .order_by(ThreadEventDB.created_at.asc())
                    )
                    bo_rows = list((await session.execute(bo_stmt)).scalars().all())
                review_status_by_tid, priority_by_tid, thread_tags_by_tid = _compute_backoffice_maps(page_tids, bo_rows)
                self.total_threads = total_rows
            else:
                # When review/tag filters are active, evaluate filtered rows in chunks.
                chunk_size = 200
                matched_total = 0
                page_start = self.current_page * self.page_size
                page_end = page_start + self.page_size
                scan_offset = 0

                while True:
                    chunk_stmt = base_stmt.limit(chunk_size).offset(scan_offset)
                    chunk_rows = list((await session.execute(chunk_stmt)).all())
                    if not chunk_rows:
                        break
                    scan_offset += len(chunk_rows)

                    chunk_tids = [str(r.thread_id) for r in chunk_rows]
                    chunk_bo_rows: list[ThreadEventDB] = []
                    if chunk_tids:
                        bo_stmt = (
                            sa.select(ThreadEventDB)
                            .where(
                                sa.and_(
                                    ThreadEventDB.thread_id.in_(chunk_tids),
                                    ThreadEventDB.event_type.like("jims.backoffice.%"),
                                )
                            )
                            .order_by(ThreadEventDB.created_at.asc())
                        )
                        chunk_bo_rows = list((await session.execute(bo_stmt)).scalars().all())
                    chunk_review_by_tid, chunk_priority_by_tid, chunk_tags_by_tid = _compute_backoffice_maps(
                        chunk_tids, chunk_bo_rows
                    )

                    for row in chunk_rows:
                        tid = str(row.thread_id)
                        review_val = chunk_review_by_tid.get(tid, "-")
                        if rf and rf != "All" and review_val != rf:
                            continue
                        if sel and len(sel.intersection(chunk_tags_by_tid.get(tid, set()))) == 0:
                            continue

                        if page_start <= matched_total < page_end:
                            rows_for_page.append(row)
                            review_status_by_tid[tid] = review_val
                            priority_by_tid[tid] = chunk_priority_by_tid.get(tid, "-")
                            thread_tags_by_tid[tid] = set(chunk_tags_by_tid.get(tid, set()))
                        matched_total += 1

                self.total_threads = matched_total
                max_page = (self.total_threads - 1) // self.page_size if self.total_threads > 0 else 0
                if self.current_page > max_page:
                    self.current_page = max_page
                    await self.get_data()  # type: ignore[operator]
                    return

            items: list[ThreadVis] = []
            for row in rows_for_page:
                tid = str(row.thread_id)
                sample = sorted(list(thread_tags_by_tid.get(tid, set())))[:3]
                items.append(
                    ThreadVis.create(
                        thread_id=tid,
                        created_at=row.created_at,
                        thread_config=row.thread_config,
                        review_status=review_status_by_tid.get(tid, "-"),
                        priority=priority_by_tid.get(tid, "-"),
                        tags_sample=sample,
                    )
                )

            self.threads = items
            self.threads_refreshing = False


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
            rx.hstack(
                rx.input(
                    value=ThreadListState.from_date,
                    type="date",
                    on_change=ThreadListState.set_from_date,
                ),
                rx.text("-"),
                rx.input(
                    value=ThreadListState.to_date,
                    type="date",
                    on_change=ThreadListState.set_to_date,
                ),
                align="center",
                spacing="1",
            ),
            rx.select(
                items=["All", "Pending", "Complete"],
                placeholder="Review: All",
                on_change=ThreadListState.set_review_filter,
                width="180px",
            ),
            rx.dialog.root(
                rx.dialog.trigger(
                    rx.button(
                        rx.cond(
                            ThreadListState.selected_tags.length() > 0,  # type: ignore[attr-defined]
                            "Tags: " + ThreadListState.selected_tags.join(", "),  # type: ignore[attr-defined]
                            "Tags: All",
                        ),
                        variant="soft",
                        color_scheme="gray",
                    )
                ),
                rx.dialog.content(
                    rx.vstack(
                        rx.hstack(
                            rx.dialog.title("Filter by Tags"),
                            rx.dialog.close(rx.button("Close", variant="ghost", size="1")),
                            justify="between",
                            align="center",
                            width="100%",
                        ),
                        rx.scroll_area(
                            rx.vstack(
                                rx.foreach(
                                    ThreadListState.available_tags,
                                    lambda t: rx.checkbox(
                                        t,
                                        checked=ThreadListState.selected_tags.contains(t),  # type: ignore[attr-defined]
                                        on_change=lambda v, tag=t: ThreadListState.toggle_tag_filter(tag=tag, value=v),  # type: ignore[operator]
                                    ),
                                ),
                                spacing="2",
                                width="100%",
                            ),
                            type="always",
                            scrollbars="vertical",
                        ),
                        rx.hstack(
                            rx.dialog.close(
                                rx.button(
                                    "Apply",
                                    on_click=[ThreadListState.reset_pagination, ThreadListState.get_data],
                                    size="1",
                                )
                            ),
                            rx.button(
                                "Clear",
                                variant="soft",
                                on_click=[ThreadListState.clear_tag_filter, ThreadListState.get_data],
                                size="1",
                            ),
                            spacing="2",
                            justify="end",
                            width="100%",
                        ),
                        spacing="3",
                    )
                ),
            ),
            rx.hstack(
                rx.select(
                    items=["Date", "Priority"],
                    on_change=ThreadListState.set_sort_by,
                    placeholder="Sort By: Date",
                    width="160px",
                ),
                rx.cond(
                    ThreadListState.sort_reverse,
                    rx.icon(
                        "arrow-down-1-0",
                        size=28,
                        stroke_width=1.5,
                        cursor="pointer",
                        flex_shrink="0",
                        on_click=ThreadListState.toggle_sort,
                    ),  # type: ignore
                    rx.icon(
                        "arrow-down-0-1",
                        size=28,
                        stroke_width=1.5,
                        cursor="pointer",
                        flex_shrink="0",
                        on_click=ThreadListState.toggle_sort,
                    ),  # type: ignore
                ),
                spacing="0",
            ),
            rx.input(
                placeholder="Search thread or interface",
                value=ThreadListState.search_text,
                on_change=ThreadListState.set_search_text,
                width="280px",
            ),
        ),
        rx.hstack(
            rx.button("Search", on_click=[ThreadListState.reset_pagination, ThreadListState.get_data]),
            rx.button(
                "Clear",
                variant="soft",
                on_click=[ThreadListState.clear_filters, ThreadListState.get_data],
            ),
            spacing="2",
        ),
        justify="between",
        width="100%",
        align="center",
        wrap="wrap",
    )

    table = rx.table.root(
        rx.table.header(
            rx.table.row(
                rx.table.column_header_cell("Thread ID"),
                rx.table.column_header_cell("Created"),
                rx.table.column_header_cell("Age"),
                rx.table.column_header_cell("Interface"),
                rx.table.column_header_cell("Tags"),
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
                    rx.table.cell(
                        rx.hstack(
                            rx.cond(t.tag1 != "", rx.badge(t.tag1, variant="soft", size="1", color_scheme="gray")),
                            rx.cond(t.tag2 != "", rx.badge(t.tag2, variant="soft", size="1", color_scheme="gray")),
                            rx.cond(t.tag3 != "", rx.badge(t.tag3, variant="soft", size="1", color_scheme="gray")),
                            spacing="1",
                        )
                    ),
                    rx.table.cell(review_badge(t.review_status)),
                    rx.table.cell(priority_badge(t.priority)),
                    style=rx.cond(
                        t.thread_id == ThreadViewState.selected_thread_id, {"backgroundColor": "var(--accent-3)"}, {}
                    ),
                ),
            ),
        ),
    )

    pagination_controls = rx.hstack(
        rx.text(ThreadListState.rows_display, size="2", color="gray"),
        rx.spacer(),
        rx.hstack(
            rx.button("⏮", variant="soft", size="1", on_click=ThreadListState.first_page, disabled=~ThreadListState.has_prev_page),  # type: ignore[operator]
            rx.button(
                "← Prev",
                variant="soft",
                size="1",
                on_click=ThreadListState.prev_page,
                disabled=~ThreadListState.has_prev_page,
            ),  # type: ignore[operator]
            rx.text(ThreadListState.page_display, size="2", style={"minWidth": "110px", "textAlign": "center"}),
            rx.button(
                "Next →",
                variant="soft",
                size="1",
                on_click=ThreadListState.next_page,
                disabled=~ThreadListState.has_next_page,
            ),  # type: ignore[operator]
            rx.button("⏭", variant="soft", size="1", on_click=ThreadListState.last_page, disabled=~ThreadListState.has_next_page),  # type: ignore[operator]
            spacing="2",
            align="center",
        ),
        width="100%",
        align="center",
        padding_top="0.5em",
    )

    def _render_event_as_msg(ev):  # type: ignore[valid-type]
        msg = {
            "id": ev.event_id,
            "content": ev.content,
            "created_at": ev.created_at_str,
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
            "event_data_str": ev.event_data_str,
            "generic_meta": ev.generic_meta,
            "dm_snapshot_id": ev.dm_snapshot_id,
        }

        tag_dialog = rx.dialog.root(
            rx.dialog.content(
                rx.vstack(
                    rx.hstack(
                        rx.dialog.title("Add Tags"),
                        rx.dialog.close(
                            rx.button("Close", variant="ghost", size="1", on_click=ThreadViewState.close_tag_dialog)
                        ),
                        justify="between",
                        align="center",
                        width="100%",
                    ),
                    rx.hstack(
                        rx.input(
                            placeholder="Add new tag...",
                            value=ThreadViewState.new_tag_text_for_event.get(ev.event_id, ""),
                            on_change=lambda v: ThreadViewState.set_new_tag_text_for_event(v, event_id=ev.event_id),
                            width="100%",
                        ),
                        rx.button(
                            "Add",
                            size="1",
                            on_click=ThreadViewState.add_new_tag_to_available(event_id=ev.event_id),
                        ),
                        spacing="2",
                        width="100%",
                    ),
                    rx.scroll_area(
                        rx.vstack(
                            rx.foreach(
                                ThreadViewState.available_tags,
                                lambda t: rx.checkbox(
                                    t,
                                    checked=ThreadViewState.selected_tags_for_event.get(ev.event_id, []).contains(t),  # type: ignore[attr-defined]
                                    on_change=lambda v, tag=t: ThreadViewState.toggle_tag_selection_for_event(
                                        tag=tag, event_id=ev.event_id, checked=v
                                    ),  # type: ignore[operator]
                                ),
                            ),
                            spacing="2",
                            width="100%",
                        ),
                        type="always",
                        scrollbars="vertical",
                        style={"width": "100%", "padding": "0"},
                    ),
                    rx.hstack(
                        rx.button(
                            "Apply",
                            size="1",
                            on_click=ThreadViewState.apply_tags_to_event(event_id=ev.event_id),  # type: ignore[call-arg,func-returns-value]
                        ),
                        spacing="2",
                        justify="end",
                        width="100%",
                    ),
                    spacing="3",
                ),
            ),
            open=ThreadViewState.tag_dialog_open_for_event == ev.event_id,
            on_open_change=ThreadViewState.handle_tag_dialog_open_change,  # type: ignore[operator]
        )

        action_line = rx.cond(
            (ev.role == "assistant") & ev.content != "",
            rx.hstack(
                rx.button(
                    "Add tag",
                    size="1",
                    on_click=ThreadViewState.open_tag_dialog(event_id=ev.event_id),  # type: ignore[call-arg,func-returns-value]
                ),
                tag_dialog,
                # Add Note (single-line input)
                rx.input(
                    placeholder="Add note...",
                    value=ThreadViewState.note_text_by_event.get(ev.event_id, ""),
                    on_change=lambda v: ThreadViewState.set_note_text_for(v, event_id=ev.event_id),  # type: ignore[call-arg,func-returns-value]
                    width="40%",
                    size="1",
                ),
                rx.select(
                    items=["Low", "Medium", "High"],
                    value=ThreadViewState.note_severity_by_event.get(ev.event_id, "Low"),
                    on_change=lambda v: ThreadViewState.set_note_severity_for(v, event_id=ev.event_id),  # type: ignore[call-arg,func-returns-value]
                    size="1",
                ),
                rx.button(
                    "Add note",
                    size="1",
                    on_click=ThreadViewState.submit_note_for(event_id=ev.event_id),  # type: ignore[call-arg,func-returns-value]
                ),
                spacing="2",
                wrap="wrap",
                width="100%",
                align="center",
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
                            rx.spacer(),
                            # Status badge
                            rx.cond(
                                c.get("status", "open") == "resolved",
                                rx.badge("Resolved", variant="soft", size="1", color_scheme="green"),
                                rx.cond(
                                    c.get("status", "open") == "closed",
                                    rx.badge("Ignored", variant="soft", size="1", color_scheme="gray"),
                                    rx.box(),
                                ),
                            ),
                            spacing="2",
                            align="center",
                            width="100%",
                        ),
                        rx.text(
                            c["note"],
                            style={
                                "whiteSpace": "pre-wrap",
                                "wordBreak": "break-word",
                            },
                        ),
                        # Actions row
                        rx.hstack(
                            rx.button(
                                "✔",
                                variant="soft",
                                size="1",
                                color_scheme="green",
                                disabled=c.get("status", "open") != "open",
                                on_click=ThreadViewState.mark_comment_resolved(comment_id=c["id"]),  # type: ignore[call-arg,func-returns-value]
                            ),
                            rx.button(
                                "✖",
                                variant="soft",
                                size="1",
                                color_scheme="gray",
                                disabled=c.get("status", "open") != "open",
                                on_click=ThreadViewState.mark_comment_closed(comment_id=c["id"]),  # type: ignore[call-arg,func-returns-value]
                            ),
                            spacing="2",
                        ),
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

        def _tag_badge(tag: str):  # type: ignore[valid-type]
            return rx.badge(
                rx.hstack(
                    rx.text(tag),
                    rx.button(
                        "×",
                        variant="ghost",
                        size="1",
                        color_scheme="gray",
                        on_click=ThreadViewState.remove_tag(event_id=ev.event_id, tag=tag),  # type: ignore[operator, call-arg,func-returns-value]
                    ),
                    spacing="1",
                ),
                variant="soft",
                size="1",
                color_scheme="gray",
            )

        tags_component = rx.hstack(
            rx.foreach(ev.visible_tags, _tag_badge),
            spacing="1",
        )

        return render_message_bubble(
            msg,
            on_toggle_details=ThreadViewState.toggle_details(event_id=ev.event_id),  # type: ignore[call-arg,func-returns-value]
            extras=extras,
            corner_tags_component=tags_component,
        )

    filters_box = rx.box(
        filters,
        margin_bottom="1em",
    )

    # Left panel (thread list with its own scroll)
    left_panel = rx.box(
        rx.vstack(
            rx.cond(
                ThreadListState.threads_refreshing,
                rx.center("Loading threads..."),
                rx.scroll_area(
                    table,
                    type="always",
                    scrollbars="vertical",
                    style={"height": "100%"},
                ),
            ),
            pagination_controls,
            width="100%",
            spacing="2",
            height="100%",
        ),
        flex="1",
        min_height="0",
        style={"height": "100%", "overflow": "hidden"},
    )

    # Right panel (conversation fills height with scroll)
    right_panel = rx.cond(
        ThreadViewState.selected_thread_id == "",
        rx.center(rx.text("Select a thread to view conversation"), style={"height": "100%"}),
        rx.box(
            rx.scroll_area(
                rx.vstack(
                    rx.foreach(ThreadViewState.events, _render_event_as_msg),
                    spacing="3",
                    width="100%",
                    padding_bottom="1em",
                    style={
                        "maxWidth": "100%",
                        "overflowX": "hidden",
                    },
                ),
                type="always",
                scrollbars="vertical",
                style={
                    "height": "100%",
                    "maxWidth": "100%",
                    "overflowX": "hidden",
                },
            ),
            flex="1",
            min_height="0",
            style={"height": "100%", "overflow": "hidden"},
        ),
    )

    return rx.vstack(
        app_header(),
        filters_box,
        rx.box(
            rx.grid(
                left_panel,
                right_panel,
                columns="2",
                spacing="4",
                sm_columns="1",
                width="100%",
                style={"height": "100%"},
            ),
            flex="1",
            min_height="0",
            width="100%",
        ),
        spacing="4",
        height="100vh",
        overflow="hidden",
        width="100%",
    )
