import pprint
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

import orjson as json
import reflex as rx
import sqlalchemy as sa
from jims_core.db import ThreadDB, ThreadEventDB
from jims_core.util import uuid7

from jims_backoffice.app_loader import get_jims_app
from jims_backoffice.util import csv_bytes, datetime_to_age

THREAD_EVENT_COLUMNS: list[str] = [
    "thread_id",
    "event_id",
    "created_at",
    "event_type",
    "event_domain",
    "event_name",
    "event_data",
]


def _thread_events_to_rows(events: list[Any]) -> list[dict[str, Any]]:
    """Flatten ThreadEventDB rows into dicts suitable for the CSV helper."""
    rows: list[dict[str, Any]] = []
    for ev in events:
        created_at = getattr(ev, "created_at", None)
        rows.append(
            {
                "thread_id": str(getattr(ev, "thread_id", "")),
                "event_id": str(getattr(ev, "event_id", "")),
                "created_at": (
                    created_at.isoformat(sep=" ") if isinstance(created_at, datetime) else str(created_at or "")
                ),
                "event_type": str(getattr(ev, "event_type", "") or ""),
                "event_domain": str(getattr(ev, "event_domain", "") or ""),
                "event_name": str(getattr(ev, "event_name", "") or ""),
                "event_data": getattr(ev, "event_data", {}) or {},
            }
        )
    return rows


@dataclass
class ThreadEventVis:
    event_id: str
    created_at: datetime
    created_at_str: str
    event_type: str
    role: str
    content: str
    tags: list[str]
    event_data_str: str
    technical_vts_queries: list[str]
    technical_cypher_queries: list[str]
    technical_models: list[tuple[str, str]]
    vts_str: str
    cypher_str: str
    models_str: str
    has_technical_info: bool
    has_vts: bool
    has_cypher: bool
    has_models: bool
    # Aggregated annotations from jims.backoffice.* events
    visible_tags: list[str] = field(default_factory=list)
    feedback_comments: list[dict[str, Any]] = field(default_factory=list)
    generic_meta: bool = False

    @classmethod
    def create(cls, event_id: Any, created_at: datetime, event_type: str, event_data: dict) -> "ThreadEventVis":
        tech: dict = event_data.get("technical_info", {})
        has_technical_info = bool(tech)

        # Role: only comm.user_* is user; all others assistant.
        role = "user" if event_type.startswith("comm.user") else "assistant"
        content: str = event_data.get("content", "") if event_type.startswith("comm.") else ""
        tags_value = event_data.get("tags")
        tags: list[str] = list(tags_value or []) if isinstance(tags_value, (list, tuple)) else []

        vts_queries: list[str] = list(tech.get("vts_queries", []) or [])
        cypher_queries: list[str] = list(tech.get("cypher_queries", []) or [])

        model_stats = tech.get("model_stats", {}) or tech.get("model_used", {}) or {}
        models_list: list[tuple[str, str]] = []
        try:
            for mk, mv in model_stats.items() if isinstance(model_stats, dict) else []:
                try:
                    models_list.append((f'"{mk}"', json.dumps(mv, option=json.OPT_INDENT_2).decode()))
                except Exception:
                    models_list.append((f'"{mk}"', str(mv)))
        except Exception:
            pass

        vts_str = "\n".join(vts_queries)
        cypher_str = "\n".join([pprint.pformat(x)[1:-1].replace("'", "") for x in cypher_queries])
        models_str = "\n".join([f"{k}: {v}" for k, v in models_list])

        # Show meta (event_data) for events that are NOT comm.* and NOT rag.query_processed
        generic_meta = False
        if not event_type.startswith("comm.") and event_type != "rag.query_processed":
            generic_meta = True

        return cls(
            event_id=str(event_id),
            created_at=created_at.replace(microsecond=0),
            created_at_str=datetime.strftime(created_at, "%Y-%m-%d %H:%M:%S"),
            event_type=event_type,
            role=role,
            content=content,
            tags=tags,
            event_data_str=json.dumps(event_data, option=json.OPT_INDENT_2).decode(),
            technical_vts_queries=vts_queries,
            technical_cypher_queries=cypher_queries,
            technical_models=models_list,
            vts_str=vts_str,
            cypher_str=cypher_str,
            models_str=models_str,
            has_technical_info=has_technical_info,
            has_vts=bool(vts_queries),
            has_cypher=bool(cypher_queries),
            has_models=bool(models_list),
            visible_tags=list(tags),
            feedback_comments=[],
            generic_meta=generic_meta,
        )


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
        )


class ThreadListState(rx.State):
    threads_refreshing: bool = True
    threads: list[ThreadVis] = []

    from_date: str = ""
    to_date: str = ""
    search_text: str = ""
    sort_reverse: bool = True
    review_filter: str = "Review: All"
    sort_by: str = "Sort by: Date"
    available_tags: list[str] = []
    selected_tags: list[str] = []
    available_interfaces: list[str] = []
    selected_interface: str = ""
    page_size: int = 20
    current_page: int = 0
    total_threads: int = 0

    bulk_export_dialog_open: bool = False

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
    async def set_selected_interface(self, value: str) -> None:
        self.selected_interface = str(value or "").strip()
        self.current_page = 0
        await self.get_data()  # type: ignore[operator]

    @rx.event
    def clear_filters(self) -> None:
        self.from_date = ""
        self.to_date = ""
        self.search_text = ""
        self.sort_reverse = True
        self.review_filter = "Review: All"
        self.sort_by = "Sort by: Date"
        self.selected_tags = []
        self.selected_interface = ""
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
        t = str(tag)
        if value:
            if t not in self.selected_tags:
                self.selected_tags = [*self.selected_tags, t]
        else:
            self.selected_tags = [x for x in self.selected_tags if x != t]

    @rx.event
    def clear_tag_filter(self) -> None:
        self.selected_tags = []
        self.current_page = 0

    @rx.event
    def reset_pagination(self) -> None:
        self.current_page = 0

    @rx.event
    def open_bulk_export_dialog(self) -> None:
        self.bulk_export_dialog_open = True

    @rx.event
    def close_bulk_export_dialog(self) -> None:
        self.bulk_export_dialog_open = False

    @rx.event
    def handle_bulk_export_dialog_open_change(self, is_open: bool) -> None:
        if not is_open:
            self.bulk_export_dialog_open = False

    @rx.event
    async def confirm_bulk_export(self):
        """Stream all matching thread_events into a CSV download."""
        self.bulk_export_dialog_open = False
        jims_app = await get_jims_app()
        async with jims_app.sessionmaker() as session:
            ids = await self._collect_filtered_thread_ids(session)
            if not ids:
                return rx.toast.info("No threads match the current filters.")
            stmt = (
                sa.select(ThreadEventDB)
                .where(ThreadEventDB.thread_id.in_(ids))
                .order_by(ThreadEventDB.thread_id.asc(), ThreadEventDB.created_at.asc())
            )
            events = (await session.execute(stmt)).scalars().all()
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        data = csv_bytes(_thread_events_to_rows(events), THREAD_EVENT_COLUMNS)
        return rx.download(data=data, filename=f"jims_threads_{ts}.csv")

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

    @rx.var
    def bulk_export_confirm_message(self) -> str:
        return (
            f"About to export thread_events for {self.total_threads} threads. Continue?"
        )

    @staticmethod
    def _compute_backoffice_maps(
        thread_ids: list[str], bo_rows: list[ThreadEventDB]
    ) -> tuple[dict[str, str], dict[str, str], dict[str, set[str]]]:
        review_status_by_tid: dict[str, str] = {tid: "-" for tid in thread_ids}
        priority_by_tid: dict[str, str] = {tid: "-" for tid in thread_ids}

        has_resolved: dict[str, bool] = {tid: False for tid in thread_ids}
        tags_by_event: dict[str, set[str]] = {}

        # Per-comment: tid, severity, final status (folded from status events in order).
        comment_info: dict[str, dict[str, Any]] = {}

        bo_rows_sorted = sorted(bo_rows, key=lambda r: getattr(r, "created_at", datetime.min))
        for ev in bo_rows_sorted:
            tid = str(ev.thread_id)
            etype = str(getattr(ev, "event_type", ""))
            if etype == "jims.backoffice.feedback":
                try:
                    sev = str((getattr(ev, "event_data", {}) or {}).get("severity", "Low"))
                except Exception:
                    sev = "Low"
                cid = str(getattr(ev, "event_id", ""))
                if cid:
                    comment_info[cid] = {"tid": tid, "severity": sev, "status": "open", "deleted": False}
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
            elif etype in (
                "jims.backoffice.comment_resolved",
                "jims.backoffice.comment_closed",
                "jims.backoffice.comment_reopened",
                "jims.backoffice.comment_deleted",
            ):
                ed = dict(getattr(ev, "event_data", {}) or {})
                cid = str(ed.get("comment_id", ""))
                if not cid or cid not in comment_info:
                    continue
                if etype == "jims.backoffice.comment_deleted":
                    comment_info[cid]["deleted"] = True
                elif etype == "jims.backoffice.comment_reopened":
                    comment_info[cid]["status"] = "open"
                elif etype == "jims.backoffice.comment_resolved":
                    comment_info[cid]["status"] = "resolved"
                else:
                    comment_info[cid]["status"] = "closed"

        # Aggregate per thread from final comment states (deleted comments are ignored).
        has_feedback: dict[str, bool] = {tid: False for tid in thread_ids}
        unresolved_by_tid: dict[str, int] = {tid: 0 for tid in thread_ids}
        priority_rank_by_tid: dict[str, int] = {tid: -1 for tid in thread_ids}
        rank_map = {"Low": 0, "Medium": 1, "High": 2}
        for info in comment_info.values():
            if info.get("deleted"):
                continue
            tid = info["tid"]
            if tid not in has_feedback:
                continue
            has_feedback[tid] = True
            if info["status"] == "open":
                unresolved_by_tid[tid] = unresolved_by_tid.get(tid, 0) + 1
                rank = rank_map.get(info["severity"], 0)
                if rank > priority_rank_by_tid.get(tid, -1):
                    priority_rank_by_tid[tid] = rank
                    priority_by_tid[tid] = {0: "Low", 1: "Medium", 2: "High"}.get(rank, "Low")

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

    def _build_filter_base_stmt(self) -> tuple[sa.Select, Any]:
        """Build the filtered base_stmt for the thread list (no pagination, no ordering).

        Applies date / search / interface filters. Review/tag filters are not applied here;
        callers that need those must do chunk-scan filtering in Python via
        `_compute_backoffice_maps`.

        Returns ``(base_stmt, activity_expr)``. The select projects
        (thread_id, created_at, thread_config, last_at, last_chat_at, priority_rank).
        """
        from_dt = self._parse_date(self.from_date)
        to_dt = self._parse_date(self.to_date)
        if to_dt is not None:
            to_dt = to_dt + timedelta(days=1)

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
        deleted_cid_expr = ThreadEventDB.event_data["comment_id"].as_string()
        deleted_cids_sq = (
            sa.select(deleted_cid_expr.label("comment_id"))
            .where(ThreadEventDB.event_type == "jims.backoffice.comment_deleted")
            .distinct()
            .subquery()
        )
        status_cid_expr = ThreadEventDB.event_data["comment_id"].as_string()
        status_events_sq = (
            sa.select(
                status_cid_expr.label("comment_id"),
                ThreadEventDB.event_type.label("etype"),
                sa.func.row_number()
                .over(
                    partition_by=status_cid_expr,
                    order_by=ThreadEventDB.created_at.desc(),
                )
                .label("rn"),
            )
            .where(
                ThreadEventDB.event_type.in_(
                    [
                        "jims.backoffice.comment_resolved",
                        "jims.backoffice.comment_closed",
                        "jims.backoffice.comment_reopened",
                    ]
                )
            )
            .subquery()
        )
        latest_closed_cids_sq = sa.select(status_events_sq.c.comment_id).where(
            sa.and_(
                status_events_sq.c.rn == 1,
                status_events_sq.c.etype.in_(
                    [
                        "jims.backoffice.comment_resolved",
                        "jims.backoffice.comment_closed",
                    ]
                ),
            )
        )

        sev_expr = ThreadEventDB.event_data["severity"].as_string()
        priority_rank_expr = sa.case((sev_expr == "High", 2), (sev_expr == "Medium", 1), else_=0)
        feedback_event_id_str = sa.cast(ThreadEventDB.event_id, sa.String)
        priority_sq = (
            sa.select(
                ThreadEventDB.thread_id.label("t_id"),
                sa.func.max(priority_rank_expr).label("priority_rank"),
            )
            .where(
                sa.and_(
                    ThreadEventDB.event_type == "jims.backoffice.feedback",
                    feedback_event_id_str.notin_(sa.select(deleted_cids_sq.c.comment_id)),
                    feedback_event_id_str.notin_(latest_closed_cids_sq),
                )
            )
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
                sa.func.lower(sa.cast(ThreadDB.thread_id, sa.String)).like(pattern)
            )

        iface_filter = (self.selected_interface or "").strip()
        if iface_filter:
            base_stmt = base_stmt.where(
                sa.or_(
                    ThreadDB.thread_config["interface"].as_string() == iface_filter,
                    ThreadDB.thread_config["channel"].as_string() == iface_filter,
                    ThreadDB.thread_config["source"].as_string() == iface_filter,
                )
            )

        return base_stmt, activity_expr

    async def _collect_filtered_thread_ids(self, session: Any) -> list[Any]:
        """Collect all thread_ids matching the active filters (review/tag included).

        Iterates page-by-page through the filtered base query, applying review_filter
        and selected_tags in Python via `_compute_backoffice_maps`.
        """
        base_stmt, activity_expr = self._build_filter_base_stmt()
        base_stmt = base_stmt.order_by(sa.desc(activity_expr))

        rf = (self.review_filter.removeprefix("Review: ") or "All").strip()
        sel = set(self.selected_tags or [])
        needs_chunk_filtering = (rf and rf != "All") or bool(sel)

        if not needs_chunk_filtering:
            filtered = base_stmt.subquery()
            id_stmt = sa.select(filtered.c.thread_id)
            return list((await session.execute(id_stmt)).scalars().all())

        matched: list[Any] = []
        chunk_size = 200
        scan_offset = 0
        while True:
            chunk_stmt = base_stmt.limit(chunk_size).offset(scan_offset)
            chunk_rows = list((await session.execute(chunk_stmt)).all())
            if not chunk_rows:
                break
            scan_offset += len(chunk_rows)
            chunk_tids = [str(r.thread_id) for r in chunk_rows]
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
            chunk_review_by_tid, _chunk_priority_by_tid, chunk_tags_by_tid = self._compute_backoffice_maps(
                chunk_tids, chunk_bo_rows
            )
            for row in chunk_rows:
                tid = str(row.thread_id)
                review_val = chunk_review_by_tid.get(tid, "-")
                if rf and rf != "All" and review_val != rf:
                    continue
                if sel and len(sel.intersection(chunk_tags_by_tid.get(tid, set()))) == 0:
                    continue
                matched.append(row.thread_id)
        return matched

    @rx.event
    async def get_data(self) -> None:
        """compiles the entire thread list table"""
        self.threads_refreshing = True
        jims_app = await get_jims_app()

        async with jims_app.sessionmaker() as session:
            tag_expr = ThreadEventDB.event_data["tag"].as_string()
            tag_stmt = sa.select(sa.distinct(tag_expr)).where(
                sa.and_(
                    ThreadEventDB.event_type.in_(["jims.backoffice.tag_added", "jims.backoffice.tag_removed"]),
                    tag_expr.is_not(None),
                )
            )
            raw_tags = (await session.execute(tag_stmt)).scalars().all()
            self.available_tags = sorted([str(t).strip() for t in raw_tags if str(t or "").strip()])

            # Populate distinct interface values from thread_config (interface | channel | source).
            iface_expr = sa.func.coalesce(
                ThreadDB.thread_config["interface"].as_string(),
                ThreadDB.thread_config["channel"].as_string(),
                ThreadDB.thread_config["source"].as_string(),
            )
            iface_stmt = sa.select(sa.distinct(iface_expr)).where(iface_expr.is_not(None))
            raw_ifaces = (await session.execute(iface_stmt)).scalars().all()
            self.available_interfaces = sorted([str(i).strip() for i in raw_ifaces if str(i or "").strip()])

            base_stmt, activity_expr = self._build_filter_base_stmt()

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
                review_status_by_tid, priority_by_tid, thread_tags_by_tid = self._compute_backoffice_maps(
                    page_tids, bo_rows
                )
                self.total_threads = total_rows
            else:
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
                    chunk_review_by_tid, chunk_priority_by_tid, chunk_tags_by_tid = self._compute_backoffice_maps(
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


class ThreadViewState(rx.State):
    loading: bool = True
    events: list[ThreadEventVis] = []
    history_page_size: int = 30
    history_visible_count: int = 30
    has_more_history: bool = False
    total_history_count: int = 0
    new_tag_text: str = ""
    note_text: str = ""
    note_severity: str = "Low"
    note_text_by_event: dict[str, str] = {}
    note_severity_by_event: dict[str, str] = {}
    selected_thread_id: str = ""
    expanded_event_id: str = ""
    tag_dialog_open_for_event: str = ""
    selected_tags_for_event: dict[str, list[str]] = {}
    new_tag_text_for_event: dict[str, str] = {}
    available_tags: list[str] = []

    async def _reload(self, reset_history_window: bool = False) -> None:
        jims_app = await get_jims_app()

        async with jims_app.sessionmaker() as session:
            if reset_history_window:
                self.history_visible_count = self.history_page_size

            base_events_stmt = (
                sa.select(ThreadEventDB)
                .where(
                    sa.and_(
                        ThreadEventDB.thread_id == self.selected_thread_id,
                        sa.not_(ThreadEventDB.event_type.like("jims.%")),
                    )
                )
                .order_by(ThreadEventDB.created_at.desc())
                .limit(self.history_visible_count)
            )
            base_events_desc = list((await session.execute(base_events_stmt)).scalars().all())
            base_events = list(reversed(base_events_desc))

            total_base_events_stmt = sa.select(sa.func.count()).where(
                sa.and_(
                    ThreadEventDB.thread_id == self.selected_thread_id,
                    sa.not_(ThreadEventDB.event_type.like("jims.%")),
                )
            )
            self.total_history_count = int((await session.execute(total_base_events_stmt)).scalar() or 0)
            self.has_more_history = self.total_history_count > len(base_events)

            target_event_ids = [str(getattr(ev, "event_id", "")) for ev in base_events if getattr(ev, "event_id", None)]

            backoffice_events: list[Any] = []
            if target_event_ids:
                target_event_expr = ThreadEventDB.event_data["target_event_id"].as_string()
                feedback_target_expr = ThreadEventDB.event_data["event_id"].as_string()
                bo_stmt = (
                    sa.select(ThreadEventDB)
                    .where(
                        sa.and_(
                            ThreadEventDB.thread_id == self.selected_thread_id,
                            sa.or_(
                                sa.and_(
                                    ThreadEventDB.event_type.in_(
                                        ["jims.backoffice.tag_added", "jims.backoffice.tag_removed"]
                                    ),
                                    target_event_expr.in_(target_event_ids),
                                ),
                                sa.and_(
                                    ThreadEventDB.event_type == "jims.backoffice.feedback",
                                    feedback_target_expr.in_(target_event_ids),
                                ),
                            ),
                        )
                    )
                    .order_by(ThreadEventDB.created_at.asc())
                )
                backoffice_events = list((await session.execute(bo_stmt)).scalars().all())
            else:
                backoffice_events = []

            feedback_ids = [
                str(getattr(ev, "event_id", ""))
                for ev in backoffice_events
                if ev.event_type == "jims.backoffice.feedback"
            ]
            if feedback_ids:
                comment_id_expr = ThreadEventDB.event_data["comment_id"].as_string()
                comment_status_stmt = (
                    sa.select(ThreadEventDB)
                    .where(
                        sa.and_(
                            ThreadEventDB.thread_id == self.selected_thread_id,
                            ThreadEventDB.event_type.in_(
                                [
                                    "jims.backoffice.comment_resolved",
                                    "jims.backoffice.comment_closed",
                                    "jims.backoffice.comment_reopened",
                                    "jims.backoffice.comment_deleted",
                                ]
                            ),
                            comment_id_expr.in_(feedback_ids),
                        )
                    )
                    .order_by(ThreadEventDB.created_at.asc())
                )
                backoffice_events.extend(list((await session.execute(comment_status_stmt)).scalars().all()))

        # Tags per original event
        tags_by_event: dict[str, set[str]] = {}
        for ev in base_events:
            eid = str(getattr(ev, "event_id", ""))
            try:
                base_tags = getattr(ev, "event_data", {}).get("tags") or []
                tags_by_event[eid] = set([str(t) for t in base_tags])
            except Exception:
                tags_by_event[eid] = set()

        for ev in backoffice_events:
            etype = str(getattr(ev, "event_type", ""))
            edata = dict(getattr(ev, "event_data", {}) or {})
            if etype == "jims.backoffice.tag_added":
                tid = str(edata.get("target_event_id", ""))
                tag = str(edata.get("tag", "")).strip()
                if tid:
                    tags_by_event.setdefault(tid, set()).add(tag)
            elif etype == "jims.backoffice.tag_removed":
                tid = str(edata.get("target_event_id", ""))
                tag = str(edata.get("tag", "")).strip()
                if tid and tag:
                    try:
                        tags_by_event.setdefault(tid, set()).discard(tag)
                    except Exception:
                        pass

        # Comments per original event + status mapping.
        # Process status events in chronological order so the last event wins
        # (e.g. reopen after resolve makes the comment open again).
        comments_by_event: dict[str, list[dict[str, Any]]] = {}
        comment_status: dict[str, str] = {}
        deleted_cids: set[str] = set()
        status_events_sorted = sorted(
            backoffice_events, key=lambda r: getattr(r, "created_at", datetime.min)
        )
        for ev in status_events_sorted:
            etype = str(getattr(ev, "event_type", ""))
            if etype == "jims.backoffice.feedback":
                edata = dict(getattr(ev, "event_data", {}) or {})
                target = str(edata.get("event_id", ""))
                if not target:
                    continue
                note_text = str(edata.get("note", ""))
                severity = str(edata.get("severity", "Low"))
                created_at = getattr(ev, "created_at", datetime.utcnow()).replace(microsecond=0)
                comments_by_event.setdefault(target, []).append(
                    {
                        "id": str(getattr(ev, "event_id", "")),
                        "note": note_text,
                        "severity": severity,
                        "created_at": datetime.strftime(created_at, "%Y-%m-%d %H:%M:%S"),
                        "status": "open",
                    }
                )
            elif etype == "jims.backoffice.comment_deleted":
                ed = dict(getattr(ev, "event_data", {}) or {})
                cid = str(ed.get("comment_id", ""))
                if cid:
                    deleted_cids.add(cid)
            elif etype in (
                "jims.backoffice.comment_resolved",
                "jims.backoffice.comment_closed",
                "jims.backoffice.comment_reopened",
            ):
                ed = dict(getattr(ev, "event_data", {}) or {})
                cid = str(ed.get("comment_id", ""))
                if not cid:
                    continue
                if etype == "jims.backoffice.comment_resolved":
                    comment_status[cid] = "resolved"
                elif etype == "jims.backoffice.comment_closed":
                    comment_status[cid] = "closed"
                else:
                    comment_status[cid] = "open"

        # Hide deleted comments entirely from the per-event lists.
        if deleted_cids:
            for target_id, lst in list(comments_by_event.items()):
                comments_by_event[target_id] = [c for c in lst if str(c.get("id", "")) not in deleted_cids]

        ev_items: list[ThreadEventVis] = []
        for bev in base_events:
            item = ThreadEventVis.create(
                event_id=bev.event_id,
                created_at=bev.created_at,
                event_type=bev.event_type,
                event_data=bev.event_data,
            )
            eid = item.event_id
            try:
                item.visible_tags = sorted(list(tags_by_event.get(eid, set())))
            except Exception:
                item.visible_tags = list(item.tags or [])
            try:
                cmts = []
                for c in comments_by_event.get(eid, []) or []:
                    c = dict(c)
                    cid = str(c.get("id", ""))
                    if cid in comment_status:
                        c["status"] = comment_status[cid]
                    cmts.append(c)
                item.feedback_comments = cmts
            except Exception:
                item.feedback_comments = []
            ev_items.append(item)

        self.events = ev_items

        all_tags: set[str] = set()
        for tags_set in tags_by_event.values():
            all_tags.update(tags_set)

        try:
            async with jims_app.sessionmaker() as session:
                tag_stmt = sa.select(ThreadEventDB.event_data).where(
                    ThreadEventDB.event_type == "jims.backoffice.tag_added"
                )
                tag_results = (await session.execute(tag_stmt)).scalars().all()
                for edata in tag_results:
                    try:
                        ed = dict(edata or {})
                        tag = str(ed.get("tag", "")).strip()
                        if tag:
                            all_tags.add(tag)
                    except Exception:
                        pass
        except Exception:
            pass

        self.available_tags = sorted(list(all_tags))
        self.loading = False

    @rx.event
    async def get_data(self):
        await self._reload(reset_history_window=True)

    @rx.event
    async def select_thread(self, thread_id: str) -> None:
        self.selected_thread_id = thread_id
        await self._reload(reset_history_window=True)

    @rx.event
    async def load_more_history(self) -> None:
        if not self.has_more_history:
            return
        self.history_visible_count += self.history_page_size
        await self._reload(reset_history_window=False)

    @rx.event
    def set_new_tag_text(self, value: str) -> None:
        self.new_tag_text = value

    @rx.event
    def set_note_text(self, value: str) -> None:
        self.note_text = value

    @rx.event
    def set_note_severity(self, value: str) -> None:
        self.note_severity = value

    @rx.event
    def toggle_details(self, event_id: str) -> None:
        self.expanded_event_id = "" if self.expanded_event_id == event_id else event_id

    @rx.event
    def set_note_text_for(self, value: str, event_id: str) -> None:
        self.note_text_by_event[event_id] = value

    @rx.event
    def set_note_severity_for(self, value: str, event_id: str) -> None:
        self.note_severity_by_event[event_id] = value

    @rx.event
    def open_tag_dialog(self, event_id: str) -> None:
        """Open tag dialog for a specific event and initialize selected tags with current tags."""
        self.tag_dialog_open_for_event = event_id
        current_event = next((e for e in self.events if e.event_id == event_id), None)
        if current_event:
            self.selected_tags_for_event[event_id] = list(current_event.visible_tags or [])
        else:
            self.selected_tags_for_event[event_id] = []

    @rx.event
    def close_tag_dialog(self) -> None:
        """Close tag dialog and clear temporary state."""
        event_id = self.tag_dialog_open_for_event
        self.tag_dialog_open_for_event = ""
        if event_id in self.new_tag_text_for_event:
            del self.new_tag_text_for_event[event_id]

    @rx.event
    def handle_tag_dialog_open_change(self, is_open: bool) -> None:
        """Handle dialog open/close state changes."""
        if not is_open:
            self.close_tag_dialog()  # type: ignore[operator]

    @rx.event
    def set_new_tag_text_for_event(self, value: str, event_id: str) -> None:
        """Set new tag text for a specific event."""
        self.new_tag_text_for_event[event_id] = value

    @rx.event
    def toggle_tag_selection_for_event(self, tag: str, event_id: str, checked: bool) -> None:
        """Toggle tag selection for a specific event."""
        selected = self.selected_tags_for_event.get(event_id, [])
        if checked:
            if tag not in selected:
                self.selected_tags_for_event[event_id] = [*selected, tag]
        else:
            self.selected_tags_for_event[event_id] = [t for t in selected if t != tag]

    @rx.event
    async def add_new_tag_to_available(self, event_id: str) -> None:
        """Add a new tag to available tags list."""
        new_tag = (self.new_tag_text_for_event.get(event_id) or "").strip()
        if new_tag and new_tag not in self.available_tags:
            self.available_tags = sorted([*self.available_tags, new_tag])
            selected = self.selected_tags_for_event.get(event_id, [])
            if new_tag not in selected:
                self.selected_tags_for_event[event_id] = [*selected, new_tag]
            self.new_tag_text_for_event[event_id] = ""

    @rx.event
    async def apply_tags_to_event(self, event_id: str):
        """Apply selected tags to an event by adding/removing tags as needed."""
        current_event = next((e for e in self.events if e.event_id == event_id), None)
        if not current_event:
            return

        current_tags = set(current_event.visible_tags or [])
        selected_tags = set(self.selected_tags_for_event.get(event_id, []))

        tags_to_add = selected_tags - current_tags
        tags_to_remove = current_tags - selected_tags

        jims_app = await get_jims_app()
        async with jims_app.sessionmaker() as session:
            thread_uuid = (
                UUID(self.selected_thread_id) if isinstance(self.selected_thread_id, str) else self.selected_thread_id
            )

            for tag in tags_to_add:
                tag_event = ThreadEventDB(
                    thread_id=thread_uuid,
                    event_id=uuid7(),
                    event_type="jims.backoffice.tag_added",
                    event_data={"target_event_id": event_id, "tag": tag},
                )
                session.add(tag_event)
                await session.flush()

            for tag in tags_to_remove:
                tag_event = ThreadEventDB(
                    thread_id=thread_uuid,
                    event_id=uuid7(),
                    event_type="jims.backoffice.tag_removed",
                    event_data={"target_event_id": event_id, "tag": tag},
                )
                session.add(tag_event)
                await session.flush()

            await session.commit()

        self.tag_dialog_open_for_event = ""
        await self._reload()
        yield ThreadListState.get_data()  # type: ignore[operator]

    @rx.event
    async def remove_tag(self, event_id: str, tag: str):
        jims_app = await get_jims_app()
        async with jims_app.sessionmaker() as session:
            tag_event = ThreadEventDB(
                thread_id=self.selected_thread_id,
                event_id=uuid7(),
                event_type="jims.backoffice.tag_removed",
                event_data={"target_event_id": event_id, "tag": tag},
            )
            session.add(tag_event)
            await session.commit()
        await self._reload()
        yield ThreadListState.get_data()  # type: ignore[operator]

    @rx.event
    async def submit_note_for(self, event_id: str):
        text = (self.note_text_by_event.get(event_id) or "").strip()
        if not text:
            return
        try:
            target = next((e for e in self.events if e.event_id == event_id), None)
            tags_list = list(getattr(target, "tags", []) or []) if target is not None else []
        except Exception:
            tags_list = []
        jims_app = await get_jims_app()
        async with jims_app.sessionmaker() as session:
            severity_val = self.note_severity_by_event.get(event_id, self.note_severity or "Low")
            note_event = ThreadEventDB(
                thread_id=self.selected_thread_id,
                event_id=uuid7(),
                event_type="jims.backoffice.feedback",
                event_data={
                    "event_id": event_id,
                    "tags": tags_list,
                    "note": text,
                    "severity": severity_val,
                },
            )
            session.add(note_event)
            await session.commit()
        try:
            del self.note_text_by_event[event_id]
        except Exception:
            pass
        try:
            del self.note_severity_by_event[event_id]
        except Exception:
            pass
        await self._reload()
        yield ThreadListState.get_data()  # type: ignore[operator]

    @rx.event
    async def mark_comment_resolved(self, comment_id: str):
        jims_app = await get_jims_app()
        async with jims_app.sessionmaker() as session:
            ev = ThreadEventDB(
                thread_id=self.selected_thread_id,
                event_id=uuid7(),
                event_type="jims.backoffice.comment_resolved",
                event_data={"comment_id": comment_id},
            )
            session.add(ev)
            await session.commit()
        await self._reload()
        yield ThreadListState.get_data()  # type: ignore[operator]

    @rx.event
    async def mark_comment_closed(self, comment_id: str):
        jims_app = await get_jims_app()
        async with jims_app.sessionmaker() as session:
            ev = ThreadEventDB(
                thread_id=self.selected_thread_id,
                event_id=uuid7(),
                event_type="jims.backoffice.comment_closed",
                event_data={"comment_id": comment_id},
            )
            session.add(ev)
            await session.commit()
        await self._reload()
        yield ThreadListState.get_data()  # type: ignore[operator]

    @rx.event
    async def reopen_comment(self, comment_id: str):
        jims_app = await get_jims_app()
        async with jims_app.sessionmaker() as session:
            ev = ThreadEventDB(
                thread_id=self.selected_thread_id,
                event_id=uuid7(),
                event_type="jims.backoffice.comment_reopened",
                event_data={"comment_id": comment_id},
            )
            session.add(ev)
            await session.commit()
        await self._reload()
        yield ThreadListState.get_data()  # type: ignore[operator]

    @rx.event
    async def delete_comment(self, comment_id: str):
        jims_app = await get_jims_app()
        async with jims_app.sessionmaker() as session:
            ev = ThreadEventDB(
                thread_id=self.selected_thread_id,
                event_id=uuid7(),
                event_type="jims.backoffice.comment_deleted",
                event_data={"comment_id": comment_id},
            )
            session.add(ev)
            await session.commit()
        await self._reload()
        yield ThreadListState.get_data()  # type: ignore[operator]

    @rx.event
    async def export_thread_csv(self):
        """Export all thread_events for the currently selected thread as CSV."""
        if not self.selected_thread_id:
            return rx.toast.info("No thread selected.")
        jims_app = await get_jims_app()
        async with jims_app.sessionmaker() as session:
            stmt = (
                sa.select(ThreadEventDB)
                .where(ThreadEventDB.thread_id == self.selected_thread_id)
                .order_by(ThreadEventDB.created_at.asc())
            )
            events = (await session.execute(stmt)).scalars().all()
        data = csv_bytes(_thread_events_to_rows(events), THREAD_EVENT_COLUMNS)
        filename = f"thread_{self.selected_thread_id}.csv"
        return rx.download(data=data, filename=filename)
