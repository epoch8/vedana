from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

import orjson as json
import reflex as rx
import sqlalchemy as sa
from jims_core.db import ThreadEventDB, get_sessionmaker
from jims_core.util import uuid7


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
    technical_info_str: str
    has_technical_info: bool
    generic_meta: bool = False
    # Aggregated annotations from jims.backoffice.* events
    visible_tags: list[str] = field(default_factory=list)
    feedback_comments: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(cls, event_id: Any, created_at: datetime, event_type: str, event_data: dict) -> "ThreadEventVis":
        tech: dict = event_data.get("technical_info", {})
        has_technical_info = bool(tech)

        # Only comm.user_message events are "user"; all others are "assistant".
        role = "user" if event_type == "comm.user_message" else "assistant"
        # Display content only for comm.* events.
        content: str = event_data.get("content", "") if event_type.startswith("comm.") else ""
        tags_value = event_data.get("tags")
        tags: list[str] = list(tags_value or []) if isinstance(tags_value, (list, tuple)) else []

        technical_info_str = json.dumps(tech, option=json.OPT_INDENT_2).decode() if tech else ""

        # Show raw event_data for any non-comm event (generic pipeline steps, etc.)
        generic_meta = not event_type.startswith("comm.")

        return cls(
            event_id=str(event_id),
            created_at=created_at.replace(microsecond=0),
            created_at_str=datetime.strftime(created_at, "%Y-%m-%d %H:%M:%S"),
            event_type=event_type,
            role=role,
            content=content,
            tags=tags,
            event_data_str=json.dumps(event_data, option=json.OPT_INDENT_2).decode(),
            technical_info_str=technical_info_str,
            has_technical_info=has_technical_info,
            visible_tags=list(tags),
            feedback_comments=[],
            generic_meta=generic_meta,
        )


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
        sessionmaker = get_sessionmaker()

        async with sessionmaker() as session:
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
                                ["jims.backoffice.comment_resolved", "jims.backoffice.comment_closed"]
                            ),
                            comment_id_expr.in_(feedback_ids),
                        )
                    )
                    .order_by(ThreadEventDB.created_at.asc())
                )
                backoffice_events.extend(list((await session.execute(comment_status_stmt)).scalars().all()))

        # 1) Aggregate tags per original event
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

        # 2) Comments per original event + status mapping
        comments_by_event: dict[str, list[dict[str, Any]]] = {}
        comment_status: dict[str, str] = {}
        for ev in backoffice_events:
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
            elif etype in ("jims.backoffice.comment_resolved", "jims.backoffice.comment_closed"):
                ed = dict(getattr(ev, "event_data", {}) or {})
                cid = str(ed.get("comment_id", ""))
                if not cid:
                    continue
                comment_status[cid] = "resolved" if etype.endswith("comment_resolved") else "closed"

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

        # Collect all available tags (current thread + all threads in the DB)
        all_tags: set[str] = set()
        for tags_set in tags_by_event.values():
            all_tags.update(tags_set)

        try:
            sessionmaker = get_sessionmaker()
            async with sessionmaker() as session:
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
        if not is_open:
            self.close_tag_dialog()  # type: ignore[operator]

    @rx.event
    def set_new_tag_text_for_event(self, value: str, event_id: str) -> None:
        self.new_tag_text_for_event[event_id] = value

    @rx.event
    def toggle_tag_selection_for_event(self, tag: str, event_id: str, checked: bool) -> None:
        selected = self.selected_tags_for_event.get(event_id, [])
        if checked:
            if tag not in selected:
                self.selected_tags_for_event[event_id] = [*selected, tag]
        else:
            self.selected_tags_for_event[event_id] = [t for t in selected if t != tag]

    @rx.event
    async def add_new_tag_to_available(self, event_id: str) -> None:
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

        sessionmaker = get_sessionmaker()
        async with sessionmaker() as session:
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
        try:
            from jims_backoffice.pages.jims_thread_list_page import ThreadListState

            yield ThreadListState.get_data()  # type: ignore[operator]
        except Exception:
            pass

    @rx.event
    async def remove_tag(self, event_id: str, tag: str):
        sessionmaker = get_sessionmaker()
        async with sessionmaker() as session:
            tag_event = ThreadEventDB(
                thread_id=self.selected_thread_id,
                event_id=uuid7(),
                event_type="jims.backoffice.tag_removed",
                event_data={"target_event_id": event_id, "tag": tag},
            )
            session.add(tag_event)
            await session.commit()
        await self._reload()
        try:
            from jims_backoffice.pages.jims_thread_list_page import ThreadListState

            yield ThreadListState.get_data()  # type: ignore[operator]
        except Exception:
            pass

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
        sessionmaker = get_sessionmaker()
        async with sessionmaker() as session:
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
        try:
            from jims_backoffice.pages.jims_thread_list_page import ThreadListState

            yield ThreadListState.get_data()  # type: ignore[operator]
        except Exception:
            pass

    @rx.event
    async def mark_comment_resolved(self, comment_id: str):
        sessionmaker = get_sessionmaker()
        async with sessionmaker() as session:
            ev = ThreadEventDB(
                thread_id=self.selected_thread_id,
                event_id=uuid7(),
                event_type="jims.backoffice.comment_resolved",
                event_data={"comment_id": comment_id},
            )
            session.add(ev)
            await session.commit()
        await self._reload()
        try:
            from jims_backoffice.pages.jims_thread_list_page import ThreadListState

            yield ThreadListState.get_data()  # type: ignore[operator]
        except Exception:
            pass

    @rx.event
    async def mark_comment_closed(self, comment_id: str):
        sessionmaker = get_sessionmaker()
        async with sessionmaker() as session:
            ev = ThreadEventDB(
                thread_id=self.selected_thread_id,
                event_id=uuid7(),
                event_type="jims.backoffice.comment_closed",
                event_data={"comment_id": comment_id},
            )
            session.add(ev)
            await session.commit()
        await self._reload()
        try:
            from jims_backoffice.pages.jims_thread_list_page import ThreadListState

            yield ThreadListState.get_data()  # type: ignore[operator]
        except Exception:
            pass
