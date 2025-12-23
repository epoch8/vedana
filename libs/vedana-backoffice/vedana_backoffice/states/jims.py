import pprint
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

import orjson as json
import reflex as rx
import sqlalchemy as sa
from jims_core.db import ThreadEventDB
from jims_core.util import uuid7

from vedana_backoffice.states.common import get_vedana_app


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
        # Parse technical_info if present
        tech: dict = event_data.get("technical_info", {})
        has_technical_info = bool(tech)

        # Extract message-like fields
        # Role: Only comm.user_message is user; all others assistant.
        role = "user" if event_type == "comm.user_message" else "assistant"
        content: str = event_data.get("content", "")
        # tags may be stored in event_data["tags"] as list[str]
        tags_value = event_data.get("tags")  # todo check fmt
        tags: list[str] = list(tags_value or []) if isinstance(tags_value, (list, tuple)) else []

        vts_queries: list[str] = list(tech.get("vts_queries", []) or [])
        cypher_queries: list[str] = list(tech.get("cypher_queries", []) or [])

        model_stats = tech.get("model_stats", {}) or tech.get("model_used", {}) or {}
        models_list: list[tuple[str, str]] = []
        try:
            # If nested dict like {model: {...}} flatten to stringified value
            for mk, mv in model_stats.items() if isinstance(model_stats, dict) else []:
                try:
                    models_list.append((f'"{mk}"', json.dumps(mv, option=json.OPT_INDENT_2).decode()))
                except Exception:
                    models_list.append((f'"{mk}"', str(mv)))
        except Exception:
            pass

        vts_str = "\n".join(vts_queries)
        cypher_str = "\n".join([pprint.pformat(x)[1:-1].replace("'", "") for x in cypher_queries])  # format to fit
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


class ThreadViewState(rx.State):
    loading: bool = True
    events: list[ThreadEventVis] = []
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

    async def _reload(self) -> None:
        vedana_app = await get_vedana_app()

        async with vedana_app.sessionmaker() as session:
            stmt = (
                sa.select(ThreadEventDB)
                .where(
                    ThreadEventDB.thread_id == self.selected_thread_id,
                )
                .order_by(ThreadEventDB.created_at.asc())
            )
            all_events = (await session.execute(stmt)).scalars().all()

        # Split base convo events vs backoffice annotations
        base_events: list[Any] = []
        backoffice_events: list[Any] = []
        for ev in all_events:
            etype = str(getattr(ev, "event_type", ""))
            if etype.startswith("jims.backoffice."):
                backoffice_events.append(ev)
            elif etype.startswith("jims."):
                # ignore other jims.* noise
                continue
            else:
                base_events.append(ev)

        # Prepare aggregations
        # 1) Tags per original event
        tags_by_event: dict[str, set[str]] = {}
        for ev in base_events:
            eid = str(getattr(ev, "event_id", ""))
            try:
                base_tags = getattr(ev, "event_data", {}).get("tags") or []
                tags_by_event[eid] = set([str(t) for t in base_tags])
            except Exception:
                tags_by_event[eid] = set()

        # Apply tag add/remove in chronological order
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
        # status by comment id (event_id of feedback)
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

        # Convert base events into visual items and attach aggregations
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

        # Present in chronological order as originally shown (created_at asc)
        self.events = ev_items

        # Collect all available tags from all threads
        all_tags: set[str] = set()
        # From current thread
        for tags_set in tags_by_event.values():
            all_tags.update(tags_set)

        # From all threads in the database
        try:
            async with vedana_app.sessionmaker() as session:
                # Query all tag_added events to get all tags ever used
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
        await self._reload()

    @rx.event
    async def select_thread(self, thread_id: str) -> None:
        self.selected_thread_id = thread_id
        await self._reload()

    # UI field updates
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

    # Per-message note editing
    # todo check if necessary or just keep jims sending only
    @rx.event
    def set_note_text_for(self, value: str, event_id: str) -> None:
        self.note_text_by_event[event_id] = value

    @rx.event
    def set_note_severity_for(self, value: str, event_id: str) -> None:
        self.note_severity_by_event[event_id] = value

    # Tag dialog management
    @rx.event
    def open_tag_dialog(self, event_id: str) -> None:
        """Open tag dialog for a specific event and initialize selected tags with current tags."""
        self.tag_dialog_open_for_event = event_id
        # Initialize selected tags with current visible tags for this event
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
        # Optionally clear temporary state
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
            # Also add to selected tags for this event
            selected = self.selected_tags_for_event.get(event_id, [])
            if new_tag not in selected:
                self.selected_tags_for_event[event_id] = [*selected, new_tag]
            # Clear the input
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

        vedana_app = await get_vedana_app()
        async with vedana_app.sessionmaker() as session:
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

        # Close dialog and reload
        self.tag_dialog_open_for_event = ""
        await self._reload()
        try:
            # local import to avoid cycles
            from vedana_backoffice.pages.jims_thread_list_page import ThreadListState

            yield ThreadListState.get_data()  # type: ignore[operator]
        except Exception:
            pass

    @rx.event
    async def remove_tag(self, event_id: str, tag: str):
        vedana_app = await get_vedana_app()
        async with vedana_app.sessionmaker() as session:
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
            from vedana_backoffice.pages.jims_thread_list_page import ThreadListState  # local import to avoid cycles

            yield ThreadListState.get_data()  # type: ignore[operator]
        except Exception:
            pass

    @rx.event
    async def submit_note_for(self, event_id: str):
        text = (self.note_text_by_event.get(event_id) or "").strip()
        if not text:
            return
        # Collect current tags from the target event if present
        try:
            target = next((e for e in self.events if e.event_id == event_id), None)
            tags_list = list(getattr(target, "tags", []) or []) if target is not None else []
        except Exception:
            tags_list = []
        vedana_app = await get_vedana_app()
        async with vedana_app.sessionmaker() as session:
            severity_val = self.note_severity_by_event.get(event_id, self.note_severity or "Low")  # todo check
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
            # todo check
            from vedana_backoffice.pages.jims_thread_list_page import ThreadListState  # local import

            yield ThreadListState.get_data()  # type: ignore[operator]
        except Exception:
            pass

    # --- Comment status actions ---
    @rx.event
    async def mark_comment_resolved(self, comment_id: str):
        vedana_app = await get_vedana_app()
        async with vedana_app.sessionmaker() as session:
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
            from vedana_backoffice.pages.jims_thread_list_page import ThreadListState

            yield ThreadListState.get_data()  # type: ignore[operator]
        except Exception:
            pass

    @rx.event
    async def mark_comment_closed(self, comment_id: str):
        vedana_app = await get_vedana_app()
        async with vedana_app.sessionmaker() as session:
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
            from vedana_backoffice.pages.jims_thread_list_page import ThreadListState

            yield ThreadListState.get_data()  # type: ignore[operator]
        except Exception:
            pass
