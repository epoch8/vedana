from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

import reflex as rx
import sqlalchemy as sa
from jims_core.db import ThreadEventDB
from vedana_core.app import make_vedana_app

from vedana_backoffice.util import datetime_to_age


@dataclass
class ThreadEventVis:
    event_id: str
    created_at: datetime
    created_at_str: str
    event_type: str
    role: str
    content: str
    tags: list[str]
    event_age: str
    event_data_list: list[tuple[str, str]]
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

    @classmethod
    def create(cls, event_id: Any, created_at: datetime, event_type: str, event_data: dict) -> "ThreadEventVis":
        import json

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
                    models_list.append((str(mk), json.dumps(mv)))
                except Exception:
                    models_list.append((str(mk), str(mv)))
        except Exception:
            pass

        vts_str = "\n".join(vts_queries)
        cypher_str = "\n".join([str(x) for x in cypher_queries])
        models_str = "\n".join([f"{k}: {v}" for k, v in models_list])

        return cls(
            event_id=str(event_id),
            created_at=created_at.replace(microsecond=0),
            created_at_str=datetime.strftime(created_at, "%Y-%m-%d %H:%M:%S"),
            event_type=event_type,
            role=role,
            content=content,
            tags=tags,
            event_age=datetime_to_age(created_at),
            event_data_list=[(str(k), str(v)) for k, v in event_data.items()],
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

    async def _reload(self) -> None:
        vedana_app = await make_vedana_app()

        async with vedana_app.sessionmaker() as session:
            stmt = sa.select(ThreadEventDB).where(
                ThreadEventDB.thread_id == self.selected_thread_id,
                sa.not_(ThreadEventDB.event_type.like("jims.%")),
            )
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

    # Persistence events: we encode admin actions as jims.backoffice.* events for now
    @rx.event
    async def add_tag(self, event_id: str) -> None:
        if not self.new_tag_text.strip():
            return
        vedana_app = await make_vedana_app()
        async with vedana_app.sessionmaker() as session:
            # Store an auxiliary event carrying tagging info
            from jims_core.util import uuid7

            tag_event = ThreadEventDB(
                thread_id=self.selected_thread_id,
                event_id=uuid7(),
                event_type="jims.backoffice.tag_added",
                event_data={"target_event_id": event_id, "tag": self.new_tag_text.strip()},
            )
            session.add(tag_event)
            await session.commit()
        self.new_tag_text = ""
        await self._reload()

    @rx.event
    async def remove_tag(self, event_id: str, tag: str) -> None:
        vedana_app = await make_vedana_app()
        async with vedana_app.sessionmaker() as session:
            from jims_core.util import uuid7

            tag_event = ThreadEventDB(
                thread_id=self.selected_thread_id,
                event_id=uuid7(),
                event_type="jims.backoffice.tag_removed",
                event_data={"target_event_id": event_id, "tag": tag},
            )
            session.add(tag_event)
            await session.commit()
        await self._reload()

    @rx.event
    async def submit_note_for(self, event_id: str) -> None:
        text = (self.note_text_by_event.get(event_id) or "").strip()
        if not text:
            return
        # Collect current tags from the target event if present
        try:
            target = next((e for e in self.events if e.event_id == event_id), None)
            tags_list = list(getattr(target, "tags", []) or []) if target is not None else []
        except Exception:
            tags_list = []
        vedana_app = await make_vedana_app()
        async with vedana_app.sessionmaker() as session:
            from jims_core.util import uuid7

            note_event = ThreadEventDB(
                thread_id=self.selected_thread_id,
                event_id=uuid7(),
                event_type="jims.backoffice.feedback",
                event_data={
                    "event_id": event_id,
                    "tags": tags_list,
                    "note": text,
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
