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
    role: str | None
    content: str | None
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
        base_data = dict(event_data or {}) if isinstance(event_data, dict) else {}
        tech = base_data.get("technical_info", {}) if isinstance(base_data, dict) else {}
        has_technical_info = bool(tech)

        # Generic key-value list for display (exclude raw technical_info if present)
        if has_technical_info and "technical_info" in base_data:
            try:
                del base_data["technical_info"]
            except Exception:
                pass
        event_data_list = [(str(k), str(v)) for k, v in base_data.items()]

        # Extract message-like fields
        # Role: Only comm.user_message is user; all others assistant.
        role = "user" if event_type == "comm.user_message" else "assistant"
        content = base_data.get("content") if isinstance(base_data, dict) else None
        # tags may be stored in event_data["tags"] as list[str]
        tags_value = base_data.get("tags") if isinstance(base_data, dict) else None
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
            role=str(role) if role else None,
            content=str(content) if content else None,
            tags=tags,
            event_age=datetime_to_age(created_at),
            event_data_list=event_data_list,
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
    selected_thread_id: str = ""
    expanded_event_id: str = ""

    async def _reload(self) -> None:
        vedana_app = await make_vedana_app()

        async with vedana_app.sessionmaker() as session:
            stmt = sa.select(ThreadEventDB).where(ThreadEventDB.thread_id == self.selected_thread_id)
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
    async def submit_note(self) -> None:
        if not self.note_text.strip():
            return
        vedana_app = await make_vedana_app()
        async with vedana_app.sessionmaker() as session:
            from jims_core.util import uuid7

            note_event = ThreadEventDB(
                thread_id=self.selected_thread_id,
                event_id=uuid7(),
                event_type="jims.backoffice.prompt_note",
                event_data={"comment": self.note_text.strip(), "severity": self.note_severity},
            )
            session.add(note_event)
            await session.commit()
        self.note_text = ""
        self.note_severity = "Low"
        await self._reload()


def _message_bubble(event: ThreadEventVis) -> rx.Component:
    # Local wrappers so type checker doesn't flag callables
    def _on_remove(tag: str):  # type: ignore[missing-type-doc]
        return ThreadViewState.remove_tag(event_id=event.event_id, tag=tag)  # type: ignore[call-arg,func-returns-value]

    def _on_toggle():  # type: ignore[missing-type-doc]
        return ThreadViewState.toggle_details(event_id=event.event_id)  # type: ignore[call-arg,func-returns-value]

    tag_badges = rx.hstack(
        rx.foreach(
            event.tags,
            lambda tag: rx.button(
                tag,
                variant="soft",
                size="1",
                color_scheme="gray",
                on_click=_on_remove(tag),
            ),
        ),
        spacing="1",
        wrap="wrap",
    )

    add_tag = rx.hstack(
        rx.input(
            placeholder="Add tag",
            value=ThreadViewState.new_tag_text,
            on_change=ThreadViewState.set_new_tag_text,
            width="160px",
        ),
        rx.button("Add", size="1", on_click=ThreadViewState.add_tag(event_id=event.event_id)),  # type: ignore[call-arg,func-returns-value]
        spacing="2",
    )

    details = rx.vstack(
        rx.button(
            "Details",
            variant="soft",
            size="1",
            on_click=_on_toggle(),
        ),
        rx.cond(
            ThreadViewState.expanded_event_id == event.event_id,
            rx.vstack(
                rx.data_list.root(
                    rx.data_list.item(rx.data_list.label("event_id"), rx.data_list.value(event.event_id)),
                    rx.data_list.item(rx.data_list.label("event_type"), rx.data_list.value(event.event_type)),
                    rx.foreach(
                        event.event_data_list,
                        lambda item: rx.data_list.item(
                            rx.data_list.label(item[0]),
                            rx.data_list.value(rx.code(item[1], font_size="11px")),
                        ),
                    ),
                    size="1",
                ),
                rx.cond(
                    event.has_technical_info,
                    rx.vstack(
                        rx.heading("Technical Info", size="2"),
                        rx.cond(
                            event.has_models,
                            rx.data_list.root(
                                rx.data_list.item(rx.data_list.label("Models"), rx.data_list.value("")),
                                rx.foreach(
                                    event.technical_models,
                                    lambda kv: rx.data_list.item(
                                        rx.data_list.label(kv[0]),
                                        rx.data_list.value(rx.code(kv[1], font_size="11px")),
                                    ),
                                ),
                                size="1",
                            ),
                        ),
                        rx.cond(
                            event.has_vts,
                            rx.vstack(
                                rx.text("VTS Queries"),
                                rx.foreach(event.technical_vts_queries, lambda q: rx.code(q, font_size="11px")),
                            ),
                        ),
                        rx.cond(
                            event.has_cypher,
                            rx.vstack(
                                rx.text("Cypher Queries"),
                                rx.foreach(event.technical_cypher_queries, lambda q: rx.code(q, font_size="11px")),
                            ),
                        ),
                        spacing="2",
                    ),
                ),
                spacing="3",
            ),
        ),
        spacing="2",
    )

    # Build shared content stack with role badge via rx.cond
    content_stack = rx.vstack(
        rx.hstack(
            rx.text(event.event_age, color="#6b7280", size="1"),
            rx.cond(
                event.role == "user",
                rx.badge("User", color_scheme="gray", variant="soft"),
                rx.badge("Assistant", color_scheme="blue", variant="soft"),
            ),
            justify="between",
            width="100%",
        ),
        rx.box(event.content, style={"whiteSpace": "pre-wrap", "color": "inherit"}),
        tag_badges,
        add_tag,
        details,
        spacing="2",
    )

    # Render two different rows based on role to avoid boolean use on Vars
    assistant_row = rx.hstack(
        rx.card(content_stack, style={"backgroundColor": "#e0f2fe"}, padding="12px", width="min(72ch, 100%)"),
        rx.spacer(),
        justify="start",
        width="100%",
    )
    user_row = rx.hstack(
        rx.spacer(),
        rx.card(content_stack, style={"backgroundColor": "#f3f4f6"}, padding="12px", width="min(72ch, 100%)"),
        justify="end",
        width="100%",
    )

    return rx.cond(event.role == "user", user_row, assistant_row)
