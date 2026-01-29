from __future__ import annotations

from typing import Any

import reflex as rx

from vedana_backoffice.states.chat import ChatState
from vedana_backoffice.states.common import AppVersionState, TelegramBotState


def telegram_link_box() -> rx.Component:
    return rx.box(
        rx.cond(
            TelegramBotState.has_bot,
            rx.link(
                rx.hstack(
                    rx.text("Telegram", font_size="1.1em"),
                    # rx.icon("external-link", stroke_width=1, size=10),
                    # spacing="1",
                ),
                href=TelegramBotState.bot_url,
                is_external=True,
            ),
        ),
        on_mount=TelegramBotState.load_bot_info,
    )


def data_model_reload_btn() -> rx.Component:
    return rx.button(
        "Reload Data Model",
        variant="soft",
        color_scheme="blue",
        on_click=ChatState.reload_data_model,
        loading=ChatState.is_refreshing_dm,
    )


def app_header() -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.hstack(
                rx.link("Vedana Backoffice", href="/", font_weight="bold", font_size="1.25em"),
                rx.markdown(AppVersionState.version),  # type: ignore[operator]
                align="center",
                spacing="3",
            ),
            rx.hstack(
                data_model_reload_btn(),
                rx.link("Data Model", href="/data-model", font_size="1.1em"),
                rx.link("ETL", href="/etl", font_size="1.1em"),
                rx.cond(
                    AppVersionState.eval_enabled,
                    rx.link("Eval", href="/eval", font_size="1.1em"),
                    rx.fragment(),
                ),
                rx.link("Chat", href="/chat", font_size="1.1em"),
                rx.link("JIMS", href="/jims", font_size="1.1em"),
                telegram_link_box(),
                rx.color_mode.button(),  # type: ignore[attr-defined]
                spacing="6",
                align="center",
            ),
            justify="between",
            align="center",
            width="100%",
        ),
        width="100%",
        padding="0.5em 1.25em",
        border_bottom="1px solid #e5e7eb",
        position="sticky",
        top="0",
        background_color=rx.color_mode_cond(light="white", dark="black"),
        style={
            "backdrop-filter": "blur(10px)",  # enables non-transparent background
            "zIndex": "1000",
        },
    )


def themed_data_table(
    *,
    data: rx.Var | list[Any],
    columns: rx.Var | list[str],
    width: str | rx.Var = "fit-content",
    max_width: str | rx.Var = "100%",
    **kwargs: Any,
) -> rx.Component:
    """Wrap rx.data_table with shared styling so it matches the app theme."""

    default_kwargs = {"pagination": True, "search": True, "sort": True}
    table_kwargs = {**default_kwargs, **kwargs}

    table_style = table_kwargs.pop("style", {})
    table_style = {"width": "fit-content", "minWidth": "100%", **table_style}

    container_style: dict[str, Any] = {
        "width": width,
        "maxWidth": max_width,
        "minWidth": "fit-content",
    }

    return rx.box(
        rx.data_table(data=data, columns=columns, style=table_style, **table_kwargs),
        class_name="datatable-surface",
        style=container_style,
    )


def table_accordion(tables: rx.Var) -> rx.Component:
    return rx.accordion.root(
        rx.foreach(
            tables,
            lambda t: rx.accordion.item(
                rx.accordion.trigger(
                    rx.hstack(
                        rx.text(t["name"]),
                        rx.badge(t["row_count"], variant="soft", size="1", color_scheme="gray"),
                        spacing="2",
                        align="center",
                    )
                ),
                rx.accordion.content(
                    themed_data_table(
                        data=t["rows"],
                        columns=t["columns"],
                        pagination=True,
                        search=True,
                        sort=True,
                        max_width="100%",
                    )
                ),
                value=t["name"],
            ),
        ),
        type="multiple",
        collapsible=True,
        variant="outline",
        width="100%",
    )


def breadcrumbs(items: list[tuple[str, str]]) -> rx.Component:
    parts: list[rx.Component] = []
    for idx, (label, href) in enumerate(items):
        parts.append(rx.link(label, href=href))
        if idx < len(items) - 1:
            parts.append(rx.text("→", color="#9ca3af"))
    return rx.hstack(*parts, spacing="2", padding_y="0.5em")
