from __future__ import annotations

from typing import Any

import reflex as rx

from vedana_backoffice.state import AppVersionState, TelegramBotState  # type: ignore[attr-defined]


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
                rx.link("ETL", href="/etl", font_size="1.1em"),
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
        },
        z_index="10",
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
    table_style = {"width": "fit-content", **table_style}

    container_style: dict[str, Any] = {"width": width, "maxWidth": max_width}

    return rx.box(
        rx.data_table(data=data, columns=columns, style=table_style, **table_kwargs),
        class_name="datatable-surface",
        style=container_style,
    )


def breadcrumbs(items: list[tuple[str, str]]) -> rx.Component:
    parts: list[rx.Component] = []
    for idx, (label, href) in enumerate(items):
        parts.append(rx.link(label, href=href))
        if idx < len(items) - 1:
            parts.append(rx.text("→", color="#9ca3af"))
    return rx.hstack(*parts, spacing="2", padding_y="0.5em")
