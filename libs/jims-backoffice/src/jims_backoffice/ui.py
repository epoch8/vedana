from __future__ import annotations

from typing import Any

import reflex as rx

from jims_backoffice.states.common import AppVersionState, DebugState, TelegramBotState


def telegram_link_box() -> rx.Component:
    """Optional Telegram bot link, mounted on demand by header consumers."""
    return rx.box(
        rx.cond(
            TelegramBotState.has_bot,
            rx.link(
                rx.hstack(
                    rx.text("Telegram", font_size="1.1em"),
                ),
                href=TelegramBotState.bot_url,
                is_external=True,
            ),
        ),
        on_mount=TelegramBotState.load_bot_info,
    )


def debug_badge() -> rx.Component:
    """Red badge indicating debug mode is active. Clickable to open API key dialog."""
    return rx.cond(
        AppVersionState.debug_mode,
        rx.tooltip(
            rx.badge(
                "DEBUG MODE",
                color_scheme="red",
                variant="solid",
                size="2",
                style={
                    "font_weight": "bold",
                    "text_transform": "uppercase",
                    "cursor": "pointer",
                },
                on_click=DebugState.open_dialog,
            ),
            content="Click to set your LiteLLM API key for debug mode model selection",
        ),
        rx.fragment(),
    )


def api_key_setup_dialog() -> rx.Component:
    """Dialog to prompt for a runtime API key used by LiteLLM in debug mode."""
    _normal_style = {"width": "100%"}
    return rx.dialog.root(
        rx.dialog.content(
            rx.dialog.title("API Key Setup"),
            rx.dialog.description(
                "Paste your LiteLLM-compatible API key here and click 'Save' to use compatible models. "
                "Remove the key to revert to default key and models.",
                margin_bottom="1em",
            ),
            rx.vstack(
                rx.vstack(
                    rx.text("Provider", font_size="2", font_weight="500"),
                    rx.select(
                        items=DebugState.provider_options,
                        value=DebugState.runtime_model_provider,
                        on_change=DebugState.set_model_provider,
                        width="100%",
                        placeholder="Select provider",
                        style=_normal_style,
                    ),
                    spacing="2",
                    width="100%",
                    align="start",
                ),
                rx.vstack(
                    rx.text("Runtime API Key", font_size="2", font_weight="500"),
                    rx.input(
                        placeholder="sk-...",
                        value=DebugState.runtime_model_api_key,
                        on_change=DebugState.set_model_api_key,
                        type="password",
                        width="100%",
                        style=_normal_style,
                    ),
                    spacing="2",
                    width="100%",
                    align="start",
                ),
                rx.hstack(
                    rx.dialog.close(
                        rx.button(
                            "Close",
                            variant="soft",
                            color_scheme="gray",
                            on_click=DebugState.close_dialog,
                        ),
                    ),
                    rx.button(
                        "Save",
                        color_scheme="blue",
                        on_click=DebugState.save_api_key,
                    ),
                    justify="end",
                    spacing="3",
                    width="100%",
                ),
                spacing="4",
                width="100%",
            ),
            style={"max_width": "450px"},
        ),
        open=DebugState.show_api_key_dialog,
    )


def app_header(
    *,
    brand_text: str = "JIMS Backoffice",
    brand_href: str = "/",
    extra_header: list[rx.Component] | None = None,
) -> rx.Component:
    """Sticky top header.

    Args:
        brand_text: Branding label rendered on the left.
        brand_href: Brand link target.
        extra_header: Optional extra components rendered in the right-side header
    """
    header_links: list[rx.Component] = [
        rx.link("Chat", href="/chat", font_size="1.1em"),
        rx.link("JIMS", href="/jims", font_size="1.1em"),
        rx.cond(
            AppVersionState.eval_enabled,
            rx.link("Eval", href="/eval", font_size="1.1em"),
            rx.fragment(),
        ),
    ]
    extra_header = extra_header or []

    right_items: list[rx.Component] = extra_header + header_links
    right_items.append(rx.color_mode.button())  # type: ignore[attr-defined]

    return rx.fragment(
        rx.box(
            rx.hstack(
                rx.hstack(
                    rx.link(brand_text, href=brand_href, font_weight="bold", font_size="1.25em"),
                    rx.markdown(AppVersionState.version),  # type: ignore[operator]
                    debug_badge(),
                    align="center",
                    spacing="3",
                ),
                rx.hstack(
                    *right_items,
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
                "backdrop-filter": "blur(10px)",
                "zIndex": "1000",
            },
        ),
        api_key_setup_dialog(),
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


def breadcrumbs(items: list[tuple[str, str]]) -> rx.Component:
    parts: list[rx.Component] = []
    for idx, (label, href) in enumerate(items):
        parts.append(rx.link(label, href=href))
        if idx < len(items) - 1:
            parts.append(rx.text("→", color="#9ca3af"))
    return rx.hstack(*parts, spacing="2", padding_y="0.5em")
