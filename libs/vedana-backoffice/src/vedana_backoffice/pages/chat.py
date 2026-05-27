"""Vedana chat page — thin wrapper around :func:`jims_backoffice.chat_page`."""

from __future__ import annotations

import jims_backoffice
import reflex as rx

from vedana_backoffice.states.chat import ChatState
from vedana_backoffice.ui import app_header

DebugState = jims_backoffice.DebugState
AppVersionState = jims_backoffice.AppVersionState


def _data_model_dialog() -> rx.Component:
    return rx.dialog.root(
        rx.dialog.trigger(
            rx.button(
                "Data model",
                variant="ghost",
                color_scheme="gray",
                size="1",
                on_click=ChatState.load_data_model_text,
            ),
        ),
        rx.dialog.content(
            rx.vstack(
                rx.hstack(
                    rx.dialog.title("Data Model"),
                    rx.spacer(),
                    rx.button(
                        "Reload",
                        variant="soft",
                        color_scheme="blue",
                        size="1",
                        on_click=ChatState.reload_data_model,
                        loading=ChatState.is_refreshing_dm,
                    ),
                    rx.dialog.close(
                        rx.button("Close", variant="ghost", color_scheme="gray", size="1"),
                    ),
                    align="center",
                    width="100%",
                ),
                rx.scroll_area(
                    rx.markdown(ChatState.data_model_text),  # type: ignore[operator]
                    type="always",
                    scrollbars="vertical",
                    style={"height": "70vh"},
                ),
                spacing="3",
                width="100%",
            ),
            max_width="70vw",
        ),
    )


def _dm_filter_extras() -> list[rx.Component]:
    return [
        _data_model_dialog(),
        rx.checkbox(
            "Filter Data Model",
            checked=ChatState.enable_dm_filtering,
            on_change=ChatState.set_enable_dm_filtering,
            size="1",
        ),
        rx.cond(
            ChatState.enable_dm_filtering & AppVersionState.debug_mode,
            rx.select(
                items=DebugState.available_models,
                value=ChatState.dm_filter_model,
                on_change=ChatState.set_dm_filter_model,
                width="16em",
                placeholder="Filter model",
            ),
            rx.text(ChatState.dm_filter_model, size="1", color="gray"),
        ),
        rx.cond(
            AppVersionState.debug_mode,
            rx.cond(
                DebugState.embeddings_model_available,
                rx.text(
                    "Embeddings: " + DebugState.embeddings_model,  # type: ignore[operator]
                    size="1",
                    color="gray",
                ),
                rx.text(
                    "Embeddings: " + DebugState.default_embeddings_model + " (unavailable for provider)",
                    size="1",
                    color="red",
                ),
            ),
            rx.fragment(),
        ),
    ]


def page() -> rx.Component:
    return jims_backoffice.chat_page(
        ChatState,
        input_bar_extras=_dm_filter_extras(),
        header=app_header,
    )
