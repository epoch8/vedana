"""Vedana eval page — thin wrapper around :func:`jims_backoffice.eval_page`."""

from __future__ import annotations

import jims_backoffice
import reflex as rx

from vedana_backoffice.states.chat import ChatState
from vedana_backoffice.states.eval import EvalState
from vedana_backoffice.ui import app_header

DebugState = jims_backoffice.DebugState
AppVersionState = jims_backoffice.AppVersionState


def _data_model_dialog() -> rx.Component:
    return rx.dialog.root(
        rx.dialog.content(
            rx.dialog.title("Data Model"),
            rx.vstack(
                rx.text(f"Model ID: {EvalState.dm_id}", size="2", color="gray"),
                rx.box(
                    rx.text(
                        rx.cond(
                            EvalState.dm_description != "",
                            EvalState.dm_description,
                            "Description not loaded",
                        ),
                        size="2",
                    ),
                    padding="1em",
                    border="1px solid var(--gray-6)",
                    border_radius="8px",
                    style={"maxHeight": "60vh", "overflow": "auto", "whiteSpace": "pre-wrap"},
                ),
                rx.dialog.close(
                    rx.button("Close", variant="soft"),
                ),
                spacing="3",
                width="100%",
            ),
            style={"maxWidth": "800px"},
        ),
        open=EvalState.data_model_dialog_open,
        on_open_change=EvalState.set_data_model_dialog_open,
    )


def _pipeline_card_extras() -> list[rx.Component]:
    """Vedana-specific bits inserted into the JIMS pipeline card.

    Adds (in this order, below the model select):
      * "View Data Model" + "Refresh Data Model" buttons.
      * Filter Data Model checkbox + filter-model select.
      * Embeddings model status (with debug-mode 'unavailable' callout).
    """
    return [
        rx.box(
            rx.text("Data model", weight="medium"),
            rx.button(
                "View Data Model",
                variant="soft",
                size="1",
                on_click=EvalState.open_data_model_dialog,
                disabled=rx.cond(EvalState.dm_id != "", False, True),  # type: ignore[arg-type]
                width="100%",
                margin_top="0.5em",
            ),
            rx.button(
                "Refresh Data Model",
                variant="soft",
                size="1",
                on_click=ChatState.reload_data_model,
                loading=ChatState.is_refreshing_dm,
                width="100%",
                margin_top="0.5em",
            ),
            width="100%",
            padding_bottom="0.75em",
        ),
        rx.hstack(
            rx.checkbox(
                "Filter Data Model",
                checked=EvalState.enable_dm_filtering,
                on_change=EvalState.set_enable_dm_filtering,
                size="2",
            ),
            rx.cond(
                EvalState.enable_dm_filtering,
                rx.cond(
                    AppVersionState.debug_mode,
                    rx.select(
                        items=DebugState.available_models,
                        value=EvalState.dm_filter_model,
                        on_change=EvalState.set_dm_filter_model,
                        width="100%",
                        placeholder="Filter model",
                    ),
                    rx.text(EvalState.dm_filter_model, size="2", color="gray"),
                ),
                rx.fragment(),
            ),
            spacing="2",
            align="center",
            wrap="wrap",
            width="100%",
            margin_top="0.5em",
        ),
        rx.box(
            rx.text("Embeddings", weight="medium"),
            rx.cond(
                AppVersionState.debug_mode,
                rx.cond(
                    DebugState.embeddings_model_available,
                    rx.text(DebugState.embeddings_model, size="3"),
                    rx.text(
                        EvalState.default_embeddings_model + " (unavailable for provider)",
                        size="3",
                        color="red",
                    ),
                ),
                rx.text(EvalState.default_embeddings_model, size="3"),
            ),
            rx.text(
                EvalState.embeddings_dim_label,
                size="1",
                color="gray",
            ),
        ),
    ]


def _questions_card_extras() -> list[rx.Component]:
    """Refresh-from-Grist button shown next to the scenario filter."""
    return [
        rx.tooltip(
            rx.button(
                "↻",
                variant="ghost",
                color_scheme="gray",
                size="1",
                on_click=EvalState.refresh_golden_dataset,
                loading=EvalState.is_running,
            ),
            content="Refresh golden dataset from Grist",
        ),
    ]


def page() -> rx.Component:
    return jims_backoffice.eval_page(
        EvalState,
        pipeline_card_extras=_pipeline_card_extras(),
        questions_card_extras=_questions_card_extras(),
        extra_dialogs=[_data_model_dialog()],
        header=app_header,
    )
