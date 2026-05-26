"""Vedana-specific UI helpers built on top of jims-backoffice components."""

from __future__ import annotations

import jims_backoffice
import reflex as rx
from jims_backoffice import (
    api_key_setup_dialog,
    breadcrumbs,
    debug_badge,
    telegram_link_box,
    themed_data_table,
)

from vedana_backoffice.states.chat import ChatState

__all__ = [
    "app_header",
    "data_model_reload_btn",
    "telegram_link_box",
    "debug_badge",
    "api_key_setup_dialog",
    "themed_data_table",
    "breadcrumbs",
]


def data_model_reload_btn() -> rx.Component:
    return rx.button(
        "Reload Data Model",
        variant="soft",
        color_scheme="blue",
        on_click=ChatState.reload_data_model,
        loading=ChatState.is_refreshing_dm,
    )


def app_header() -> rx.Component:
    """Vedana branded header that re-injects DM reload + telegram + nav links."""
    return jims_backoffice.app_header(
        brand_text="Vedana Backoffice",
        brand_href="/",
        extra_header=[
            data_model_reload_btn(),
            telegram_link_box(),
            rx.link("ETL", href="/etl", font_size="1.1em"),
        ],
    )
