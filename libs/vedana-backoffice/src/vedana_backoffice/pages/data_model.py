import reflex as rx

from vedana_backoffice.states.data_model import DataModelState
from vedana_backoffice.ui import app_header, themed_data_table


def _branch_row(branch: rx.Var, snapshot_id: rx.Var) -> rx.Component:
    return rx.hstack(
        rx.text(branch, color="gray"),
        rx.spacer(),
        rx.text(
            rx.cond(
                snapshot_id,
                f"Snapshot: {snapshot_id}",
                "Snapshot: —",
            ),
            color="gray",
        ),
        spacing="3",
        width="100%",
    )


def _table_accordion(tables: rx.Var) -> rx.Component:
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


def page() -> rx.Component:
    return rx.box(
        app_header(),
        rx.vstack(
            rx.hstack(
                rx.heading("Data Model", size="4"),
                rx.spacer(),
                rx.button(
                    "Refresh status",
                    size="1",
                    variant="soft",
                    on_click=DataModelState.load_status,
                    loading=DataModelState.is_loading,
                ),
                width="100%",
                align="center",
            ),
            rx.heading("Branch status", size="3"),
            _branch_row(DataModelState.dev_branch, DataModelState.dev_snapshot_id),
            _branch_row(DataModelState.prod_branch, DataModelState.prod_snapshot_id),
            rx.cond(
                DataModelState.error_message != "",
                rx.callout(DataModelState.error_message, icon="triangle_alert", color_scheme="red"),
                rx.fragment(),
            ),
            rx.hstack(
                rx.button(
                    "Sync Prod ← Dev",
                    size="1",
                    on_click=DataModelState.sync_prod_with_dev,
                    color_scheme="green",
                ),
                spacing="3",
            ),
            rx.heading("Diff between branches", size="3"),
            rx.hstack(
                rx.select(
                    items=[DataModelState.dev_branch, DataModelState.prod_branch],
                    value=DataModelState.diff_branch_left,
                    on_change=DataModelState.set_diff_branch_left,
                    width="10em",
                ),
                rx.text("vs", color="gray"),
                rx.select(
                    items=[DataModelState.dev_branch, DataModelState.prod_branch],
                    value=DataModelState.diff_branch_right,
                    on_change=DataModelState.set_diff_branch_right,
                    width="10em",
                ),
                rx.button(
                    "Load diff",
                    size="1",
                    on_click=DataModelState.load_diff,
                    loading=DataModelState.diff_is_loading,
                ),
                spacing="3",
                align="center",
                width="100%",
            ),
            rx.cond(
                DataModelState.diff_error_message != "",
                rx.callout(DataModelState.diff_error_message, icon="triangle_alert", color_scheme="red"),
                rx.fragment(),
            ),
            _table_accordion(DataModelState.diff_tables),
            rx.heading("Browse snapshots", size="3"),
            rx.hstack(
                rx.select(
                    items=[DataModelState.dev_branch, DataModelState.prod_branch],
                    value=DataModelState.view_branch,
                    on_change=DataModelState.set_view_branch,
                    width="10em",
                ),
                rx.input(
                    placeholder="Snapshot id (optional)",
                    value=DataModelState.view_snapshot_input,
                    on_change=DataModelState.set_view_snapshot_input,
                    width="14em",
                ),
                rx.button(
                    "Load snapshot",
                    size="1",
                    on_click=DataModelState.load_view_snapshot,
                    loading=DataModelState.view_is_loading,
                ),
                spacing="3",
                align="center",
                width="100%",
            ),
            rx.cond(
                DataModelState.view_error_message != "",
                rx.callout(DataModelState.view_error_message, icon="triangle_alert", color_scheme="red"),
                rx.fragment(),
            ),
            _table_accordion(DataModelState.view_tables),
            spacing="4",
            padding="1.5em",
            width="100%",
        ),
        width="100%",
    )
