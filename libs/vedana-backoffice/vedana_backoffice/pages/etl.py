import reflex as rx

from vedana_backoffice.components.etl_graph import etl_graph
from vedana_backoffice.states.etl import EtlState
from vedana_backoffice.ui import app_header


def _graph_card() -> rx.Component:
    return rx.card(
        rx.hstack(
            rx.heading("ETL Pipeline", size="4"),
            rx.spacer(),
            rx.hstack(
                rx.hstack(
                    rx.text("Data view", size="1", color="gray"),
                    rx.switch(checked=EtlState.data_view, on_change=EtlState.set_data_view),
                    spacing="2",
                    align="center",
                ),
                rx.text("Flow", size="1", color="gray"),
                rx.select(
                    items=EtlState.available_flows,
                    value=EtlState.selected_flow,
                    on_change=EtlState.set_flow,
                    width="12em",
                ),
                rx.text("Stage", size="1", color="gray"),
                rx.select(
                    items=EtlState.available_stages,
                    value=EtlState.selected_stage,
                    on_change=EtlState.set_stage,
                    width="12em",
                ),
                rx.button("Reset", variant="soft", size="1", on_click=EtlState.reset_filters),
                rx.button("Run Selected", size="1", on_click=EtlState.run_selected, loading=EtlState.is_running),
                rx.tooltip(
                    rx.button(
                        "↻",
                        variant="ghost",
                        color_scheme="gray",
                        size="1",
                        on_click=EtlState.load_pipeline_metadata,
                    ),
                    content="Reload metadata",
                ),
                spacing="3",
                align="center",
            ),
            align="center",
            width="100%",
        ),
        rx.scroll_area(
            rx.box(
                etl_graph(),
                style={
                    "position": "relative",
                    "minWidth": EtlState.graph_width_css,
                    "minHeight": EtlState.graph_height_css,
                },
            ),
            type="always",
            scrollbars="both",
            style={"height": "65vh", "width": "100%"},
        ),
        padding="1em",
        width="100%",
    )


def _pipeline_panel() -> rx.Component:
    return rx.vstack(
        _graph_card(),
        rx.cond(EtlState.logs_open, _logs_bottom(), rx.box(width="100%")),
        spacing="1",
        width="100%",
    )


def _pipeline_tabs() -> rx.Component:
    return rx.tabs.root(
        rx.tabs.list(
            rx.foreach(
                EtlState.available_pipelines,
                lambda name: rx.tabs.trigger(
                    rx.cond(name, name, EtlState.default_pipeline_name),
                    value=name,
                ),
            ),
            style={"gap": "0.75rem"},
        ),
        rx.foreach(
            EtlState.available_pipelines,
            lambda name: rx.tabs.content(
                _pipeline_panel(),
                value=name,
                style={"width": "100%"},
            ),
        ),
        value=EtlState.selected_pipeline,
        on_change=EtlState.set_pipeline,
        default_value=EtlState.default_pipeline_name,
        style={"width": "100%"},
    )


def _logs_bottom() -> rx.Component:
    return rx.card(
        rx.hstack(
            rx.heading("Logs", size="3"),
            rx.spacer(),
            rx.button("Hide", variant="ghost", color_scheme="gray", size="1", on_click=EtlState.toggle_logs),
            align="center",
            width="100%",
        ),
        rx.scroll_area(
            rx.vstack(rx.foreach(EtlState.logs, lambda m: rx.text(m))),
            type="always",
            scrollbars="vertical",
            style={"height": "22vh"},
        ),
        padding="1em",
        width="100%",
    )


def _preview_styled_table() -> rx.Component:
    """Table with row styling for changes view."""
    return rx.table.root(
        rx.table.header(
            rx.table.row(
                rx.foreach(
                    EtlState.preview_columns,
                    lambda c: rx.table.column_header_cell(c),
                )
            )
        ),
        rx.table.body(
            rx.foreach(
                EtlState.preview_rows,
                lambda r: rx.table.row(
                    rx.foreach(
                        EtlState.preview_columns,
                        lambda c: rx.table.cell(
                            rx.text(
                                r.get(c, "—"),
                                style={
                                    "whiteSpace": "nowrap",
                                    "textOverflow": "ellipsis",
                                    "overflow": "hidden",
                                    "maxWidth": "400px",
                                },
                            )
                        ),
                    ),
                    style=r.get("row_style", {}),
                ),
            )
        ),
        variant="surface",
        style={"width": "100%", "tableLayout": "auto"},
    )


def _table_preview_dialog() -> rx.Component:
    return rx.dialog.root(
        rx.dialog.content(
            rx.vstack(
                rx.hstack(
                    rx.dialog.title(
                        rx.cond(
                            EtlState.preview_display_name,
                            EtlState.preview_display_name,
                            rx.cond(EtlState.preview_table_name, EtlState.preview_table_name, ""),
                        ),
                        size="4",
                    ),
                    rx.spacer(),
                    rx.hstack(
                        rx.text("Last run changes", size="1", color="gray"),
                        rx.switch(
                            checked=EtlState.preview_changes_only,
                            on_change=EtlState.toggle_preview_changes_only,
                            size="1",
                        ),
                        spacing="2",
                        align="center",
                    ),
                    rx.dialog.close(
                        rx.button("Close", variant="ghost", color_scheme="gray", size="1"),
                    ),
                    align="center",
                    width="100%",
                ),
                rx.cond(
                    EtlState.has_preview,
                    rx.vstack(
                        rx.scroll_area(
                            _preview_styled_table(),
                            type="always",
                            scrollbars="both",
                            style={"maxHeight": "68vh", "maxWidth": "calc(90vw - 3em)"},
                        ),
                        # Server-side pagination controls with legend
                        rx.hstack(
                            rx.text(EtlState.preview_rows_display, size="2", color="gray"),
                            # Color legend (only shown in changes view)
                            rx.cond(
                                EtlState.preview_changes_only,
                                rx.hstack(
                                    rx.hstack(
                                        rx.box(
                                            style={
                                                "width": "12px",
                                                "height": "12px",
                                                "backgroundColor": "rgba(34,197,94,0.12)",
                                                "borderRadius": "2px",
                                            }
                                        ),
                                        rx.text("Added", size="1", color="gray"),
                                        spacing="1",
                                        align="center",
                                    ),
                                    rx.hstack(
                                        rx.box(
                                            style={
                                                "width": "12px",
                                                "height": "12px",
                                                "backgroundColor": "rgba(245,158,11,0.12)",
                                                "borderRadius": "2px",
                                            }
                                        ),
                                        rx.text("Updated", size="1", color="gray"),
                                        spacing="1",
                                        align="center",
                                    ),
                                    rx.hstack(
                                        rx.box(
                                            style={
                                                "width": "12px",
                                                "height": "12px",
                                                "backgroundColor": "rgba(239,68,68,0.12)",
                                                "borderRadius": "2px",
                                            }
                                        ),
                                        rx.text("Deleted", size="1", color="gray"),
                                        spacing="1",
                                        align="center",
                                    ),
                                    spacing="3",
                                    align="center",
                                ),
                                rx.box(),
                            ),
                            rx.spacer(),
                            rx.hstack(
                                rx.button(
                                    "⏮",
                                    variant="soft",
                                    size="1",
                                    on_click=EtlState.preview_first_page,
                                    disabled=~EtlState.preview_has_prev,
                                ),
                                rx.button(
                                    "← Prev",
                                    variant="soft",
                                    size="1",
                                    on_click=EtlState.preview_prev_page,
                                    disabled=~EtlState.preview_has_prev,
                                ),
                                rx.text(
                                    EtlState.preview_page_display,
                                    size="2",
                                    style={"minWidth": "100px", "textAlign": "center"},
                                ),
                                rx.button(
                                    "Next →",
                                    variant="soft",
                                    size="1",
                                    on_click=EtlState.preview_next_page,
                                    disabled=~EtlState.preview_has_next,
                                ),
                                rx.button(
                                    "⏭",
                                    variant="soft",
                                    size="1",
                                    on_click=EtlState.preview_last_page,
                                    disabled=~EtlState.preview_has_next,
                                ),
                                spacing="2",
                                align="center",
                            ),
                            width="100%",
                            align="center",
                            padding_top="0.5em",
                        ),
                        width="100%",
                        spacing="2",
                    ),
                    rx.cond(
                        EtlState.preview_changes_only,
                        rx.box(rx.text("No changes in last run")),
                        rx.box(rx.text("No data")),
                    ),
                ),
                spacing="3",
                width="100%",
            ),
            style={
                "maxWidth": "90vw",
                "maxHeight": "85vh",
                "width": "fit-content",
                "minWidth": "400px",
            },
        ),
        open=EtlState.preview_open,
        on_open_change=EtlState.set_preview_open,
    )


def page() -> rx.Component:
    return rx.vstack(
        app_header(),
        _pipeline_tabs(),
        _table_preview_dialog(),
        align="start",
        spacing="1",
        padding="1em",
        width="100%",
    )
