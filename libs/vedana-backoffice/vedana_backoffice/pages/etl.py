import reflex as rx

from vedana_backoffice.components.etl_graph import etl_graph
from vedana_backoffice.state import EtlState
from vedana_backoffice.ui import app_header, breadcrumbs


def _filters() -> rx.Component:
    return rx.vstack(
        rx.heading("Filters", size="3"),
        rx.text("Flow"),
        rx.select(
            items=EtlState.available_flows,
            value=EtlState.selected_flow,
            on_change=EtlState.set_flow,
            width="16em",
        ),
        rx.text("Stage"),
        rx.select(
            items=EtlState.available_stages,
            value=EtlState.selected_stage,
            on_change=EtlState.set_stage,
            width="16em",
        ),
        rx.hstack(
            rx.button("Run Selected", on_click=EtlState.run_selected, loading=EtlState.is_running),
            rx.button("Reload Metadata", on_click=EtlState.load_pipeline_metadata),
            spacing="3",
        ),
        spacing="3",
    )


def _graph_card() -> rx.Component:
    return rx.card(
        rx.hstack(
            rx.heading("Pipeline", size="4"),
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
                spacing="2",
                align="center",
            ),
            align="center",
            width="100%",
        ),
        rx.box(etl_graph(), style={"height": "65vh", "width": "100%"}),
        padding="1em",
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


def _table_preview() -> rx.Component:
    return rx.cond(
        EtlState.preview_open,
        rx.card(
            rx.hstack(
                rx.heading(rx.cond(EtlState.preview_table_name, EtlState.preview_table_name, "Table"), size="3"),
                rx.spacer(),
                rx.button("Close", variant="ghost", color_scheme="gray", size="1", on_click=EtlState.close_preview),
                align="center",
                width="100%",
            ),
            rx.cond(
                EtlState.has_preview,
                rx.table.root(
                    rx.table.header(
                        rx.table.row(rx.foreach(EtlState.preview_columns, lambda c: rx.table.column_header_cell(c)))
                    ),
                    rx.table.body(
                        rx.foreach(
                            EtlState.preview_rows,
                            lambda r: rx.table.row(
                                rx.foreach(EtlState.preview_columns, lambda c: rx.table.cell(rx.text(r.get(c, ""))))
                            ),
                        )
                    ),
                    variant="surface",
                ),
                rx.box(rx.text("No data")),
            ),
            padding="1em",
            width="400px",
            height="100%",
        ),
        rx.box(),
    )


def _topbar() -> rx.Component:
    return rx.hstack(
        rx.heading("ETL Pipeline"),
        rx.spacer(),
        rx.hstack(rx.button("Logs", variant="soft", size="1", on_click=EtlState.toggle_logs), spacing="2"),
        align="center",
        width="100%",
    )


def page() -> rx.Component:
    main_row = rx.flex(
        rx.box(
            _graph_card(),
            flex="1",
            min_width="0",
        ),
        rx.cond(EtlState.preview_open, rx.box(_table_preview(), width="420px"), rx.box()),
        gap="1em",
        width="100%",
        align="start",
    )

    return rx.vstack(
        app_header(),
        breadcrumbs([("Main", "/"), ("ETL", "/etl")]),
        _topbar(),
        main_row,
        rx.cond(EtlState.logs_open, _logs_bottom(), rx.box()),
        align="start",
        spacing="1",
        padding="1em",
        width="100%",
    )
