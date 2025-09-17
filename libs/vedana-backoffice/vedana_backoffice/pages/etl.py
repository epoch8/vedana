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
            rx.text("steps", size="1", color="gray"),
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
    return rx.card(
        rx.heading("Tables", size="3"),
        rx.hstack(
            rx.select(
                items=EtlState.available_tables,
                placeholder="Select table",
                on_change=EtlState.preview_table,
                value=EtlState.preview_table_name,
                width="20em",
            ),
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
            rx.box(),
        ),
        padding="1em",
    )


def _sidebar() -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.heading("Controls", size="4"),
                rx.spacer(),
                rx.button(
                    "Hide",
                    variant="ghost",
                    color_scheme="gray",
                    size="1",
                    on_click=EtlState.toggle_sidebar,
                ),
                align="center",
                width="100%",
            ),
            _filters(),
            _table_preview(),
            rx.card(
                rx.vstack(
                    rx.heading("Run History", size="3"),
                    rx.text("Coming soon"),
                    spacing="2",
                ),
                variant="surface",
            ),
            spacing="4",
            width="100%",
        ),
        padding="1em",
        width="340px",
        height="100%",
    )


def _topbar() -> rx.Component:
    return rx.hstack(
        rx.heading("ETL Pipeline"),
        rx.spacer(),
        rx.hstack(
            rx.button("Sidebar", variant="soft", size="1", on_click=EtlState.toggle_sidebar),
            rx.button("Logs", variant="soft", size="1", on_click=EtlState.toggle_logs),
            spacing="2",
        ),
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
        rx.cond(
            EtlState.sidebar_open,
            rx.box(_sidebar(), width="340px"),
            rx.box(
                rx.card(
                    rx.button("Show Controls", size="1", on_click=EtlState.toggle_sidebar),
                    padding="0.5em",
                ),
                width="160px",
            ),
        ),
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
