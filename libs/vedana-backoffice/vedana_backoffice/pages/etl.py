import reflex as rx

from vedana_backoffice.state import EtlState
from vedana_backoffice.ui import app_header, breadcrumbs


def _filters() -> rx.Component:
    return rx.vstack(
        rx.heading("Filters", size="4"),
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


def _steps_table() -> rx.Component:
    def row(step: dict) -> rx.Component:
        return rx.table.row(
            rx.table.cell(rx.text(step.get("name", ""))),
            # rx.table.cell(rx.text(step.get("type", ""))),
            rx.table.cell(rx.text(step.get("inputs_str", ""))),
            rx.table.cell(rx.text(step.get("outputs_str", ""))),
            rx.table.cell(rx.text(step.get("labels_str", ""))),
            rx.table.cell(
                rx.button(
                    "Run",
                    on_click=EtlState.run_one_step(index=step["index"]),
                    loading=EtlState.is_running,
                )
            ),
        )

    return rx.card(
        rx.heading("Steps", size="4"),
        rx.table.root(
            rx.table.header(
                rx.table.row(
                    rx.table.column_header_cell("Name"),
                    # rx.table.column_header_cell("Type"),
                    rx.table.column_header_cell("Inputs"),
                    rx.table.column_header_cell("Outputs"),
                    rx.table.column_header_cell("Labels"),
                    rx.table.column_header_cell("Actions"),
                )
            ),
            rx.table.body(rx.foreach(EtlState.filtered_steps, row)),
            variant="surface",
        ),
        padding="1em",
    )


def _logs() -> rx.Component:
    return rx.card(
        rx.heading("Logs", size="4"),
        rx.scroll_area(
            rx.vstack(rx.foreach(EtlState.logs, lambda m: rx.text(m))),
            type="always",
            scrollbars="vertical",
            style={"height": "240px"},
        ),
        padding="1em",
    )


def _table_preview() -> rx.Component:
    return rx.card(
        rx.heading("Tables", size="4"),
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


def page() -> rx.Component:
    return rx.vstack(
        app_header(),
        breadcrumbs([("Main", "/"), ("ETL", "/etl")]),
        rx.heading("ETL Pipeline"),
        rx.grid(
            _filters(),
            _logs(),
            _steps_table(),
            _table_preview(),
            columns="2",
        ),
        align="start",
        padding="1em",
    )
