import reflex as rx
from datetime import datetime

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
                rx.cond(
                    EtlState.k8s_enabled,
                    rx.hstack(
                        rx.text("Execution", size="1", color="gray"),
                        rx.select(
                            items=["local", "k8s"],
                            value=EtlState.execution_mode,
                            on_change=EtlState.set_execution_mode,
                            width="8em",
                        ),
                        spacing="2",
                        align="center",
                    ),
                    rx.box(),
                ),
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


def _k8s_jobs_card() -> rx.Component:
    """Card showing Kubernetes jobs status and management."""

    def _job_row(job: dict) -> rx.Component:
        """Render a single job row."""
        return rx.table.row(
            rx.table.cell(
                rx.text(
                    job["name"],
                    size="2",
                    style={"fontFamily": "monospace"},
                )
            ),
            rx.table.cell(
                rx.badge(
                    rx.cond(
                        job["status"] == "completed",
                        "COMPLETED",
                        rx.cond(
                            job["status"] == "running",
                            "RUNNING",
                            rx.cond(
                                job["status"] == "failed",
                                "FAILED",
                                rx.cond(
                                    job["status"] == "pending",
                                    "PENDING",
                                    "UNKNOWN",
                                ),
                            ),
                        ),
                    ),
                    color_scheme=rx.cond(
                        job["status"] == "completed",
                        "green",
                        rx.cond(
                            job["status"] == "running",
                            "blue",
                            rx.cond(
                                job["status"] == "failed",
                                "red",
                                "gray",
                            ),
                        ),
                    ),
                    size="1",
                )
            ),
            rx.table.cell(
                rx.text(
                    job.get("created_str", "—"),
                    size="1",
                    color="gray",
                )
            ),
            rx.table.cell(
                rx.hstack(
                    rx.button(
                        "Logs",
                        variant="ghost",
                        size="1",
                        on_click=EtlState.view_k8s_job_logs(job_name=job["name"]),  # type: ignore[arg-type,call-arg,func-returns-value]
                    ),
                    rx.button(
                        "Delete",
                        variant="ghost",
                        size="1",
                        color_scheme="red",
                        on_click=EtlState.delete_k8s_job(job_name=job["name"]),  # type: ignore[arg-type,call-arg,func-returns-value]
                    ),
                    spacing="1",
                )
            ),
        )

    return rx.cond(
        EtlState.k8s_jobs_open,
        rx.card(
            rx.hstack(
                rx.heading("Kubernetes Jobs", size="3"),
                rx.spacer(),
                rx.hstack(
                    rx.button(
                        "↻",
                        variant="ghost",
                        color_scheme="gray",
                        size="1",
                        on_click=EtlState.load_k8s_jobs,
                        loading=EtlState.k8s_jobs_loading,
                    ),
                    rx.button("Hide", variant="ghost", color_scheme="gray", size="1", on_click=EtlState.toggle_k8s_jobs),
                    spacing="2",
                    align="center",
                ),
                align="center",
                width="100%",
            ),
            rx.cond(
                EtlState.k8s_jobs_loading,
                rx.box(rx.text("Loading jobs...", size="2", color="gray"), padding="1em"),
                rx.cond(
                    EtlState.k8s_jobs,
                    rx.scroll_area(
                        rx.table.root(
                            rx.table.header(
                                rx.table.row(
                                    rx.table.column_header_cell("Job Name"),
                                    rx.table.column_header_cell("Status"),
                                    rx.table.column_header_cell("Created"),
                                    rx.table.column_header_cell("Actions"),
                                )
                            ),
                            rx.table.body(
                                rx.foreach(EtlState.k8s_jobs, _job_row),
                            ),
                            variant="surface",
                            style={"width": "100%"},
                        ),
                        type="always",
                        scrollbars="vertical",
                        style={"maxHeight": "30vh"},
                    ),
                    rx.box(rx.text("No Kubernetes jobs found", size="2", color="gray"), padding="1em"),
                ),
            ),
            padding="1em",
            width="100%",
        ),
        rx.card(
            rx.hstack(
                rx.heading("Kubernetes Jobs", size="3"),
                rx.spacer(),
                rx.button("Show", variant="ghost", color_scheme="gray", size="1", on_click=EtlState.toggle_k8s_jobs),
                align="center",
                width="100%",
            ),
            padding="1em",
            width="100%",
        ),
    )


def _pipeline_panel() -> rx.Component:
    return rx.vstack(
        _graph_card(),
        _k8s_jobs_card(),
        _logs_bottom(),
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
    return rx.cond(
        EtlState.logs_open,
        rx.card(
            rx.hstack(
                rx.heading("Logs", size="3"),
                rx.spacer(),
                rx.button("Hide", variant="ghost", color_scheme="gray", size="1", on_click=EtlState.toggle_logs),
                align="center",
                width="100%",
            ),
            rx.tabs.root(
                rx.tabs.list(
                    rx.tabs.trigger("System Logs", value="system"),
                    rx.cond(
                        EtlState.viewing_k8s_job_name,
                        rx.tabs.trigger(
                            rx.text(
                                rx.cond(
                                    EtlState.viewing_k8s_job_name,
                                    rx.text(EtlState.viewing_k8s_job_name),
                                    "K8s Job Logs",
                                ),
                                size="2",
                            ),
                            value="k8s_job",
                        ),
                        rx.box(),
                    ),
                ),
                rx.tabs.content(
                    rx.scroll_area(
                        rx.vstack(rx.foreach(EtlState.logs, lambda m: rx.text(m))),
                        type="always",
                        scrollbars="vertical",
                        style={"height": "44vh"},
                    ),
                    value="system",
                ),
                rx.cond(
                    EtlState.viewing_k8s_job_name,
                    rx.tabs.content(
                        rx.scroll_area(
                            rx.vstack(rx.foreach(EtlState.k8s_job_logs, lambda m: rx.text(m))),
                            type="always",
                            scrollbars="vertical",
                            style={"height": "44vh"},
                        ),
                        value="k8s_job",
                    ),
                    rx.box(),
                ),
                value=EtlState.log_view_mode,
                on_change=EtlState.set_log_view_mode,
                default_value="system",
            ),
            padding="1em",
            width="100%",
        ),
        rx.card(
            rx.hstack(
                rx.heading("Logs", size="3"),
                rx.spacer(),
                rx.button("Show", variant="ghost", color_scheme="gray", size="1", on_click=EtlState.toggle_logs),
                align="center",
                width="100%",
            ),
            padding="1em",
            width="100%",
        ),
    )


def _preview_styled_table() -> rx.Component:
    """Table with row styling for changes view and expandable cells."""

    def _expandable_cell(row: dict[str, rx.Var], col: rx.Var) -> rx.Component:
        """Create an expandable/collapsible cell for long text content."""
        row_id = row.get("row_id", "")
        return rx.table.cell(
            rx.box(
                rx.cond(
                    row.get("expanded", False),
                    rx.text(
                        row.get(col, "—"),  # type: ignore[call-overload]
                        size="1",
                        white_space="pre-wrap",
                        style={"wordBreak": "break-word"},
                    ),
                    rx.text(
                        row.get(col, "—"),  # type: ignore[call-overload]
                        size="1",
                        style={
                            "display": "-webkit-box",
                            "WebkitLineClamp": "2",
                            "WebkitBoxOrient": "vertical",
                            "overflow": "hidden",
                            "textOverflow": "ellipsis",
                            "maxWidth": "400px",
                            "wordBreak": "break-word",
                        },
                    ),
                ),
                cursor="pointer",
                on_click=EtlState.toggle_preview_row_expand(row_id=row_id),  # type: ignore[arg-type,call-arg,func-returns-value]
                style={"minWidth": "0", "width": "100%"},
            ),
            style={"minWidth": "0"},
        )

    def _make_row_renderer(row: dict[str, rx.Var]):
        """Create a column renderer that captures the row context."""
        return lambda col: _expandable_cell(row, col)

    def _row(r: dict[str, rx.Var]) -> rx.Component:
        return rx.table.row(
            rx.foreach(EtlState.preview_columns, _make_row_renderer(r)),
            style=r.get("row_style", {}),
        )

    return rx.table.root(
        rx.table.header(
            rx.table.row(
                rx.foreach(
                    EtlState.preview_columns,
                    lambda c: rx.table.column_header_cell(c),
                )
            )
        ),
        rx.table.body(rx.foreach(EtlState.preview_rows, _row)),
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
