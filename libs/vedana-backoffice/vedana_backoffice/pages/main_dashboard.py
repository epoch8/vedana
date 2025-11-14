import reflex as rx

from vedana_backoffice.state import DashboardState
from vedana_backoffice.ui import app_header


def _stat_tile(title: str, value: rx.Var | str, subtitle: str = "", color: str = "indigo") -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.text(title, size="1", color="gray"),
            rx.heading(value, size="6"),
            rx.cond(subtitle != "", rx.text(subtitle, size="1", color="gray"), rx.box()),
            spacing="1",
            width="100%",
        ),
        variant="surface",
        width="100%",
        padding="1em",
        style={"borderTop": f"3px solid var(--{color}-9)"},
    )


def _graph_stats_card() -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.heading("Graph stats", size="3"),
                rx.spacer(),
                rx.badge(
                    rx.cond(
                        (DashboardState.nodes_total_diff == 0) & (DashboardState.edges_total_diff == 0),  # type: ignore[operator]
                        "OK",
                        "Warning!",
                    ),
                    color_scheme=rx.cond(  # type: ignore[operator]
                        (DashboardState.nodes_total_diff == 0) & (DashboardState.edges_total_diff == 0),
                        "green",
                        "red",
                    ),
                    variant="soft",
                ),
                align="center",
                width="100%",
            ),
            rx.grid(
                _stat_tile(
                    "Total Nodes",
                    DashboardState.graph_total_nodes,  # type: ignore[arg-type]
                    subtitle=rx.cond(  # type: ignore[arg-type]
                        DashboardState.nodes_total_diff == 0,  # type: ignore[operator]
                        "Matches pipeline",
                        rx.cond(  # type: ignore[arg-type]
                            DashboardState.nodes_total_diff > 0,
                            rx.text(f"+{DashboardState.nodes_total_diff} vs ETL", color_scheme="red", weight="bold"),
                            rx.text(f"{DashboardState.nodes_total_diff} vs ETL", color_scheme="red", weight="bold"),
                        ),
                    ),
                    color="green",
                ),
                _stat_tile(
                    "Total Edges",
                    DashboardState.graph_total_edges,  # type: ignore[arg-type]
                    subtitle=rx.cond(  # type: ignore[arg-type]
                        DashboardState.edges_total_diff == 0,  # type: ignore[operator]
                        "Matches pipeline",
                        rx.cond(  # type: ignore[operator]
                            DashboardState.edges_total_diff > 0,
                            rx.text(f"+{DashboardState.edges_total_diff} vs ETL", color_scheme="red", weight="bold"),
                            rx.text(f"{DashboardState.edges_total_diff} vs ETL", color_scheme="red", weight="bold"),
                        ),
                    ),
                    color="green",
                ),
                columns="2",
                spacing="4",
                width="100%",
            ),
            rx.grid(
                _stat_tile("Nodes Added", DashboardState.new_nodes, color="indigo"),  # type: ignore[arg-type]
                _stat_tile("Edges Added", DashboardState.new_edges, color="indigo"),  # type: ignore[arg-type]
                _stat_tile("Nodes Updated", DashboardState.updated_nodes, color="amber"),  # type: ignore[arg-type]
                _stat_tile("Edges Updated", DashboardState.updated_edges, color="amber"),  # type: ignore[arg-type]
                _stat_tile("Nodes Deleted", DashboardState.deleted_nodes, color="red"),  # type: ignore[arg-type]
                _stat_tile("Edges Deleted", DashboardState.deleted_edges, color="red"),  # type: ignore[arg-type]
                columns="2",
                spacing="4",
                width="100%",
            ),
            spacing="3",
            width="100%",
        ),
        padding="1em",
        width="100%",
        style={"height": "100%", "display": "flex", "flexDirection": "column"},
    )


def _changes_preview_popover() -> rx.Component:
    return rx.cond(
        DashboardState.changes_preview_open,  # type: ignore[operator]
        rx.popover.root(
            rx.popover.trigger(
                rx.box(
                    style={
                        "position": "absolute",
                        "left": DashboardState.changes_preview_anchor_left,  # type: ignore[arg-type]
                        "top": DashboardState.changes_preview_anchor_top,  # type: ignore[arg-type]
                        "width": "1px",
                        "height": "1px",
                        "pointerEvents": "none",
                    }
                )
            ),
            rx.popover.content(
                rx.vstack(
                    rx.hstack(
                        rx.heading(
                            rx.cond(
                                DashboardState.changes_preview_table_name,  # type: ignore[operator]
                                DashboardState.changes_preview_table_name,  # type: ignore[arg-type]
                                "",
                            ),
                            size="3",
                        ),
                        rx.spacer(),
                        rx.popover.close(
                            rx.button("Close", variant="ghost", color_scheme="gray", size="1"),
                        ),
                        align="center",
                        width="100%",
                    ),
                    rx.cond(
                        DashboardState.changes_has_preview,  # type: ignore[operator]
                        rx.scroll_area(
                            rx.table.root(
                                rx.table.header(
                                    rx.table.row(
                                        rx.foreach(
                                            DashboardState.changes_preview_columns,  # type: ignore[arg-type]
                                            lambda c: rx.table.column_header_cell(c),
                                        )
                                    )
                                ),
                                rx.table.body(
                                    rx.foreach(
                                        DashboardState.changes_preview_rows,  # type: ignore[arg-type]
                                        lambda r: rx.table.row(
                                            rx.foreach(
                                                DashboardState.changes_preview_columns,  # type: ignore[arg-type]
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
                                style={"width": "100%", "tableLayout": "fixed"},
                            ),
                            type="always",
                            scrollbars="vertical",
                            style={"maxHeight": "70vh", "width": "100%"},
                        ),
                        rx.box(rx.text("No data")),
                    ),
                    spacing="2",
                    padding="1em",
                    width="fit-content",
                    min_width="400px",
                ),
                side="right",
                align="center",
                size="3",
                avoid_collisions=True,
                collision_padding=20,
                style={
                    "width": "fit-content",
                    "maxWidth": "85vw",
                },
            ),
            open=True,
            on_open_change=DashboardState.set_changes_preview_open,  # type: ignore[arg-type]
        ),
        rx.box(),
    )


def _graph_stats_expanded_card() -> rx.Component:
    return rx.card(
        rx.grid(
            _per_label_stats_table("Graph Nodes by Label", DashboardState.graph_nodes_by_label, "nodes"),  # type: ignore[arg-type]
            _per_label_stats_table("Graph Edges by Type", DashboardState.graph_edges_by_type, "edges"),  # type: ignore[arg-type]
            columns="1",  # type: ignore[arg-type]
            spacing="4",  # type: ignore[arg-type]
            width="100%",
            style={"height": "100%"},
        ),
        padding="1em",
        style={"height": "100%", "display": "flex", "flexDirection": "column"},
    )


def _ingest_card() -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.heading("Ingest Activity", size="3"),
                rx.spacer(),
                # rx.text("Generator tables", size="1", color="gray"),
                align="center",
                width="100%",
            ),
            rx.grid(
                _stat_tile("New Entries", DashboardState.ingest_new_total, color="indigo"),  # type: ignore[arg-type]
                _stat_tile("Updated Entries", DashboardState.ingest_updated_total, color="amber"),  # type: ignore[arg-type]
                _stat_tile("Deleted Entries", DashboardState.ingest_deleted_total, color="red"),  # type: ignore[arg-type]
                columns="3",
                spacing="4",
                width="100%",
            ),
            rx.table.root(
                rx.table.header(
                    rx.table.row(
                        rx.table.column_header_cell("Table"),
                        rx.table.column_header_cell("Total"),
                        rx.table.column_header_cell("Added"),
                        rx.table.column_header_cell("Updated"),
                        rx.table.column_header_cell("Deleted"),
                    )
                ),
                rx.table.body(
                    rx.foreach(
                        DashboardState.ingest_breakdown,  # type: ignore[arg-type]
                        lambda r: rx.table.row(
                            rx.table.cell(rx.text(r.get("table", ""))),
                            # todo order by ?
                            rx.table.cell(rx.text(r.get("total", 0))),  # type: ignore[arg-type]
                            rx.table.cell(rx.text(r.get("added", 0))),  # type: ignore[arg-type]
                            rx.table.cell(rx.text(r.get("updated", 0))),  # type: ignore[arg-type]
                            rx.table.cell(rx.text(r.get("deleted", 0))),  # type: ignore[arg-type]
                            on_click=DashboardState.open_changes_preview(table_name=r.get("table", "")),  # type: ignore
                            style={"cursor": "pointer"},
                        ),
                    )
                ),
                variant="surface",
                style={"width": "100%", "tableLayout": "fixed"},
            ),
            spacing="3",
            width="100%",
        ),
        padding="1em",
        width="100%",
        style={"height": "100%", "display": "flex", "flexDirection": "column"},
    )


def _per_label_stats_table(title: str, rows: rx.Var | list[dict], kind: str) -> rx.Component:
    def _row(r: dict) -> rx.Component:
        label = r.get("label", "")
        return rx.table.row(
            rx.table.cell(rx.text(label)),
            rx.table.cell(rx.text(r.get("graph_count", 0))),
            rx.table.cell(rx.text(r.get("etl_count", 0))),
            rx.table.cell(rx.text(r.get("added", 0))),
            rx.table.cell(rx.text(r.get("updated", 0))),
            rx.table.cell(rx.text(r.get("deleted", 0))),
            on_click=rx.cond(
                kind == "nodes",
                DashboardState.open_graph_per_label_changes_preview(kind="nodes", label=label),  # type: ignore
                DashboardState.open_graph_per_label_changes_preview(kind="edges", label=label),  # type: ignore
            ),
            style={"cursor": "pointer"},
        )

    return rx.vstack(
        rx.hstack(rx.heading(title, size="3"), align="center", justify="between", width="100%"),
        rx.box(
            rx.scroll_area(
                rx.table.root(
                    rx.table.header(
                        rx.table.row(
                            rx.table.column_header_cell("Label"),
                            rx.table.column_header_cell("Graph"),
                            rx.table.column_header_cell("ETL"),
                            rx.table.column_header_cell("Added"),
                            rx.table.column_header_cell("Updated"),
                            rx.table.column_header_cell("Deleted"),
                        )
                    ),
                    rx.table.body(rx.foreach(rows, _row)),
                    variant="surface",
                    style={"width": "100%", "tableLayout": "fixed"},
                ),
                type="always",
                scrollbars="vertical",
                style={
                    "position": "absolute",
                    "top": 0,
                    "bottom": 0,
                    "left": 0,
                    "right": 0,
                },
            ),
            style={
                "position": "relative",
                "flex": "1 1 0",
                "minHeight": 0,
                "width": "100%",
            },
        ),
        spacing="2",
        style={"height": "100%", "display": "flex", "flexDirection": "column"},
    )


def page() -> rx.Component:
    return rx.vstack(
        app_header(),
        rx.card(
            rx.vstack(
                rx.hstack(
                    rx.hstack(
                        rx.text("ETL Overview", weight="medium"),
                        align="center",
                        spacing="3",
                    ),
                    rx.spacer(),
                    rx.hstack(
                        rx.text("Timeframe (days):", size="1", color="gray"),
                        rx.select(
                            items=DashboardState.time_window_options,  # type: ignore[arg-type]
                            on_change=DashboardState.set_time_window_days,  # type: ignore[arg-type]
                            width="8em",
                        ),
                        rx.button(
                            rx.cond(DashboardState.loading, "Refreshing…", "Refresh"),  # type: ignore[operator]
                            on_click=DashboardState.load_dashboard,  # type: ignore[arg-type]
                            loading=DashboardState.loading,
                            size="2",
                        ),
                        spacing="3",
                        align="center",
                    ),
                    align="center",
                    width="100%",
                ),
                rx.grid(
                    _ingest_card(),
                    _graph_stats_card(),
                    _graph_stats_expanded_card(),
                    columns="3",
                    spacing="4",
                    width="100%",
                    align="start",
                    style={"gridTemplateColumns": "3fr 2fr 5fr", "height": "90vh", "align-items": "stretch"},
                ),
            ),
            padding="1em",
            width="100%",
        ),
        rx.cond(
            DashboardState.error_message != "",  # type: ignore[operator]
            rx.callout(DashboardState.error_message, color_scheme="red", variant="soft"),  # type: ignore[arg-type]
            rx.box(),
        ),
        _changes_preview_popover(),
        spacing="4",
        align="start",
        padding="1em",
        width="100%",
    )
