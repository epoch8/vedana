import reflex as rx

from vedana_backoffice.state import EtlState


def _node_card(node: dict) -> rx.Component:
    is_table = node.get("node_type") == "table"  # step for transform, table for table
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.hstack(
                    rx.cond(
                        is_table,
                        rx.box(),
                        rx.badge(node.get("step_type", ""), color_scheme="indigo", variant="soft"),
                    ),
                    spacing="2",
                ),
                rx.text(node.get("name", ""), weight="medium"),
                justify="between",
                width="100%",
            ),
            rx.text(node.get("labels_str", ""), size="1", color="gray"),
            rx.cond(
                is_table,
                rx.hstack(
                    rx.tooltip(rx.text(node.get("last_run", "—"), size="1", color="gray"), content="last update time"),
                    rx.hstack(
                        rx.text(node.get("row_count", "—"), size="1", color="gray", weight="bold"),
                        rx.text(node.get("last_add", "—"), size="1", color="green", weight="bold"),
                        rx.text("/", size="1", color="gray", weight="bold"),
                        rx.text(node.get("last_rm", "—"), size="1", color="red", weight="bold"),
                        spacing="1",
                    ),
                    width="100%",
                    justify="between",
                ),
                rx.box(),
            ),
            spacing="2",
            width="100%",
        ),
        padding="0.75em",
        style={
            "position": "absolute",
            "left": node.get("left", "0px"),
            "top": node.get("top", "0px"),
            "width": node.get("width", "420px"),
            "height": "auto",
            "border": node.get("border_css", "1px solid #e5e7eb"),
            "overflow": "visible",
            "boxSizing": "border-box",
        },
        variant="surface",
        on_click=rx.cond(
            is_table,
            # no direct return values here and that's ok, handled in state
            EtlState.preview_table(table_name=node.get("name", "")),  # type: ignore
            EtlState.toggle_node_selection(index=node.get("index_value")),  # type: ignore
        ),
    )


def etl_graph() -> rx.Component:
    svg = rx.box(
        rx.html(EtlState.graph_svg),
        style={
            "position": "absolute",
            "left": 0,
            "top": 0,
            "pointerEvents": "none",
            "width": "100%",
            "height": "100%",
        },
    )

    nodes_layer = rx.box(
        rx.foreach(EtlState.graph_nodes, _node_card),
        style={
            "position": "absolute",
            "left": 0,
            "top": 0,
            "width": "100%",
            "height": "100%",
        },
    )

    return rx.box(svg, nodes_layer, style={"position": "relative", "width": "100%", "height": "100%"})
