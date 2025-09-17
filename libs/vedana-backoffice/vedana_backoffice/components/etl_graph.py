import reflex as rx

from vedana_backoffice.state import EtlState


def _node_card(node: dict) -> rx.Component:
    node_index = node.get("index", -1)
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.badge(node.get("index_str", "#"), color_scheme="gray", variant="soft"),
                rx.text(node.get("name", ""), weight="medium"),
                justify="between",
                width="100%",
            ),
            rx.text(node.get("labels_str", ""), size="1", color="gray"),
            rx.hstack(
                rx.text("last:", size="1", color="gray"),
                rx.text(node.get("last_run", "—"), size="1", color="gray"),
                spacing="1",
            ),
            rx.hstack(
                rx.button(
                    "Run",
                    size="1",
                    variant="surface",
                    on_click=EtlState.run_one_step(index=node_index),
                    loading=EtlState.is_running,
                ),
                spacing="2",
            ),
            spacing="2",
            width="100%",
        ),
        padding="0.75em",
        style={
            "position": "absolute",
            "left": node.get("left", "0px"),
            "top": node.get("top", "0px"),
            "width": node.get("width", "220px"),
            "height": node.get("height", "90px"),
            "border": node.get("border_css", "1px solid #e5e7eb"),
            "overflow": "hidden",
            "cursor": "pointer",
        },
        variant="surface",
        on_click=EtlState.toggle_node_selection(index=node_index),
    )


def etl_graph() -> rx.Component:
    svg = rx.box(
        rx.html(EtlState.graph_svg),
        style={"position": "absolute", "left": 0, "top": 0, "pointerEvents": "none", "width": "100%", "height": "100%"},
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
