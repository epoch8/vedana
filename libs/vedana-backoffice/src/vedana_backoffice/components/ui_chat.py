import reflex as rx

from vedana_backoffice.states.data_model import DataModelState
from vedana_backoffice.ui import themed_data_table, table_accordion


def render_message_bubble(
    msg: dict,
    on_toggle_details,
    extras: rx.Component | None = None,
    corner_tags_component: rx.Component | None = None,
) -> rx.Component:  # type: ignore[valid-type]
    """Render a chat-style message bubble.

    Expects msg dict with keys:
      - content, is_assistant (bool-like), created_at_fmt or created_at
      - has_tech, has_logs, show_details
      - has_models, has_vts, has_cypher, models_str, vts_str, cypher_str (optional)
      - logs_str (optional)
    """

    tech_block = rx.cond(
        msg.get("has_tech"),
        rx.card(
            rx.vstack(
                rx.cond(
                    msg.get("has_models"),
                    rx.vstack(
                        rx.text("Models", weight="medium"),
                        rx.code_block(
                            msg.get("models_str", ""),
                            font_size="11px",
                            language="json",
                            # (bug in reflex?) code_block does not pass some custom styling (wordBreak, whiteSpace)
                            # https://github.com/reflex-dev/reflex/issues/6051
                            code_tag_props={"style": {"whiteSpace": "pre-wrap"}},
                            style={
                                "display": "block",
                                "maxWidth": "100%",
                                "boxSizing": "border-box",
                            },
                        ),
                        spacing="1",
                        width="100%",
                    ),
                ),
                rx.cond(
                    msg.get("has_vts"),
                    rx.vstack(
                        rx.text("VTS Queries", weight="medium"),
                        rx.code_block(
                            msg.get("vts_str", ""),
                            font_size="11px",
                            language="python",
                            code_tag_props={"style": {"whiteSpace": "pre-wrap"}},
                            style={
                                "display": "block",
                                "maxWidth": "100%",
                                "boxSizing": "border-box",
                            },
                        ),
                        spacing="1",
                        width="100%",
                    ),
                ),
                rx.cond(
                    msg.get("has_cypher"),
                    rx.vstack(
                        rx.text("Cypher Queries", weight="medium"),
                        rx.code_block(
                            msg.get("cypher_str", ""),
                            font_size="11px",
                            language="cypher",
                            code_tag_props={"style": {"whiteSpace": "pre-wrap"}},
                            style={
                                "display": "block",
                                "maxWidth": "100%",
                                "boxSizing": "border-box",
                            },
                        ),
                        spacing="1",
                        width="100%",
                    ),
                ),
                rx.cond(
                    msg.get("dm_snapshot_id"),
                    rx.hstack(
                        rx.text("Data Model ID: "),
                        rx.text(msg.get("dm_snapshot_id")),
                        rx.popover.root(
                            rx.popover.trigger(
                                rx.button(
                                    "Open",
                                    size="1",
                                    variant="soft",
                                    on_click=DataModelState.open_quick_view(
                                        snapshot_id=msg.get("dm_snapshot_id")
                                    ),  # type: ignore[arg-type,call-arg,func-returns-value]
                                )
                            ),
                            rx.popover.content(
                                rx.vstack(
                                    rx.hstack(
                                        rx.text("Snapshot", weight="medium"),
                                        rx.spacer(),
                                        rx.text(DataModelState.quick_view_snapshot_id, color="gray"),
                                        width="100%",
                                    ),
                                    rx.cond(
                                        DataModelState.quick_view_error_message != "",
                                        rx.callout(
                                            DataModelState.quick_view_error_message,
                                            icon="triangle_alert",
                                            color_scheme="red",
                                        ),
                                        rx.fragment(),
                                    ),
                                    rx.cond(
                                        DataModelState.quick_view_is_loading,
                                        rx.center(rx.spinner(size="3"), height="200px"),
                                        table_accordion(DataModelState.quick_view_tables),  # type: ignore[arg-type]
                                    ),
                                    spacing="3",
                                    width="100%",
                                ),
                                style={"maxWidth": "90vw", "maxHeight": "80vh", "overflow": "auto"},
                            ),
                            open=DataModelState.quick_view_open,
                            on_open_change=DataModelState.set_quick_view_open,
                        ),
                        rx.popover.root(
                            rx.popover.trigger(
                                rx.button(
                                    "Show diff",
                                    size="1",
                                    variant="soft",
                                    on_click=DataModelState.open_quick_diff(
                                        snapshot_id=msg.get("dm_snapshot_id"),
                                        compare_branch=DataModelState.prod_branch,
                                    ),  # type: ignore[arg-type,call-arg,func-returns-value]
                                )
                            ),
                            rx.popover.content(
                                rx.vstack(
                                    rx.hstack(
                                        rx.text("Diff vs prod", weight="medium"),
                                        rx.spacer(),
                                        rx.text(DataModelState.quick_diff_snapshot_id, color="gray"),
                                        width="100%",
                                    ),
                                    rx.cond(
                                        DataModelState.quick_diff_error_message != "",
                                        rx.callout(
                                            DataModelState.quick_diff_error_message,
                                            icon="triangle_alert",
                                            color_scheme="red",
                                        ),
                                        rx.fragment(),
                                    ),
                                    rx.cond(
                                        DataModelState.quick_diff_is_loading,
                                        rx.center(rx.spinner(size="3"), height="200px"),
                                        table_accordion(DataModelState.quick_diff_tables),  # type: ignore[arg-type]
                                    ),
                                    spacing="3",
                                    width="100%",
                                ),
                                style={"maxWidth": "90vw", "maxHeight": "80vh", "overflow": "auto"},
                            ),
                            open=DataModelState.quick_diff_open,
                            on_open_change=DataModelState.set_quick_diff_open,
                        ),
                        spacing="2",
                        align="center",
                    )
                ),
                spacing="2",
                width="100%",
            ),
            padding="0.75em",
            variant="surface",
        ),
        rx.box(),
    )

    generic_details_block = rx.cond(
        msg.get("generic_meta"),
        rx.code_block(
            msg.get("event_data_str", ""),
            font_size="11px",
            language="json",
            code_tag_props={"style": {"whiteSpace": "pre-wrap"}},
            style={
                "display": "block",
                "maxWidth": "100%",
                "boxSizing": "border-box",
            },
        ),
        rx.box(),
    )

    logs_block = rx.cond(
        msg.get("has_logs"),
        rx.card(
            rx.vstack(
                rx.text("Logs", weight="medium"),
                rx.scroll_area(
                    rx.code_block(
                        msg.get("logs_str", ""),
                        font_size="11px",
                        language="log",
                        wrap_long_lines=True,
                        style={
                            "display": "block",
                            "maxWidth": "100%",
                            "boxSizing": "border-box",
                        },
                        code_tag_props={"style": {"whiteSpace": "pre-wrap"}},  # styling workaround
                    ),
                    type="always",
                    scrollbars="vertical",
                    style={
                        "maxHeight": "25vh",
                        "width": "100%",
                    },
                ),
                spacing="1",
                width="100%",
            ),
            padding="0.75em",
            width="100%",
            variant="surface",
        ),
        rx.box(),
    )

    details_block = rx.vstack(
        tech_block,
        generic_details_block,
        logs_block,
        spacing="2",
        width="100%",
    )

    # Tag badges for feedback
    tags_box = corner_tags_component or rx.box()

    header_left = rx.hstack(
        # event type label at the very left
        rx.cond(
            msg.get("tag_label", "") != "",
            rx.badge(msg.get("tag_label", ""), variant="soft", size="1", color_scheme="gray"),
            rx.box(),
        ),
        rx.cond(
            msg.get("has_tech") | msg.get("has_logs") | msg.get("generic_meta"),  # type: ignore[operator]
            rx.button(
                "Details",
                variant="ghost",
                color_scheme="gray",
                size="1",
                on_click=on_toggle_details,  # type: ignore[arg-type]
            ),
        ),
        rx.text(msg.get("created_at", ""), size="1", color="gray"),
        spacing="2",
        align="center",
    )

    body = rx.vstack(
        rx.hstack(header_left, tags_box, justify="between", width="100%", align="center"),  # header
        rx.text(
            msg.get("content", ""),
            style={
                "whiteSpace": "pre-wrap",
                "wordBreak": "break-word",
            },
        ),
        rx.cond(msg.get("show_details"), details_block),
        rx.cond(extras is not None, extras or rx.box()),
        spacing="2",
        width="100%",
        style={
            "maxWidth": "100%",
        },
    )

    assistant_bubble = rx.card(
        body,
        padding="0.75em",
        style={
            "maxWidth": "70%",  # 35 vw is 70% of 50% parent card width in vw terms
            "backgroundColor": "#11182714",
            "border": "1px solid #e5e7eb",
            "borderRadius": "12px",
            "wordBreak": "break-word",
            "overflowX": "hidden",
        },
    )
    user_bubble = rx.card(
        body,
        padding="0.75em",
        style={
            "maxWidth": "70%",
            "backgroundColor": "#3b82f614",
            "border": "1px solid #e5e7eb",
            "borderRadius": "12px",
            "wordBreak": "break-word",
            "overflowX": "hidden",
        },
    )

    return rx.cond(
        msg.get("is_assistant"),
        rx.hstack(
            rx.avatar(fallback="A", size="2", radius="full"),
            assistant_bubble,
            spacing="2",
            width="100%",
            justify="start",
            align="start",
        ),
        rx.hstack(
            user_bubble,
            rx.avatar(fallback="U", size="2", radius="full"),
            spacing="2",
            width="100%",
            justify="end",
            align="start",
        ),
    )
