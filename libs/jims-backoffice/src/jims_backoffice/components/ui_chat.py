import reflex as rx


def render_message_bubble(
    msg: dict,
    on_toggle_details,
    extras: rx.Component | None = None,
    corner_tags_component: rx.Component | None = None,
) -> rx.Component:  # type: ignore[valid-type]
    """Render a chat-style message bubble.

    Expects msg dict with keys:
      - content, is_assistant (bool-like), created_at
      - show_details
      - has_logs, logs_str (optional)
      - generic_meta, event_data_str (optional)
      - tag_label (optional badge)
    """

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
                        code_tag_props={"style": {"whiteSpace": "pre-wrap"}},
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
        generic_details_block,
        logs_block,
        spacing="2",
        width="100%",
    )

    tags_box = corner_tags_component or rx.box()

    header_left = rx.hstack(
        rx.cond(
            msg.get("tag_label", "") != "",
            rx.badge(msg.get("tag_label", ""), variant="soft", size="1", color_scheme="gray"),
            rx.box(),
        ),
        rx.cond(
            msg.get("has_logs") | msg.get("generic_meta"),  # type: ignore[operator]
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
        rx.hstack(header_left, tags_box, justify="between", width="100%", align="center"),
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
            "maxWidth": "70%",
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
