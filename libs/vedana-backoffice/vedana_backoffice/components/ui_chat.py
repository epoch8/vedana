import reflex as rx


def render_message_bubble(
    msg: dict,
    on_toggle_details,
    extras: rx.Component | None = None,
    corner_tags_component: rx.Component | None = None,
) -> rx.Component:  # type: ignore[valid-type]
    """Render a chat-style message bubble.

    Expects msg dict with keys:
      - content, is_assistant (bool-like), created_at_fmt or created_at
      - has_tech, show_details
      - has_models, has_vts, has_cypher, models_str, vts_str, cypher_str (optional)
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
                            style={
                                "whiteSpace": "pre-wrap",
                                "wordBreak": "break-all",
                                "overflowX": "auto",
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
                            style={
                                "whiteSpace": "pre-wrap",
                                "wordBreak": "break-all",
                                "overflowX": "auto",
                                "display": "block",
                                "maxWidth": "100%",
                                "boxSizing": "border-box",
                            },
                        ),
                        style={
                            "whiteSpace": "pre-wrap",
                            "wordBreak": "break-all",
                        },
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
                            style={
                                "whiteSpace": "pre-wrap",
                                "wordBreak": "break-all",
                                "overflowX": "auto",
                                "display": "block",
                                "maxWidth": "100%",
                                "boxSizing": "border-box",
                            },
                        ),
                        spacing="1",
                        width="100%",
                    ),
                ),
                spacing="2",
                width="100%",
            ),
            padding="0.75em",
            variant="surface",
            style={
                "maxWidth": "100%",
                "overflowX": "auto",
            },
        ),
        rx.box(),
    )

    generic_details_block = rx.cond(
        msg.get("generic_meta"),
        rx.card(
            rx.vstack(
                rx.code_block(
                    msg.get("event_data_str", ""),
                    font_size="11px",
                    language="json",
                    style={
                        "whiteSpace": "pre-wrap",
                        "word-break": "break-all",
                        "overflowX": "auto",
                        "display": "block",
                        "maxWidth": "100%",
                        "boxSizing": "border-box",
                    },
                ),
                spacing="1",
                width="100%",
            ),
            padding="0.75em",
            variant="surface",
            width="100%",
            style={
                "maxWidth": "100%",
                "overflowX": "auto",
            },
        ),
        rx.box(),
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
            msg.get("has_tech") | msg.get("generic_meta"),  # type: ignore[operator]
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
        rx.cond(msg.get("show_details"), tech_block),
        rx.cond(
            msg.get("show_details") & msg.get("generic_meta"),  # type: ignore[operator]
            generic_details_block,
        ),
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
            "maxWidth": "35vw",  # 70% of 50% parent card width in vw terms
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
            "maxWidth": "35vw",  # 70% of 50% parent card width in vw terms
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
