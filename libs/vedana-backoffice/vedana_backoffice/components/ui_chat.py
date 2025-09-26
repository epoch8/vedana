import reflex as rx


def render_message_bubble(
    msg: dict,
    on_toggle_details,
    extras: rx.Component | None = None,
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
                        rx.code(msg.get("models_str", ""), font_size="11px"),
                        spacing="1",
                        width="100%",
                    ),
                ),
                rx.cond(
                    msg.get("has_vts"),
                    rx.vstack(
                        rx.text("VTS Queries", weight="medium"),
                        rx.code(msg.get("vts_str", ""), font_size="11px"),
                        spacing="1",
                        width="100%",
                    ),
                ),
                rx.cond(
                    msg.get("has_cypher"),
                    rx.vstack(
                        rx.text("Cypher Queries", weight="medium"),
                        rx.code(msg.get("cypher_str", ""), font_size="11px"),
                        spacing="1",
                        width="100%",
                    ),
                ),
                spacing="2",
                width="100%",
            ),
            padding="0.75em",
            variant="surface",
        ),
        rx.box(),
    )

    header = rx.hstack(
        rx.cond(
            msg.get("has_tech"),
            rx.button(
                "Details",
                variant="ghost",
                color_scheme="gray",
                size="1",
                on_click=on_toggle_details,  # type: ignore[arg-type]
            ),
        ),
        rx.text(msg.get("created_at_fmt", msg.get("created_at", "")), size="1", color="gray"),
        justify="between",
        width="100%",
    )

    body_children = []
    # Optional tag label; avoid Python truthiness on Vars
    tag_label = msg.get("tag_label", "")
    body_children.append(
        rx.cond(
            tag_label != "",
            rx.badge(tag_label, variant="soft", size="1", color_scheme="gray"),
            rx.box(),
        )
    )
    body_children.extend(
        [
            rx.text(msg.get("content", "")),
            header,
            rx.cond(msg.get("show_details"), tech_block),
            rx.cond(extras is not None, extras or rx.box()),
        ]
    )

    body = rx.vstack(
        *body_children,
        spacing="2",
        width="100%",
    )

    assistant_bubble = rx.card(
        body,
        padding="0.75em",
        style={
            "maxWidth": "70%",
            "backgroundColor": "#11182714",
            "border": "1px solid #e5e7eb",
            "borderRadius": "12px",
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
