import reflex as rx

from vedana_backoffice.state import ChatState
from vedana_backoffice.ui import app_header, breadcrumbs


def _message_row(msg: dict) -> rx.Component:
    tech_block = rx.cond(
        msg["has_tech"] & msg["show_details"],
        rx.card(
            rx.vstack(
                rx.cond(
                    msg.get("has_models"),
                    rx.vstack(rx.text("Models"), rx.code(msg.get("models_str", ""), font_size="11px")),
                ),
                rx.cond(
                    msg.get("has_vts"),
                    rx.vstack(rx.text("VTS Queries"), rx.code(msg.get("vts_str", ""), font_size="11px")),
                ),
                rx.cond(
                    msg.get("has_cypher"),
                    rx.vstack(rx.text("Cypher Queries"), rx.code(msg.get("cypher_str", ""), font_size="11px")),
                ),
                spacing="2",
            ),
            padding="0.75em",
        ),
        rx.box(),
    )

    bubble = rx.card(
        rx.vstack(
            rx.text(msg["content"]),
            rx.cond(
                msg["has_tech"],
                rx.button(
                    "Details",
                    variant="soft",
                    color_scheme="gray",
                    size="1",
                    on_click=ChatState.toggle_details_by_id(msg["id"]),
                ),
            ),
            tech_block,
            spacing="2",
        ),
        padding="0.75em",
        style={"maxWidth": "60%"},
    )

    return rx.cond(
        msg["is_assistant"],
        rx.hstack(rx.box(), bubble, width="100%"),
        rx.hstack(bubble, rx.box(), width="100%"),
    )


def page() -> rx.Component:
    return rx.vstack(
        app_header(),
        breadcrumbs([("Main", "/"), ("Chatbot", "/chat")]),
        rx.vstack(
            rx.scroll_area(
                rx.vstack(
                    rx.foreach(ChatState.messages, _message_row),
                    spacing="3",
                    width="100%",
                ),
                type="always",
                scrollbars="vertical",
                style={"height": "60vh"},
            ),
            rx.cond(
                ChatState.is_running,
                rx.hstack(
                    rx.spinner(),
                    rx.text("Generating response..."),
                    spacing="2",
                ),
            ),
            rx.form.root(
                rx.hstack(
                    rx.input(
                        placeholder="Type your message...",
                        value=ChatState.input_text,
                        on_change=ChatState.set_input,
                        width="100%",
                    ),
                    rx.button("Send", type="submit", loading=ChatState.is_running),
                    spacing="2",
                    width="100%",
                ),
                on_submit=ChatState.send,
            ),
            spacing="3",
            width="100%",
        ),
        align="start",
        spacing="3",
        padding="1em",
    )
