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

    # Common bubble content
    bubble_content = rx.vstack(
        rx.text(msg["content"]),
        rx.hstack(
            rx.cond(
                msg["has_tech"],
                rx.button(
                    "Details",
                    variant="ghost",
                    color_scheme="gray",
                    size="1",
                    on_click=ChatState.toggle_details_by_id(msg["id"]),
                ),
            ),
            rx.text(msg.get("created_at_fmt", msg["created_at"]), size="1", color="gray"),
            justify="between",
            width="100%",
        ),
        tech_block,
        spacing="2",
        width="100%",
    )

    assistant_bubble = rx.card(
        bubble_content,
        padding="0.75em",
        style={
            "maxWidth": "70%",
            "backgroundColor": "#11182714",
            "border": "1px solid #e5e7eb",
            "borderRadius": "12px",
        },
    )
    user_bubble = rx.card(
        bubble_content,
        padding="0.75em",
        style={
            "maxWidth": "70%",
            "backgroundColor": "#3b82f614",
            "border": "1px solid #e5e7eb",
            "borderRadius": "12px",
        },
    )

    return rx.cond(
        msg["is_assistant"],
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


def page() -> rx.Component:
    return rx.vstack(
        app_header(),
        breadcrumbs([("Main", "/"), ("Chatbot", "/chat")]),
        rx.flex(
            # Messages scroll region
            rx.box(
                rx.scroll_area(
                    rx.vstack(
                        rx.foreach(ChatState.messages, _message_row),
                        spacing="3",
                        width="100%",
                    ),
                    type="always",
                    scrollbars="vertical",
                    style={"height": "100%"},
                ),
                flex="1",
                min_height="0",
                width="100%",
            ),
            # Typing indicator
            rx.cond(
                ChatState.is_running,
                rx.hstack(
                    rx.spinner(),
                    rx.text("Generating response..."),
                    spacing="2",
                    padding_y="0.5em",
                    width="100%",
                ),
            ),
            # Sticky action + input bar
            rx.box(
                rx.vstack(
                    rx.hstack(
                        rx.button(
                            "Clear history",
                            variant="ghost",
                            color_scheme="gray",
                            size="1",
                            on_click=ChatState.reset_session,
                        ),
                        rx.dialog.root(
                            rx.dialog.trigger(
                                rx.button(
                                    "Data model",
                                    variant="ghost",
                                    color_scheme="gray",
                                    size="1",
                                    on_click=ChatState.load_data_model_text,
                                ),
                            ),
                            rx.dialog.content(
                                rx.vstack(
                                    rx.hstack(
                                        rx.dialog.title("Current Data Model"),
                                        rx.spacer(),
                                        rx.dialog.close(
                                            rx.button("Close", variant="ghost", color_scheme="gray", size="1"),
                                        ),
                                        align="center",
                                        width="100%",
                                    ),
                                    rx.scroll_area(
                                        rx.markdown(ChatState.data_model_text),
                                        type="always",
                                        scrollbars="vertical",
                                        style={"height": "50vh"},
                                    ),
                                    rx.hstack(
                                        rx.box(
                                            rx.text(
                                                rx.cond(
                                                    ChatState.data_model_last_sync != "",
                                                    "Last sync: " + ChatState.data_model_last_sync,
                                                    "Last sync: —",
                                                ),
                                                size="1",
                                                color="gray",
                                            ),
                                            width="100%",
                                        ),
                                        rx.button(
                                            "Reload from Grist",
                                            variant="classic",
                                            color_scheme="gray",
                                            size="2",
                                            loading=ChatState.is_refreshing_dm,
                                            on_click=ChatState.refresh_data_model,
                                        ),
                                        justify="end",
                                        align="center",
                                        width="100%",
                                    ),
                                    spacing="3",
                                    width="100%",
                                ),
                                max_width="900px",
                            ),
                        ),
                        align="end",
                        width="100%",
                    ),
                    rx.form.root(
                        rx.hstack(
                            rx.input(
                                placeholder="Type your message…",
                                value=ChatState.input_text,
                                on_change=ChatState.set_input,
                                width="100%",
                            ),
                            rx.button("Send", type="submit", loading=ChatState.is_running),
                            spacing="2",
                            width="100%",
                        ),
                        on_submit=ChatState.send,
                        width="100%",
                    ),
                    spacing="2",
                    width="100%",
                ),
                position="sticky",
                bottom="0",
                padding_top="0.5em",
                padding_bottom="0.5em",
                style={"backgroundColor": "inherit"},
                width="100%",
            ),
            direction="column",
            gap="0.75em",
            flex="1",
            min_height="0",
            width="100%",
        ),
        align="start",
        spacing="2",
        padding="1em",
        height="100vh",
        overflow="hidden",
        on_mount=ChatState.mount,
        width="100%",
    )
