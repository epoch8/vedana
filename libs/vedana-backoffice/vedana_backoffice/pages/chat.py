import reflex as rx

from vedana_backoffice.components.ui_chat import render_message_bubble
from vedana_backoffice.state import ChatState
from vedana_backoffice.ui import app_header


def _message_row(msg: dict) -> rx.Component:
    return render_message_bubble(
        msg,
        on_toggle_details=ChatState.toggle_details_by_id(message_id=msg["id"]),  # type: ignore[call-arg,func-returns-value]
    )


def page() -> rx.Component:
    return rx.vstack(
        app_header(),
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
                            disabled=rx.cond(
                                ChatState.chat_thread_id == "",
                                True,  # no thread_id --> nothing to reset
                                False,  # thread_id present --> can be reset
                            ),
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
                                        rx.dialog.title("Data Model"),
                                        rx.spacer(),
                                        rx.button(
                                            "Reload",
                                            variant="soft",
                                            color_scheme="blue",
                                            size="1",
                                            on_click=ChatState.reload_data_model,
                                            loading=ChatState.is_refreshing_dm,
                                        ),
                                        rx.dialog.close(
                                            rx.button("Close", variant="ghost", color_scheme="gray", size="1"),
                                        ),
                                        align="center",
                                        width="100%",
                                    ),
                                    rx.scroll_area(
                                        rx.markdown(ChatState.data_model_text),  # type: ignore[operator]
                                        type="always",
                                        scrollbars="vertical",
                                        style={"height": "70vh"},
                                    ),
                                    spacing="3",
                                    width="100%",
                                ),
                                max_width="70vw",
                            ),
                        ),
                        rx.spacer(),
                        rx.hstack(
                            rx.cond(
                                ChatState.chat_thread_id != "",
                                rx.hstack(
                                    rx.text("thread id: ", size="1", color="gray"),
                                    rx.button(
                                        ChatState.chat_thread_id,
                                        variant="soft",
                                        size="1",
                                        on_click=ChatState.open_jims_thread,
                                    ),
                                    spacing="1",
                                ),
                            ),
                            rx.text(f"model: {ChatState.model}", size="1", color="gray"),
                            rx.cond(
                                ChatState.total_conversation_cost > 0,
                                rx.text(
                                    "total cost: " + ChatState.total_conversation_cost_str,
                                    size="1",
                                    color="gray",
                                ),
                            ),
                            spacing="2",
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
