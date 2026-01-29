import reflex as rx

from vedana_backoffice.components.ui_chat import render_message_bubble
from vedana_backoffice.states.chat import ChatState
from vedana_backoffice.ui import app_header
from vedana_core.settings import settings as core_settings


def _message_row(msg: dict) -> rx.Component:
    return render_message_bubble(
        msg,
        on_toggle_details=ChatState.toggle_details_by_id(message_id=msg["id"]),  # type: ignore[call-arg,func-returns-value]
    )


def page() -> rx.Component:
    return rx.flex(
        app_header(),
        rx.box(
            rx.scroll_area(
                rx.vstack(
                    rx.foreach(ChatState.messages, _message_row),
                    spacing="3",
                    width="100%",
                    padding="1em",
                ),
                type="always",
                scrollbars="vertical",
                style={"height": "100%"},
            ),
            flex="1",
            min_height="0",
            width="100%",
            overflow="hidden",
        ),
        # Typing indicator (above input bar)
        rx.cond(
            ChatState.is_running,
            rx.hstack(
                rx.spinner(),
                rx.text("Generating response..."),
                spacing="2",
                padding_x="1em",
                padding_y="0.5em",
                width="100%",
            ),
        ),
        # Fixed input bar at bottom
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
                                rx.hstack(
                                    rx.text(f"Branch: {ChatState.data_model_branch}", color="gray"),
                                    rx.spacer(),
                                    rx.text(
                                        rx.cond(
                                            ChatState.data_model_snapshot_id != "",
                                            f"Snapshot: {ChatState.data_model_snapshot_id}",
                                            "Snapshot: —",
                                        ),
                                        color="gray",
                                    ),
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
                    rx.select(
                        items=[
                            core_settings.config_plane_dev_branch,
                            core_settings.config_plane_prod_branch,
                        ],
                        value=ChatState.data_model_branch,
                        on_change=ChatState.set_data_model_branch,
                        width="10em",
                    ),
                    rx.input(
                        placeholder="Snapshot id (optional)",
                        value=ChatState.data_model_snapshot_input,
                        on_change=ChatState.set_data_model_snapshot_input,
                        width="14em",
                    ),
                    rx.checkbox(
                        "Filter Data Model",
                        checked=ChatState.enable_dm_filtering,
                        on_change=ChatState.set_enable_dm_filtering,
                        size="1",
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
                        rx.select(
                            items=["openai", "openrouter"],
                            value=ChatState.provider,
                            on_change=ChatState.set_provider,
                            width="10em",
                            placeholder="Provider",
                        ),
                        rx.cond(
                            ChatState.provider == "openrouter",
                            rx.input(
                                placeholder=rx.cond(
                                    ChatState.default_openrouter_key_present,
                                    "(Optional) custom OPENROUTER_API_KEY",
                                    "(Required) OPENROUTER_API_KEY",
                                ),
                                type="password",
                                value=ChatState.custom_openrouter_key,
                                on_change=ChatState.set_custom_openrouter_key,
                                width="36em",
                                required=rx.cond(ChatState.default_openrouter_key_present, False, True),
                            ),
                        ),
                        rx.select(
                            items=ChatState.available_models,
                            value=ChatState.model,
                            on_change=ChatState.set_model,
                            width="16em",
                            placeholder="Select model",
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
            padding="1em",
            style={
                "backgroundColor": "var(--color-background)",
                "borderTop": "1px solid var(--gray-a5)",
                "flexShrink": "0",
                "position": "relative",
                "zIndex": "1000",
            },
            width="100%",
        ),
        direction="column",
        height="100vh",
        width="100%",
        overflow="hidden",
        on_mount=ChatState.mount,
    )
