from __future__ import annotations

from typing import Callable

import reflex as rx

from jims_backoffice.states.chat import ChatState
from jims_backoffice.states.common import DebugState
from jims_backoffice.ui import app_header


def chat_page(
    state_cls: type[ChatState],
    *,
    input_bar_extras: list[rx.Component] | None = None,
    header: Callable[[], rx.Component] | None = None,
) -> rx.Component:
    """Generic chat page.

    Args:
        state_cls: Concrete subclass of :class:`ChatState` that the page wires to.
        input_bar_extras: Optional extra components rendered in the input-bar
            top row (e.g. data-model dialog button, filter checkbox).
        header: Optional header factory; defaults to :func:`app_header`.
    """
    extras = list(input_bar_extras or [])
    header_component = (header or app_header)()

    def _message_row(msg: dict) -> rx.Component:
        return rx.fragment()  # placeholder, will be replaced below

    # Local closure so rx.foreach has access to state_cls.
    def _render_message(msg: dict) -> rx.Component:
        from jims_backoffice.components.ui_chat import render_message_bubble

        return render_message_bubble(
            msg,
            on_toggle_details=state_cls.toggle_details_by_id(message_id=msg["id"]),  # type: ignore[call-arg,func-returns-value]
        )

    return rx.flex(
        header_component,
        rx.box(
            rx.scroll_area(
                rx.vstack(
                    rx.foreach(state_cls.messages, _render_message),
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
        rx.cond(
            state_cls.is_running,
            rx.hstack(
                rx.spinner(),
                rx.text("Generating response..."),
                spacing="2",
                padding_x="1em",
                padding_y="0.5em",
                width="100%",
            ),
        ),
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.button(
                        "Clear history",
                        variant="ghost",
                        color_scheme="gray",
                        size="1",
                        on_click=state_cls.reset_session,
                        disabled=rx.cond(
                            state_cls.chat_thread_id == "",
                            True,
                            False,
                        ),
                    ),
                    *extras,
                    rx.spacer(),
                    rx.hstack(
                        rx.cond(
                            state_cls.chat_thread_id != "",
                            rx.hstack(
                                rx.text("thread id: ", size="1", color="gray"),
                                rx.button(
                                    state_cls.chat_thread_id,
                                    variant="soft",
                                    size="1",
                                    on_click=state_cls.open_jims_thread,
                                ),
                                spacing="1",
                            ),
                        ),
                        rx.cond(
                            state_cls.total_conversation_cost > 0,
                            rx.text(
                                "total cost: " + state_cls.total_conversation_cost_str,
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
                            value=state_cls.input_text,
                            on_change=state_cls.set_input,
                            width="100%",
                        ),
                        rx.cond(
                            state_cls.model_selection_allowed,
                            rx.select(
                                items=DebugState.available_models,
                                value=state_cls.model,
                                on_change=state_cls.set_model,
                                width="20em",
                                placeholder="Select model",
                            ),
                            rx.badge(state_cls.model, variant="surface", color_scheme="gray", size="3"),
                        ),
                        rx.button("Send", type="submit", loading=state_cls.is_running),
                        spacing="2",
                        width="100%",
                    ),
                    on_submit=state_cls.send,
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
        on_mount=state_cls.mount,
    )
