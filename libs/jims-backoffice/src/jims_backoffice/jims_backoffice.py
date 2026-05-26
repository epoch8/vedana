"""Standalone demo Reflex app for jims-backoffice.

Run with ``JIMS_APP=module:attr`` environment variable pointing to a callable
or :class:`jims_core.app.JimsApp` instance. The module-attribute reference is
resolved by :mod:`jims_core.util.load_jims_app`.

Concrete state classes are declared here to materialise the
``jims_backoffice.ChatState`` / ``EvalState`` mixins as Reflex substates.
Other apps (e.g. ``vedana_backoffice``) provide their own concrete subclasses.
"""

from __future__ import annotations

import reflex as rx

from jims_backoffice import (
    ChatState,
    EvalState,
    ThreadListState,
    chat_page,
    eval_page,
    jims_thread_list_page,
    register_chat_state,
    register_eval_state,
)


class JimsChatState(ChatState, rx.State):
    """Concrete Chat state for the jims-backoffice demo."""


class JimsEvalState(EvalState, rx.State):
    """Concrete Eval state for the jims-backoffice demo."""


register_chat_state(JimsChatState)
register_eval_state(JimsEvalState)


def _chat_page() -> rx.Component:
    return chat_page(JimsChatState)


def _eval_page() -> rx.Component:
    return eval_page(JimsEvalState)


def _jims_page() -> rx.Component:
    return jims_thread_list_page()


app = rx.App(stylesheets=["/styles.css"])
app.add_page(_chat_page, route="/", title="JIMS Backoffice — Chat", on_load=JimsChatState.reset_session)
app.add_page(_chat_page, route="/chat", title="Chat", on_load=JimsChatState.reset_session)
app.add_page(_jims_page, route="/jims", title="JIMS", on_load=ThreadListState.get_data)
app.add_page(_eval_page, route="/eval", title="Evaluation", on_load=JimsEvalState.load_eval_data)
