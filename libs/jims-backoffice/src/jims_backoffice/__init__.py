"""JIMS backoffice library.

Provides reusable Reflex states and page factories for any JIMS app:

- :func:`chat_page` — generic chat UI parameterized by a :class:`ChatState` mixin.
- :func:`eval_page` — generic evaluation UI parameterized by an :class:`EvalState` mixin.
- :func:`jims_thread_list_page` — JIMS thread browser.
- :func:`app_header` — sticky top header with optional extra slots.
- :func:`register_app` — register a JIMS app reference (``"module:attr"``) the
  states should resolve when run.
- :func:`register_chat_state` / :func:`register_eval_state` — register the
  concrete state classes used at runtime so ``DebugState`` can dispatch refresh
  events to them.
"""

from jims_backoffice.app_loader import get_jims_app, register_app
from jims_backoffice.pages.chat import chat_page
from jims_backoffice.pages.eval import eval_page
from jims_backoffice.pages.jims_thread_list import jims_thread_list_page
from jims_backoffice.states.chat import ChatState
from jims_backoffice.states.common import (
    AppVersionState,
    DebugState,
    TelegramBotState,
    register_chat_state,
    register_eval_state,
)
from jims_backoffice.states.eval import EvalState
from jims_backoffice.states.jims import ThreadEventVis, ThreadListState, ThreadViewState, ThreadVis
from jims_backoffice.ui import (
    api_key_setup_dialog,
    app_header,
    breadcrumbs,
    debug_badge,
    telegram_link_box,
    themed_data_table,
)

__all__ = [
    "AppVersionState",
    "ChatState",
    "DebugState",
    "EvalState",
    "TelegramBotState",
    "ThreadEventVis",
    "ThreadListState",
    "ThreadVis",
    "ThreadViewState",
    "api_key_setup_dialog",
    "app_header",
    "breadcrumbs",
    "chat_page",
    "debug_badge",
    "eval_page",
    "get_jims_app",
    "jims_thread_list_page",
    "register_app",
    "register_chat_state",
    "register_eval_state",
    "telegram_link_box",
    "themed_data_table",
]
