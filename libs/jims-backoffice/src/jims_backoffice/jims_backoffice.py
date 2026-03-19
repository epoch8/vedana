import reflex as rx

from jims_backoffice.pages.chat import page as chat_page
from jims_backoffice.pages.eval import page as eval_page
from jims_backoffice.pages.jims import ThreadListState, jims_thread_list_page
from jims_backoffice.states.chat import ChatState
from jims_backoffice.states.eval import EvalState

app = rx.App(stylesheets=["/styles.css"])
app.add_page(chat_page, route="/chat", title="Chat", on_load=ChatState.reset_session)
app.add_page(jims_thread_list_page, route="/jims", title="JIMS", on_load=ThreadListState.get_data)
app.add_page(eval_page, route="/eval", title="Evaluation", on_load=EvalState.mount)
