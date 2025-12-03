import reflex as rx

from vedana_backoffice.pages.chat import page as chat_page
from vedana_backoffice.pages.etl import page as etl_page
from vedana_backoffice.pages.jims_thread_list_page import jims_thread_list_page
from vedana_backoffice.pages.main_dashboard import page as main_dashboard_page
from vedana_backoffice.states.chat import ChatState
from vedana_backoffice.states.common import EVAL_ENABLED
from vedana_backoffice.states.etl import EtlState
from vedana_backoffice.states.main_dashboard import DashboardState

app = rx.App(stylesheets=["/styles.css"])
app.add_page(main_dashboard_page, route="/", title="Vedana Backoffice", on_load=DashboardState.load_dashboard)
app.add_page(etl_page, route="/etl", title="ETL", on_load=EtlState.load_pipeline_metadata)
app.add_page(chat_page, route="/chat", title="Chat", on_load=ChatState.reset_session)
app.add_page(jims_thread_list_page, route="/jims", title="JIMS")

# Eval page is only available when GRIST_TEST_SET_DOC_ID is configured
if EVAL_ENABLED:
    from vedana_backoffice.pages.eval import page as eval_page
    from vedana_backoffice.states.eval import EvalState

    app.add_page(eval_page, route="/eval", title="Evaluation", on_load=EvalState.load_eval_data)
