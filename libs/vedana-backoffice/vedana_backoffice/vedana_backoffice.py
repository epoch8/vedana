import reflex as rx

from vedana_backoffice.pages import etl
from vedana_backoffice.pages import eval as eval_page
from vedana_backoffice.pages.jims_thread_list_page import jims_thread_list_page
from vedana_backoffice.pages.jims_thread_page import jims_thread_page
from vedana_backoffice.state import EtlState
from vedana_backoffice.ui import app_header


def index() -> rx.Component:
    return rx.vstack(
        app_header(),
        rx.center(
            rx.vstack(
                rx.heading("Welcome", font_size="1.25em"),
                rx.text("Admin console for ETL, Evaluation, JIMS, and Chatbot"),
                rx.hstack(
                    rx.link("ETL", href="/etl"),
                    rx.link("Evaluation", href="/eval"),
                    rx.link("JIMS", href="/jims"),
                    spacing="4",
                ),
                align="start",
                spacing="4",
                width="100%",
            ),
            padding="2em",
            width="100%",
        ),
        width="100%",
    )


app = rx.App()
app.add_page(index, route="/", title="Vedana Backoffice")
app.add_page(etl.page, route="/etl", title="ETL", on_load=EtlState.load_pipeline_metadata)
app.add_page(eval_page.page, route="/eval", title="Evaluation")
app.add_page(jims_thread_list_page, route="/jims", title="JIMS")
app.add_page(jims_thread_page)
# app.add_page(chat.page, route="/chat", title="Chatbot")
