import reflex as rx

from vedana_backoffice.pages import chat, etl, eval as eval_page, jims
from vedana_backoffice.state import EtlState


def index() -> rx.Component:
    return rx.center(
        rx.vstack(
            rx.heading("Vedana Backoffice", font_size="1.5em"),
            rx.text("Admin console for ETL, Evaluation, JIMS, and Chatbot"),
            rx.hstack(
                rx.link("ETL", href="/etl"),
                rx.link("Evaluation", href="/eval"),
                rx.link("JIMS", href="/jims"),
                rx.link("Chatbot", href="/chat"),
                spacing="4",
            ),
            align="start",
            spacing="4",
            width="100%",
        ),
        padding="2em",
        width="100%",
    )


app = rx.App()
app.add_page(index, route="/", title="Vedana Backoffice")
app.add_page(etl.page, route="/etl", title="ETL", on_load=EtlState.load_pipeline_metadata)
app.add_page(eval_page.page, route="/eval", title="Evaluation")
app.add_page(jims.page, route="/jims", title="JIMS")
app.add_page(chat.page, route="/chat", title="Chatbot")
