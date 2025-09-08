import reflex as rx

from vedana_backoffice.ui import app_header, breadcrumbs


def page() -> rx.Component:
    return rx.vstack(
        app_header(),
        breadcrumbs([("Main", "/"), ("Evaluation", "/eval")]),
        rx.heading("Evaluation Tests"),
        rx.text("Launch question-answering pipeline tests (coming soon)"),
        align="start",
        spacing="3",
        padding="1em",
    )
