import reflex as rx


def page() -> rx.Component:
    return rx.vstack(
        rx.heading("Evaluation Tests"),
        rx.text("Launch question-answering pipeline tests (coming soon)"),
        align="start",
        spacing="3",
        padding="1em",
    )


