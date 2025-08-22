import reflex as rx


def page() -> rx.Component:
    return rx.vstack(
        rx.heading("Chatbot"),
        rx.text("Manual chat with tool-call introspection (coming soon)"),
        align="start",
        spacing="3",
        padding="1em",
    )
