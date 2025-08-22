import reflex as rx


def page() -> rx.Component:
    return rx.vstack(
        rx.heading("JIMS Sessions"),
        rx.text("Browse sessions and details (coming soon)"),
        align="start",
        spacing="3",
        padding="1em",
    )
