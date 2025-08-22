import reflex as rx


def page() -> rx.Component:
    return rx.vstack(
        rx.heading("ETL Pipeline"),
        rx.text("Control Vedana ETL pipeline branches (coming soon)"),
        align="start",
        spacing="3",
        padding="1em",
    )


