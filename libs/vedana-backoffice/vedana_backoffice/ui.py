import reflex as rx
import os

VERSION = str(f"`{os.environ.get('VERSION', 'unspecified_version')}`")


def app_header() -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.hstack(
                rx.link("Vedana Backoffice", href="/", font_weight="bold", font_size="1.25em"),
                rx.markdown(VERSION),
                align="center",
                spacing="3",
            ),
            rx.hstack(
                rx.link("ETL", href="/etl", font_size="1.1em"),
                rx.link("Chat", href="/chat", font_size="1.1em"),
                rx.link("JIMS", href="/jims", font_size="1.1em"),
                rx.color_mode.button(),
                spacing="6",
                align="center",
            ),
            justify="between",
            align="center",
            width="100%",
        ),
        width="100%",
        padding="""0.75em 1.25em""",
        border_bottom="1px solid #e5e7eb",
        position="sticky",
        top="0",
        style={"backgroundColor": "inherit"},
        z_index="10",
    )


def breadcrumbs(items: list[tuple[str, str]]) -> rx.Component:
    parts: list[rx.Component] = []
    for idx, (label, href) in enumerate(items):
        parts.append(rx.link(label, href=href))
        if idx < len(items) - 1:
            parts.append(rx.text("→", color="#9ca3af"))
    return rx.hstack(*parts, spacing="2", padding_y="0.5em")
