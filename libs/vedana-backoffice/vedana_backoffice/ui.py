import reflex as rx


def app_header() -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.link("Vedana Backoffice", href="/", font_weight="bold", font_size="1.25em"),
            justify="start",
            width="100%",
        ),
        width="100%",
        padding="0.75em 1em",
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
