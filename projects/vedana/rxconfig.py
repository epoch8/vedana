import reflex as rx

config = rx.Config(  # type: ignore
    api_url="",
    app_name="vedana_backoffice",
    plugins=[
        rx.plugins.TailwindV3Plugin(),
    ],
)
