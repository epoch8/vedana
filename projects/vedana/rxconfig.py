import reflex as rx

config = rx.Config(  # type: ignore
    app_name="vedana_backoffice",
    plugins=[
        rx.plugins.sitemap.SitemapPlugin(),
    ],
)
