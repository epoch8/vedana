import reflex as rx

config = rx.Config(  # type: ignore
    app_name="vedana_admin",
    plugins=[
        rx.plugins.sitemap.SitemapPlugin(),
    ],
)
