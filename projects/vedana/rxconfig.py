import reflex

config = reflex.Config(  # type: ignore
    app_name="vedana_backoffice",
    plugins=[
        reflex.plugins.TailwindV3Plugin(),
        reflex.plugins.sitemap.SitemapPlugin(),
    ],
)
