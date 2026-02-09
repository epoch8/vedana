import reflex

config = reflex.Config(  # type: ignore
    app_name="vedana_backoffice",
    show_built_with_reflex=False,
    plugins=[
        reflex.plugins.TailwindV3Plugin(),
        reflex.plugins.sitemap.SitemapPlugin(),
    ],
)
