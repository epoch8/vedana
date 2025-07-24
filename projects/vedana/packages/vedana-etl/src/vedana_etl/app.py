from datapipe_app import DatapipeAPI

from vedana_etl.config import ds, catalog
from vedana_etl.catalog import compile_catalog, init_catalog, default_custom_tables
from vedana_etl.pipeline import get_pipeline, default_custom_steps


# base app - no extra tables / steps
catalog = init_catalog(catalog, compile_catalog(default_custom_tables))
pipeline = get_pipeline(custom_steps=default_custom_steps)

app = DatapipeAPI(ds, catalog, pipeline)
