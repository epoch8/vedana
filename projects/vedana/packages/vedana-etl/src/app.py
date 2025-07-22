from datapipe_app import DatapipeAPI

from src.config import ds
from src.catalog import init_catalog, default_custom_tables
from src.pipeline import get_pipeline, default_custom_steps


# base app - no extra tables / steps
catalog = init_catalog(catalog_extra_tables=default_custom_tables)
pipeline = get_pipeline(custom_steps=default_custom_steps)

app = DatapipeAPI(ds, catalog, pipeline)
