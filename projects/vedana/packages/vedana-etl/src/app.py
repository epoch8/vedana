from datapipe_app import DatapipeAPI

from src.config import ds
from src.catalog import init_catalog
from src.pipeline import get_pipeline

# base app - no extra tables / steps
catalog = init_catalog()
pipeline = get_pipeline()

app = DatapipeAPI(ds, catalog, pipeline)
