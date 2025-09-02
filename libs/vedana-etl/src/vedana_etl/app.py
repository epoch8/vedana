from datapipe.compute import Catalog
from datapipe_app import DatapipeAPI

from vedana_etl.config import ds
from vedana_etl.pipeline import default_custom_steps, get_pipeline

# base app - no extra tables / steps
pipeline = get_pipeline(custom_steps=default_custom_steps)

app = DatapipeAPI(ds, Catalog({}), pipeline)
