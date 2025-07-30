import json
from functools import partial

from datapipe.compute import Catalog
from datapipe.datatable import DataStore
from datapipe.store.database import DBConn

from vedana_etl.settings import settings

MEMGRAPH_CONN_ARGS = {
    "uri": settings.memgraph_uri,
    "auth": (settings.memgraph_user, settings.memgraph_pwd),
}

DBCONN_DATAPIPE = DBConn(
    connstr=settings.db_conn_uri, create_engine_kwargs=dict(json_serializer=partial(json.dumps, ensure_ascii=False))
)


ds = DataStore(DBCONN_DATAPIPE)
catalog = Catalog({})
