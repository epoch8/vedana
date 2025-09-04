from datapipe.compute import Table
from datapipe.store.database import TableStoreDB
from datapipe.store.neo4j import Neo4JStore
from sqlalchemy import Boolean, Column, Float, String

import vedana_etl.schemas as schemas
from vedana_etl.config import DBCONN_DATAPIPE, MEMGRAPH_CONN_ARGS

dm_links = Table(
    name="dm_links",
    store=TableStoreDB(
        dbconn=DBCONN_DATAPIPE,
        name="dm_links",
        data_sql_schema=[
            Column("anchor1", String, primary_key=True),
            Column("anchor2", String, primary_key=True),
            Column("sentence", String, primary_key=True),
            Column("description", String),
            Column("query", String),
            Column("anchor1_link_column_name", String),
            Column("anchor2_link_column_name", String),
            Column("has_direction", Boolean, default=False),
        ],
    ),
)

# ---
# Evaluation pipeline

eval_tables = {
    "dm_version": (
        Table(
            store=TableStoreDB(
                dbconn=DBCONN_DATAPIPE,
                name="dm_version",
                data_sql_schema=schemas.DM_VERSIONING_TABLE_SCHEMA,
            )
        )
    ),
    "eval_gds": (
        Table(
            store=TableStoreDB(
                dbconn=DBCONN_DATAPIPE,
                name="eval_gds",
                data_sql_schema=schemas.EVAL_GDS_SCHEMA,
            )
        )
    ),
}


dm_attributes = Table(
    name="dm_attributes_v2",
    store=TableStoreDB(
        dbconn=DBCONN_DATAPIPE,
        name="dm_attributes_v2",
        data_sql_schema=[
            Column("attribute_name", String, primary_key=True),
            Column("anchor", String, primary_key=True),
            Column("description", String),
            Column("link", String),
            Column("data_example", String),
            Column("embeddable", Boolean),
            Column("query", String),
            Column("dtype", String),
            Column("embed_threshold", Float),
        ],
    ),
)


dm_anchors = Table(
    name="dm_anchors",
    store=TableStoreDB(
        dbconn=DBCONN_DATAPIPE,
        name="dm_anchors",
        data_sql_schema=[
            Column("noun", String, primary_key=True),
            Column("description", String),
            Column("id_example", String),
            Column("query", String),
        ],
    ),
)


grist_nodes = Table(
    name="grist_nodes",
    store=TableStoreDB(
        dbconn=DBCONN_DATAPIPE,
        name="grist_nodes",
        data_sql_schema=schemas.GENERIC_NODE_DATA_SCHEMA,
    ),
)

grist_edges = Table(
    name="grist_edges",
    store=TableStoreDB(
        dbconn=DBCONN_DATAPIPE,
        name="grist_edges",
        data_sql_schema=schemas.GENERIC_EDGE_DATA_SCHEMA,
    ),
)

grist_nodes_filtered = Table(
    name="grist_nodes_filtered",
    store=TableStoreDB(
        dbconn=DBCONN_DATAPIPE,
        name="grist_nodes_filtered",
        data_sql_schema=schemas.GENERIC_NODE_DATA_SCHEMA,
    ),
)

grist_edges_filtered = Table(
    name="grist_edges_filtered",
    store=TableStoreDB(
        dbconn=DBCONN_DATAPIPE,
        name="grist_edges_filtered",
        data_sql_schema=schemas.GENERIC_EDGE_DATA_SCHEMA,
    ),
)

nodes = Table(
    name="nodes",
    store=TableStoreDB(
        dbconn=DBCONN_DATAPIPE,
        name="nodes",
        data_sql_schema=schemas.GENERIC_NODE_DATA_SCHEMA,
    ),
)

edges = Table(
    name="edges",
    store=TableStoreDB(
        dbconn=DBCONN_DATAPIPE,
        name="edges",
        data_sql_schema=schemas.GENERIC_EDGE_DATA_SCHEMA,
    ),
)

memgraph_indexes = Table(
    name="memgraph_indexes",
    store=TableStoreDB(
        dbconn=DBCONN_DATAPIPE,
        name="memgraph_indexes",
        data_sql_schema=[
            Column("attribute_name", String, primary_key=True),
        ],
    ),
)

memgraph_vector_indexes = Table(
    name="memgraph_vector_indexes",
    store=TableStoreDB(
        dbconn=DBCONN_DATAPIPE,
        name="memgraph_vector_indexes",
        data_sql_schema=[
            Column("attribute_name", String, primary_key=True),
            Column("anchor", String),
            Column("link", String),
        ],
    ),
)

memgraph_nodes = Table(
    name="memgraph_nodes",
    store=Neo4JStore(
        connection_kwargs=MEMGRAPH_CONN_ARGS,
        data_sql_schema=schemas.GENERIC_NODE_DATA_SCHEMA,
    ),
)

memgraph_edges = Table(
    name="memgraph_edges",
    store=Neo4JStore(
        connection_kwargs=MEMGRAPH_CONN_ARGS,
        data_sql_schema=schemas.GENERIC_EDGE_DATA_SCHEMA,
    ),
)
