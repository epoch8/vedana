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

dm_anchor_attributes = Table(
    name="dm_anchor_attributes",
    store=TableStoreDB(
        dbconn=DBCONN_DATAPIPE,
        name="dm_anchor_attributes",
        data_sql_schema=[
            Column("anchor", String, primary_key=True),
            Column("attribute_name", String, primary_key=True),
            Column("description", String),
            Column("data_example", String),
            Column("embeddable", Boolean),
            Column("query", String),
            Column("dtype", String),
            Column("embed_threshold", Float),
        ],
    ),
)

dm_link_attributes = Table(
    name="dm_link_attributes",
    store=TableStoreDB(
        dbconn=DBCONN_DATAPIPE,
        name="dm_link_attributes",
        data_sql_schema=[
            Column("link", String, primary_key=True),
            Column("attribute_name", String, primary_key=True),
            Column("description", String),
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

dm_queries = Table(
    name="dm_queries",
    store=TableStoreDB(
        dbconn=DBCONN_DATAPIPE,
        name="dm_queries",
        data_sql_schema=[
            Column("query_name", String, primary_key=True),
            Column("query_example", String),
        ],
    ),
)

dm_prompts = Table(
    name="dm_prompts",
    store=TableStoreDB(
        dbconn=DBCONN_DATAPIPE,
        name="dm_prompts",
        data_sql_schema=[
            Column("name", String, primary_key=True),
            Column("text", String),
        ],
    ),
)

dm_conversation_lifecycle = Table(
    name="dm_conversation_lifecycle",
    store=TableStoreDB(
        dbconn=DBCONN_DATAPIPE,
        name="dm_conversation_lifecycle",
        data_sql_schema=[
            Column("event", String, primary_key=True),
            Column("text", String),
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

# --- Tables used as input for memgraph ---

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

# --- Memgraph-related tables ---

memgraph_anchor_indexes = Table(
    name="memgraph_anchor_indexes",
    store=TableStoreDB(
        dbconn=DBCONN_DATAPIPE,
        name="memgraph_anchor_indexes",
        data_sql_schema=[
            Column("anchor", String, primary_key=True),
        ],
    ),
)

memgraph_link_indexes = Table(
    name="memgraph_link_indexes",
    store=TableStoreDB(
        dbconn=DBCONN_DATAPIPE,
        name="memgraph_link_indexes",
        data_sql_schema=[
            Column("link", String, primary_key=True),
        ],
    ),
)

memgraph_anchor_vector_indexes = Table(
    name="memgraph_anchor_vector_indexes",
    store=TableStoreDB(
        dbconn=DBCONN_DATAPIPE,
        name="memgraph_anchor_vector_indexes",
        data_sql_schema=[
            Column("anchor", String, primary_key=True),
            Column("attribute_name", String, primary_key=True),
        ],
    ),
)

memgraph_link_vector_indexes = Table(
    name="memgraph_link_vector_indexes",
    store=TableStoreDB(
        dbconn=DBCONN_DATAPIPE,
        name="memgraph_link_vector_indexes",
        data_sql_schema=[
            Column("link", String, primary_key=True),
            Column("attribute_name", String, primary_key=True),
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
