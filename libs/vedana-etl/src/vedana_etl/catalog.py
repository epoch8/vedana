from datapipe.compute import Catalog, Table
from datapipe.store.database import TableStoreDB
from datapipe.store.neo4j import Neo4JStore

import vedana_etl.schemas as schemas
from vedana_etl.config import DBCONN_DATAPIPE, MEMGRAPH_CONN_ARGS

data_model_tables = {
    "dm_links": (
        Table(
            store=TableStoreDB(
                dbconn=DBCONN_DATAPIPE,
                name="dm_links",
                data_sql_schema=schemas.DM_LINKS_SCHEMA,
            )
        )
    ),
    "dm_attributes": (
        Table(
            store=TableStoreDB(
                dbconn=DBCONN_DATAPIPE,
                name="dm_attributes",
                data_sql_schema=schemas.DM_ATTRIBUTES_SCHEMA,
            )
        )
    ),
    "dm_anchors": (
        Table(
            store=TableStoreDB(
                dbconn=DBCONN_DATAPIPE,
                name="dm_anchors",
                data_sql_schema=schemas.DM_ANCHORS_SCHEMA,
            )
        )
    ),
}

grist_tables = {
    "grist_nodes": (
        Table(
            store=TableStoreDB(
                dbconn=DBCONN_DATAPIPE,
                name="grist_nodes",
                data_sql_schema=schemas.GENERIC_NODE_DATA_SCHEMA,
            )
        )
    ),
    "grist_edges": (
        Table(
            store=TableStoreDB(
                dbconn=DBCONN_DATAPIPE,
                name="grist_edges",
                data_sql_schema=schemas.GENERIC_EDGE_DATA_SCHEMA,
            )
        )
    ),
    "grist_nodes_filtered": (
        Table(
            store=TableStoreDB(
                dbconn=DBCONN_DATAPIPE,
                name="grist_nodes_filtered",
                data_sql_schema=schemas.GENERIC_NODE_DATA_SCHEMA,
            )
        )
    ),
    "grist_edges_filtered": (
        Table(
            store=TableStoreDB(
                dbconn=DBCONN_DATAPIPE,
                name="grist_edges_filtered",
                data_sql_schema=schemas.GENERIC_EDGE_DATA_SCHEMA,
            )
        )
    ),
}

# ---
# This part is customisable (can be replaced with a connection of other branches

default_custom_tables: dict[str, Table] = {}

# ---

memgraph_tables = {
    "nodes": (
        Table(
            store=TableStoreDB(
                dbconn=DBCONN_DATAPIPE,
                name="nodes",
                data_sql_schema=schemas.GENERIC_NODE_DATA_SCHEMA,
            )
        )
    ),
    "edges": (
        Table(
            store=TableStoreDB(
                dbconn=DBCONN_DATAPIPE,
                name="edges",
                data_sql_schema=schemas.GENERIC_EDGE_DATA_SCHEMA,
            )
        )
    ),
    "memgraph_indexes": (
        Table(
            store=TableStoreDB(
                dbconn=DBCONN_DATAPIPE,
                name="memgraph_indexes",
                data_sql_schema=schemas.MEMGRAPH_INDEXES_SCHEMA,
            )
        )
    ),
    "memgraph_vector_indexes": (
        Table(
            store=TableStoreDB(
                dbconn=DBCONN_DATAPIPE,
                name="memgraph_vector_indexes",
                data_sql_schema=schemas.MEMGRAPH_VECTOR_INDEXES_SCHEMA,
            )
        )
    ),
    "memgraph_nodes": (
        Table(
            store=Neo4JStore(
                connection_kwargs=MEMGRAPH_CONN_ARGS,
                data_sql_schema=schemas.GENERIC_NODE_DATA_SCHEMA,
            )
        )
    ),
    "memgraph_edges": (
        Table(
            store=Neo4JStore(
                connection_kwargs=MEMGRAPH_CONN_ARGS,
                data_sql_schema=schemas.GENERIC_EDGE_DATA_SCHEMA,
            )
        )
    ),
}

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


def compile_catalog(catalog_extra_tables: dict):
    catalog_dict = {
        **data_model_tables,
        **grist_tables,
        **catalog_extra_tables,
        **memgraph_tables,
    }
    return catalog_dict


def init_catalog(catalog: Catalog, catalog_tables: dict):
    for subcatalog in [catalog_tables]:
        for table_name, table_store in subcatalog.items():
            catalog.add_datatable(table_name, table_store)
    return catalog
