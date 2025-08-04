from sqlalchemy import Boolean, Column, Float, String
from sqlalchemy.dialects.postgresql import JSONB

# Grist data model

DM_ANCHORS_SCHEMA: list[Column] = [
    Column("noun", String, primary_key=True),
    Column("description", String),
    Column("id_example", String),
    Column("query", String),
]

DM_ATTRIBUTES_SCHEMA: list[Column] = [
    Column("attribute_name", String, primary_key=True),
    Column("description", String),
    Column("anchor", String),
    Column("link", String),
    Column("data_example", String),
    Column("embeddable", Boolean),
    Column("query", String),
    Column("dtype", String),
    Column("embed_threshold", Float),
]

DM_LINKS_SCHEMA: list[Column] = [
    Column("anchor1", String, primary_key=True),
    Column("anchor2", String, primary_key=True),
    Column("sentence", String, primary_key=True),
    Column("description", String),
    Column("query", String),
    Column("anchor1_link_column_name", String),
    Column("anchor2_link_column_name", String),
    Column("has_direction", Boolean, default=False),
]

# Memgraph input schema

GENERIC_NODE_DATA_SCHEMA: list[Column] = [
    Column("node_id", String, primary_key=True),
    Column("node_type", String, primary_key=True),
    Column("attributes", JSONB),
]

GENERIC_EDGE_DATA_SCHEMA: list[Column] = [
    Column("from_node_id", String, primary_key=True),
    Column("to_node_id", String, primary_key=True),
    Column("from_node_type", String, primary_key=True),
    Column("to_node_type", String, primary_key=True),
    Column("edge_label", String, primary_key=True),
    Column("attributes", JSONB),
]

# Memgraph helpers - indices

MEMGRAPH_INDEXES_SCHEMA: list[Column] = [
    Column("attribute_name", String, primary_key=True),
]

MEMGRAPH_VECTOR_INDEXES_SCHEMA: list[Column] = [
    Column("attribute_name", String, primary_key=True),
    Column("anchor", String),
    Column("link", String),
]
