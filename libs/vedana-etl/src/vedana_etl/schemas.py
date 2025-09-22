from sqlalchemy import Column, String
from sqlalchemy.dialects.postgresql import JSONB

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

# ---
# Evaluation pipeline schemas

DM_VERSIONING_TABLE_SCHEMA: list[Column] = [
    Column("dm_id", String, primary_key=True),
    Column("dm_description", String),
]

EVAL_GDS_SCHEMA: list[Column] = [
    Column("gds_question", String, primary_key=True),
    Column("gds_answer", String),
    Column("question_context", String),
]
