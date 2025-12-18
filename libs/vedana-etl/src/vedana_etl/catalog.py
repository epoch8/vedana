from datapipe.compute import Table
from datapipe.store.database import TableStoreDB
from datapipe.store.neo4j import Neo4JStore
from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, Column, Float, Integer, String
from vedana_core.settings import settings as core_settings

import vedana_etl.schemas as schemas
from vedana_etl.config import DBCONN_DATAPIPE, MEMGRAPH_CONN_ARGS
from vedana_etl.settings import settings
from vedana_etl.store import GristStore

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

# --- Node/Edge tables with generated embeddings ---

embedded_nodes = Table(
    name="embedded_nodes",
    store=TableStoreDB(
        dbconn=DBCONN_DATAPIPE,
        name="embedded_nodes",
        data_sql_schema=schemas.GENERIC_NODE_DATA_SCHEMA,
    ),
)

embedded_edges = Table(
    name="embedded_edges",
    store=TableStoreDB(
        dbconn=DBCONN_DATAPIPE,
        name="embedded_edges",
        data_sql_schema=schemas.GENERIC_EDGE_DATA_SCHEMA,
    ),
)

# --- VTS (pgvector) ---
# embedding size column is fixed for indexing and is defined through settings. Definition is then fixed in migrations

rag_anchor_embeddings = Table(
    name="rag_anchor_embeddings",
    store=TableStoreDB(
        dbconn=DBCONN_DATAPIPE,
        name="rag_anchor_embeddings",
        data_sql_schema=[
            Column("node_id", String, primary_key=True),
            Column("attribute_name", String, primary_key=True),
            Column("label", String, nullable=False),
            Column("attribute_value", String),
            Column("embedding", Vector(dim=core_settings.embeddings_dim), nullable=False),
            Column("embedding_model", String, nullable=False),
        ],
    ),
)

rag_edge_embeddings = Table(
    name="rag_edge_embeddings",
    store=TableStoreDB(
        dbconn=DBCONN_DATAPIPE,
        name="rag_edge_embeddings",
        data_sql_schema=[
            Column("from_node_id", String, primary_key=True),
            Column("to_node_id", String, primary_key=True),
            Column("edge_label", String, primary_key=True),
            Column("attribute_name", String, primary_key=True),
            Column("attribute_value", String),
            Column("embedding", Vector(dim=core_settings.embeddings_dim), nullable=False),
            Column("embedding_model", String, nullable=False),
        ],
    ),
)

# --- Eval pipeline ---

dm_version = Table(
    name="dm_version",
    store=GristStore(
        server=core_settings.grist_server_url,
        api_key=core_settings.grist_api_key,
        doc_id=settings.grist_test_set_doc_id,
        table="Dm_version",
        data_sql_schema=[
            Column("dm_id", String, primary_key=True),
            Column("dm_description", String),
        ],
    ),
)

llm_embeddings_config = Table(
    name="llm_embeddings_config",
    store=TableStoreDB(
        dbconn=DBCONN_DATAPIPE,
        name="llm_embeddings_config",
        data_sql_schema=[
            Column("embeddings_model", String, primary_key=True),
            Column("embeddings_dim", Integer, primary_key=True),
        ],
    ),
)

llm_pipeline_config = Table(
    name="llm_pipeline_config",
    store=TableStoreDB(
        dbconn=DBCONN_DATAPIPE,
        name="llm_pipeline_config",
        data_sql_schema=[
            Column("pipeline_model", String, primary_key=True),
        ],
    ),
)

eval_gds = Table(
    name="eval_gds",
    store=TableStoreDB(
        dbconn=DBCONN_DATAPIPE,
        name="eval_gds",
        data_sql_schema=[
            Column("gds_question", String, primary_key=True),
            Column("gds_answer", String),
            Column("question_context", String),
        ],
    ),
)

judge_config = Table(
    name="judge_config",
    store=GristStore(
        server=core_settings.grist_server_url,
        api_key=core_settings.grist_api_key,
        doc_id=settings.grist_test_set_doc_id,
        table="Judge_config",
        data_sql_schema=[
            Column("judge_model", String, primary_key=True),
            Column("judge_prompt_id", String, primary_key=True),
            Column("judge_prompt", String),
        ],
    ),
)

eval_llm_answers = Table(
    name="eval_llm_answers",
    store=TableStoreDB(
        dbconn=DBCONN_DATAPIPE,
        name="eval_llm_answers",
        data_sql_schema=[
            Column("dm_id", String, primary_key=True),
            Column("pipeline_model", String, primary_key=True),
            Column("embeddings_model", String, primary_key=True),
            Column("embeddings_dim", Integer, primary_key=True),
            Column("gds_question", String, primary_key=True),
            Column("question_context", String),
            Column("llm_answer", String),
            Column("tool_calls", String),
            Column("test_date", String),
        ],
    ),
)

tests = Table(
    name="tests",
    store=GristStore(
        server=core_settings.grist_server_url,
        api_key=core_settings.grist_api_key,
        doc_id=settings.grist_test_set_doc_id,
        table=settings.tests_table_name,
        data_sql_schema=[
            Column("judge_model", String, primary_key=True),
            Column("judge_prompt_id", String, primary_key=True),
            Column("dm_id", String, primary_key=True),
            Column("pipeline_model", String, primary_key=True),
            Column("embeddings_model", String, primary_key=True),
            Column("embeddings_dim", Integer, primary_key=True),
            Column("gds_question", String, primary_key=True),
            Column("question_context", String),
            Column("gds_answer", String),
            Column("llm_answer", String),
            Column("tool_calls", String),
            Column("test_status", String),
            Column("eval_judge_comment", String),
            Column("test_date", String),
        ],
    ),
)
