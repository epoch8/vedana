"""
Tables used in system evaluation pipeline, moved here due to configuration requirements (settings / core_settings)
"""

from datapipe.compute import Table
from datapipe.store.database import TableStoreDB

from sqlalchemy import Column, Integer, String
from vedana_etl.store import GristStore

from vedana_etl.settings import settings
from vedana_core.settings import settings as core_settings
from vedana_etl.config import DBCONN_DATAPIPE


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
