from dataclasses import dataclass

import sqlalchemy.ext.asyncio as sa_aio
from async_lru import alru_cache
from jims_core.app import JimsApp
from loguru import logger

from vedana_core.data_model import DataModel
from vedana_core.db import get_sessionmaker
from vedana_core.graph import Graph, MemgraphGraphPgvectorVts
from vedana_core.rag_pipeline import RagPipeline, StartPipeline
from vedana_core.settings import settings as core_settings


@dataclass
class VedanaApp:
    sessionmaker: sa_aio.async_sessionmaker[sa_aio.AsyncSession]

    graph: Graph
    data_model: DataModel
    pipeline: RagPipeline
    start_pipeline: StartPipeline


@alru_cache
async def make_vedana_app() -> VedanaApp:
    sessionmaker = get_sessionmaker()

    graph = MemgraphGraphPgvectorVts(
        core_settings.memgraph_uri,
        core_settings.memgraph_user,
        core_settings.memgraph_pwd,
        sessionmaker=sessionmaker,
    )

    # Jims setup
    sessionmaker = get_sessionmaker()

    data_model = DataModel(sessionmaker=sessionmaker)

    pipeline = RagPipeline(
        graph=graph,
        data_model=data_model,
        logger=logger,
        threshold=0.8,
    )

    start_pipeline = StartPipeline(data_model=data_model)

    return VedanaApp(
        sessionmaker=sessionmaker,
        graph=graph,
        data_model=data_model,
        pipeline=pipeline,
        start_pipeline=start_pipeline,
    )


@alru_cache
async def make_jims_app() -> JimsApp:
    vedana_app = await make_vedana_app()

    app = JimsApp(
        sessionmaker=vedana_app.sessionmaker,
        pipeline=vedana_app.pipeline,
        conversation_start_pipeline=vedana_app.start_pipeline,
    )

    return app


# This creates a async coroutine which will be evaluated in the event loop
app = make_jims_app()
