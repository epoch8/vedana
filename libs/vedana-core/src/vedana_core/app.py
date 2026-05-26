from dataclasses import dataclass

import sqlalchemy.ext.asyncio as sa_aio
from async_lru import alru_cache
from jims_core.app import JimsApp
# from jims_core.schema import Pipeline
from loguru import logger

from vedana_core.data_model import DataModel
from vedana_core.db import get_sessionmaker
from vedana_core.graph import Graph, MemgraphGraph
from vedana_core.vts import VectorStore, PGVectorStore
from vedana_core.rag_pipeline import RagPipeline
from vedana_core.start_pipeline import StartPipeline
from vedana_core.settings import settings as core_settings


@dataclass
class VedanaApp:
    sessionmaker: sa_aio.async_sessionmaker[sa_aio.AsyncSession]
    pipeline: RagPipeline
    start_pipeline: StartPipeline

    graph: Graph
    vts: VectorStore
    data_model: DataModel

    def __post_init__(self):
        self.jims_app = JimsApp(
            sessionmaker=self.sessionmaker,
            pipeline=self.pipeline,
            conversation_start_pipeline=self.start_pipeline,
        )

@alru_cache
async def make_vedana_app() -> VedanaApp:
    sessionmaker = get_sessionmaker()

    graph = MemgraphGraph(
        core_settings.memgraph_uri,
        core_settings.memgraph_user,
        core_settings.memgraph_pwd,
    )

    vts = PGVectorStore(
        sessionmaker=sessionmaker,
    )

    data_model = DataModel(sessionmaker=sessionmaker)

    pipeline = RagPipeline(
        graph=graph,
        vts=vts,
        data_model=data_model,
        logger=logger,
        threshold=0.8,
    )

    start_pipeline = StartPipeline(data_model=data_model)

    return VedanaApp(
        sessionmaker=sessionmaker,
        graph=graph,
        vts=vts,
        data_model=data_model,
        pipeline=pipeline,
        start_pipeline=start_pipeline,
    )


async def make_jims_app() -> JimsApp:
    return (await make_vedana_app()).jims_app


# Coroutine consumed once by JIMS_APP=vedana_core.app:app standalone entrypoints.
app = make_jims_app()
