from async_lru import alru_cache
from jims_core.app import JimsApp
from loguru import logger

from vedana_core.data_model import DataModel
from vedana_core.db import get_sessionmaker
from vedana_core.graph import MemgraphGraph
from vedana_core.rag_pipeline import RagPipeline, StartPipeline
from vedana_core.settings import settings as core_settings


@alru_cache
async def make_vedana_app() -> JimsApp:
    graph = MemgraphGraph(core_settings.memgraph_uri, core_settings.memgraph_user, core_settings.memgraph_pwd)

    data_model = await DataModel.load_from_graph(graph)
    if data_model is None:
        logger.info("No DataModel found in graph – loading from Grist …")
        data_model = DataModel.load_grist_online(
            core_settings.grist_data_model_doc_id,
            grist_server=core_settings.grist_server_url,
            api_key=core_settings.grist_api_key,
        )

        try:
            await data_model.update_data_model_node(graph)
        except Exception as e:
            logger.warning(f"Unable to cache DataModel in graph: {e}")

    pipeline = RagPipeline(
        graph=graph,
        data_model=data_model,
        logger=logger,
        threshold=0.8,
        temperature=0.1,
    )

    start_pipeline = StartPipeline(data_model=data_model)

    # Jims setup
    sessionmaker = get_sessionmaker()

    app = JimsApp(
        sessionmaker=sessionmaker,
        pipeline=pipeline,
        conversation_start_pipeline=start_pipeline,
    )

    return app


# This creates a async coroutine which will be evaluated in the event loop
app = make_vedana_app()
