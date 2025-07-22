import logging
from contextlib import asynccontextmanager

import fastapi
import gradio as gr
import sentry_sdk
import uvicorn
from opentelemetry import trace
from opentelemetry.propagate import set_global_textmap
from opentelemetry.sdk.trace import TracerProvider
from prometheus_client import start_http_server
from sentry_sdk.integrations.opentelemetry import SentryPropagator, SentrySpanProcessor

from vedana.data_model import DataModel
from vedana.db import get_sessionmaker
from vedana.embeddings import OpenaiEmbeddingProvider
from vedana.gradio_ui import create_gradio_interface, init_async_stuff
from vedana.graph import MemgraphGraph
from vedana.importers.fast import DataModelLoader
from vedana.settings import settings as s

logging.basicConfig(
    level=(logging.DEBUG if s.debug else logging.INFO),
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    init_async_stuff()

    embeds = OpenaiEmbeddingProvider(s.embeddings_cache_path, s.embeddings_dim)
    app.state.embed_provider = embeds

    yield
    embeds.close()


def make_jims_app() -> fastapi.FastAPI:
    # load data
    graph = MemgraphGraph(s.memgraph_uri, s.memgraph_user, s.memgraph_pwd)

    data_model = DataModel.load_from_graph(graph)
    if data_model is None:
        logger.info("No DataModel found in graph – loading from Grist …")
        data_model = DataModel.load_grist_online(
            s.grist_data_model_doc_id, grist_server=s.grist_server_url, api_key=s.grist_api_key
        )

        try:
            DataModelLoader(data_model, graph).update_data_model_node()
        except Exception as e:
            logger.warning(f"Unable to cache DataModel in graph: {e}")

    embed_provider = OpenaiEmbeddingProvider(s.embeddings_cache_path, s.embeddings_dim)

    # load JIMS
    sessionmaker = get_sessionmaker()
    init_async_stuff()

    from vedana.gradio_ui import loop

    # gradio setup
    iface = create_gradio_interface(
        graph=graph,
        embed_provider=embed_provider,
        data_model=data_model,
        sessionmaker=sessionmaker,
        loop=loop,
    )
    iface.queue(default_concurrency_limit=10)

    # app setup + health for k8s
    app = fastapi.FastAPI(lifespan=lifespan)

    # TODO remove
    @app.get("/health")
    async def health_check():
        return {"status": "ok"}

    @app.get("/healthz")
    async def healthz_check():
        return {"status": "ok"}

    app = gr.mount_gradio_app(app, iface, path="/")
    return app


def main():
    sentry_sdk.init(
        send_default_pii=True,
        traces_sample_rate=1.0,
        instrumenter="otel",
    )

    sentry_sdk.set_tag("medium", "gradio")

    provider = TracerProvider()
    provider.add_span_processor(SentrySpanProcessor())
    trace.set_tracer_provider(provider)
    set_global_textmap(SentryPropagator())

    start_http_server(8000)

    app = make_jims_app()
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
