import asyncio
import logging

import fastapi
import gradio as gr
import sentry_sdk
import uvicorn
from opentelemetry import trace
from opentelemetry.propagate import set_global_textmap
from opentelemetry.sdk.trace import TracerProvider
from prometheus_client import start_http_server
from sentry_sdk.integrations.opentelemetry import SentryPropagator, SentrySpanProcessor
from vedana_core.data_model import DataModel
from vedana_core.db import get_sessionmaker
from vedana_core.graph import MemgraphGraph
from vedana_core.settings import settings as s

from vedana_gradio.gradio_ui import create_gradio_interface

logging.basicConfig(
    level=(logging.DEBUG if s.debug else logging.INFO),
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


async def make_jims_app() -> fastapi.FastAPI:
    # load data
    graph = MemgraphGraph(s.memgraph_uri, s.memgraph_user, s.memgraph_pwd)

    data_model = await DataModel.load_from_graph(graph)
    if data_model is None:
        logger.info("No DataModel found in graph – loading from Grist …")
        data_model = DataModel.load_grist_online(
            s.grist_data_model_doc_id, grist_server=s.grist_server_url, api_key=s.grist_api_key
        )

        try:
            await data_model.update_data_model_node(graph)
        except Exception as e:
            logger.warning(f"Unable to cache DataModel in graph: {e}")

    # load JIMS
    sessionmaker = get_sessionmaker()

    # gradio setup
    iface = await create_gradio_interface(
        graph=graph,
        data_model=data_model,
        sessionmaker=sessionmaker,
    )
    iface.queue(default_concurrency_limit=10)

    # app setup + health for k8s
    app = fastapi.FastAPI()

    # TODO remove
    @app.get("/health")
    async def health_check():
        return {"status": "ok"}

    @app.get("/healthz")
    async def healthz_check():
        return {"status": "ok"}

    app = gr.mount_gradio_app(app, iface, path="")
    return app


async def main():
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

    app = await make_jims_app()

    # Create uvicorn config for async startup
    config = uvicorn.Config(app, host="0.0.0.0", port=7860)
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
