import asyncio
import logging

import sentry_sdk
from aiohttp import web
from jims_telegram import TelegramController
from opentelemetry import trace
from opentelemetry.propagate import set_global_textmap
from opentelemetry.sdk.trace import TracerProvider
from sentry_sdk.integrations.opentelemetry import SentryPropagator, SentrySpanProcessor

from vedana_core.data_model import DataModel
from vedana_core.db import get_sessionmaker
from vedana_core.embeddings import OpenaiEmbeddingProvider
from vedana_core.graph import MemgraphGraph
from vedana_core.rag_pipeline import RagPipeline, StartPipeline
from vedana_core.settings import settings as s

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)


async def healthcheck(host: str = "0.0.0.0", port: int = 8000):
    async def health(request: web.Request):
        return web.json_response({"status": "ok"})

    app = web.Application()
    app.add_routes([web.get("/health", health)])
    app.add_routes([web.get("/healthz", health)])

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    return runner


async def main_tg():
    sentry_sdk.init(
        send_default_pii=True,
        traces_sample_rate=1.0,
        instrumenter="otel",
    )

    sentry_sdk.set_tag("medium", "telegram")

    provider = TracerProvider()
    provider.add_span_processor(SentrySpanProcessor())
    trace.set_tracer_provider(provider)
    set_global_textmap(SentryPropagator())

    try:
        # data setup
        data_model = DataModel.load_grist_online(
            s.grist_data_model_doc_id, grist_server=s.grist_server_url, api_key=s.grist_api_key
        )
        graph = MemgraphGraph(s.memgraph_uri, s.memgraph_user, s.memgraph_pwd)
        embed_provider = OpenaiEmbeddingProvider(s.embeddings_cache_path, s.embeddings_dim)

        pipeline = RagPipeline(
            graph=graph,
            embed_provider=embed_provider,
            data_model=data_model,
            logger=logger,
            threshold=0.8,
            temperature=0.1,
        )

        start_pipeline = StartPipeline(data_model=data_model)

        # Jims setup
        sessionmaker = get_sessionmaker()
        logger.info("Starting Telegram bot...")
        app = TelegramController(
            sessionmaker=sessionmaker, pipeline=pipeline, conversation_start_pipeline=start_pipeline
        )

        await healthcheck()
        await app.run()

    except Exception as e:
        logger.exception(f"Failed to start Telegram bot: {e}")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main_tg())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
        exit(1)
