import asyncio
from typing import Any, Awaitable
from uuid import UUID

import click
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from jims_core.app import JimsApp
from jims_core.thread.thread_controller import ThreadController
from jims_core.util import (
    load_jims_app,
    setup_monitoring_and_tracing_with_sentry,
    setup_prometheus_metrics,
    setup_verbose_logging,
    uuid7,
)
from loguru import logger
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    contact_id: str
    message: str
    thread_id: UUID | None = None  # if none provided - create new thread
    thread_config: dict[str, Any] = Field(default_factory=lambda: {"interface": "api"})
    event_type: str = "comm.user_message"
    run_conversation_start_on_new_thread: bool = False


class ApiEvent(BaseModel):
    event_type: str
    event_data: dict[str, Any]


class ChatResponse(BaseModel):
    thread_id: UUID
    created_new_thread: bool
    assistant_messages: list[str]
    events: list[ApiEvent]


def _extract_token(authorization: str | None) -> str | None:
    if authorization is None:
        return None

    if authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1]

    return authorization


def _auth_dependency(api_key: str | None):
    async def require_auth(authorization: str | None = Header(default=None, alias="Authorization")) -> None:
        if api_key is None:
            return

        token = _extract_token(authorization)
        if token != api_key:
            raise HTTPException(status_code=401, detail="Unauthorized")

    return require_auth


async def _resolve_jims_app(app_name: str) -> JimsApp:
    loaded_app = load_jims_app(app_name)
    if isinstance(loaded_app, Awaitable):
        loaded_app = await loaded_app
    return loaded_app


def create_api(jims_app: JimsApp, api_key: str | None) -> FastAPI:
    app = FastAPI(title="JIMS API", version="0.1.0")
    require_auth = _auth_dependency(api_key)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/v1/chat", response_model=ChatResponse, dependencies=[Depends(require_auth)])
    async def chat(req: ChatRequest) -> ChatResponse:
        created_new_thread = False
        ctl: ThreadController

        if req.thread_id is not None:
            ctl = await ThreadController.from_thread_id(jims_app.sessionmaker, req.thread_id)
            if ctl is None:
                raise HTTPException(status_code=404, detail=f"Thread '{req.thread_id}' not found")
        else:
            ctl = await ThreadController.latest_thread_from_contact_id(jims_app.sessionmaker, req.contact_id)

        if ctl is None:
            created_new_thread = True
            thread_id = req.thread_id or uuid7()
            ctl = await jims_app.new_thread(
                contact_id=req.contact_id,
                thread_id=thread_id,
                thread_config=req.thread_config,
            )

            if req.run_conversation_start_on_new_thread and jims_app.conversation_start_pipeline is not None:
                await ctl.run_pipeline_with_context(jims_app.conversation_start_pipeline)

        await ctl.store_event_dict(
            event_id=uuid7(),
            event_type=req.event_type,
            event_data={"role": "user", "content": req.message},
        )

        try:
            outgoing_events = await ctl.run_pipeline_with_context(jims_app.pipeline)
        except Exception as exc:
            logger.exception("Pipeline execution failed")
            raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

        assistant_messages = [
            str(ev.event_data.get("content", ""))
            for ev in outgoing_events
            if ev.event_type == "comm.assistant_message" and isinstance(ev.event_data, dict)
        ]

        response_events = [
            ApiEvent(
                event_type=ev.event_type,
                event_data=ev.event_data if isinstance(ev.event_data, dict) else {},
                # todo filter comm.* events only?
            )
            for ev in outgoing_events
        ]

        return ChatResponse(
            thread_id=ctl.thread.thread_id,
            created_new_thread=created_new_thread,
            assistant_messages=assistant_messages,
            events=response_events,
        )

    return app


@click.command()
@click.option("--app", type=click.STRING, default="app", help="JIMS app in module:attr format.")
@click.option("--host", type=click.STRING, default="0.0.0.0")
@click.option("--port", type=click.INT, default=8080)
@click.option("--api-key", type=click.STRING, default=None, help="Optional bearer token for API auth.")
@click.option("--enable-sentry", is_flag=True, help="Enable tracing to Sentry", default=False)
@click.option("--metrics-port", type=click.INT, default=8000)
@click.option("--verbose", is_flag=True, default=False)
def cli(
    app: str,
    host: str,
    port: int,
    api_key: str | None,
    enable_sentry: bool,
    metrics_port: int,
    verbose: bool,
) -> None:
    if verbose:
        setup_verbose_logging()

    setup_prometheus_metrics(port=metrics_port)

    if enable_sentry:
        setup_monitoring_and_tracing_with_sentry()

    async def run_api() -> None:
        jims_app = await _resolve_jims_app(app)
        api = create_api(jims_app, api_key=api_key)
        server = uvicorn.Server(uvicorn.Config(api, host=host, port=port, log_level="info"))
        await server.serve()

    try:
        asyncio.run(run_api())
    except KeyboardInterrupt:
        logger.info("API stopped by user")
    except Exception as exc:
        logger.exception(f"API crashed: {exc}")
        raise SystemExit(1) from exc


def main() -> None:
    cli(auto_envvar_prefix="JIMS")


if __name__ == "__main__":
    main()
