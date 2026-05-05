
import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Awaitable

import click
from fastapi.security import APIKeyQuery
import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from jims_core.app import JimsApp

from jims_core.thread.thread_controller import ThreadController
from jims_core.util import load_jims_app, setup_monitoring_and_tracing_with_sentry, setup_prometheus_metrics, setup_verbose_logging, uuid7
from jims_chatwoot.settings import chatwoot_settings, api_settings
from loguru import logger
import uvicorn

app = FastAPI()


_CHANNEL_WEBWIDGET = "Channel::WebWidget"
_CHANNEL_TELEGRAM = "Channel::Telegram"


def detect_channel(data: dict[str, Any]) -> str | None:
    if data.get("event") == "webwidget_triggered":
        return _CHANNEL_WEBWIDGET
    conversation = data.get("conversation")
    if conversation:
        return conversation.get("channel")
    return data.get("channel")


def extract_metadata(data: dict[str, Any]) -> dict[str, Any]:
    try:
        telegram_username = data["conversation"]["meta"]["sender"]["additional_attributes"].get("username")
    except (KeyError, TypeError):
        telegram_username = None

    return {
        "contact_id": (data.get("sender") or {}).get("id"),
        "contact_name": (data.get("sender") or {}).get("name"),
        "account_id": (data.get("account") or {}).get("id"),
        "inbox_id": (data.get("inbox") or {}).get("id"),
        "conversation_id": (data.get("conversation") or {}).get("id"),
        "telegram_username": telegram_username,
        "messages": (data.get("conversation") or {}).get("messages"),
    }


def is_closed(data: dict[str, Any]) -> bool:
    custom_attrs = (data.get("conversation") or {}).get("custom_attributes") or {}
    return bool(custom_attrs.get("closed", False))


def attachment_intent(messages: list[dict] | None) -> str | None:
    if not messages:
        return None
    for message in messages:
        for attachment in message.get("attachments") or []:
            file_type = attachment.get("file_type")
            if file_type == "audio":
                return chatwoot_settings.audio_content_intent
            if file_type:
                return chatwoot_settings.media_content_intent
    return None


async def send_response(
    conversation_id: str,
    bot_response: str,
    buttons: list[dict[str, Any]],
    httpx_client: httpx.AsyncClient,
) -> None:
    if not bot_response and not buttons:
        return
    base = f"{chatwoot_settings.url}/api/v1/accounts/{chatwoot_settings.account_id}"
    url = f"{base}/conversations/{conversation_id}/messages"
    headers = {"api_access_token": chatwoot_settings.admin_access_token}

    if buttons:
        body: dict[str, Any] = {
            "content": bot_response,
            "content_type": "input_select",
            "content_attributes": {
                "items": [{"title": b.get("title", ""), "value": b.get("payload", "")} for b in buttons],
            },
            "message_type": "outgoing",
            "private": False,
        }
    else:
        body = {"content": bot_response, "message_type": "outgoing", "private": False}

    try:
        resp = await httpx_client.post(url, json=body, headers=headers)
        if resp.status_code not in (status.HTTP_200_OK, status.HTTP_201_CREATED):
            logger.error(f"Chatwoot send_response failed: {resp.status_code} {resp.text}")
    except httpx.HTTPError as exc:
        logger.error(f"Chatwoot send_response error: {exc}")


async def create_conversation(
    inbox_id: Any,
    contact_id: Any,
    source_id: Any,
    httpx_client: httpx.AsyncClient
) -> int | None:
    url = f"{chatwoot_settings.url}/api/v1/accounts/{chatwoot_settings.account_id}/conversations"
    body = {"inbox_id": str(inbox_id), "contact_id": str(contact_id), "source_id": str(source_id)}
    headers = {"api_access_token": chatwoot_settings.agent_bot_access_token}
    resp = await httpx_client.post(url, json=body, headers=headers)
    if resp.status_code != status.HTTP_200_OK:
        logger.error(f"Chatwoot create_conversation failed: {resp.status_code} {resp.text}")
        return None
    return resp.json().get("id")


async def _resolve_jims_app(app_name: str) -> JimsApp:
    loaded_app = load_jims_app(app_name)
    if isinstance(loaded_app, Awaitable):
        loaded_app = await loaded_app
    return loaded_app


_query_scheme = APIKeyQuery(name="token", auto_error=False)


def verify_token(
    token: str | None = Security(_query_scheme),
) -> None:
    if token != api_settings.token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")


Auth = Annotated[None, Depends(verify_token)]


def create_api(jims_app: JimsApp) -> FastAPI:
    http_client = httpx.AsyncClient()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        await http_client.aclose()

    app = FastAPI(title="JIMS Chatwoot", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/webhooks/chatwoot", status_code=status.HTTP_200_OK, response_model=str)
    async def handle_chatwoot(_: Auth, request: Request) -> str:
        data: dict[str, Any] = await request.json()
        event = data.get("event")
        channel = detect_channel(data)
        logger.info(f"Chatwoot webhook: event={event!r} channel={channel!r}")

        if is_closed(data):
            logger.info("Skipping: dialog is closed")
            return "Ok"

        text: str | None = None
        conversation_id: str | None = None
        metadata: dict[str, Any] = {}

        if (
            event == "webwidget_triggered"
            and data.get("current_conversation") is None
            and not (data.get("event_info") or {}).get("pre_chat_form_enabled")
        ):
            event_info = data.get("event_info") or {}
            text = event_info.get("start_message") or "/get_started"
            inbox_id = (data.get("inbox") or {}).get("id")
            contact_id = (data.get("contact") or {}).get("id")
            source_id = data.get("source_id")
            conv_id = await create_conversation(
                inbox_id,
                contact_id,
                source_id,
                httpx_client=http_client,
            )
            if conv_id is None:
                return "Ok"
            conversation_id = str(conv_id)
            metadata = {"contact_id": contact_id, "account_id": contact_id, "conversation_id": conv_id}
        elif event == "message_updated" and data.get("content_attributes"):
            try:
                text = data["content_attributes"]["submitted_values"][0]["value"]
            except (KeyError, IndexError, TypeError):
                logger.warning("Chatwoot button click: could not extract intent value")
                return "Ok"
            metadata = extract_metadata(data)
            conversation_id = str(metadata.get("conversation_id", ""))
        elif event == "message_created":
            if data.get("message_type") == "outgoing":
                return "Ok"
            conv_status = (data.get("conversation") or {}).get("status")
            if channel == _CHANNEL_WEBWIDGET and conv_status != "pending":
                return "Ok"
            if channel == _CHANNEL_TELEGRAM and conv_status not in ("pending", "open"):
                return "Ok"
            metadata = extract_metadata(data)
            conversation_id = str(metadata.get("conversation_id", ""))
            text = data.get("content") or ""
        else:
            logger.debug(f"Chatwoot: ignoring event={event!r}")
            return "Ok"

        attach_intent = attachment_intent(metadata.get("messages"))  # type: ignore[arg-type]
        if attach_intent:
            text = attach_intent

        if not text or not conversation_id:
            logger.info("Chatwoot: empty text or conversation_id, skipping")
            return "Ok"

        logger.info(f"Chatwoot: conversation_id={conversation_id!r} text={text!r}")

        ctl = await ThreadController.from_external_id(
            jims_app.sessionmaker, external_id=conversation_id
        )

        if ctl is None:
            ctl = await jims_app.new_thread_via_external_id(
                external_id=conversation_id,
                thread_config={},
            )

        await ctl.store_event_dict(
            event_id=uuid7(),
            event_type=f"chatwoot.{event}",
            event_data={"role": "user", "content": text},
        )
        await ctl.store_user_message(event_id=uuid7(), content=text)

        try:
            outgoing_events = await ctl.run_pipeline_with_context(jims_app.pipeline)
        except Exception as exc:
            logger.exception("Pipeline execution failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Pipeline error: {exc}"
            ) from exc

        bot_messages = [
            ev.event_data
            for ev in outgoing_events
            if ev.event_type == "comm.assistant_message_with_buttons"
        ]

        if not bot_messages:
            logger.warning(f"Pipeline produced no assistant messages for conversation_id={conversation_id!r}")
            return "Ok"

        for assistant_message in bot_messages:
            content = assistant_message.get("content")
            buttons = assistant_message.get("buttons", [])

            if not content:
                logger.warning(f"Assistant message has empty content for conversation_id={conversation_id!r}")
                continue

            await send_response(
                conversation_id,
                bot_response=content,
                buttons=buttons,
                httpx_client=http_client,
            )
        return "Ok"

    return app


@click.command()
@click.option("--app", type=click.STRING, default="app", help="JIMS app in module:attr format.")
@click.option("--host", type=click.STRING, default="0.0.0.0")
@click.option("--port", type=click.INT, default=8080)
@click.option("--enable-sentry", is_flag=True, help="Enable tracing to Sentry", default=False)
@click.option("--metrics-port", type=click.INT, default=8000)
@click.option("--verbose", is_flag=True, default=False)
@click.option("--enable-stell-api", is_flag=True, default=False)
def cli(
    app: str,
    host: str,
    port: int,
    enable_sentry: bool,
    metrics_port: int,
    verbose: bool,
    enable_stell_api: bool,
) -> None:
    if verbose:
        setup_verbose_logging()

    setup_prometheus_metrics(port=metrics_port)

    if enable_sentry:
        setup_monitoring_and_tracing_with_sentry()


    async def run_api() -> None:
        jims_app = await _resolve_jims_app(app)
        api = create_api(jims_app)
        if enable_stell_api:
            from stell_core.api import stell_api
            
            api.mount("/stell-api", stell_api)
            logger.info("Stell API mounted at /stell-api")

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
    cli()


if __name__ == "__main__":
    main()
