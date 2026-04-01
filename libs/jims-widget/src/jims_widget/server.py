import json
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jims_core.app import JimsApp
from jims_core.thread.thread_controller import ThreadController
from jims_core.util import uuid7
from loguru import logger

STATIC_DIR = Path(__file__).parent / "static"


def create_widget_app(jims_app: JimsApp, cors_origins: list[str] | None = None) -> FastAPI:
    app = FastAPI(title="JIMS Widget", version="0.1.0")

    if cors_origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    async def demo_page() -> HTMLResponse:
        html_path = STATIC_DIR / "demo.html"
        return HTMLResponse(content=html_path.read_text())

    @app.websocket("/ws/chat")
    async def websocket_chat(
        websocket: WebSocket,
        thread_id: str | None = Query(default=None),
        contact_id: str | None = Query(default=None),
    ) -> None:
        await websocket.accept()

        effective_contact_id = contact_id or f"widget:{uuid7()}"
        ctl: ThreadController | None = None

        if thread_id:
            try:
                ctl = await ThreadController.from_thread_id(jims_app.sessionmaker, UUID(thread_id))
            except Exception:
                pass

        new_thread = False
        if ctl is None:
            new_thread = True
            tid = uuid7()
            ctl = await jims_app.new_thread(
                contact_id=effective_contact_id,
                thread_id=tid,
                thread_config={"interface": "widget"},
            )
        if new_thread and jims_app.conversation_start_pipeline is not None:
            try:
                start_events = await ctl.run_pipeline_with_context(jims_app.conversation_start_pipeline)
                # Extract assistant messages from conversation_start_pipeline
                start_messages = [
                    str(ev.event_data.get("content", ""))
                    for ev in start_events
                    if ev.event_type == "comm.assistant_message" and isinstance(ev.event_data, dict)
                ]
                # Send the start message immediately over WebSocket
                if start_messages:
                    start_text = "\n\n".join(start_messages)
                    await websocket.send_json(
                        {"text": start_text, "thread_id": str(ctl.thread.thread_id)}
                    )
                    logger.info(f"Sent conversation start message: {start_text}")
            except Exception as exc:
                logger.warning(f"conversation_start_pipeline failed: {exc}")

        logger.info(f"Widget WS connected: thread={ctl.thread.thread_id} contact={effective_contact_id}")

        try:
            while True:
                raw = await websocket.receive_text()
                user_text = _extract_user_text(raw)
                if not user_text:
                    await websocket.send_json({"error": "Empty message"})
                    continue

                try:
                    assistant_text = await _handle_message(jims_app, ctl, user_text)
                    await websocket.send_json(
                        {"text": assistant_text, "thread_id": str(ctl.thread.thread_id)}
                    )
                except Exception as exc:
                    logger.exception(f"Pipeline error on thread {ctl.thread.thread_id}")
                    await websocket.send_json({"error": f"Processing error: {exc}"})

        except WebSocketDisconnect:
            logger.info(f"Widget WS disconnected: thread={ctl.thread.thread_id}")

    return app


def _extract_user_text(raw: str) -> str | None:
    """Extract user message text from a DeepChat WebSocket frame.

    DeepChat may send:
      - A JSON object: {"messages": [{"role":"user","text":"hi"}]}
      - A double-encoded JSON string: "{\"messages\":[{\"role\":\"user\",\"text\":\"hi\"}]}"
      - A plain string (the message text itself)
      - A JSON-encoded string: "\"hello\""
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw.strip() or None

    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return data.strip() or None

    if isinstance(data, dict):
        messages = data.get("messages", [])
        if messages:
            last = messages[-1]
            if isinstance(last, dict):
                text = last.get("text", "")
                if text and isinstance(text, str):
                    result = text.strip() or None
                    return result

        # Fallback: check for direct text/message/content keys
        for key in ("text", "message", "content"):
            val = data.get(key)
            if val and isinstance(val, str):
                return val.strip() or None

    return raw.strip() or None


async def _handle_message(jims_app: JimsApp, ctl: ThreadController, user_text: str) -> str:
    await ctl.store_user_message(uuid7(), user_text)
    outgoing_events = await ctl.run_pipeline_with_context(jims_app.pipeline)

    assistant_messages = [
        str(ev.event_data.get("content", ""))
        for ev in outgoing_events
        if ev.event_type == "comm.assistant_message" and isinstance(ev.event_data, dict)
    ]

    return "\n\n".join(assistant_messages) if assistant_messages else "(no response)"
