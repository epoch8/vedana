import html
import json
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jims_core.app import JimsApp
from jims_core.thread.schema import CommunicationEvent
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

        if ctl is not None:
            logger.info(f"Widget WS connected: thread={ctl.thread.thread_id} contact={effective_contact_id}")
        else:
            logger.info(f"Widget WS connected: no thread yet, contact={effective_contact_id}")

        try:
            while True:
                raw = await websocket.receive_text()
                user_text = _extract_user_text(raw)
                if not user_text:
                    await websocket.send_json({"error": "Empty message"})
                    continue

                try:
                    if ctl is None:
                        ctl = await jims_app.new_thread(
                            contact_id=effective_contact_id,
                            thread_id=uuid7(),
                            thread_config={"interface": "widget"},
                        )
                        logger.info(
                            f"Widget thread created on first message: "
                            f"thread={ctl.thread.thread_id} contact={effective_contact_id}"
                        )
                    payload = await _handle_message(jims_app, ctl, user_text)
                    await websocket.send_json(payload)
                except Exception as exc:
                    thread_for_log = str(ctl.thread.thread_id) if ctl is not None else "<not-created>"
                    logger.exception(f"Pipeline error on thread {thread_for_log}")
                    await websocket.send_json({"error": f"Processing error: {exc}"})

        except WebSocketDisconnect:
            if ctl is not None:
                logger.info(f"Widget WS disconnected: thread={ctl.thread.thread_id}")
            else:
                logger.info(f"Widget WS disconnected before first message: contact={effective_contact_id}")

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


def _normalize_button_rows(buttons: list) -> list[list[dict]]:
    """Match jims-telegram row shape: list of rows, each row a list of {text, id?}."""
    if not buttons:
        return []
    if all(isinstance(b, dict) for b in buttons):
        return [buttons]  # type: ignore[list-item]
    rows: list[list[dict]] = []
    for row in buttons:
        if isinstance(row, list):
            rows.append([b for b in row if isinstance(b, dict)])
    return rows


def _buttons_to_widget_html(buttons: list) -> str:
    """Build Deep Chat–friendly HTML for callback buttons (wired in jims-widget.js)."""
    rows = _normalize_button_rows(buttons)
    if not rows:
        return ""
    parts = [
        '<div class="jims-widget-button-stack" style="display:flex;flex-direction:column;align-items:center;'
        'gap:10px;margin-top:8px;width:100%;box-sizing:border-box;">'
    ]
    for row in rows:
        parts.append(
            '<div class="jims-widget-button-row" style="display:flex;flex-wrap:wrap;gap:8px;'
            'justify-content:center;width:100%;">'
        )
        for b in row:
            label_v = b.get("text")
            if label_v is None:
                continue
            label = html.escape(str(label_v))
            raw_id = b.get("id") or b.get("callback_data") or label_v
            raw_s = str(raw_id)
            callback = raw_s if raw_s.startswith("btn:") else f"btn:{raw_s}"
            safe_cb = html.escape(callback, quote=True)
            parts.append(
                '<button type="button" class="deep-chat-button jims-widget-callback-btn" '
                'style="min-width:12rem;max-width:100%;padding:10px 18px;font-weight:600;" '
                f'data-jims-callback="{safe_cb}">{label}</button>'
            )
        parts.append("</div>")
    parts.append("</div>")
    return "".join(parts)


def _widget_response_from_outgoing(outgoing_events: list) -> dict[str, str] | list[dict[str, str]]:
    """Turn pipeline outgoing comm.* events into Deep Chat websocket payload(s).

    Deep Chat does not render `html` when the same JSON object also has `text` — it keeps
    only one type per message. When we have both, send a JSON array: text bubble first,
    then an html-only bubble for buttons (see deep-chat bundle: text vs html branch).
    """
    text_chunks: list[str] = []
    html_chunks: list[str] = []

    for ev in outgoing_events:
        if not isinstance(ev.event_data, dict):
            continue
        data = ev.event_data
        if ev.event_type == "comm.assistant_message":
            content = data.get("content")
            if content:
                text_chunks.append(str(content))
        elif ev.event_type == "comm.assistant_message_with_buttons":
            content = data.get("content")
            if content:
                text_chunks.append(str(content))
            btn_html = _buttons_to_widget_html(data.get("buttons") or [])
            if btn_html:
                html_chunks.append(btn_html)
        elif ev.event_type == "comm.assistant_buttons":
            content = data.get("content")
            if content:
                text_chunks.append(str(content))
            btn_html = _buttons_to_widget_html(data.get("buttons") or [])
            if btn_html:
                html_chunks.append(btn_html)

    text = "\n\n".join(text_chunks) if text_chunks else ""
    html_combined = "".join(html_chunks) if html_chunks else ""

    if not text and not html_combined:
        return {"text": "(no response)"}

    if html_combined:
        parts: list[dict[str, str]] = []
        if text.strip():
            parts.append({"text": text})
        parts.append({"html": html_combined})
        return parts if len(parts) > 1 else parts[0]

    return {"text": text}


async def _handle_message(jims_app: JimsApp, ctl: ThreadController, user_text: str) -> dict[str, str] | list[dict[str, str]]:
    if user_text.startswith("btn:"):
        await ctl.store_event_dict(
            event_id=uuid7(),
            event_type="comm.user_button_click",
            event_data=dict(CommunicationEvent(role="user", content=user_text)),
        )
    else:
        await ctl.store_user_message(uuid7(), user_text)
    outgoing_events = await ctl.run_pipeline_with_context(jims_app.pipeline)
    return _widget_response_from_outgoing(outgoing_events)
