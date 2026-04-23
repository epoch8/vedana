"""Integration tests for widget FastAPI app and WebSocket."""

import pytest
from httpx import AsyncClient
from starlette.testclient import TestClient


@pytest.mark.asyncio
async def test_healthz(client: AsyncClient) -> None:
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_demo_page(client: AsyncClient) -> None:
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "deep-chat" in resp.text.lower() or "chat" in resp.text.lower()


@pytest.mark.asyncio
async def test_static_widget_js(client: AsyncClient) -> None:
    resp = await client.get("/static/jims-widget.js")
    assert resp.status_code == 200
    assert "jims-widget" in resp.text or "data-server" in resp.text


def test_websocket_chat(app) -> None:
    with TestClient(app) as client:
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"messages": [{"role": "user", "text": "Hello"}]})
            data = ws.receive_json()
            assert "text" in data
            assert "Echo: Hello" in data["text"]


def test_websocket_empty_message_returns_error(app) -> None:
    with TestClient(app) as client:
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_text("")
            data = ws.receive_json()
            assert "error" in data
            assert "Empty" in data["error"]
