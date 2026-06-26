"""Integration tests for JIMS API /api/v1/chat endpoint."""

import json

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_chat_new_conversation(client: AsyncClient) -> None:
    """Request without thread_id creates a new thread and returns created_new_thread=True."""
    resp = await client.post(
        "/api/v1/chat",
        json={
            "contact_id": "test:user1",
            "message": "Hello",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "thread_id" in data
    assert data["created_new_thread"] is True
    assert data["assistant_messages"] == ["Echo: Hello"]
    assert len(data["events"]) >= 1
    assert any(e["event_type"] == "comm.assistant_message" for e in data["events"])


@pytest.mark.asyncio
async def test_chat_existing_thread(client: AsyncClient) -> None:
    """Request with thread_id continues an existing conversation."""
    # First request: create a thread
    resp1 = await client.post(
        "/api/v1/chat",
        json={
            "contact_id": "test:user2",
            "message": "First message",
        },
    )
    assert resp1.status_code == 200
    thread_id = resp1.json()["thread_id"]
    assert resp1.json()["created_new_thread"] is True

    # Second request: continue the same thread
    resp2 = await client.post(
        "/api/v1/chat",
        json={
            "contact_id": "test:user2",
            "message": "Second message",
            "thread_id": thread_id,
        },
    )
    assert resp2.status_code == 200
    data = resp2.json()
    assert data["thread_id"] == thread_id
    assert data["created_new_thread"] is False
    assert data["assistant_messages"] == ["Echo: Second message"]


async def _read_sse_events(resp) -> list[tuple[str, dict]]:
    """Parse an SSE response body into a list of (event, data) pairs."""
    events: list[tuple[str, dict]] = []
    event_name = "message"
    data_lines: list[str] = []

    async for line in resp.aiter_lines():
        if line == "":
            if data_lines:
                events.append((event_name, json.loads("\n".join(data_lines))))
            event_name = "message"
            data_lines = []
        elif line.startswith("event:"):
            event_name = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").strip())

    return events


@pytest.mark.asyncio
async def test_chat_stream_new_conversation(client: AsyncClient) -> None:
    """Streaming endpoint yields a final 'result' event with the pipeline's response."""
    async with client.stream(
        "POST",
        "/api/v1/chat/stream",
        json={"contact_id": "test:user3", "message": "Hello"},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = await _read_sse_events(resp)

    assert events, "expected at least one SSE event"
    event_name, data = events[-1]
    assert event_name == "result"
    assert data["created_new_thread"] is True
    assert data["assistant_messages"] == ["Echo: Hello"]


@pytest.mark.asyncio
async def test_chat_stream_status_updates(status_client: AsyncClient) -> None:
    """Streaming endpoint relays agent status updates before the final result."""
    async with status_client.stream(
        "POST",
        "/api/v1/chat/stream",
        json={"contact_id": "test:user4", "message": "Hello"},
    ) as resp:
        assert resp.status_code == 200
        events = await _read_sse_events(resp)

    status_events = [(data["status"], data["type"]) for name, data in events if name == "status"]
    assert status_events == [
        ("Thinking...", "thinking"),
        ("Searching knowledge base...", "searching_knowledge_base"),
    ]

    assert events[-1][0] == "result"
    assert events[-1][1]["assistant_messages"] == ["Echo: Hello"]
