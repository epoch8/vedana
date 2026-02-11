"""Integration tests for JIMS API /api/v1/chat endpoint."""

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
