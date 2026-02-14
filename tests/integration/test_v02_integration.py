"""V0.2 Integration Tests: Full stack adapter → ASGI app → A2A protocol.

Tests the complete request flow without mocking internal components:
    Client → ASGI App → DefaultRequestHandler → AdapterAgentExecutor → Adapter

Uses httpx.AsyncClient with ASGI transport to test against the real Starlette
app without starting a network server, verifying A2A protocol compliance
end-to-end.
"""

import json
import pytest
import httpx

from a2a_adapter import BaseA2AAdapter, AdapterMetadata, to_a2a, build_agent_card


# ──── Test Adapters ────


class EchoAdapter(BaseA2AAdapter):
    """Simple echo adapter for integration testing."""

    async def invoke(self, user_input, context_id=None, **kwargs):
        return f"Echo: {user_input}"

    def get_metadata(self):
        return AdapterMetadata(
            name="Echo Agent",
            description="Echoes back user input",
            version="1.0.0",
            skills=[{"id": "echo", "name": "Echo", "description": "Echoes input"}],
        )


class StreamEchoAdapter(BaseA2AAdapter):
    """Streaming echo adapter for integration testing."""

    async def invoke(self, user_input, context_id=None, **kwargs):
        return f"Echo: {user_input}"

    async def stream(self, user_input, context_id=None, **kwargs):
        for word in user_input.split():
            yield word + " "

    def get_metadata(self):
        return AdapterMetadata(
            name="Stream Echo Agent",
            description="Streaming echo",
            streaming=True,
        )


class FailAdapter(BaseA2AAdapter):
    """Adapter that always fails — for error handling tests."""

    async def invoke(self, user_input, context_id=None, **kwargs):
        raise RuntimeError("intentional failure")


# ──── Fixtures ────


@pytest.fixture
def echo_app():
    return to_a2a(EchoAdapter())


@pytest.fixture
def stream_app():
    return to_a2a(StreamEchoAdapter())


@pytest.fixture
def fail_app():
    return to_a2a(FailAdapter())


# ──── Helpers ────


def make_send_payload(text: str = "Hello") -> dict:
    """Create a minimal A2A message/send JSON-RPC payload."""
    return {
        "jsonrpc": "2.0",
        "id": "test-1",
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "messageId": "msg-1",
                "parts": [{"kind": "text", "text": text}],
            }
        },
    }


def make_stream_payload(text: str = "Hello") -> dict:
    """Create a minimal A2A message/stream JSON-RPC payload."""
    return {
        "jsonrpc": "2.0",
        "id": "test-1",
        "method": "message/stream",
        "params": {
            "message": {
                "role": "user",
                "messageId": "msg-1",
                "parts": [{"kind": "text", "text": text}],
            }
        },
    }


# ═══════════════════════════════════════════════════
# Agent Card Tests
# ═══════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_agent_card_endpoint(echo_app):
    """/.well-known/agent.json should return the agent card."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=echo_app),
        base_url="http://testserver",
    ) as client:
        resp = await client.get("/.well-known/agent.json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Echo Agent"
        assert data["description"] == "Echoes back user input"
        assert data["version"] == "1.0.0"
        assert len(data["skills"]) == 1
        assert data["skills"][0]["id"] == "echo"


# ═══════════════════════════════════════════════════
# message/send Tests
# ═══════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_message_send_basic(echo_app):
    """message/send should return a completed task with echo response."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=echo_app),
        base_url="http://testserver",
    ) as client:
        resp = await client.post("/", json=make_send_payload("Hi there"))
        assert resp.status_code == 200

        data = resp.json()
        assert "result" in data
        result = data["result"]
        assert result["status"]["state"] == "completed"

        # Agent response is in status.message (the completion message)
        status_msg = result["status"].get("message", {})
        assert status_msg.get("role") == "agent"
        text_parts = [
            p["text"] for p in status_msg.get("parts", [])
            if p.get("kind") == "text"
        ]
        assert any("Echo: Hi there" in t for t in text_parts)

        # Artifacts should also contain the response
        artifacts = result.get("artifacts", [])
        assert len(artifacts) > 0
        artifact_texts = [
            p["text"] for a in artifacts
            for p in a.get("parts", [])
            if p.get("kind") == "text"
        ]
        assert any("Echo: Hi there" in t for t in artifact_texts)


@pytest.mark.asyncio
async def test_message_send_error_handling(fail_app):
    """When adapter raises, task should be in failed state."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fail_app),
        base_url="http://testserver",
    ) as client:
        resp = await client.post("/", json=make_send_payload("anything"))
        assert resp.status_code == 200

        data = resp.json()
        result = data["result"]
        assert result["status"]["state"] == "failed"


# ═══════════════════════════════════════════════════
# message/stream Tests (SSE)
# ═══════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_message_stream_returns_sse(stream_app):
    """message/stream should return Server-Sent Events."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=stream_app),
        base_url="http://testserver",
    ) as client:
        resp = await client.post("/", json=make_stream_payload("hello world"))
        assert resp.status_code == 200
        # SSE response should contain event data
        content_type = resp.headers.get("content-type", "")
        assert "text/event-stream" in content_type

        # Parse SSE events
        events = []
        for line in resp.text.split("\n"):
            line = line.strip()
            if line.startswith("data:"):
                event_data = json.loads(line[5:].strip())
                events.append(event_data)

        # Should have at least one event
        assert len(events) > 0


# ═══════════════════════════════════════════════════
# Custom Agent Card Tests
# ═══════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_custom_agent_card():
    """to_a2a with card overrides should serve the custom card."""
    app = to_a2a(
        EchoAdapter(),
        name="Custom Echo",
        url="http://myserver:5000",
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        resp = await client.get("/.well-known/agent.json")
        data = resp.json()
        assert data["name"] == "Custom Echo"
        assert data["url"] == "http://myserver:5000"


# ═══════════════════════════════════════════════════
# Multiple Requests (Concurrency) Test
# ═══════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_multiple_sequential_requests(echo_app):
    """Multiple requests should each get independent responses."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=echo_app),
        base_url="http://testserver",
    ) as client:
        for i in range(3):
            resp = await client.post(
                "/", json=make_send_payload(f"Message {i}")
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["result"]["status"]["state"] == "completed"


# ═══════════════════════════════════════════════════
# JSON-RPC Error Tests
# ═══════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_invalid_method(echo_app):
    """Invalid JSON-RPC method should return an error response."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=echo_app),
        base_url="http://testserver",
    ) as client:
        payload = {
            "jsonrpc": "2.0",
            "id": "test-bad",
            "method": "nonexistent/method",
            "params": {},
        }
        resp = await client.post("/", json=payload)
        # The SDK should return an error response
        data = resp.json()
        assert "error" in data


# ═══════════════════════════════════════════════════
# Build Agent Card Standalone Tests
# ═══════════════════════════════════════════════════


def test_build_card_integration():
    """build_agent_card should produce valid AgentCard from adapter metadata."""
    adapter = EchoAdapter()
    card = build_agent_card(adapter)

    assert card.name == "Echo Agent"
    assert card.description == "Echoes back user input"
    assert card.capabilities.streaming is False
    assert len(card.skills) == 1


def test_build_card_streaming_adapter():
    adapter = StreamEchoAdapter()
    card = build_agent_card(adapter)

    assert card.name == "Stream Echo Agent"
    assert card.capabilities.streaming is True
