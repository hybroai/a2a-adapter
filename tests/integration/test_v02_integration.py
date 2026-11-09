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


@pytest.mark.asyncio
async def test_agent_card_push_notifications(echo_app):
    """Push notifications are off by default."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=echo_app),
        base_url="http://testserver",
    ) as client:
        resp = await client.get("/.well-known/agent.json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["capabilities"]["pushNotifications"] is False


@pytest.mark.asyncio
async def test_agent_card_push_notifications_opt_in():
    app = to_a2a(EchoAdapter(), push_notifications=True)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        data = (await client.get("/.well-known/agent.json")).json()
    assert data["capabilities"]["pushNotifications"] is True


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
        text_parts = [p["text"] for p in status_msg.get("parts", []) if p.get("kind") == "text"]
        assert any("Echo: Hi there" in t for t in text_parts)

        # Artifacts should also contain the response
        artifacts = result.get("artifacts", [])
        assert len(artifacts) > 0
        artifact_texts = [
            p["text"] for a in artifacts for p in a.get("parts", []) if p.get("kind") == "text"
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
# streaming=False on a streaming adapter
# ═══════════════════════════════════════════════════


@pytest.fixture
async def streaming_off_client():
    app = to_a2a(StreamEchoAdapter(), streaming=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        yield client


async def test_streaming_off_card_does_not_advertise_streaming(streaming_off_client):
    data = (await streaming_off_client.get("/.well-known/agent.json")).json()
    assert not data["capabilities"].get("streaming")


async def test_streaming_off_rejects_v03_message_stream(streaming_off_client):
    resp = await streaming_off_client.post("/", json=make_stream_payload("hello world"))
    assert resp.status_code == 200
    events = [
        json.loads(line[5:]) for line in resp.text.splitlines() if line.strip().startswith("data:")
    ]
    assert len(events) == 1
    assert "Streaming is not supported" in events[0]["error"]["message"]


async def test_streaming_off_rejects_v1_send_streaming_message(streaming_off_client):
    payload = {
        "jsonrpc": "2.0",
        "id": "test-1",
        "method": "SendStreamingMessage",
        "params": {
            "message": {
                "role": "ROLE_USER",
                "messageId": "msg-1",
                "parts": [{"text": "hello world"}],
            }
        },
    }
    resp = await streaming_off_client.post("/", json=payload, headers={"A2A-Version": "1.0"})
    assert resp.json()["error"]["code"] == -32004


async def test_streaming_off_message_send_completes(streaming_off_client):
    resp = await streaming_off_client.post("/", json=make_send_payload("hello world"))
    result = resp.json()["result"]
    assert result["status"]["state"] == "completed"
    text = "".join(p.get("text", "") for p in result["status"]["message"]["parts"])
    assert text == "hello world "


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
            resp = await client.post("/", json=make_send_payload(f"Message {i}"))
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


# ═══════════════════════════════════════════════════
# Shutdown kills CLI process trees (POSIX)
# ═══════════════════════════════════════════════════


@pytest.mark.skipif(__import__("os").name != "posix", reason="POSIX process groups")
@pytest.mark.asyncio
async def test_lifespan_shutdown_kills_cli_grandchildren(tmp_path):
    import asyncio

    from tests.process_helpers import (
        TreeCliAdapter,
        cancel_and_wait,
        kill_pids,
        wait_dead,
        wait_for_pids,
    )

    app = to_a2a(TreeCliAdapter(tmp_path))
    request = None
    grandchild = None

    async def run_lifespan():
        nonlocal request, grandchild
        async with app.router.lifespan_context(app):
            client = httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            )
            request = asyncio.create_task(client.post("/", json=make_send_payload("hi")))
            _, grandchild = await wait_for_pids(tmp_path)
        # Lifespan exit ran adapter.close()

    try:
        await asyncio.wait_for(run_lifespan(), 5)
        assert await wait_dead(grandchild), "grandchild survived the lifespan shutdown"
    finally:
        kill_pids(tmp_path)
        await cancel_and_wait(request)


# ═══════════════════════════════════════════════════
# Bearer-token auth
# ═══════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_authenticated_send_and_stream():
    app = to_a2a(StreamEchoAdapter(), auth_token="s3cret")
    auth = {"Authorization": "Bearer s3cret"}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        sent = await client.post("/", json=make_send_payload("hi there"), headers=auth)
        assert sent.status_code == 200
        task = sent.json()["result"]
        assert task["status"]["state"] == "completed"

        streamed = await client.post("/", json=make_stream_payload("a b"), headers=auth)
        assert streamed.status_code == 200
        assert "text/event-stream" in streamed.headers["content-type"]
        assert '"completed"' in streamed.text or "completed" in streamed.text

        get = {"jsonrpc": "2.0", "id": "g", "method": "tasks/get", "params": {"id": task["id"]}}
        assert (await client.post("/", json=get)).status_code == 401
        assert (await client.post("/", json=get, headers=auth)).status_code == 200


@pytest.mark.asyncio
async def test_activity_log_records_rejected_request():
    import io

    from a2a_adapter.terminal import ActivityLog, Verbosity

    stream = io.StringIO()
    log = ActivityLog(stream, color=False, verbosity=Verbosity.DEBUG, width=200)
    app = to_a2a(EchoAdapter(), auth_token="s3cret", activity=log)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.post("/", json=make_send_payload("private-payload-xyz"))
    assert response.status_code == 401
    output = stream.getvalue()
    rows = [line for line in output.splitlines() if "401" in line]
    assert len(rows) == 1
    assert "POST /" in rows[0]
    assert "private-payload-xyz" not in output


@pytest.mark.asyncio
async def test_push_notification_delivered_to_allowed_url(monkeypatch):
    """Push on, private receivers allowed, auth on: a task POSTs to the receiver."""
    import asyncio

    import a2a_adapter.server as server_module

    received: list[httpx.Request] = []
    real = server_module._make_push_client

    def make(**kwargs):
        transport = httpx.MockTransport(lambda r: received.append(r) or httpx.Response(200))
        return real(transport=transport, **kwargs)

    monkeypatch.setattr(server_module, "_make_push_client", make)
    app = to_a2a(
        EchoAdapter(), push_notifications=True, allow_private_push_urls=True, auth_token="s3cret"
    )
    payload = make_send_payload("ping")
    payload["params"]["configuration"] = {
        "pushNotificationConfig": {"url": "http://receiver.test/hook"}
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
        headers={"Authorization": "Bearer s3cret"},
    ) as client:
        reply = (await client.post("/", json=payload)).json()
        for _ in range(50):
            if received:
                break
            await asyncio.sleep(0.01)
    assert reply["result"]["status"]["state"] == "completed"
    assert received, "the receiver got the task update"
    assert str(received[0].url) == "http://receiver.test/hook"
    assert reply["result"]["id"] in received[-1].content.decode()
