"""A LangGraph graph that pauses on ``interrupt()``, served over A2A.

Full stack: httpx -> to_a2a ASGI app -> DefaultRequestHandler ->
AdapterAgentExecutor -> LangGraphAdapter -> real compiled graph with an
in-memory checkpointer. Covers SendMessage and SendStreamingMessage (A2A 1.0).
"""

import json

import httpx
import pytest

pytest.importorskip("langgraph")

from a2a_adapter import to_a2a  # noqa: E402
from a2a_adapter.integrations.langgraph import LangGraphAdapter  # noqa: E402
from tests.langgraph_graphs import approval_graph  # noqa: E402

V1 = {"A2A-Version": "1.0"}


def _payload(method, text, message_id, task_id=None, context_id=None):
    message = {"role": "ROLE_USER", "messageId": message_id, "parts": [{"text": text}]}
    if task_id:
        message["taskId"] = task_id
    if context_id:
        message["contextId"] = context_id
    return {"jsonrpc": "2.0", "id": message_id, "method": method, "params": {"message": message}}


def _sse_results(resp):
    return [
        json.loads(line[5:])["result"]
        for line in resp.text.splitlines()
        if line.startswith("data:")
    ]


async def _send(client, text, message_id, **ids):
    """SendMessage; returns the task."""
    resp = await client.post("/", json=_payload("SendMessage", text, message_id, **ids), headers=V1)
    body = resp.json()
    assert "error" not in body, body
    return body["result"]["task"]


async def _send_streaming(client, text, message_id, **ids):
    """SendStreamingMessage; returns (final status, task id, context id, events)."""
    resp = await client.post(
        "/", json=_payload("SendStreamingMessage", text, message_id, **ids), headers=V1
    )
    events = _sse_results(resp)
    assert events, resp.text
    assert not any("error" in e for e in events), resp.text
    last = events[-1].get("statusUpdate") or events[-1].get("task")
    task_id = last.get("taskId") or last.get("id")
    return last["status"], task_id, last["contextId"], events


def _text(status):
    return "".join(p.get("text", "") for p in status.get("message", {}).get("parts", []))


@pytest.fixture
async def client_for():
    clients = []

    async def make(**adapter_kwargs):
        adapter = LangGraphAdapter(graph=approval_graph(), **adapter_kwargs)
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=to_a2a(adapter)), base_url="http://testserver"
        )
        clients.append(client)
        return client

    yield make
    for client in clients:
        await client.aclose()


# ──── Reporting the pause ────


@pytest.mark.parametrize("streaming", [True, False], ids=["stream-path", "invoke-path"])
async def test_send_message_reports_input_required(client_for, streaming):
    client = await client_for(streaming=streaming)

    task = await _send(client, "write a draft", "m1")

    assert task["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
    assert _text(task["status"]) == "Approve the draft?"


async def test_send_streaming_message_reports_input_required(client_for):
    client = await client_for()

    status, _, _, _ = await _send_streaming(client, "write a draft", "m1")

    assert status["state"] == "TASK_STATE_INPUT_REQUIRED"
    assert _text(status) == "Approve the draft?"


# ──── Cancelling a paused task ────


async def _cancel(client, task_id):
    resp = await client.post(
        "/",
        json={"jsonrpc": "2.0", "id": "c1", "method": "CancelTask", "params": {"id": task_id}},
        headers=V1,
    )
    body = resp.json()
    assert "error" not in body, body
    return body["result"]


async def _get(client, task_id):
    resp = await client.post(
        "/",
        json={"jsonrpc": "2.0", "id": "g1", "method": "GetTask", "params": {"id": task_id}},
        headers=V1,
    )
    return resp.json()["result"]


async def test_cancel_input_required_task(client_for):
    client = await client_for()
    task = await _send(client, "write a draft", "m1")
    assert task["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"

    canceled = await _cancel(client, task["id"])

    assert canceled["status"]["state"] == "TASK_STATE_CANCELED"
    assert (await _get(client, task["id"]))["status"]["state"] == "TASK_STATE_CANCELED"
