"""Full-stack tests for the activity rail: client → middleware → executor → rows.

Runs the real Starlette app in-process through httpx's ASGI transport with
fake adapters, so no agent CLI, model or network is involved.
"""

import asyncio
import io
import json
import logging
import re

import httpx
import pytest

from a2a_adapter import AdapterMetadata, BaseA2AAdapter, to_a2a
from a2a_adapter.executor import AdapterAgentExecutor
from a2a_adapter.terminal import ActivityLog, RailLogHandler, Verbosity

ANSI = re.compile(r"\x1b\[[0-9;]*m")


class EchoAdapter(BaseA2AAdapter):
    async def invoke(self, user_input, context_id=None, **kwargs):
        return f"Echo: {user_input}"

    def get_metadata(self):
        return AdapterMetadata(name="Echo Agent")


class StreamAdapter(BaseA2AAdapter):
    async def invoke(self, user_input, context_id=None, **kwargs):
        return user_input

    async def stream(self, user_input, context_id=None, **kwargs):
        for word in user_input.split():
            yield word + " "

    def supports_streaming(self):
        return True


class FailAdapter(BaseA2AAdapter):
    async def invoke(self, user_input, context_id=None, **kwargs):
        raise RuntimeError("intentional failure")


def _payload(method, text="Hello", **params):
    params = params or {
        "message": {
            "role": "user",
            "messageId": "msg-1",
            "parts": [{"kind": "text", "text": text}],
        }
    }
    return {"jsonrpc": "2.0", "id": "req-1", "method": method, "params": params}


def _rows(stream):
    return [ANSI.sub("", line) for line in stream.getvalue().splitlines()]


def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")


def _app(adapter, verbosity=Verbosity.NORMAL):
    stream = io.StringIO()
    log = ActivityLog(stream, color=False, verbosity=verbosity, width=200)
    return to_a2a(adapter, activity=log), stream


async def _settle():
    # Rows from the executor's own task can land just after the response.
    for _ in range(20):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_agent_card_fetch_is_one_row():
    app, stream = _app(EchoAdapter())
    async with _client(app) as client:
        resp = await client.get("/.well-known/agent-card.json")
    assert resp.status_code == 200
    (row,) = _rows(stream)
    assert "← agent-card" in row
    assert "GET /.well-known/agent-card.json" in row


@pytest.mark.asyncio
async def test_message_send_rows_follow_the_task():
    app, stream = _app(EchoAdapter())
    async with _client(app) as client:
        resp = await client.post("/", json=_payload("message/send", "Hi there"))
        await _settle()
    assert resp.json()["result"]["status"]["state"] == "completed"
    rows = _rows(stream)
    assert len(rows) == 3, rows
    assert "← message/send" in rows[0] and "○ submitted" in rows[0]
    assert '"Hi there"' in rows[0]
    assert "◐ working" in rows[1]
    assert "→ message/send" in rows[2] and "✓ completed" in rows[2]
    assert '"Echo: Hi there"' in rows[2]


@pytest.mark.asyncio
async def test_message_stream_counts_chunks():
    app, stream = _app(StreamAdapter())
    async with _client(app) as client:
        resp = await client.post("/", json=_payload("message/stream", "one two three"))
        await _settle()
    assert "text/event-stream" in resp.headers["content-type"]
    rows = _rows(stream)
    assert "← message/stream" in rows[0]
    completed = next(row for row in rows if "completed" in row)
    assert "→ message/stream" in completed
    assert '3 chunks · "one two three"' in completed


@pytest.mark.asyncio
async def test_failed_task_row_and_no_duplicate_error_log():
    stream = io.StringIO()
    log = ActivityLog(stream, color=False, width=200)
    app = to_a2a(FailAdapter(), activity=log)
    handler = RailLogHandler(log)
    executor_logger = logging.getLogger("a2a_adapter.executor")
    executor_logger.addHandler(handler)
    try:
        async with _client(app) as client:
            resp = await client.post("/", json=_payload("message/send"))
            await _settle()
    finally:
        executor_logger.removeHandler(handler)
    assert resp.json()["result"]["status"]["state"] == "failed"
    rows = _rows(stream)
    failed = [row for row in rows if "intentional failure" in row]
    assert len(failed) == 1, rows
    assert "✗ message/send" in failed[0] and "✗ failed" in failed[0]
    assert "RuntimeError: intentional failure" in failed[0]


@pytest.mark.asyncio
async def test_tasks_get_shows_the_task_state():
    app, stream = _app(EchoAdapter())
    async with _client(app) as client:
        sent = await client.post("/", json=_payload("message/send"))
        task_id = sent.json()["result"]["id"]
        await _settle()
        await client.post("/", json=_payload("tasks/get", id=task_id))
    row = _rows(stream)[-1]
    assert "← tasks/get" in row
    assert task_id[:4] in row
    assert "✓ completed" in row


@pytest.mark.asyncio
async def test_rest_send_is_named_like_jsonrpc():
    app, stream = _app(EchoAdapter())
    body = {"message": {"role": "ROLE_USER", "messageId": "m1", "parts": [{"text": "rest hi"}]}}
    async with _client(app) as client:
        resp = await client.post("/message:send", json=body, headers={"A2A-Version": "1.0"})
        await _settle()
    assert resp.status_code == 200, resp.text
    rows = _rows(stream)
    assert "← message/send" in rows[0] and '"rest hi"' in rows[0]


@pytest.mark.asyncio
async def test_jsonrpc_error_is_an_error_row():
    app, stream = _app(EchoAdapter())
    async with _client(app) as client:
        await client.post("/", json=_payload("tasks/get", id="does-not-exist"))
    (row,) = _rows(stream)
    assert row.lstrip().split()[1] == "✗"
    assert "tasks/get" in row
    assert "not found" in row.lower()


@pytest.mark.asyncio
async def test_quiet_only_reports_failures():
    app, stream = _app(EchoAdapter(), verbosity=Verbosity.QUIET)
    async with _client(app) as client:
        await client.get("/.well-known/agent-card.json")
        await client.post("/", json=_payload("message/send"))
        await client.post("/", json=_payload("tasks/get", id="missing"))
        await _settle()
    rows = _rows(stream)
    assert len(rows) == 1 and "tasks/get" in rows[0]


@pytest.mark.asyncio
async def test_debug_prints_request_and_response_payloads():
    app, stream = _app(EchoAdapter(), verbosity=Verbosity.DEBUG)
    async with _client(app) as client:
        await client.post("/", json=_payload("message/send", "debug me"))
        await _settle()
    text = "\n".join(_rows(stream))
    assert "request" in text and "response" in text
    assert '"method": "message/send"' in text
    assert '"state": "completed"' in text


@pytest.mark.asyncio
async def test_debug_prints_stream_events():
    app, stream = _app(StreamAdapter(), verbosity=Verbosity.DEBUG)
    async with _client(app) as client:
        await client.post("/", json=_payload("message/stream", "a b"))
        await _settle()
    text = "\n".join(_rows(stream))
    assert text.count("event") >= 2


@pytest.mark.asyncio
async def test_observer_errors_never_break_the_task():
    class Broken:
        def __getattr__(self, name):
            def fail(*args, **kwargs):
                raise RuntimeError("display broke")

            return fail

    from a2a.server.agent_execution import RequestContext
    from a2a.server.context import ServerCallContext
    from a2a.server.events import EventQueue
    from a2a.types import Message, Part, Role, SendMessageRequest, TaskState

    executor = AdapterAgentExecutor(EchoAdapter(), observer=Broken())
    message = Message(role=Role.ROLE_USER, message_id="m", parts=[Part(text="hi")])
    context = RequestContext(
        call_context=ServerCallContext(),
        request=SendMessageRequest(message=message),
        task_id="t1",
        context_id="c1",
    )
    queue = EventQueue()
    await executor.execute(context, queue)
    events = []
    while not queue.queue.empty():
        events.append(await queue.dequeue_event())
    final = events[-1]
    assert final.status.state == TaskState.TASK_STATE_COMPLETED


def test_to_a2a_without_activity_is_unchanged():
    app = to_a2a(EchoAdapter())
    assert not any(m.cls.__name__ == "ActivityMiddleware" for m in app.user_middleware)


def test_debug_payload_json_is_pretty():
    stream = io.StringIO()
    log = ActivityLog(stream, color=False, verbosity=Verbosity.DEBUG, width=120)
    log.payload("request", json.dumps({"a": 1, "b": [1, 2]}).encode())
    lines = _rows(stream)
    assert any('"a": 1' in line for line in lines)
    assert all(line.startswith(" " * 14) for line in lines[1:])


@pytest.mark.asyncio
async def test_debug_response_payload_follows_its_row():
    app, stream = _app(EchoAdapter(), verbosity=Verbosity.DEBUG)
    async with _client(app) as client:
        await client.get("/.well-known/agent-card.json")
    rows = _rows(stream)
    row = next(i for i, line in enumerate(rows) if "← agent-card" in line)
    assert row < rows.index(" " * 14 + "response")


# ── Review round 2 ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_protocol_errors_print_once_through_the_real_sdk_logs():
    from a2a_adapter.terminal import TerminalDisplay

    stream = io.StringIO()
    display = TerminalDisplay(None, stream=stream, color=False, width=200, version="1")
    app = to_a2a(EchoAdapter(), activity=display.activity)
    display.install_logging("warning")
    try:
        async with _client(app) as client:
            await client.get("/tasks/nope", headers={"A2A-Version": "1.0"})
            await client.post("/", json=_payload("tasks/get", id="nope"))
    finally:
        display.uninstall_logging()
    rows = [row for row in _rows(stream) if "not found" in row.lower()]
    assert len(rows) == 2, rows
    assert all("✗ tasks/get" in row for row in rows)


@pytest.mark.asyncio
async def test_follow_up_message_to_an_active_task_gets_its_own_rows():
    class Slow(BaseA2AAdapter):
        async def invoke(self, user_input, context_id=None, **kwargs):
            await asyncio.sleep(0.2)
            return "R " + user_input

    def message(text, **extra):
        return {
            "role": "user",
            "messageId": text,
            "parts": [{"kind": "text", "text": text}],
            **extra,
        }

    app, stream = _app(Slow())
    log = app.user_middleware[0].kwargs["log"]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t", timeout=10
    ) as client:
        first = asyncio.create_task(
            client.post(
                "/",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "message/send",
                    "params": {"message": message("one")},
                },
            )
        )
        for _ in range(100):
            await asyncio.sleep(0.01)
            if log._task_starts:
                break
        task_id = next(iter(log._task_starts))
        await client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "message/send",
                "params": {"message": message("two", taskId=task_id)},
            },
        )
        await first
        await asyncio.sleep(0.3)
    rows = _rows(stream)
    # The SDK answers the follow-up at once with the task as it stands and
    # runs the queued message afterwards: one request row for that answer,
    # then the follow-up's own task rows, each printed once.
    answered = [row for row in rows if "← message/send" in row and "submitted" not in row]
    assert len(answered) == 1, rows
    assert sum('"two"' in row for row in rows) == 1, rows
    assert sum('"R two"' in row and "→ message/send" in row for row in rows) == 1, rows
    assert sum('"R one"' in row for row in rows) == 1, rows


# ── Review round 3 ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_streaming_to_a_non_streaming_agent_prints_one_error():
    from a2a_adapter.terminal import TerminalDisplay

    stream = io.StringIO()
    display = TerminalDisplay(None, stream=stream, color=False, width=200, version="1")
    app = to_a2a(EchoAdapter(), activity=display.activity)
    display.install_logging("warning")
    try:
        async with _client(app) as client:
            await client.post("/", json=_payload("message/stream", "hi"))
    finally:
        display.uninstall_logging()
    rows = [row for row in _rows(stream) if "not supported" in row.lower()]
    assert len(rows) == 1, _rows(stream)
    assert "✗ message/stream" in rows[0]


@pytest.mark.asyncio
async def test_follow_up_after_its_request_finished_keeps_its_own_label():
    from a2a_adapter.terminal import TerminalDisplay

    class Slow(StreamAdapter):
        async def stream(self, user_input, context_id=None, **kwargs):
            await asyncio.sleep(0.2)
            yield "R " + user_input

    def message(text, **extra):
        return {
            "role": "user",
            "messageId": text,
            "parts": [{"kind": "text", "text": text}],
            **extra,
        }

    stream = io.StringIO()
    display = TerminalDisplay(None, stream=stream, color=False, width=200, version="1")
    app = to_a2a(Slow(), activity=display.activity)
    log = display.activity
    display.install_logging("warning")
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t", timeout=10
        ) as client:
            first = asyncio.create_task(
                client.post(
                    "/",
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "message/stream",
                        "params": {"message": message("one")},
                    },
                )
            )
            for _ in range(100):
                await asyncio.sleep(0.01)
                if log._task_starts:
                    break
            task_id = next(iter(log._task_starts))
            await client.post(
                "/",
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "message/send",
                    "params": {"message": message("two", taskId=task_id)},
                },
            )
            await first
            await asyncio.sleep(0.4)
    finally:
        display.uninstall_logging()
    rows = _rows(stream)
    two = [row for row in rows if '"two"' in row or '"R two"' in row]
    assert len(two) == 2, rows
    assert all("message/send" in row for row in two), rows
    # A follow-up to an existing task is not an error.
    assert not any("already exists" in row for row in rows), rows
    assert log._requests == {} and log._task_requests == {}
