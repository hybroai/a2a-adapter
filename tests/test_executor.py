"""Tests for AdapterAgentExecutor — the bridge between adapters and A2A SDK."""

import asyncio
import contextlib
import uuid
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from a2a.server.agent_execution import RequestContext
from a2a.server.context import ServerCallContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    CancelTaskRequest,
    GetTaskRequest,
    Message,
    Part,
    Role,
    SendMessageRequest,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
)

from a2a_adapter.base_adapter import BaseA2AAdapter
from a2a_adapter.executor import AdapterAgentExecutor
from a2a_adapter.server import build_agent_card, to_a2a

from .conftest import (
    FailingAdapter,
    FailingStreamAdapter,
    StreamingStubAdapter,
    StubAdapter,
)


async def _collect_events(queue: EventQueue) -> list:
    """Drain all pending events from an EventQueue."""
    events = []
    # V1.0 SDK: dequeue_event() has no parameters, queue.queue is the underlying asyncio.Queue
    while not queue.queue.empty():
        try:
            event = await queue.dequeue_event()
            events.append(event)
        except (asyncio.QueueEmpty, Exception):
            break
    return events


# ── _is_empty_chunk ──────────────────────────────────────────


class TestIsEmptyChunk:
    """Regression tests for the empty-chunk filter that prevents
    empty or contentless Parts from reaching the wire.
    """

    def test_empty_string_is_empty(self):
        assert AdapterAgentExecutor._is_empty_chunk("") is True

    def test_nonempty_string_is_not_empty(self):
        assert AdapterAgentExecutor._is_empty_chunk("hello") is False

    def test_whitespace_only_is_not_empty(self):
        assert AdapterAgentExecutor._is_empty_chunk(" ") is False

    def test_empty_text_part_is_empty(self):
        part = Part(text="")
        assert AdapterAgentExecutor._is_empty_chunk(part) is True

    def test_nonempty_text_part_is_not_empty(self):
        part = Part(text="hi")
        assert AdapterAgentExecutor._is_empty_chunk(part) is False

    def test_none_is_not_empty(self):
        assert AdapterAgentExecutor._is_empty_chunk(None) is False

    def test_integer_is_not_empty(self):
        assert AdapterAgentExecutor._is_empty_chunk(42) is False

    def test_bare_part_is_empty(self):
        """Part() with no oneof content variant set should be empty."""
        assert AdapterAgentExecutor._is_empty_chunk(Part()) is True

    def test_url_part_is_not_empty(self):
        """Part with a url variant is meaningful content."""
        part = Part(url="https://example.com")
        assert AdapterAgentExecutor._is_empty_chunk(part) is False

    def test_raw_part_is_not_empty(self):
        """Part with inline binary data is meaningful content."""
        part = Part(raw=b"data")
        assert AdapterAgentExecutor._is_empty_chunk(part) is False


# ── _to_parts ────────────────────────────────────────────────


class TestToParts:
    def test_string_to_parts(self):
        parts = AdapterAgentExecutor._to_parts("hello")
        assert len(parts) == 1
        assert parts[0].text == "hello"

    def test_part_passthrough(self):
        part = Part(text="world")
        parts = AdapterAgentExecutor._to_parts(part)
        assert len(parts) == 1
        assert parts[0] is part

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="Unexpected type"):
            AdapterAgentExecutor._to_parts(12345)


# ── _concatenate_chunks ──────────────────────────────────────


class TestConcatenateChunks:
    def setup_method(self):
        self.executor = AdapterAgentExecutor(StubAdapter())

    def test_string_chunks(self):
        assert self.executor._concatenate_chunks(["a", "b", "c"]) == "abc"

    def test_part_chunks(self):
        chunks = [Part(text="x"), Part(text="y")]
        assert self.executor._concatenate_chunks(chunks) == "xy"

    def test_mixed_chunks(self):
        chunks = ["a", Part(text="b")]
        assert self.executor._concatenate_chunks(chunks) == "ab"

    def test_empty_list(self):
        assert self.executor._concatenate_chunks([]) == "[Streamed non-text content]"


# ── _extract_text_from_parts ─────────────────────────────────


class TestExtractTextFromParts:
    def setup_method(self):
        self.executor = AdapterAgentExecutor(StubAdapter())

    def test_text_parts(self):
        parts = [Part(text="foo"), Part(text="bar")]
        assert self.executor._extract_text_from_parts(parts) == "foo bar"

    def test_no_text_parts(self):
        part = Part()
        assert self.executor._extract_text_from_parts([part]) == "[Non-text response]"


# ── execute (invoke path) ────────────────────────────────────


class TestExecuteInvoke:
    @pytest.fixture
    def ctx(self):
        ctx = MagicMock(spec=RequestContext)
        ctx.task_id = "task-inv"
        ctx.context_id = "ctx-inv"
        ctx.get_user_input.return_value = "ping"
        return ctx

    async def test_invoke_emits_artifact_and_complete(self, ctx, event_queue):
        adapter = StubAdapter(response="pong")
        executor = AdapterAgentExecutor(adapter)

        await executor.execute(ctx, event_queue)

        events = await _collect_events(event_queue)
        event_types = [type(e).__name__ for e in events]
        assert "TaskStatusUpdateEvent" in event_types
        assert "TaskArtifactUpdateEvent" in event_types

    async def test_invoke_multimodal_response(self, ctx, event_queue):
        class MultimodalAdapter(BaseA2AAdapter):
            async def invoke(self, user_input, context_id=None, **kwargs):
                return [
                    Part(text="text response"),
                ]

        executor = AdapterAgentExecutor(MultimodalAdapter())
        await executor.execute(ctx, event_queue)

        events = await _collect_events(event_queue)
        event_types = [type(e).__name__ for e in events]
        assert "TaskArtifactUpdateEvent" in event_types

    async def test_invoke_error_emits_failed(self, ctx, event_queue):
        executor = AdapterAgentExecutor(FailingAdapter())

        await executor.execute(ctx, event_queue)

        events = await _collect_events(event_queue)
        status_events = [e for e in events if type(e).__name__ == "TaskStatusUpdateEvent"]
        assert any(hasattr(e, "status") and getattr(e.status, "state", None) for e in status_events)


# ── execute (streaming path) ─────────────────────────────────


class TestExecuteStreaming:
    @pytest.fixture
    def ctx(self):
        ctx = MagicMock(spec=RequestContext)
        ctx.task_id = "task-stream"
        ctx.context_id = "ctx-stream"
        ctx.get_user_input.return_value = "hi"
        return ctx

    async def test_streaming_emits_incremental_artifacts(self, ctx, event_queue):
        adapter = StreamingStubAdapter(["Hello", ", ", "world!"])
        executor = AdapterAgentExecutor(adapter)

        await executor.execute(ctx, event_queue)

        events = await _collect_events(event_queue)
        artifact_events = [e for e in events if type(e).__name__ == "TaskArtifactUpdateEvent"]
        assert len(artifact_events) == 3

    async def test_streaming_filters_empty_chunks(self, ctx, event_queue):
        """Empty string chunks must be silently dropped."""
        adapter = StreamingStubAdapter(["Hello", "", "", "world"])
        executor = AdapterAgentExecutor(adapter)

        await executor.execute(ctx, event_queue)

        events = await _collect_events(event_queue)
        artifact_events = [e for e in events if type(e).__name__ == "TaskArtifactUpdateEvent"]
        assert len(artifact_events) == 2

    async def test_streaming_filters_empty_text_parts(self, ctx, event_queue):
        """Part(text="") must also be filtered."""
        chunks = [
            "Hello",
            Part(text=""),
            "world",
        ]
        adapter = StreamingStubAdapter(chunks)
        executor = AdapterAgentExecutor(adapter)

        await executor.execute(ctx, event_queue)

        events = await _collect_events(event_queue)
        artifact_events = [e for e in events if type(e).__name__ == "TaskArtifactUpdateEvent"]
        assert len(artifact_events) == 2

    async def test_streaming_all_empty_produces_no_artifacts(self, ctx, event_queue):
        """A stream of only empty chunks should still complete gracefully."""
        adapter = StreamingStubAdapter(["", ""])
        executor = AdapterAgentExecutor(adapter)

        await executor.execute(ctx, event_queue)

        events = await _collect_events(event_queue)
        artifact_events = [e for e in events if type(e).__name__ == "TaskArtifactUpdateEvent"]
        assert len(artifact_events) == 0

        status_events = [e for e in events if type(e).__name__ == "TaskStatusUpdateEvent"]
        assert len(status_events) >= 1

    async def test_streaming_single_chunk(self, ctx, event_queue):
        adapter = StreamingStubAdapter(["only"])
        executor = AdapterAgentExecutor(adapter)

        await executor.execute(ctx, event_queue)

        events = await _collect_events(event_queue)
        artifact_events = [e for e in events if type(e).__name__ == "TaskArtifactUpdateEvent"]
        assert len(artifact_events) == 1

    async def test_streaming_part_objects(self, ctx, event_queue):
        chunks = [
            Part(text="foo"),
            Part(text="bar"),
        ]
        adapter = StreamingStubAdapter(chunks)
        executor = AdapterAgentExecutor(adapter)

        await executor.execute(ctx, event_queue)

        events = await _collect_events(event_queue)
        artifact_events = [e for e in events if type(e).__name__ == "TaskArtifactUpdateEvent"]
        assert len(artifact_events) == 2

    async def test_streaming_failure_closes_artifact(self, ctx, event_queue):
        """When the stream raises after yielding chunks, the open artifact
        must be closed (last_chunk=True) and the task must enter failed state."""
        adapter = FailingStreamAdapter(["Hello", " world"])
        executor = AdapterAgentExecutor(adapter)

        await executor.execute(ctx, event_queue)

        events = await _collect_events(event_queue)
        artifact_events = [e for e in events if type(e).__name__ == "TaskArtifactUpdateEvent"]
        assert len(artifact_events) >= 1

        status_events = [e for e in events if type(e).__name__ == "TaskStatusUpdateEvent"]
        states = [
            getattr(e.status, "state", None)
            for e in status_events
            if hasattr(e, "status") and hasattr(getattr(e, "status", None), "state")
        ]
        assert TaskState.TASK_STATE_FAILED in states

    async def test_streaming_failure_flushes_held_chunk(self, ctx, event_queue):
        """On failure the held-back chunk is sent once to close the artifact;
        no earlier chunk is duplicated."""
        adapter = FailingStreamAdapter(["A", "B", "C"])
        executor = AdapterAgentExecutor(adapter)

        await executor.execute(ctx, event_queue)

        events = await _collect_events(event_queue)
        artifact_events = [e for e in events if type(e).__name__ == "TaskArtifactUpdateEvent"]
        assert [e.artifact.parts[0].text for e in artifact_events] == ["A", "B", "C"]
        assert [e.append for e in artifact_events] == [False, True, True]
        assert [e.last_chunk for e in artifact_events] == [False, False, True]
        assert len({e.artifact.artifact_id for e in artifact_events}) == 1

    async def test_streaming_failure_after_single_chunk(self, ctx, event_queue):
        """A single chunk followed by a failure is delivered as a closed
        artifact instead of being dropped."""
        adapter = FailingStreamAdapter(["only"])
        executor = AdapterAgentExecutor(adapter)

        await executor.execute(ctx, event_queue)

        events = await _collect_events(event_queue)
        artifact_events = [e for e in events if type(e).__name__ == "TaskArtifactUpdateEvent"]
        assert [e.artifact.parts[0].text for e in artifact_events] == ["only"]
        assert artifact_events[0].append is False
        assert artifact_events[0].last_chunk is True

    async def test_streaming_failure_no_chunks_still_fails(self, ctx, event_queue):
        """When stream() raises immediately (no chunks emitted),
        the task should enter failed state without artifact cleanup."""
        adapter = FailingStreamAdapter([])
        executor = AdapterAgentExecutor(adapter)

        await executor.execute(ctx, event_queue)

        events = await _collect_events(event_queue)
        artifact_events = [e for e in events if type(e).__name__ == "TaskArtifactUpdateEvent"]
        assert len(artifact_events) == 0

        status_events = [e for e in events if type(e).__name__ == "TaskStatusUpdateEvent"]
        states = [
            getattr(e.status, "state", None)
            for e in status_events
            if hasattr(e, "status") and hasattr(getattr(e, "status", None), "state")
        ]
        assert TaskState.TASK_STATE_FAILED in states


# ── cancel ────────────────────────────────────────────────────


class TestCancel:
    async def test_cancel_delegates_to_adapter(self, request_context, event_queue):
        adapter = StubAdapter()
        adapter.cancel = AsyncMock()
        executor = AdapterAgentExecutor(adapter)

        await executor.cancel(request_context, event_queue)
        adapter.cancel.assert_awaited_once()

    async def test_cancel_emits_canceled(self, request_context, event_queue):
        adapter = StubAdapter()
        adapter.cancel = AsyncMock()
        observer = MagicMock()
        executor = AdapterAgentExecutor(adapter, observer=observer)

        await executor.cancel(request_context, event_queue)

        events = await _collect_events(event_queue)
        assert [e.status.state for e in events] == [TaskState.TASK_STATE_CANCELED]
        observer.task_canceled.assert_called_once()

    async def test_a_failed_adapter_cancel_is_not_reported_canceled(
        self, request_context, event_queue
    ):
        """The work may still be running, so the task is marked failed, once,
        by the executor; the error does not propagate to the SDK."""
        adapter = StubAdapter()
        adapter.cancel = AsyncMock(side_effect=RuntimeError("cancel failed"))
        observer = MagicMock()
        executor = AdapterAgentExecutor(adapter, observer=observer)

        await executor.cancel(request_context, event_queue)

        events = await _collect_events(event_queue)
        assert [e.status.state for e in events] == [TaskState.TASK_STATE_FAILED]
        assert events[0].status.message.parts[0].text == "Cancel failed: cancel failed"
        observer.task_canceled.assert_not_called()
        assert "cancel failed" in observer.task_failed.call_args.args[2]

    async def test_an_interrupted_execute_waits_for_the_cancel(self, request_context, event_queue):
        """The SDK cancels execute() before it calls cancel() and closes the
        queue when execute() returns, so execute() must outlast cancel()."""
        started, release = asyncio.Event(), asyncio.Event()

        class Adapter(StubAdapter):
            async def invoke(self, user_input, context_id=None, **kwargs):
                started.set()
                await asyncio.Event().wait()

            async def cancel(self, **kwargs):
                await release.wait()

        executor = AdapterAgentExecutor(Adapter())
        run = asyncio.create_task(executor.execute(request_context, event_queue))
        await started.wait()

        async def sdk_cancel():  # ActiveTask.cancel: cancel the run, then call cancel()
            run.cancel()
            await executor.cancel(request_context, event_queue)

        cancel = asyncio.create_task(sdk_cancel())
        await asyncio.sleep(0.05)
        assert not run.done()

        release.set()
        await cancel
        with pytest.raises(asyncio.CancelledError):
            await run
        events = await _collect_events(event_queue)
        states = [e.status.state for e in events if e.HasField("status")]
        assert states[-1] == TaskState.TASK_STATE_CANCELED


class TestCancelOverJsonRpc:
    """tasks/cancel through the SDK request handler."""

    class Hanging(BaseA2AAdapter):
        def __init__(self, cancel_error=None, cancel_awaits=False):
            self.started = asyncio.Event()
            self.cancel_error = cancel_error
            self.cancel_awaits = cancel_awaits

        async def invoke(self, user_input, context_id=None, **kwargs):
            self.started.set()
            await asyncio.Event().wait()

        async def cancel(self, context_id=None, **kwargs):
            if self.cancel_awaits:  # e.g. a round trip to the robot
                await asyncio.sleep(0.05)
            if self.cancel_error:
                raise self.cancel_error

    async def _cancel(self, adapter):
        app = to_a2a(adapter)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://agent"
        ) as client:

            async def rpc(method, params):
                body = {
                    "jsonrpc": "2.0",
                    "id": uuid.uuid4().hex,
                    "method": method,
                    "params": params,
                }
                return (await client.post("/", json=body)).json()

            message = {
                "role": "user",
                "messageId": uuid.uuid4().hex,
                "parts": [{"kind": "text", "text": "hi"}],
            }
            sent = await rpc(
                "message/send", {"message": message, "configuration": {"blocking": False}}
            )
            task_id = sent["result"]["id"]
            await asyncio.wait_for(adapter.started.wait(), 2)
            cancelled = await rpc("tasks/cancel", {"id": task_id})
            for _ in range(100):  # the failed state is stored after the cancel reply
                task = (await rpc("tasks/get", {"id": task_id}))["result"]
                if task["status"]["state"] in ("canceled", "failed"):
                    break
                await asyncio.sleep(0.01)
            return cancelled, task

    @pytest.mark.parametrize("cancel_awaits", [False, True])
    async def test_cancel_reports_the_task_canceled(self, cancel_awaits):
        cancelled, task = await self._cancel(self.Hanging(cancel_awaits=cancel_awaits))
        assert cancelled["result"]["status"]["state"] == "canceled"
        assert task["status"]["state"] == "canceled"

    @pytest.mark.parametrize("cancel_awaits", [False, True])
    async def test_a_failed_adapter_cancel_fails_the_task(self, cancel_awaits):
        adapter = self.Hanging(RuntimeError("robot did not stop"), cancel_awaits=cancel_awaits)
        cancelled, task = await self._cancel(adapter)
        assert cancelled["result"]["status"]["state"] == "failed"
        assert task["status"]["state"] == "failed"
        texts = [part["text"] for part in task["status"]["message"]["parts"]]
        assert texts == ["Cancel failed: robot did not stop"]


# ── tasks/cancel through the real request handler ─────────────

SUBMITTED = TaskState.TASK_STATE_SUBMITTED
WORKING = TaskState.TASK_STATE_WORKING
CANCELED = TaskState.TASK_STATE_CANCELED
FAILED = TaskState.TASK_STATE_FAILED
COMPLETED = TaskState.TASK_STATE_COMPLETED


def _status_text(status) -> str | None:
    if not status.HasField("message"):
        return None
    return "".join(part.text for part in status.message.parts)


def _describe(payload) -> tuple:
    """A stream payload as a comparable tuple, read from public SDK types."""
    if isinstance(payload, Task):
        return ("task", payload.status.state)
    if isinstance(payload, TaskStatusUpdateEvent):
        return ("status", payload.status.state, _status_text(payload.status))
    if isinstance(payload, TaskArtifactUpdateEvent):
        text = "".join(part.text for part in payload.artifact.parts)
        return ("artifact", text, payload.append, payload.last_chunk)
    return (type(payload).__name__,)


def _states(payloads) -> list:
    """Status-carrying payloads as (kind, state) pairs, artifacts as ("artifact",)."""
    return [d[:2] if d[0] != "artifact" else d[:1] for d in map(_describe, payloads)]


async def _run_and_cancel(
    adapter,
    *,
    observer=None,
    payloads_before_cancel: int = 2,
    wait_stream: bool = True,
    cancel_timeout: float = 5,
):
    """Stream a message through DefaultRequestHandler, then cancel the task.

    Returns ``(cancel result or exception, stream payloads, stream task,
    stored task)``. With ``wait_stream`` the stream is awaited first (for
    runs that unwind); otherwise the stored task is read right after the
    cancel, while a stuck run keeps the stream open.
    """
    executor = AdapterAgentExecutor(adapter, observer)
    handler = DefaultRequestHandler(
        executor, InMemoryTaskStore(), build_agent_card(adapter, streaming=True)
    )
    call = ServerCallContext()
    message = Message(role=Role.ROLE_USER, message_id=uuid.uuid4().hex, parts=[Part(text="hi")])
    payloads: list = []

    async def consume():
        async for payload in handler.on_message_send_stream(
            SendMessageRequest(message=message), call
        ):
            payloads.append(payload)

    stream = asyncio.create_task(consume())
    await asyncio.wait_for(adapter.started.wait(), 5)
    for _ in range(500):
        if len(payloads) >= payloads_before_cancel:
            break
        await asyncio.sleep(0.01)
    task_id = payloads[0].id
    try:
        result = await asyncio.wait_for(
            handler.on_cancel_task(CancelTaskRequest(id=task_id), call), cancel_timeout
        )
    except Exception as e:  # the old behaviour: the cancel RPC raised
        result = e
    if wait_stream:
        await asyncio.wait_for(asyncio.shield(stream), 5)
    stored = await handler.on_get_task(GetTaskRequest(id=task_id), call)
    return result, payloads, stream, stored


class Stubborn(BaseA2AAdapter):
    """invoke() ignores CancelledError until the test sets ``release``."""

    def __init__(self, cancel_error: Exception | None = None):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancel_error = cancel_error

    async def invoke(self, user_input, context_id=None, **kwargs):
        self.started.set()
        while not self.release.is_set():
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                continue
        return "late"

    async def cancel(self, context_id=None, **kwargs):
        if self.cancel_error is not None:
            raise self.cancel_error


async def _finish(adapter: Stubborn, stream: asyncio.Task) -> None:
    """Let a stubborn run finish and the stream end, whatever the test saw."""
    adapter.release.set()
    with contextlib.suppress(Exception, asyncio.CancelledError):
        await asyncio.wait_for(asyncio.shield(stream), 5)


class TestCancelThroughHandler:
    async def test_a_failed_adapter_cancel_sends_one_failed(self):
        adapter = Stubborn(RuntimeError("robot did not stop"))
        observer = MagicMock()
        result, payloads, stream, stored = await _run_and_cancel(
            adapter, observer=observer, wait_stream=False
        )
        try:
            assert isinstance(result, Task), result
            assert result.status.state == FAILED
            assert stored.status.state == FAILED
            assert _status_text(stored.status) == "Cancel failed: robot did not stop"
            observer.task_failed.assert_called_once()
            observer.task_canceled.assert_not_called()
        finally:
            await _finish(adapter, stream)
        assert stream.exception() is None
        assert [_describe(p) for p in payloads[:3]] == [
            ("task", SUBMITTED),
            ("status", WORKING, None),
            ("status", FAILED, "Cancel failed: robot did not stop"),
        ]
