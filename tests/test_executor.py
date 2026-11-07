"""Tests for AdapterAgentExecutor — the bridge between adapters and A2A SDK."""

import asyncio
import contextlib
import os
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

from a2a_adapter import executor as executor_module
from a2a_adapter.base_adapter import BaseA2AAdapter
from a2a_adapter.executor import AdapterAgentExecutor
from a2a_adapter.server import build_agent_card, to_a2a

from . import process_helpers
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

    async def test_a_failed_cancel_with_no_run_reports_canceled(
        self, request_context, event_queue, caplog
    ):
        """No execute() is in flight, so nothing of ours is running: the
        task is canceled once, and the adapter error is only logged."""
        adapter = StubAdapter()
        adapter.cancel = AsyncMock(side_effect=RuntimeError("cancel failed"))
        observer = MagicMock()
        executor = AdapterAgentExecutor(adapter, observer=observer)

        with caplog.at_level("WARNING", logger="a2a_adapter.executor"):
            await executor.cancel(request_context, event_queue)

        events = await _collect_events(event_queue)
        assert [e.status.state for e in events] == [TaskState.TASK_STATE_CANCELED]
        observer.task_canceled.assert_called_once()
        observer.task_failed.assert_not_called()
        assert "cancel failed" in caplog.text

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
    async def test_a_failed_cancel_of_a_run_that_stopped_reports_canceled(self, cancel_awaits):
        adapter = self.Hanging(RuntimeError("already gone"), cancel_awaits=cancel_awaits)
        cancelled, task = await self._cancel(adapter)
        assert cancelled["result"]["status"]["state"] == "canceled"
        assert task["status"]["state"] == "canceled"

    async def test_a_failed_cancel_of_a_stuck_run_fails_the_task(self, monkeypatch):
        monkeypatch.setattr(executor_module, "_CANCEL_SETTLE_TIMEOUT", 0.05)
        adapter = Stubborn(RuntimeError("robot did not stop"))
        try:
            cancelled, task = await self._cancel(adapter)
        finally:
            adapter.release.set()
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
    ready=None,
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
    await asyncio.wait_for(ready() if ready else adapter.started.wait(), 5)
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
    async def test_a_failed_adapter_cancel_sends_one_failed(self, monkeypatch):
        monkeypatch.setattr(executor_module, "_CANCEL_SETTLE_TIMEOUT", 0.05)
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
        # The late result is dropped: no artifact or completed after failed.
        assert [_describe(p) for p in payloads] == [
            ("task", SUBMITTED),
            ("status", WORKING, None),
            ("status", FAILED, "Cancel failed: robot did not stop"),
        ]
        observer.task_completed.assert_not_called()

    async def test_an_error_raised_while_cancelling_does_not_fail_the_task(self):
        class Converts(BaseA2AAdapter):
            def __init__(self):
                self.started = asyncio.Event()

            async def invoke(self, user_input, context_id=None, **kwargs):
                self.started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    raise RuntimeError("converted") from None

        observer = MagicMock()
        result, payloads, _, stored = await _run_and_cancel(Converts(), observer=observer)

        assert result.status.state == CANCELED
        assert _states(payloads) == [("task", SUBMITTED), ("status", WORKING), ("status", CANCELED)]
        assert stored.status.state == CANCELED
        observer.task_failed.assert_not_called()

    async def test_an_error_raised_while_cancelling_a_stream_sends_nothing_after_canceled(self):
        class ConvertsStream(BaseA2AAdapter):
            def __init__(self):
                self.started = asyncio.Event()

            async def invoke(self, user_input, context_id=None, **kwargs):
                raise NotImplementedError

            async def stream(self, user_input, context_id=None, **kwargs):
                yield "a"
                yield "b"
                self.started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    raise RuntimeError("converted") from None

        result, payloads, _, stored = await _run_and_cancel(
            ConvertsStream(), payloads_before_cancel=3
        )

        assert result.status.state == CANCELED
        # The artifact stays open: nothing may follow the terminal state.
        assert [_describe(p) for p in payloads] == [
            ("task", SUBMITTED),
            ("status", WORKING, None),
            ("artifact", "a", False, False),
            ("status", CANCELED, None),
        ]
        assert stored.status.state == CANCELED

    async def test_a_result_returned_while_cancelling_is_dropped(self):
        class Swallows(BaseA2AAdapter):
            def __init__(self):
                self.started = asyncio.Event()

            async def invoke(self, user_input, context_id=None, **kwargs):
                self.started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    return "late"

        observer = MagicMock()
        result, payloads, _, stored = await _run_and_cancel(Swallows(), observer=observer)

        assert result.status.state == CANCELED
        assert _states(payloads) == [("task", SUBMITTED), ("status", WORKING), ("status", CANCELED)]
        assert stored.status.state == CANCELED
        assert not stored.artifacts
        observer.task_completed.assert_not_called()

    @pytest.mark.parametrize("cancel_awaits", [False, True])
    async def test_a_failed_cancel_after_the_run_stopped_reports_canceled(
        self, cancel_awaits, caplog
    ):
        class Unwinds(BaseA2AAdapter):
            def __init__(self):
                self.started = asyncio.Event()

            async def invoke(self, user_input, context_id=None, **kwargs):
                self.started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    await asyncio.sleep(0.01)  # cleanup before unwinding
                    raise

            async def cancel(self, context_id=None, **kwargs):
                if cancel_awaits:  # e.g. a remote "stop" call
                    await asyncio.sleep(0.01)
                raise RuntimeError("already gone")

        observer = MagicMock()
        with caplog.at_level("WARNING", logger="a2a_adapter.executor"):
            result, payloads, _, stored = await _run_and_cancel(Unwinds(), observer=observer)

        assert result.status.state == CANCELED
        assert _states(payloads) == [("task", SUBMITTED), ("status", WORKING), ("status", CANCELED)]
        assert stored.status.state == CANCELED
        observer.task_canceled.assert_called_once()
        observer.task_failed.assert_not_called()
        assert "already gone" in caplog.text

    @pytest.mark.skipif(not hasattr(os, "killpg"), reason="POSIX process groups")
    async def test_a_failed_cancel_of_a_cli_run_reports_canceled(self, tmp_path):
        class RaisingCli(process_helpers.TreeCliAdapter):
            async def cancel(self, context_id=None, **kwargs):
                await super().cancel(context_id, **kwargs)
                raise RuntimeError("cancel bookkeeping failed")

        adapter = RaisingCli(tmp_path)
        pids: list[int] = []

        async def ready():
            pids.extend(await process_helpers.wait_for_pids(tmp_path))

        try:
            result, payloads, _, stored = await _run_and_cancel(adapter, ready=ready)
            assert result.status.state == CANCELED
            assert _states(payloads) == [
                ("task", SUBMITTED),
                ("status", WORKING),
                ("status", CANCELED),
            ]
            assert stored.status.state == CANCELED
            sh_pid, grandchild_pid = pids
            assert await process_helpers.wait_dead(sh_pid)
            assert await process_helpers.wait_dead(grandchild_pid)
        finally:
            process_helpers.kill_pids(tmp_path)

    async def test_a_cancelled_cancel_still_sends_a_terminal_state(self, monkeypatch):
        """The cancel RPC itself is cancelled during the settle wait (e.g. a
        client disconnect): the decision and its enqueue still finish."""
        monkeypatch.setattr(executor_module, "_CANCEL_SETTLE_TIMEOUT", 0.5)
        adapter = Stubborn(RuntimeError("robot did not stop"))
        result, payloads, stream, _ = await _run_and_cancel(
            adapter, wait_stream=False, cancel_timeout=0.1
        )
        try:
            assert isinstance(result, TimeoutError), result
            for _ in range(300):
                if any(_describe(p)[:2] == ("status", FAILED) for p in payloads):
                    break
                await asyncio.sleep(0.01)
        finally:
            await _finish(adapter, stream)
        assert [_describe(p) for p in payloads] == [
            ("task", SUBMITTED),
            ("status", WORKING, None),
            ("status", FAILED, "Cancel failed: robot did not stop"),
        ]

    @pytest.mark.parametrize("cancel_error", [None, RuntimeError("robot did not stop")])
    async def test_a_stream_that_ignores_the_cancel_sends_nothing_after_it(
        self, cancel_error, monkeypatch
    ):
        monkeypatch.setattr(executor_module, "_CANCEL_SETTLE_TIMEOUT", 0.05)

        class StubbornStream(BaseA2AAdapter):
            def __init__(self):
                self.started = asyncio.Event()
                self.release = asyncio.Event()

            async def invoke(self, user_input, context_id=None, **kwargs):
                raise NotImplementedError

            async def stream(self, user_input, context_id=None, **kwargs):
                yield "a"
                self.started.set()
                while not self.release.is_set():
                    try:
                        await self.release.wait()
                    except asyncio.CancelledError:
                        continue
                yield "late1"
                yield "late2"

            async def cancel(self, context_id=None, **kwargs):
                if cancel_error is not None:
                    raise cancel_error

        adapter = StubbornStream()
        result, payloads, stream, _ = await _run_and_cancel(adapter, wait_stream=False)
        try:
            expected = CANCELED if cancel_error is None else FAILED
            assert result.status.state == expected
        finally:
            await _finish(adapter, stream)
        assert stream.exception() is None
        assert _states(payloads) == [("task", SUBMITTED), ("status", WORKING), ("status", expected)]


class _BlockingQueue:
    """An event queue whose enqueue blocks on ``gate`` for matching events."""

    def __init__(self, blocks):
        self.events: list = []
        self.blocks = blocks
        self.blocked = asyncio.Event()
        self.gate = asyncio.Event()

    async def enqueue_event(self, event):
        if self.blocks(event) and not self.gate.is_set():
            self.blocked.set()
            await self.gate.wait()
        self.events.append(event)


def _is_status(state):
    return lambda e: isinstance(e, TaskStatusUpdateEvent) and e.status.state == state


class TestTerminalEnqueueRacingCancel:
    async def test_a_completed_state_being_enqueued_survives_the_cancel(self, request_context):
        adapter = StubAdapter(response="done")
        adapter.cancel = AsyncMock()
        adapter._release_task = MagicMock()
        observer = MagicMock()
        executor = AdapterAgentExecutor(adapter, observer=observer)
        queue = _BlockingQueue(_is_status(COMPLETED))

        run = asyncio.create_task(executor.execute(request_context, queue))
        await asyncio.wait_for(queue.blocked.wait(), 5)

        run.cancel()  # what ActiveTask.cancel does, in this order
        cancel = asyncio.create_task(executor.cancel(request_context, queue))
        await asyncio.sleep(0.05)
        queue.gate.set()
        await asyncio.wait_for(cancel, 5)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(run, 5)

        states = [e.status.state for e in queue.events if isinstance(e, TaskStatusUpdateEvent)]
        assert states == [WORKING, COMPLETED]
        adapter.cancel.assert_not_awaited()
        adapter._release_task.assert_called_once_with(request_context.task_id)
        observer.task_completed.assert_called_once()
        observer.task_canceled.assert_not_called()

    async def test_a_streaming_run_unwinds_before_cancel_decides(self, request_context):
        executor_ref: list[AdapterAgentExecutor] = []
        finished_when_cleaned: list[bool] = []

        class Gen(BaseA2AAdapter):
            async def invoke(self, user_input, context_id=None, **kwargs):
                raise NotImplementedError

            async def stream(self, user_input, context_id=None, **kwargs):
                try:
                    yield "a"
                    yield "b"
                    yield "c"
                finally:
                    run = executor_ref[0]._runs[request_context.task_id]
                    finished_when_cleaned.append(run.finished.is_set())

            async def cancel(self, context_id=None, **kwargs):
                raise RuntimeError("already gone")

        executor = AdapterAgentExecutor(Gen())
        executor_ref.append(executor)
        queue = _BlockingQueue(lambda e: isinstance(e, TaskArtifactUpdateEvent))

        run = asyncio.create_task(executor.execute(request_context, queue))
        await asyncio.wait_for(queue.blocked.wait(), 5)
        run.cancel()
        cancel = asyncio.create_task(executor.cancel(request_context, queue))
        await asyncio.wait_for(cancel, 5)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(run, 5)

        assert finished_when_cleaned == [False]
        states = [e.status.state for e in queue.events if isinstance(e, TaskStatusUpdateEvent)]
        assert states == [WORKING, CANCELED]


class _WaitsBeforeCli(process_helpers.TreeCliAdapter):
    """A CLI adapter that is cancelled before it starts its CLI."""

    def __init__(self, tmp_path):
        super().__init__(tmp_path)
        self.started = asyncio.Event()

    async def stream(self, user_input, context_id=None, **kwargs):
        self.started.set()
        await asyncio.Event().wait()
        async for chunk in super().stream(user_input, context_id, **kwargs):
            yield chunk


class TestReleaseTask:
    async def test_a_cancel_before_the_cli_starts_leaves_no_marker(self, tmp_path):
        adapter = _WaitsBeforeCli(tmp_path)
        result, _, _, _ = await _run_and_cancel(adapter)  # waits for the stream to end

        assert result.status.state == CANCELED
        assert adapter._cancelled_tasks == set()
        assert adapter._killed_tasks == set()

    async def test_a_cancel_after_the_run_ended_leaves_no_marker(
        self, tmp_path, request_context, event_queue
    ):
        class Quick(process_helpers.TreeCliAdapter):
            def supports_streaming(self):
                return False

            async def invoke(self, user_input, context_id=None, **kwargs):
                return "done"

        adapter = Quick(tmp_path)
        executor = AdapterAgentExecutor(adapter)
        await executor.execute(request_context, event_queue)
        await executor.cancel(request_context, event_queue)

        assert adapter._cancelled_tasks == set()
        assert adapter._killed_tasks == set()

    async def test_a_cancel_with_no_run_releases_once(self, request_context, event_queue):
        adapter = StubAdapter()
        adapter._release_task = MagicMock()
        await AdapterAgentExecutor(adapter).cancel(request_context, event_queue)
        adapter._release_task.assert_called_once_with(request_context.task_id)

    async def test_a_cancel_during_a_run_releases_once_after_both_finished(
        self, request_context, event_queue
    ):
        order: list[str] = []
        started = asyncio.Event()

        class Adapter(StubAdapter):
            async def invoke(self, user_input, context_id=None, **kwargs):
                started.set()
                await asyncio.Event().wait()

            async def cancel(self, **kwargs):
                await asyncio.sleep(0.01)
                order.append("cancel done")

            def _release_task(self, task_id):
                order.append("released")

        executor = AdapterAgentExecutor(Adapter())
        run = asyncio.create_task(executor.execute(request_context, event_queue))
        await started.wait()
        run.cancel()
        await executor.cancel(request_context, event_queue)
        with pytest.raises(asyncio.CancelledError):
            await run

        assert order == ["cancel done", "released"]

    async def test_a_failing_release_does_not_change_the_task_state(
        self, request_context, event_queue
    ):
        adapter = StubAdapter()
        adapter._release_task = MagicMock(side_effect=RuntimeError("cleanup broke"))
        await AdapterAgentExecutor(adapter).cancel(request_context, event_queue)

        events = await _collect_events(event_queue)
        assert [e.status.state for e in events] == [CANCELED]


class TestCancelEdgeCases:
    async def test_a_failing_completed_enqueue_is_not_mistaken_for_a_cancel(
        self, request_context, caplog
    ):
        """Outside a cancel, a failed terminal enqueue still propagates (the
        SDK then marks the task failed), as before; it is not swallowed as
        "raised during cancel"."""

        class RefusesCompleted(_BlockingQueue):
            async def enqueue_event(self, event):
                if _is_status(COMPLETED)(event):
                    raise RuntimeError("store refused")
                self.events.append(event)

        queue = RefusesCompleted(lambda e: False)
        observer = MagicMock()
        executor = AdapterAgentExecutor(StubAdapter(), observer=observer)
        with caplog.at_level("INFO", logger="a2a_adapter.executor"):
            with pytest.raises(RuntimeError):
                await executor.execute(request_context, queue)

        assert "store refused" in caplog.text
        assert "cancel in progress" not in caplog.text
        observer.task_completed.assert_not_called()

    async def test_a_cancel_given_up_on_drops_the_late_result(self, request_context, monkeypatch):
        """cancel() is cancelled and its decision outlasts the bounded wait:
        the stuck run must not send completed when it finally returns."""
        monkeypatch.setattr(executor_module, "_CANCEL_SETTLE_TIMEOUT", 0.05)
        adapter = Stubborn()
        cancel_gate = asyncio.Event()

        async def slow_cancel(**kwargs):
            await cancel_gate.wait()

        adapter.cancel = slow_cancel
        released: list[str] = []
        adapter._release_task = lambda task_id: released.append(
            "after decision" if cancel_gate.is_set() else "before decision"
        )
        observer = MagicMock()
        executor = AdapterAgentExecutor(adapter, observer=observer)
        queue = _BlockingQueue(lambda e: False)

        run = asyncio.create_task(executor.execute(request_context, queue))
        await asyncio.wait_for(adapter.started.wait(), 5)
        run.cancel()
        cancel = asyncio.create_task(executor.cancel(request_context, queue))
        await asyncio.sleep(0.01)
        cancel.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(cancel, 5)

        adapter.release.set()  # the stuck run returns "late"
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(run, 5)
        cancel_gate.set()  # the abandoned decision finishes
        for _ in range(100):
            if not executor._cancelling and len(queue.events) > 2:
                break
            await asyncio.sleep(0.01)

        states = [e.status.state for e in queue.events if isinstance(e, TaskStatusUpdateEvent)]
        assert COMPLETED not in states
        observer.task_completed.assert_not_called()
        # The abandoned decision releases once it finished, so a marker
        # its adapter.cancel() queued does not leak.
        assert released[-1] == "after decision"
