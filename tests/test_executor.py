"""Tests for AdapterAgentExecutor — the bridge between adapters and A2A SDK."""

import asyncio
import contextlib
import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from a2a.server.agent_execution import RequestContext
from a2a.server.context import ServerCallContext
from a2a.server.events import EventQueue
from a2a.server.events.event_queue_v2 import EventQueueSource
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
from google.protobuf.json_format import MessageToJson

from a2a_adapter import executor as executor_module
from a2a_adapter.base_adapter import BaseA2AAdapter
from a2a_adapter.exceptions import UserFacingError
from a2a_adapter.executor import AdapterAgentExecutor
from a2a_adapter.integrations._framework import run_with_timeout
from a2a_adapter.integrations.claude_code import ClaudeCodeAdapter
from a2a_adapter.integrations.langgraph import LangGraphAdapter
from a2a_adapter.integrations.n8n import N8nAdapter
from a2a_adapter.integrations.ollama import OllamaAdapter
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
        assert texts == [f"Cancel failed (task {task['id']}). Details are in the server log."]


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
    executor=None,
    store=None,
):
    """Stream a message through DefaultRequestHandler, then cancel the task.

    Returns ``(cancel result or exception, stream payloads, stream task,
    stored task)``. With ``wait_stream`` the stream is awaited first (for
    runs that unwind); otherwise the stored task is read right after the
    cancel, while a stuck run keeps the stream open.
    """
    executor = executor or AdapterAgentExecutor(adapter, observer)
    handler = DefaultRequestHandler(
        executor, store or InMemoryTaskStore(), build_agent_card(adapter, streaming=True)
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


def _refuse_status(monkeypatch, refused) -> None:
    """Make the agent queue raise on the executor's enqueue of one state.

    The SDK's own failed state carries no message and still lands.
    """
    original = EventQueueSource.enqueue_event

    async def enqueue_event(self, event):
        if _is_status(refused)(event) and (refused != FAILED or event.status.HasField("message")):
            raise RuntimeError("store refused")
        await original(self, event)

    monkeypatch.setattr(EventQueueSource, "enqueue_event", enqueue_event)


async def _stored_state(store, task_id, settled_on=FAILED):
    """The stored state, polled: the stream can raise before the consumer saved it."""
    for _ in range(200):
        state = (await store.get(task_id, ServerCallContext())).status.state
        if state == settled_on:
            break
        await asyncio.sleep(0.01)
    return state


async def _send_to_the_end(adapter, observer):
    """Run one message through the real handler; return (error, stored state)."""
    store = InMemoryTaskStore()
    handler = DefaultRequestHandler(
        AdapterAgentExecutor(adapter, observer), store, build_agent_card(adapter, streaming=True)
    )
    message = Message(role=Role.ROLE_USER, message_id=uuid.uuid4().hex, parts=[Part(text="hi")])
    error = None
    try:
        async for _ in handler.on_message_send_stream(
            SendMessageRequest(message=message), ServerCallContext()
        ):
            pass
    except RuntimeError as e:
        error = e
    # The stream can raise before its first payload, so ask the observer.
    return error, await _stored_state(store, observer.task_submitted.call_args.args[0])


class TestCancelThroughHandler:
    @pytest.mark.parametrize(
        ("cancel_error", "refused", "reported"),
        [
            (None, CANCELED, "RuntimeError: store refused"),
            (
                RuntimeError("robot did not stop"),
                FAILED,
                "cancel failed: RuntimeError: robot did not stop",
            ),
        ],
    )
    async def test_a_refused_enqueue_in_cancel_leaves_the_sdk_failed_state_as_the_only_one(
        self, monkeypatch, cancel_error, refused, reported
    ):
        """cancel() raises, so the SDK marks the task failed: the observer
        hears failed, never canceled, and the run still going sends nothing
        after it, even once it finishes."""
        monkeypatch.setattr(executor_module, "_CANCEL_SETTLE_TIMEOUT", 0.05)
        _refuse_status(monkeypatch, refused)
        adapter = Stubborn(cancel_error)
        observer = MagicMock()
        executor = AdapterAgentExecutor(adapter, observer)
        store = InMemoryTaskStore()
        result, payloads, stream, _ = await _run_and_cancel(
            adapter, executor=executor, store=store, wait_stream=False
        )
        try:
            assert isinstance(result, RuntimeError), result
            observer.task_canceled.assert_not_called()
        finally:
            await _finish(adapter, stream)
        for _ in range(500):
            if not executor._runs:
                break
            await asyncio.sleep(0.01)
        assert not executor._runs

        observer.task_completed.assert_not_called()
        observer.task_failed.assert_called_once()
        assert observer.task_failed.call_args.args[2] == reported
        assert await _stored_state(store, payloads[0].id) == FAILED

    async def test_a_refused_failed_enqueue_reports_the_adapter_error_once(self, monkeypatch):
        _refuse_status(monkeypatch, FAILED)
        observer = MagicMock()

        error, stored = await _send_to_the_end(FailingAdapter(), observer)

        assert isinstance(error, RuntimeError)
        assert stored == FAILED
        observer.task_failed.assert_called_once()
        assert observer.task_failed.call_args.args[2] == "RuntimeError: boom"

    async def test_a_refused_completed_enqueue_reports_failed_not_completed(self, monkeypatch):
        """TaskUpdater latches on the first terminal attempt, so the executor
        cannot send failed instead: the SDK's failed state is the result."""
        _refuse_status(monkeypatch, COMPLETED)
        observer = MagicMock()

        error, stored = await _send_to_the_end(StubAdapter(), observer)

        assert isinstance(error, RuntimeError)
        assert stored == FAILED
        observer.task_completed.assert_not_called()
        observer.task_failed.assert_called_once()
        assert observer.task_failed.call_args.args[2] == "RuntimeError: store refused"

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
            assert _status_text(stored.status) == _generic("Cancel failed", stored.id)
            observer.task_failed.assert_called_once()
            observer.task_canceled.assert_not_called()
        finally:
            await _finish(adapter, stream)
        assert stream.exception() is None
        # The late result is dropped: no artifact or completed after failed.
        assert [_describe(p) for p in payloads] == [
            ("task", SUBMITTED),
            ("status", WORKING, None),
            ("status", FAILED, _generic("Cancel failed", payloads[0].id)),
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
            ("status", FAILED, _generic("Cancel failed", payloads[0].id)),
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
    async def test_a_refused_enqueue_under_a_cancelled_run_is_not_reported(self, request_context):
        class BlocksThenRefuses(_BlockingQueue):
            async def enqueue_event(self, event):
                await super().enqueue_event(event)
                if self.blocks(event):
                    raise RuntimeError("store refused")

        observer = MagicMock()
        queue = BlocksThenRefuses(_is_status(FAILED))
        executor = AdapterAgentExecutor(FailingAdapter(), observer=observer)
        run = asyncio.create_task(executor.execute(request_context, queue))
        await asyncio.wait_for(queue.blocked.wait(), 5)

        run.cancel()
        await asyncio.sleep(0)
        queue.gate.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(run, 5)

        observer.task_failed.assert_not_called()

    async def test_failed_is_reported_only_once_it_was_enqueued(self, request_context):
        observer = MagicMock()
        queue = _BlockingQueue(_is_status(FAILED))
        executor = AdapterAgentExecutor(FailingAdapter(), observer=observer)
        run = asyncio.create_task(executor.execute(request_context, queue))
        await asyncio.wait_for(queue.blocked.wait(), 5)

        observer.task_failed.assert_not_called()

        queue.gate.set()
        await asyncio.wait_for(run, 5)
        observer.task_failed.assert_called_once()

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


# ── Exception text sent to clients ────────────────────────────

SECRET = "secret-token-XYZ at /etc/private"


def _generic(prefix: str, task_id: str) -> str:
    return f"{prefix} (task {task_id}). Details are in the server log."


def _wire(payload) -> str:
    return MessageToJson(payload)


class _Raises(BaseA2AAdapter):
    def __init__(self, error: Exception):
        self.error = error

    async def invoke(self, user_input, context_id=None, **kwargs):
        raise self.error


class _RaisesMidStream(_Raises):
    async def stream(self, user_input, context_id=None, **kwargs):
        yield "partial"
        raise self.error


def _handler(adapter, observer=None, **executor_kwargs):
    return DefaultRequestHandler(
        AdapterAgentExecutor(adapter, observer, **executor_kwargs),
        InMemoryTaskStore(),
        build_agent_card(adapter, streaming=True),
    )


def _user_message() -> Message:
    return Message(role=Role.ROLE_USER, message_id=uuid.uuid4().hex, parts=[Part(text="hi")])


async def _send(adapter, observer=None, **executor_kwargs):
    """message/send through the real handler; return (result, stored task)."""
    handler = _handler(adapter, observer, **executor_kwargs)
    call = ServerCallContext()
    result = await handler.on_message_send(SendMessageRequest(message=_user_message()), call)
    return result, await handler.on_get_task(GetTaskRequest(id=result.id), call)


async def _stream(adapter, observer=None, **executor_kwargs):
    """message/stream through the real handler; return (payloads, stored task)."""
    handler = _handler(adapter, observer, **executor_kwargs)
    call = ServerCallContext()
    payloads = [
        payload
        async for payload in handler.on_message_send_stream(
            SendMessageRequest(message=_user_message()), call
        )
    ]
    return payloads, await handler.on_get_task(GetTaskRequest(id=payloads[0].id), call)


class _CustomUserError(UserFacingError):
    pass


class TestExceptionTextOnTheWire:
    async def test_send_and_get_hide_the_exception(self):
        result, stored = await _send(_Raises(RuntimeError(SECRET)))
        assert result.status.state == FAILED
        assert _status_text(result.status) == _generic("Agent execution failed", result.id)
        for payload in (result, stored):
            assert "secret-token" not in _wire(payload)

    @pytest.mark.parametrize("adapter_type", [_Raises, _RaisesMidStream])
    async def test_stream_and_get_hide_the_exception(self, adapter_type):
        payloads, stored = await _stream(adapter_type(RuntimeError(SECRET)))
        assert stored.status.state == FAILED
        assert _status_text(stored.status) == _generic("Agent execution failed", stored.id)
        assert all("secret-token" not in _wire(p) for p in [*payloads, stored])

    async def test_a_failing_cancel_hides_the_exception(self, monkeypatch):
        monkeypatch.setattr(executor_module, "_CANCEL_SETTLE_TIMEOUT", 0.05)
        adapter = Stubborn(RuntimeError(SECRET))
        result, payloads, stream, stored = await _run_and_cancel(adapter, wait_stream=False)
        try:
            assert stored.status.state == FAILED
            assert _status_text(stored.status) == _generic("Cancel failed", stored.id)
            assert all("secret-token" not in _wire(p) for p in [result, *payloads, stored])
        finally:
            await _finish(adapter, stream)

    async def test_the_log_and_observer_keep_the_exception(self, caplog):
        observer = MagicMock()
        with caplog.at_level("ERROR", logger="a2a_adapter.executor"):
            await _send(_Raises(RuntimeError(SECRET)), observer)
        record = next(r for r in caplog.records if SECRET in r.getMessage())
        assert record.exc_info is not None and record.exc_info[1].args == (SECRET,)
        assert observer.task_failed.call_args.args[2] == f"RuntimeError: {SECRET}"

    async def test_a_failing_cancel_logs_the_traceback(self, monkeypatch, caplog):
        monkeypatch.setattr(executor_module, "_CANCEL_SETTLE_TIMEOUT", 0.05)
        adapter = Stubborn(RuntimeError(SECRET))
        observer = MagicMock()
        with caplog.at_level("WARNING", logger="a2a_adapter.executor"):
            _, _, stream, _ = await _run_and_cancel(adapter, observer=observer, wait_stream=False)
        await _finish(adapter, stream)
        record = next(r for r in caplog.records if SECRET in r.getMessage())
        assert record.exc_info is not None and record.exc_info[1].args == (SECRET,)
        assert observer.task_failed.call_args.args[2] == f"cancel failed: RuntimeError: {SECRET}"

    @pytest.mark.parametrize("error_type", [UserFacingError, _CustomUserError])
    async def test_user_facing_errors_pass_through(self, error_type):
        result, stored = await _send(_Raises(error_type("Quota used up")))
        assert _status_text(stored.status) == "Agent execution failed: Quota used up"
        assert _status_text(result.status) == "Agent execution failed: Quota used up"

    async def test_a_user_facing_cancel_error_passes_through(self, monkeypatch):
        monkeypatch.setattr(executor_module, "_CANCEL_SETTLE_TIMEOUT", 0.05)
        adapter = Stubborn(UserFacingError("Still stopping"))
        _, _, stream, stored = await _run_and_cancel(adapter, wait_stream=False)
        await _finish(adapter, stream)
        assert _status_text(stored.status) == "Cancel failed: Still stopping"

    async def test_expose_error_details_restores_the_text(self):
        _, stored = await _send(_Raises(RuntimeError(SECRET)), expose_error_details=True)
        assert _status_text(stored.status) == f"Agent execution failed: {SECRET}"

    async def test_expose_error_details_restores_the_cancel_text(self, monkeypatch):
        monkeypatch.setattr(executor_module, "_CANCEL_SETTLE_TIMEOUT", 0.05)
        adapter = Stubborn(RuntimeError(SECRET))
        executor = AdapterAgentExecutor(adapter, expose_error_details=True)
        _, _, stream, stored = await _run_and_cancel(adapter, executor=executor, wait_stream=False)
        await _finish(adapter, stream)
        assert _status_text(stored.status) == f"Cancel failed: {SECRET}"

    def test_user_facing_error_is_exported_as_a_runtime_error(self):
        import a2a_adapter

        assert a2a_adapter.UserFacingError is UserFacingError
        assert "UserFacingError" in a2a_adapter.__all__
        assert issubclass(UserFacingError, RuntimeError)


# Each builder returns an adapter whose failure carries the secret, with only
# the transport or subprocess faked.
def _n8n_500(_tmp_path):
    def respond(request):
        return httpx.Response(500, text=f"server said {SECRET}")

    adapter = N8nAdapter(webhook_url="http://n8n.test/webhook/secret-token-path", max_retries=0)
    adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    return adapter


def _ollama_connect_error(_tmp_path):
    def respond(request):
        raise httpx.ConnectError(SECRET)

    adapter = OllamaAdapter(base_url="http://secret-token-host:11434")
    adapter.client._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    return adapter


def _langgraph_bad_mapper(_tmp_path):
    def bad_mapper(raw, context_id):
        raise ValueError(SECRET)

    return LangGraphAdapter(graph=MagicMock(), input_mapper=bad_mapper, streaming=False)


class TestRealAdaptersHideErrors:
    @pytest.mark.parametrize(
        "build", [_n8n_500, _ollama_connect_error, _langgraph_bad_mapper], ids=lambda b: b.__name__
    )
    async def test_transport_and_mapper_errors(self, build, tmp_path, caplog):
        with caplog.at_level("ERROR", logger="a2a_adapter.executor"):
            result, stored = await _send(build(tmp_path))
        assert stored.status.state == FAILED
        assert _status_text(stored.status) == _generic("Agent execution failed", stored.id)
        assert "secret-token" not in _wire(result) + _wire(stored)
        assert any("secret-token" in r.getMessage() for r in caplog.records)

    async def test_a_mapper_failure_keeps_its_cause_in_the_log(self, tmp_path, caplog):
        with caplog.at_level("ERROR", logger="a2a_adapter.executor"):
            await _send(_langgraph_bad_mapper(tmp_path))
        record = next(r for r in caplog.records if r.exc_info)
        assert isinstance(record.exc_info[1].__cause__, ValueError)

    async def test_cli_stderr(self, tmp_path):
        adapter = ClaudeCodeAdapter(
            working_dir=str(tmp_path), session_store_path=str(tmp_path / "sessions.json")
        )
        adapter.supports_streaming = lambda: False
        proc = AsyncMock()
        proc.returncode = 1
        proc.pid = 12345
        proc.kill = MagicMock()
        proc.communicate = AsyncMock(return_value=(b"", f"fatal: {SECRET}".encode()))
        with patch("a2a_adapter.base_adapter.create_subprocess_exec", return_value=proc):
            result, stored = await _send(adapter)
        assert stored.status.state == FAILED
        assert "secret-token" not in _wire(result) + _wire(stored)


async def test_an_adapter_timeout_reaches_the_client_readable():
    class Slow(BaseA2AAdapter):
        async def invoke(self, user_input, context_id=None, **kwargs):
            return await run_with_timeout(asyncio.sleep(5), 0.01, "slow thing")

    _, stored = await _send(Slow())
    assert _status_text(stored.status) == (
        "Agent execution failed: slow thing timed out after 0.01 seconds"
    )


async def test_an_http_library_timeout_stays_hidden():
    _, stored = await _send(_Raises(httpx.ReadTimeout(SECRET)))
    assert _status_text(stored.status) == _generic("Agent execution failed", stored.id)
