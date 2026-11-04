"""Tests for AdapterAgentExecutor — the bridge between adapters and A2A SDK."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.types import Part, TaskState

from a2a_adapter.base_adapter import BaseA2AAdapter
from a2a_adapter.executor import AdapterAgentExecutor

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
        status_events = [
            e
            for e in events
            if type(e).__name__ == "TaskStatusUpdateEvent"
        ]
        assert any(
            hasattr(e, "status") and getattr(e.status, "state", None)
            for e in status_events
        )


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
        artifact_events = [
            e for e in events if type(e).__name__ == "TaskArtifactUpdateEvent"
        ]
        assert len(artifact_events) == 3

    async def test_streaming_filters_empty_chunks(self, ctx, event_queue):
        """Empty string chunks must be silently dropped."""
        adapter = StreamingStubAdapter(["Hello", "", "", "world"])
        executor = AdapterAgentExecutor(adapter)

        await executor.execute(ctx, event_queue)

        events = await _collect_events(event_queue)
        artifact_events = [
            e for e in events if type(e).__name__ == "TaskArtifactUpdateEvent"
        ]
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
        artifact_events = [
            e for e in events if type(e).__name__ == "TaskArtifactUpdateEvent"
        ]
        assert len(artifact_events) == 2

    async def test_streaming_all_empty_produces_no_artifacts(self, ctx, event_queue):
        """A stream of only empty chunks should still complete gracefully."""
        adapter = StreamingStubAdapter(["", ""])
        executor = AdapterAgentExecutor(adapter)

        await executor.execute(ctx, event_queue)

        events = await _collect_events(event_queue)
        artifact_events = [
            e for e in events if type(e).__name__ == "TaskArtifactUpdateEvent"
        ]
        assert len(artifact_events) == 0

        status_events = [
            e for e in events if type(e).__name__ == "TaskStatusUpdateEvent"
        ]
        assert len(status_events) >= 1

    async def test_streaming_single_chunk(self, ctx, event_queue):
        adapter = StreamingStubAdapter(["only"])
        executor = AdapterAgentExecutor(adapter)

        await executor.execute(ctx, event_queue)

        events = await _collect_events(event_queue)
        artifact_events = [
            e for e in events if type(e).__name__ == "TaskArtifactUpdateEvent"
        ]
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
        artifact_events = [
            e for e in events if type(e).__name__ == "TaskArtifactUpdateEvent"
        ]
        assert len(artifact_events) == 2

    async def test_streaming_failure_closes_artifact(self, ctx, event_queue):
        """When the stream raises after yielding chunks, the open artifact
        must be closed (last_chunk=True) and the task must enter failed state."""
        adapter = FailingStreamAdapter(["Hello", " world"])
        executor = AdapterAgentExecutor(adapter)

        await executor.execute(ctx, event_queue)

        events = await _collect_events(event_queue)
        artifact_events = [
            e for e in events if type(e).__name__ == "TaskArtifactUpdateEvent"
        ]
        assert len(artifact_events) >= 1

        status_events = [
            e for e in events if type(e).__name__ == "TaskStatusUpdateEvent"
        ]
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
        artifact_events = [
            e for e in events if type(e).__name__ == "TaskArtifactUpdateEvent"
        ]
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
        artifact_events = [
            e for e in events if type(e).__name__ == "TaskArtifactUpdateEvent"
        ]
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
        artifact_events = [
            e for e in events if type(e).__name__ == "TaskArtifactUpdateEvent"
        ]
        assert len(artifact_events) == 0

        status_events = [
            e for e in events if type(e).__name__ == "TaskStatusUpdateEvent"
        ]
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

    async def test_cancel_handles_adapter_error(self, request_context, event_queue):
        adapter = StubAdapter()
        adapter.cancel = AsyncMock(side_effect=RuntimeError("cancel failed"))
        executor = AdapterAgentExecutor(adapter)

        await executor.cancel(request_context, event_queue)
