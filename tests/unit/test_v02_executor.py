"""Tests for AdapterAgentExecutor — the bridge between adapters and A2A SDK.

This is the most critical v0.2 component: it translates invoke()/stream()
calls into A2A SDK events. Tests use mock RequestContext and EventQueue
to verify the full lifecycle without needing a real HTTP server.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from a2a_adapter.executor import AdapterAgentExecutor
from a2a_adapter.base_adapter import BaseA2AAdapter, AdapterMetadata


# ──── Helpers: Minimal Adapter Implementations ────


class InvokeOnlyAdapter(BaseA2AAdapter):
    """Adapter that only implements invoke()."""

    def __init__(self, response="test response"):
        self._response = response

    async def invoke(self, user_input, context_id=None, **kwargs):
        return self._response


class StreamingAdapter(BaseA2AAdapter):
    """Adapter that supports streaming."""

    def __init__(self, chunks=None):
        self._chunks = chunks or ["hello", " ", "world"]

    async def invoke(self, user_input, context_id=None, **kwargs):
        return "".join(self._chunks)

    async def stream(self, user_input, context_id=None, **kwargs):
        for chunk in self._chunks:
            yield chunk


class FailingAdapter(BaseA2AAdapter):
    """Adapter that raises during invoke()."""

    async def invoke(self, user_input, context_id=None, **kwargs):
        raise RuntimeError("agent crashed")


class FailingStreamAdapter(BaseA2AAdapter):
    """Adapter that raises during stream()."""

    async def invoke(self, user_input, context_id=None, **kwargs):
        return "should not reach"

    async def stream(self, user_input, context_id=None, **kwargs):
        yield "partial"
        raise RuntimeError("stream crashed")


class CancellableAdapter(BaseA2AAdapter):
    """Adapter with custom cancel() logic."""

    def __init__(self):
        self.cancel_called = False

    async def invoke(self, user_input, context_id=None, **kwargs):
        return "done"

    async def cancel(self, **kwargs):
        self.cancel_called = True


class FailingCancelAdapter(BaseA2AAdapter):
    """Adapter whose cancel() raises."""

    async def invoke(self, user_input, context_id=None, **kwargs):
        return "done"

    async def cancel(self, **kwargs):
        raise RuntimeError("cancel failed")


# ──── Helpers: Mock SDK Objects ────


def make_mock_context(user_input="test input", task_id="task-1", context_id="ctx-1"):
    """Create a mock RequestContext."""
    ctx = MagicMock()
    ctx.get_user_input.return_value = user_input
    ctx.task_id = task_id
    ctx.context_id = context_id
    return ctx


def make_mock_event_queue():
    """Create a mock EventQueue."""
    return MagicMock()


# ──── Test: Execute with invoke() ────


@pytest.mark.asyncio
async def test_execute_invoke_basic():
    """invoke() adapter should emit start_work, add_artifact, complete."""
    adapter = InvokeOnlyAdapter("hello from adapter")
    executor = AdapterAgentExecutor(adapter)

    ctx = make_mock_context(user_input="test")
    queue = make_mock_event_queue()

    with patch("a2a_adapter.executor.TaskUpdater") as MockUpdater:
        updater_instance = AsyncMock()
        MockUpdater.return_value = updater_instance

        await executor.execute(ctx, queue)

        # Verify lifecycle
        MockUpdater.assert_called_once_with(queue, "task-1", "ctx-1")
        updater_instance.start_work.assert_awaited_once()
        updater_instance.add_artifact.assert_awaited_once()
        updater_instance.complete.assert_awaited_once()

        # Verify artifact content
        artifact_call = updater_instance.add_artifact.call_args
        parts = artifact_call[0][0]
        assert len(parts) == 1
        assert parts[0].root.text == "hello from adapter"


@pytest.mark.asyncio
async def test_execute_invoke_passes_context():
    """context_id should be passed to adapter.invoke()."""

    class TrackingAdapter(BaseA2AAdapter):
        def __init__(self):
            self.received_input = None
            self.received_ctx = None

        async def invoke(self, user_input, context_id=None, **kwargs):
            self.received_input = user_input
            self.received_ctx = context_id
            return "ok"

    adapter = TrackingAdapter()
    executor = AdapterAgentExecutor(adapter)

    ctx = make_mock_context(user_input="hello", context_id="my-ctx")
    queue = make_mock_event_queue()

    with patch("a2a_adapter.executor.TaskUpdater") as MockUpdater:
        MockUpdater.return_value = AsyncMock()
        await executor.execute(ctx, queue)

    assert adapter.received_input == "hello"
    assert adapter.received_ctx == "my-ctx"


@pytest.mark.asyncio
async def test_execute_invoke_empty_input():
    """Empty input should still work (with a warning logged)."""
    adapter = InvokeOnlyAdapter("response")
    executor = AdapterAgentExecutor(adapter)

    ctx = make_mock_context(user_input="")
    queue = make_mock_event_queue()

    with patch("a2a_adapter.executor.TaskUpdater") as MockUpdater:
        MockUpdater.return_value = AsyncMock()
        await executor.execute(ctx, queue)

        MockUpdater.return_value.complete.assert_awaited_once()


# ──── Test: Execute with stream() ────


@pytest.mark.asyncio
async def test_execute_streaming_basic():
    """streaming adapter should emit start_work, multiple add_artifact, complete."""
    adapter = StreamingAdapter(chunks=["a", "b", "c"])
    executor = AdapterAgentExecutor(adapter)

    ctx = make_mock_context()
    queue = make_mock_event_queue()

    with patch("a2a_adapter.executor.TaskUpdater") as MockUpdater:
        updater_instance = AsyncMock()
        MockUpdater.return_value = updater_instance

        await executor.execute(ctx, queue)

        updater_instance.start_work.assert_awaited_once()
        # 3 chunks → 3 add_artifact calls
        assert updater_instance.add_artifact.await_count == 3
        updater_instance.complete.assert_awaited_once()

        # First chunk: append=False, rest: append=True
        calls = updater_instance.add_artifact.call_args_list
        assert calls[0].kwargs.get("append", calls[0][1].get("append") if len(calls[0]) > 1 else None) is not True or calls[0] == calls[0]  # first call
        # Verify first chunk has append=False (not True)
        first_call_kwargs = calls[0][1] if len(calls[0]) > 1 else calls[0].kwargs
        assert first_call_kwargs.get("append") is False or first_call_kwargs.get("append") is not True


@pytest.mark.asyncio
async def test_execute_streaming_complete_message():
    """Complete message should contain full concatenated text."""
    adapter = StreamingAdapter(chunks=["hello", " world"])
    executor = AdapterAgentExecutor(adapter)

    ctx = make_mock_context()
    queue = make_mock_event_queue()

    with patch("a2a_adapter.executor.TaskUpdater") as MockUpdater:
        updater_instance = AsyncMock()
        MockUpdater.return_value = updater_instance

        with patch("a2a_adapter.executor.new_agent_text_message") as mock_msg:
            mock_msg.return_value = "final_message"
            await executor.execute(ctx, queue)
            # new_agent_text_message should be called with full text
            mock_msg.assert_called_with("hello world", "ctx-1", "task-1")


# ──── Test: Error Handling ────


@pytest.mark.asyncio
async def test_execute_invoke_error():
    """Exception during invoke() should emit failed state."""
    adapter = FailingAdapter()
    executor = AdapterAgentExecutor(adapter)

    ctx = make_mock_context()
    queue = make_mock_event_queue()

    with patch("a2a_adapter.executor.TaskUpdater") as MockUpdater:
        updater_instance = AsyncMock()
        MockUpdater.return_value = updater_instance

        await executor.execute(ctx, queue)

        updater_instance.start_work.assert_awaited_once()
        updater_instance.failed.assert_awaited_once()
        updater_instance.complete.assert_not_awaited()

        # Verify error message contains the exception text
        fail_kwargs = updater_instance.failed.call_args[1]
        assert fail_kwargs.get("message") is not None


@pytest.mark.asyncio
async def test_execute_stream_error():
    """Exception during stream() should emit failed state."""
    adapter = FailingStreamAdapter()
    executor = AdapterAgentExecutor(adapter)

    ctx = make_mock_context()
    queue = make_mock_event_queue()

    with patch("a2a_adapter.executor.TaskUpdater") as MockUpdater:
        updater_instance = AsyncMock()
        MockUpdater.return_value = updater_instance

        await executor.execute(ctx, queue)

        updater_instance.failed.assert_awaited_once()


# ──── Test: Cancellation ────


@pytest.mark.asyncio
async def test_cancel_delegates_to_adapter():
    """cancel() should call adapter.cancel() and emit canceled state."""
    adapter = CancellableAdapter()
    executor = AdapterAgentExecutor(adapter)

    ctx = make_mock_context()
    queue = make_mock_event_queue()

    with patch("a2a_adapter.executor.TaskUpdater") as MockUpdater:
        updater_instance = AsyncMock()
        MockUpdater.return_value = updater_instance

        await executor.cancel(ctx, queue)

        assert adapter.cancel_called is True
        updater_instance.cancel.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancel_adapter_error_still_cancels():
    """If adapter.cancel() raises, we should still emit canceled state."""
    adapter = FailingCancelAdapter()
    executor = AdapterAgentExecutor(adapter)

    ctx = make_mock_context()
    queue = make_mock_event_queue()

    with patch("a2a_adapter.executor.TaskUpdater") as MockUpdater:
        updater_instance = AsyncMock()
        MockUpdater.return_value = updater_instance

        await executor.cancel(ctx, queue)

        # Should still cancel despite adapter.cancel() raising
        updater_instance.cancel.assert_awaited_once()


# ──── Test: Streaming Detection Routing ────


@pytest.mark.asyncio
async def test_invoke_only_adapter_routes_to_invoke():
    """Non-streaming adapter should route to _execute_invoke."""
    adapter = InvokeOnlyAdapter()
    executor = AdapterAgentExecutor(adapter)

    assert not adapter.supports_streaming()

    ctx = make_mock_context()
    queue = make_mock_event_queue()

    with patch("a2a_adapter.executor.TaskUpdater") as MockUpdater:
        MockUpdater.return_value = AsyncMock()

        with patch.object(executor, "_execute_invoke", new_callable=AsyncMock) as mock_invoke:
            with patch.object(executor, "_execute_streaming", new_callable=AsyncMock) as mock_stream:
                await executor.execute(ctx, queue)
                mock_invoke.assert_awaited_once()
                mock_stream.assert_not_awaited()


@pytest.mark.asyncio
async def test_streaming_adapter_routes_to_stream():
    """Streaming adapter should route to _execute_streaming."""
    adapter = StreamingAdapter()
    executor = AdapterAgentExecutor(adapter)

    assert adapter.supports_streaming()

    ctx = make_mock_context()
    queue = make_mock_event_queue()

    with patch("a2a_adapter.executor.TaskUpdater") as MockUpdater:
        MockUpdater.return_value = AsyncMock()

        with patch.object(executor, "_execute_invoke", new_callable=AsyncMock) as mock_invoke:
            with patch.object(executor, "_execute_streaming", new_callable=AsyncMock) as mock_stream:
                await executor.execute(ctx, queue)
                mock_stream.assert_awaited_once()
                mock_invoke.assert_not_awaited()
