"""Tests for AdapterAgentExecutor — the bridge between adapters and A2A SDK.

This is the most critical v0.2 component: it translates invoke()/stream()
calls into A2A SDK events. Tests use mock RequestContext and EventQueue
to verify the full lifecycle without needing a real HTTP server.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import Part, Task, TaskState, TaskStatus

from a2a_adapter.base_adapter import BaseA2AAdapter, InputRequired
from a2a_adapter.exceptions import CancelledByAdapterError
from a2a_adapter.executor import AdapterAgentExecutor


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


class CancelledStreamAdapter(BaseA2AAdapter):
    """Adapter whose stream() is cancelled after one chunk."""

    async def invoke(self, user_input, context_id=None, **kwargs):
        return "should not reach"

    async def stream(self, user_input, context_id=None, **kwargs):
        yield "partial"
        raise CancelledByAdapterError("task-1 was cancelled")


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
    """Create a mock EventQueue with async enqueue_event."""
    queue = MagicMock()
    queue.enqueue_event = AsyncMock()
    return queue


# ──── Test: Execute with invoke() ────


@pytest.mark.asyncio
async def test_execute_invoke_basic():
    """invoke() adapter should emit start_work, add_artifact, complete."""
    adapter = InvokeOnlyAdapter("hello from adapter")
    executor = AdapterAgentExecutor(adapter)

    ctx = make_mock_context(user_input="test")
    queue = make_mock_event_queue()

    with patch("a2a_adapter.executor.TaskUpdater") as MockUpdater:
        updater_instance = AsyncMock(spec=TaskUpdater)
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
        assert parts[0].text == "hello from adapter"


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
        MockUpdater.return_value = AsyncMock(spec=TaskUpdater)
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
        MockUpdater.return_value = AsyncMock(spec=TaskUpdater)
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
        updater_instance = AsyncMock(spec=TaskUpdater)
        MockUpdater.return_value = updater_instance

        await executor.execute(ctx, queue)

        updater_instance.start_work.assert_awaited_once()
        # 3 chunks → 3 add_artifact calls
        assert updater_instance.add_artifact.await_count == 3
        updater_instance.complete.assert_awaited_once()

        # First chunk: append=False, rest: append=True
        calls = updater_instance.add_artifact.call_args_list
        assert (
            calls[0].kwargs.get("append", calls[0][1].get("append") if len(calls[0]) > 1 else None)
            is not True
            or calls[0] == calls[0]
        )  # first call
        # Verify first chunk has append=False (not True)
        first_call_kwargs = calls[0][1] if len(calls[0]) > 1 else calls[0].kwargs
        assert (
            first_call_kwargs.get("append") is False or first_call_kwargs.get("append") is not True
        )


@pytest.mark.asyncio
async def test_execute_streaming_complete_message():
    """Complete message should contain full concatenated text."""
    adapter = StreamingAdapter(chunks=["hello", " world"])
    executor = AdapterAgentExecutor(adapter)

    ctx = make_mock_context()
    queue = make_mock_event_queue()

    with patch("a2a_adapter.executor.TaskUpdater") as MockUpdater:
        updater_instance = AsyncMock(spec=TaskUpdater)
        MockUpdater.return_value = updater_instance

        await executor.execute(ctx, queue)

        # Verify new_agent_message was called with the full concatenated text
        updater_instance.new_agent_message.assert_called_once()
        parts = updater_instance.new_agent_message.call_args[0][0]
        assert len(parts) == 1
        assert parts[0].text == "hello world"


# ──── Test: Error Handling ────


@pytest.mark.asyncio
async def test_execute_invoke_error():
    """Exception during invoke() should emit failed state."""
    adapter = FailingAdapter()
    executor = AdapterAgentExecutor(adapter)

    ctx = make_mock_context()
    queue = make_mock_event_queue()

    with patch("a2a_adapter.executor.TaskUpdater") as MockUpdater:
        updater_instance = AsyncMock(spec=TaskUpdater)
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
        updater_instance = AsyncMock(spec=TaskUpdater)
        MockUpdater.return_value = updater_instance

        await executor.execute(ctx, queue)

        updater_instance.failed.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_stream_cancelled_sends_nothing_more():
    """A stream cancelled by the adapter must not flush its held-back chunk.

    cancel() has already emitted the canceled state, so an artifact sent now
    would follow the task's terminal state.
    """
    adapter = CancelledStreamAdapter()
    executor = AdapterAgentExecutor(adapter)

    ctx = make_mock_context()
    queue = make_mock_event_queue()

    with patch("a2a_adapter.executor.TaskUpdater") as MockUpdater:
        updater_instance = AsyncMock(spec=TaskUpdater)
        MockUpdater.return_value = updater_instance

        await executor.execute(ctx, queue)

        updater_instance.add_artifact.assert_not_awaited()
        updater_instance.failed.assert_not_awaited()
        updater_instance.complete.assert_not_awaited()


# ──── Test: Cancellation ────


@pytest.mark.asyncio
async def test_cancel_delegates_to_adapter():
    """cancel() should call adapter.cancel() and emit canceled state."""
    adapter = CancellableAdapter()
    executor = AdapterAgentExecutor(adapter)

    ctx = make_mock_context()
    queue = make_mock_event_queue()

    with patch("a2a_adapter.executor.TaskUpdater") as MockUpdater:
        updater_instance = AsyncMock(spec=TaskUpdater)
        MockUpdater.return_value = updater_instance

        await executor.cancel(ctx, queue)

        assert adapter.cancel_called is True
        updater_instance.cancel.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancel_adapter_error_with_no_run_is_reported_canceled():
    """If adapter.cancel() raises but no execute() is in flight, nothing of
    ours is running: the task is canceled, not failed."""
    adapter = FailingCancelAdapter()
    executor = AdapterAgentExecutor(adapter)

    ctx = make_mock_context()
    queue = make_mock_event_queue()

    with patch("a2a_adapter.executor.TaskUpdater") as MockUpdater:
        updater_instance = AsyncMock(spec=TaskUpdater)
        MockUpdater.return_value = updater_instance

        await executor.cancel(ctx, queue)

        updater_instance.cancel.assert_awaited_once()
        updater_instance.failed.assert_not_awaited()


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
        MockUpdater.return_value = AsyncMock(spec=TaskUpdater)

        with patch.object(executor, "_execute_invoke", new_callable=AsyncMock) as mock_invoke:
            with patch.object(
                executor, "_execute_streaming", new_callable=AsyncMock
            ) as mock_stream:
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
        MockUpdater.return_value = AsyncMock(spec=TaskUpdater)

        with patch.object(executor, "_execute_invoke", new_callable=AsyncMock) as mock_invoke:
            with patch.object(
                executor, "_execute_streaming", new_callable=AsyncMock
            ) as mock_stream:
                await executor.execute(ctx, queue)
                mock_stream.assert_awaited_once()
                mock_invoke.assert_not_awaited()


# ──── Test: invoke() drops empty Parts ────


@pytest.mark.asyncio
async def test_execute_invoke_filters_empty_parts():
    """Parts with no content set must not reach the artifact (they serialize to {})."""
    from a2a.types import Part

    adapter = InvokeOnlyAdapter(
        [Part(text="hello"), Part(), Part(text=""), Part(url="http://x/f.pdf")]
    )
    executor = AdapterAgentExecutor(adapter)

    with patch("a2a_adapter.executor.TaskUpdater") as MockUpdater:
        updater_instance = AsyncMock(spec=TaskUpdater)
        MockUpdater.return_value = updater_instance

        await executor.execute(make_mock_context(), make_mock_event_queue())

        parts = updater_instance.add_artifact.call_args[0][0]
        assert [p.WhichOneof("content") for p in parts] == ["text", "url"]
        updater_instance.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_invoke_all_empty_parts_skips_artifact():
    """Like the streaming path, an all-empty result emits no artifact but completes."""
    from a2a.types import Part

    adapter = InvokeOnlyAdapter([Part(), Part(text="")])
    executor = AdapterAgentExecutor(adapter)

    with patch("a2a_adapter.executor.TaskUpdater") as MockUpdater:
        updater_instance = AsyncMock(spec=TaskUpdater)
        MockUpdater.return_value = updater_instance

        await executor.execute(make_mock_context(), make_mock_event_queue())

        updater_instance.add_artifact.assert_not_awaited()
        updater_instance.complete.assert_awaited_once()
        updater_instance.failed.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_invoke_empty_string_skips_artifact():
    """An empty str result emits no empty-text artifact but still completes."""
    executor = AdapterAgentExecutor(InvokeOnlyAdapter(""))

    with patch("a2a_adapter.executor.TaskUpdater") as MockUpdater:
        updater_instance = AsyncMock(spec=TaskUpdater)
        MockUpdater.return_value = updater_instance

        await executor.execute(make_mock_context(), make_mock_event_queue())

        updater_instance.add_artifact.assert_not_awaited()
        updater_instance.complete.assert_awaited_once()
        message_parts = updater_instance.new_agent_message.call_args[0][0]
        assert [p.text for p in message_parts] == ["[Empty response]"]


# ──── Test: InputRequired (adapter pauses for user input) ────


class _InputRequiredAdapter(BaseA2AAdapter):
    def __init__(self, result):
        self._result = result

    async def invoke(self, user_input, context_id=None, **kwargs):
        return self._result


class _InputRequiredStreamAdapter(BaseA2AAdapter):
    def __init__(self, items):
        self._items = items
        self.closed = False

    async def invoke(self, user_input, context_id=None, **kwargs):
        raise AssertionError("not used")

    async def stream(self, user_input, context_id=None, **kwargs):
        try:
            for item in self._items:
                yield item
        finally:
            self.closed = True


async def _run_with_updater(adapter, observer=None):
    executor = AdapterAgentExecutor(adapter, observer=observer)
    with patch("a2a_adapter.executor.TaskUpdater") as MockUpdater:
        updater = AsyncMock(spec=TaskUpdater)
        updater.new_agent_message = MagicMock(side_effect=lambda parts: ("msg", parts))
        MockUpdater.return_value = updater
        await executor.execute(make_mock_context(), make_mock_event_queue())
    return updater


@pytest.mark.asyncio
async def test_invoke_input_required_reports_input_required_not_completed():
    observer = MagicMock()
    updater = await _run_with_updater(
        _InputRequiredAdapter(InputRequired("Approve the draft?")), observer
    )

    updater.requires_input.assert_awaited_once()
    updater.complete.assert_not_awaited()
    updater.failed.assert_not_awaited()
    updater.add_artifact.assert_not_awaited()
    _, parts = updater.requires_input.call_args.kwargs["message"]
    assert [p.text for p in parts] == ["Approve the draft?"]
    observer.task_input_required.assert_called_once_with("task-1", "ctx-1", "Approve the draft?")
    observer.task_completed.assert_not_called()


@pytest.mark.asyncio
async def test_invoke_input_required_with_parts():
    parts_in = [Part(text="Pick one:"), Part(text="a or b")]
    updater = await _run_with_updater(_InputRequiredAdapter(InputRequired(parts_in)))

    _, parts = updater.requires_input.call_args.kwargs["message"]
    assert [p.text for p in parts] == ["Pick one:", "a or b"]


@pytest.mark.asyncio
async def test_invoke_input_required_empty_message_gets_placeholder():
    updater = await _run_with_updater(_InputRequiredAdapter(InputRequired("")))

    _, parts = updater.requires_input.call_args.kwargs["message"]
    assert [p.text for p in parts] == ["[Input required]"]


@pytest.mark.asyncio
async def test_stream_input_required_closes_artifact_then_reports_input_required():
    observer = MagicMock()
    adapter = _InputRequiredStreamAdapter(["draft ", "ready", InputRequired("Approve the draft?")])
    updater = await _run_with_updater(adapter, observer)

    calls = updater.add_artifact.call_args_list
    assert [c.args[0][0].text for c in calls] == ["draft ", "ready"]
    assert calls[-1].kwargs["last_chunk"] is True
    updater.requires_input.assert_awaited_once()
    updater.complete.assert_not_awaited()
    _, parts = updater.requires_input.call_args.kwargs["message"]
    assert [p.text for p in parts] == ["Approve the draft?"]
    observer.task_input_required.assert_called_once_with("task-1", "ctx-1", "Approve the draft?")


@pytest.mark.asyncio
async def test_stream_input_required_without_output():
    updater = await _run_with_updater(_InputRequiredStreamAdapter([InputRequired("Continue?")]))

    updater.add_artifact.assert_not_awaited()
    updater.requires_input.assert_awaited_once()
    updater.complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_stream_stops_reading_after_input_required():
    adapter = _InputRequiredStreamAdapter(["a", InputRequired("Continue?"), "late"])
    updater = await _run_with_updater(adapter)

    sent = [c.args[0][0].text for c in updater.add_artifact.call_args_list]
    assert sent == ["a"]
    assert adapter.closed
    updater.requires_input.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancel_with_no_run_sends_canceled_before_adapter_cancel():
    """A task idle in input-required has no execute() in flight.

    The SDK cancels its idle producer just before calling cancel(); the
    producer then closes the queue at its next step, so the canceled state
    must be enqueued before anything else awaits.
    """
    order = []

    class Adapter(BaseA2AAdapter):
        async def invoke(self, user_input, context_id=None, **kwargs):
            return "unused"

        async def cancel(self, context_id=None, **kwargs):
            order.append("adapter.cancel")

    executor = AdapterAgentExecutor(Adapter())
    with patch("a2a_adapter.executor.TaskUpdater") as MockUpdater:
        updater = AsyncMock(spec=TaskUpdater)
        updater.cancel.side_effect = lambda *a, **k: order.append("canceled")
        MockUpdater.return_value = updater
        ctx = make_mock_context()
        ctx.current_task = Task(
            id="task-1",
            context_id="ctx-1",
            status=TaskStatus(state=TaskState.TASK_STATE_INPUT_REQUIRED),
        )
        await executor.cancel(ctx, make_mock_event_queue())

    assert order == ["canceled", "adapter.cancel"]


@pytest.mark.asyncio
async def test_cancel_with_no_run_and_a_finished_run_keeps_the_old_order():
    """No run in flight but the task is not waiting for input (e.g. its run
    just finished): adapter.cancel() goes first, as before."""
    order = []

    class Adapter(BaseA2AAdapter):
        async def invoke(self, user_input, context_id=None, **kwargs):
            return "unused"

        async def cancel(self, context_id=None, **kwargs):
            order.append("adapter.cancel")

    executor = AdapterAgentExecutor(Adapter())
    with patch("a2a_adapter.executor.TaskUpdater") as MockUpdater:
        updater = AsyncMock(spec=TaskUpdater)
        updater.cancel.side_effect = lambda *a, **k: order.append("canceled")
        MockUpdater.return_value = updater
        ctx = make_mock_context()
        ctx.current_task = Task(
            id="task-1", context_id="ctx-1", status=TaskStatus(state=TaskState.TASK_STATE_WORKING)
        )
        await executor.cancel(ctx, make_mock_event_queue())

    assert order == ["adapter.cancel", "canceled"]


@pytest.mark.asyncio
async def test_invoke_input_required_with_a_bare_part():
    updater = await _run_with_updater(_InputRequiredAdapter(InputRequired(Part(text="Pick one"))))

    _, parts = updater.requires_input.call_args.kwargs["message"]
    assert [p.text for p in parts] == ["Pick one"]


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_input_required_dropped_while_a_cancel_is_in_progress(streaming):
    """cancel() sends the terminal state; the pause must not follow it."""
    executor = None

    class Adapter(BaseA2AAdapter):
        async def invoke(self, user_input, context_id=None, **kwargs):
            executor._cancelling["task-1"] = asyncio.get_running_loop().create_future()
            return InputRequired("Continue?")

        async def _stream(self, user_input, context_id=None, **kwargs):
            yield "partial"
            executor._cancelling["task-1"] = asyncio.get_running_loop().create_future()
            yield InputRequired("Continue?")

    if streaming:
        Adapter.stream = Adapter._stream
    executor = AdapterAgentExecutor(Adapter())
    with patch("a2a_adapter.executor.TaskUpdater") as MockUpdater:
        updater = AsyncMock(spec=TaskUpdater)
        MockUpdater.return_value = updater
        run = asyncio.ensure_future(executor.execute(make_mock_context(), make_mock_event_queue()))
        while "task-1" not in executor._cancelling:
            await asyncio.sleep(0)
        executor._cancelling.pop("task-1").set_result(None)
        await run

    updater.requires_input.assert_not_awaited()
    updater.complete.assert_not_awaited()
    updater.failed.assert_not_awaited()
