"""
Unit tests for LangGraphAgentAdapter.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from a2a_adapter.integrations.langgraph import LangGraphAgentAdapter
from a2a.types import Message, MessageSendParams, Task, TaskState, TextPart, Role, Part
from a2a.server.tasks import InMemoryTaskStore


def make_message_send_params(text: str, context_id: str | None = None) -> MessageSendParams:
    """Helper to create MessageSendParams with correct A2A types."""
    return MessageSendParams(
        message=Message(
            message_id="test-msg-id",
            role=Role.user,
            parts=[Part(root=TextPart(text=text))],
            context_id=context_id,
        )
    )


class TestLangGraphAdapterBasic:
    """Basic functionality tests for LangGraphAgentAdapter."""

    @pytest.mark.asyncio
    async def test_to_framework_with_messages_input_key(self):
        """Test that to_framework creates messages format when input_key is 'messages'."""
        mock_graph = MagicMock()
        adapter = LangGraphAgentAdapter(graph=mock_graph, input_key="messages")

        params = make_message_send_params("hello world")
        payload = await adapter.to_framework(params)

        assert "messages" in payload
        assert len(payload["messages"]) == 1
        # Check message content (could be HumanMessage or dict)
        msg = payload["messages"][0]
        if hasattr(msg, "content"):
            assert msg.content == "hello world"
        else:
            assert msg["content"] == "hello world"

    @pytest.mark.asyncio
    async def test_to_framework_with_simple_input_key(self):
        """Test that to_framework uses simple key when not 'messages'."""
        mock_graph = MagicMock()
        adapter = LangGraphAgentAdapter(graph=mock_graph, input_key="query")

        params = make_message_send_params("test query")
        payload = await adapter.to_framework(params)

        assert payload["query"] == "test query"
        assert "messages" not in payload

    @pytest.mark.asyncio
    async def test_to_framework_handles_multiple_parts(self):
        """Test that multiple text parts are joined correctly."""
        mock_graph = MagicMock()
        adapter = LangGraphAgentAdapter(graph=mock_graph, input_key="input")

        # Create params with multiple text parts
        params = MessageSendParams(
            message=Message(
                message_id="test-msg-id",
                role=Role.user,
                parts=[
                    Part(root=TextPart(text="part one")),
                    Part(root=TextPart(text="part two")),
                ],
                context_id=None,
            )
        )
        payload = await adapter.to_framework(params)

        assert "part one" in payload["input"]
        assert "part two" in payload["input"]


class TestLangGraphAdapterContextId:
    """Tests for context_id handling in LangGraphAgentAdapter."""

    @pytest.mark.asyncio
    async def test_from_framework_preserves_context_id(self):
        """Test that from_framework preserves context_id on the response Message."""
        mock_graph = MagicMock()
        adapter = LangGraphAgentAdapter(graph=mock_graph)

        params = make_message_send_params("hello", context_id="ctx-123")
        framework_output = {"output": "response text"}

        result = await adapter.from_framework(framework_output, params)

        assert isinstance(result, Message)
        assert result.context_id == "ctx-123"
        assert result.role == Role.agent

    @pytest.mark.asyncio
    async def test_from_framework_handles_missing_context_id(self):
        """Test that from_framework handles missing context_id gracefully."""
        mock_graph = MagicMock()
        adapter = LangGraphAgentAdapter(graph=mock_graph)

        params = make_message_send_params("hello", context_id=None)
        framework_output = {"output": "response text"}

        result = await adapter.from_framework(framework_output, params)

        assert isinstance(result, Message)
        assert result.context_id is None


class TestLangGraphAdapterOutputExtraction:
    """Tests for output extraction from various LangGraph state types."""

    @pytest.mark.asyncio
    async def test_from_framework_extracts_with_output_key(self):
        """Test extraction using configured output_key."""
        mock_graph = MagicMock()
        adapter = LangGraphAgentAdapter(graph=mock_graph, output_key="result")

        params = make_message_send_params("hello")
        framework_output = {"result": "extracted value", "other": "ignored"}

        result = await adapter.from_framework(framework_output, params)

        assert result.parts[0].root.text == "extracted value"

    @pytest.mark.asyncio
    async def test_from_framework_extracts_from_messages(self):
        """Test extraction from messages key in state."""
        mock_graph = MagicMock()
        adapter = LangGraphAgentAdapter(graph=mock_graph)

        # Mock AIMessage-like object
        mock_ai_message = MagicMock()
        mock_ai_message.content = "AI response from messages"

        params = make_message_send_params("hello")
        framework_output = {"messages": [mock_ai_message]}

        result = await adapter.from_framework(framework_output, params)

        assert result.parts[0].root.text == "AI response from messages"

    @pytest.mark.asyncio
    async def test_from_framework_extracts_common_keys(self):
        """Test extraction from common output keys."""
        mock_graph = MagicMock()
        adapter = LangGraphAgentAdapter(graph=mock_graph)

        params = make_message_send_params("hello")

        # Test 'output' key
        result = await adapter.from_framework({"output": "from output"}, params)
        assert result.parts[0].root.text == "from output"

        # Test 'response' key
        result = await adapter.from_framework({"response": "from response"}, params)
        assert result.parts[0].root.text == "from response"

        # Test 'answer' key
        result = await adapter.from_framework({"answer": "from answer"}, params)
        assert result.parts[0].root.text == "from answer"

    @pytest.mark.asyncio
    async def test_from_framework_extracts_dict_messages(self):
        """Test extraction from dict-format messages."""
        mock_graph = MagicMock()
        adapter = LangGraphAgentAdapter(graph=mock_graph)

        params = make_message_send_params("hello")
        framework_output = {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "dict message response"},
            ]
        }

        result = await adapter.from_framework(framework_output, params)

        assert result.parts[0].root.text == "dict message response"


class TestLangGraphAdapterCallFramework:
    """Tests for call_framework execution."""

    @pytest.mark.asyncio
    async def test_call_framework_invokes_graph(self):
        """Test that call_framework calls graph.ainvoke()."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={"output": "graph result"})

        adapter = LangGraphAgentAdapter(graph=mock_graph, input_key="input")
        params = make_message_send_params("test")
        framework_input = {"input": "test message"}

        result = await adapter.call_framework(framework_input, params)

        mock_graph.ainvoke.assert_called_once_with(framework_input)
        assert result == {"output": "graph result"}


class TestLangGraphAdapterHandle:
    """Tests for the complete handle flow."""

    @pytest.mark.asyncio
    async def test_handle_end_to_end(self):
        """Test complete handle() flow from params to message."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={"output": "processed response"})

        adapter = LangGraphAgentAdapter(graph=mock_graph, input_key="input")
        params = make_message_send_params("test message", context_id="e2e-ctx")

        result = await adapter.handle(params)

        assert isinstance(result, Message)
        assert result.role == Role.agent
        assert result.context_id == "e2e-ctx"
        assert result.parts[0].root.text == "processed response"


class TestLangGraphAdapterAsyncMode:
    """Tests for async task mode in LangGraphAgentAdapter."""

    def test_supports_async_tasks_false_by_default(self):
        """Test that async mode is disabled by default."""
        mock_graph = MagicMock()
        adapter = LangGraphAgentAdapter(graph=mock_graph)
        assert adapter.supports_async_tasks() is False

    def test_supports_async_tasks_true_when_enabled(self):
        """Test that async mode can be enabled."""
        mock_graph = MagicMock()
        adapter = LangGraphAgentAdapter(graph=mock_graph, async_mode=True)
        assert adapter.supports_async_tasks() is True

    def test_async_mode_creates_task_store(self):
        """Test that async mode creates an InMemoryTaskStore by default."""
        mock_graph = MagicMock()
        adapter = LangGraphAgentAdapter(graph=mock_graph, async_mode=True)
        assert adapter.task_store is not None
        assert isinstance(adapter.task_store, InMemoryTaskStore)

    def test_async_mode_accepts_custom_task_store(self):
        """Test that a custom TaskStore can be provided."""
        mock_graph = MagicMock()
        custom_store = InMemoryTaskStore()
        adapter = LangGraphAgentAdapter(
            graph=mock_graph,
            async_mode=True,
            task_store=custom_store,
        )
        assert adapter.task_store is custom_store

    @pytest.mark.asyncio
    async def test_get_task_raises_when_not_async_mode(self):
        """Test that get_task raises RuntimeError when not in async mode."""
        mock_graph = MagicMock()
        adapter = LangGraphAgentAdapter(graph=mock_graph)

        with pytest.raises(RuntimeError, match="only available in async mode"):
            await adapter.get_task("some-task-id")

    @pytest.mark.asyncio
    async def test_cancel_task_raises_when_not_async_mode(self):
        """Test that cancel_task raises RuntimeError when not in async mode."""
        mock_graph = MagicMock()
        adapter = LangGraphAgentAdapter(graph=mock_graph)

        with pytest.raises(RuntimeError, match="only available in async mode"):
            await adapter.cancel_task("some-task-id")

    @pytest.mark.asyncio
    async def test_handle_async_returns_task_immediately(self):
        """Test that handle() returns a Task immediately in async mode."""
        mock_graph = MagicMock()

        async def slow_invoke(input_dict):
            await asyncio.sleep(0.5)
            return {"output": "delayed response"}

        mock_graph.ainvoke = slow_invoke

        adapter = LangGraphAgentAdapter(graph=mock_graph, async_mode=True)
        params = make_message_send_params("test message", context_id="async-ctx")

        # handle() should return immediately with a Task
        result = await adapter.handle(params)

        # Verify we got a Task back
        assert isinstance(result, Task)
        assert result.status.state == TaskState.working
        assert result.context_id == "async-ctx"

        # Clean up
        await adapter.close()

    @pytest.mark.asyncio
    async def test_async_task_completes_in_background(self):
        """Test that the task completes in the background and can be polled."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={"output": "workflow result"})

        adapter = LangGraphAgentAdapter(graph=mock_graph, async_mode=True)
        params = make_message_send_params("test message")

        # Get initial task
        task = await adapter.handle(params)
        task_id = task.id

        # Wait for background task to complete
        await asyncio.sleep(0.1)

        # Poll for completed task
        completed_task = await adapter.get_task(task_id)

        assert completed_task is not None
        assert completed_task.status.state == TaskState.completed
        assert completed_task.status.message is not None
        assert "workflow result" in completed_task.status.message.parts[0].root.text

        await adapter.close()

    @pytest.mark.asyncio
    async def test_async_task_handles_failure(self):
        """Test that failed workflows result in failed task state."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("Graph failed"))

        adapter = LangGraphAgentAdapter(graph=mock_graph, async_mode=True)
        params = make_message_send_params("test message")

        # Get initial task
        task = await adapter.handle(params)
        task_id = task.id

        # Wait for background task to fail
        await asyncio.sleep(0.1)

        # Poll for failed task
        failed_task = await adapter.get_task(task_id)

        assert failed_task is not None
        assert failed_task.status.state == TaskState.failed
        assert failed_task.status.message is not None
        assert "failed" in failed_task.status.message.parts[0].root.text.lower()

        await adapter.close()

    @pytest.mark.asyncio
    async def test_cancel_task_marks_task_as_canceled(self):
        """Test that cancel_task() marks the task as canceled."""
        mock_graph = MagicMock()

        async def slow_invoke(input_dict):
            await asyncio.sleep(10)
            return {"output": "result"}

        mock_graph.ainvoke = slow_invoke

        adapter = LangGraphAgentAdapter(graph=mock_graph, async_mode=True)
        params = make_message_send_params("test message")

        # Get initial task
        task = await adapter.handle(params)
        task_id = task.id

        # Cancel the task
        canceled_task = await adapter.cancel_task(task_id)

        assert canceled_task is not None
        assert canceled_task.status.state == TaskState.canceled

        await adapter.close()

    @pytest.mark.asyncio
    async def test_get_task_returns_none_for_unknown_id(self):
        """Test that get_task returns None for unknown task ID."""
        mock_graph = MagicMock()
        adapter = LangGraphAgentAdapter(graph=mock_graph, async_mode=True)

        result = await adapter.get_task("nonexistent-task-id")
        assert result is None

        await adapter.close()

    @pytest.mark.asyncio
    async def test_task_timeout(self):
        """Test that tasks timeout after async_timeout seconds."""
        mock_graph = MagicMock()

        async def very_slow_invoke(input_dict):
            await asyncio.sleep(10)
            return {"output": "result"}

        mock_graph.ainvoke = very_slow_invoke

        adapter = LangGraphAgentAdapter(
            graph=mock_graph,
            async_mode=True,
            async_timeout=1,  # 1 second timeout for testing
        )
        params = make_message_send_params("test message")

        # Get initial task
        task = await adapter.handle(params)
        task_id = task.id

        # Wait for timeout to occur
        await asyncio.sleep(1.5)

        # Task should be failed due to timeout
        timed_out_task = await adapter.get_task(task_id)

        assert timed_out_task is not None
        assert timed_out_task.status.state == TaskState.failed
        assert "timed out" in timed_out_task.status.message.parts[0].root.text.lower()

        await adapter.close()


class TestLangGraphAdapterStreaming:
    """Tests for streaming functionality."""

    def test_supports_streaming_with_astream(self):
        """Test that supports_streaming returns True when graph has astream."""
        mock_graph = MagicMock()
        mock_graph.astream = MagicMock()

        adapter = LangGraphAgentAdapter(graph=mock_graph)
        assert adapter.supports_streaming() is True

    def test_supports_streaming_without_astream(self):
        """Test that supports_streaming returns False when graph lacks astream."""
        mock_graph = MagicMock(spec=[])  # No astream method

        adapter = LangGraphAgentAdapter(graph=mock_graph)
        assert adapter.supports_streaming() is False

    @pytest.mark.asyncio
    async def test_handle_stream_yields_events(self):
        """Test that handle_stream yields proper SSE events."""
        # Create mock graph with async generator
        async def mock_astream(input_dict):
            states = [
                {"output": "Hello"},
                {"output": "Hello World"},
                {"output": "Hello World!"},
            ]
            for state in states:
                yield state

        mock_graph = MagicMock()
        mock_graph.astream = mock_astream

        adapter = LangGraphAgentAdapter(graph=mock_graph)
        params = make_message_send_params("test", context_id="stream-ctx")

        events = []
        async for event in adapter.handle_stream(params):
            events.append(event)

        # Should have message events plus a done event
        message_events = [e for e in events if e["event"] == "message"]
        done_events = [e for e in events if e["event"] == "done"]

        assert len(message_events) >= 1  # At least one message event
        assert len(done_events) == 1


class TestLangGraphAdapterLifecycle:
    """Tests for adapter lifecycle management."""

    @pytest.mark.asyncio
    async def test_close_cancels_pending_background_tasks(self):
        """Test that close() cancels all pending background tasks."""
        mock_graph = MagicMock()

        async def slow_invoke(input_dict):
            await asyncio.sleep(10)
            return {"output": "result"}

        mock_graph.ainvoke = slow_invoke

        adapter = LangGraphAgentAdapter(graph=mock_graph, async_mode=True)
        params = make_message_send_params("test")

        # Start a task
        task = await adapter.handle(params)

        # Verify background task is running
        assert task.id in adapter._background_tasks

        # Close the adapter
        await adapter.close()

        # Verify background tasks are cleared
        assert len(adapter._background_tasks) == 0

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test that adapter works as async context manager."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={"output": "result"})

        async with LangGraphAgentAdapter(graph=mock_graph) as adapter:
            params = make_message_send_params("test")
            result = await adapter.handle(params)
            assert isinstance(result, Message)


class TestLangGraphAdapterLegacySupport:
    """Tests for legacy params.messages format support."""

    @pytest.mark.asyncio
    async def test_to_framework_handles_legacy_messages_format(self):
        """Test that to_framework handles legacy messages array format."""
        mock_graph = MagicMock()
        adapter = LangGraphAgentAdapter(graph=mock_graph, input_key="input")

        # Create legacy-style params
        legacy_message = MagicMock()
        legacy_message.content = "legacy content"

        params = MagicMock(spec=MessageSendParams)
        params.message = None
        params.messages = [legacy_message]

        payload = await adapter.to_framework(params)

        assert payload["input"] == "legacy content"
