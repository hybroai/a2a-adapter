"""
Unit tests for CrewAIAgentAdapter.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from a2a_adapter.integrations.crewai import CrewAIAgentAdapter
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


class TestCrewAIAdapterBasic:
    """Basic functionality tests for CrewAIAgentAdapter."""

    @pytest.mark.asyncio
    async def test_to_framework_extracts_message_text(self):
        """Test that to_framework extracts message text correctly."""
        mock_crew = MagicMock()
        adapter = CrewAIAgentAdapter(crew=mock_crew)

        params = make_message_send_params("hello world")
        payload = await adapter.to_framework(params)

        assert payload["message"] == "hello world"
        assert payload["inputs"] == "hello world"

    @pytest.mark.asyncio
    async def test_to_framework_custom_inputs_key(self):
        """Test that custom inputs_key is used."""
        mock_crew = MagicMock()
        adapter = CrewAIAgentAdapter(crew=mock_crew, inputs_key="query")

        params = make_message_send_params("test query")
        payload = await adapter.to_framework(params)

        assert payload["query"] == "test query"
        assert payload["message"] == "test query"

    @pytest.mark.asyncio
    async def test_to_framework_handles_multiple_parts(self):
        """Test that multiple text parts are joined correctly."""
        mock_crew = MagicMock()
        adapter = CrewAIAgentAdapter(crew=mock_crew)

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

        assert "part one" in payload["message"]
        assert "part two" in payload["message"]


class TestCrewAIAdapterContextId:
    """Tests for context_id handling in CrewAIAgentAdapter."""

    @pytest.mark.asyncio
    async def test_to_framework_extracts_context_id(self):
        """Test that to_framework extracts context_id from params.message."""
        mock_crew = MagicMock()
        adapter = CrewAIAgentAdapter(crew=mock_crew)

        params = make_message_send_params("hello", context_id="ctx-123")
        payload = await adapter.to_framework(params)

        assert payload["context_id"] == "ctx-123"

    @pytest.mark.asyncio
    async def test_from_framework_preserves_context_id(self):
        """Test that from_framework preserves context_id on the response Message."""
        mock_crew = MagicMock()
        adapter = CrewAIAgentAdapter(crew=mock_crew)

        params = make_message_send_params("hello", context_id="ctx-789")
        framework_output = "crew result"

        result = await adapter.from_framework(framework_output, params)

        assert isinstance(result, Message)
        assert result.context_id == "ctx-789"
        assert result.role == Role.agent

    @pytest.mark.asyncio
    async def test_from_framework_handles_missing_context_id(self):
        """Test that from_framework handles missing context_id gracefully."""
        mock_crew = MagicMock()
        adapter = CrewAIAgentAdapter(crew=mock_crew)

        params = make_message_send_params("hello", context_id=None)
        framework_output = "crew result"

        result = await adapter.from_framework(framework_output, params)

        assert isinstance(result, Message)
        assert result.context_id is None


class TestCrewAIAdapterOutputExtraction:
    """Tests for output extraction from various CrewAI output types."""

    @pytest.mark.asyncio
    async def test_from_framework_extracts_string_output(self):
        """Test extraction from simple string output."""
        mock_crew = MagicMock()
        adapter = CrewAIAgentAdapter(crew=mock_crew)

        params = make_message_send_params("hello")
        result = await adapter.from_framework("simple string response", params)

        assert result.parts[0].root.text == "simple string response"

    @pytest.mark.asyncio
    async def test_from_framework_extracts_crew_output_raw(self):
        """Test extraction from CrewOutput object with raw attribute."""
        mock_crew = MagicMock()
        adapter = CrewAIAgentAdapter(crew=mock_crew)

        # Mock CrewOutput-like object
        mock_crew_output = MagicMock()
        mock_crew_output.raw = "raw crew output"

        params = make_message_send_params("hello")
        result = await adapter.from_framework(mock_crew_output, params)

        assert result.parts[0].root.text == "raw crew output"

    @pytest.mark.asyncio
    async def test_from_framework_extracts_crew_output_result(self):
        """Test extraction from CrewOutput object with result attribute."""
        mock_crew = MagicMock()
        adapter = CrewAIAgentAdapter(crew=mock_crew)

        # Mock CrewOutput-like object without raw but with result
        mock_crew_output = MagicMock(spec=["result"])
        mock_crew_output.result = "result crew output"

        params = make_message_send_params("hello")
        result = await adapter.from_framework(mock_crew_output, params)

        assert result.parts[0].root.text == "result crew output"

    @pytest.mark.asyncio
    async def test_from_framework_extracts_dict_output(self):
        """Test extraction from dict output."""
        mock_crew = MagicMock()
        adapter = CrewAIAgentAdapter(crew=mock_crew)

        params = make_message_send_params("hello")
        framework_output = {"output": "dict output value"}

        result = await adapter.from_framework(framework_output, params)

        assert result.parts[0].root.text == "dict output value"


class TestCrewAIAdapterCallFramework:
    """Tests for call_framework execution."""

    @pytest.mark.asyncio
    async def test_call_framework_uses_kickoff_async(self):
        """Test that call_framework calls crew.kickoff_async()."""
        mock_crew = MagicMock()
        mock_crew.kickoff_async = AsyncMock(return_value="crew result")

        adapter = CrewAIAgentAdapter(crew=mock_crew)
        params = make_message_send_params("test")
        framework_input = {"inputs": "test message", "message": "test message"}

        result = await adapter.call_framework(framework_input, params)

        mock_crew.kickoff_async.assert_called_once_with(inputs=framework_input)
        assert result == "crew result"

    @pytest.mark.asyncio
    async def test_call_framework_fallback_to_sync(self):
        """Test that call_framework falls back to sync kickoff if async not available."""
        mock_crew = MagicMock()
        # Remove kickoff_async to simulate old CrewAI
        del mock_crew.kickoff_async
        mock_crew.kickoff = MagicMock(return_value="sync result")

        adapter = CrewAIAgentAdapter(crew=mock_crew)
        params = make_message_send_params("test")
        framework_input = {"inputs": "test"}

        result = await adapter.call_framework(framework_input, params)

        assert result == "sync result"


class TestCrewAIAdapterHandle:
    """Tests for the complete handle flow."""

    @pytest.mark.asyncio
    async def test_handle_end_to_end(self):
        """Test complete handle() flow from params to message."""
        mock_crew = MagicMock()
        mock_crew.kickoff_async = AsyncMock(return_value="processed response")

        adapter = CrewAIAgentAdapter(crew=mock_crew)
        params = make_message_send_params("test message", context_id="e2e-ctx")

        result = await adapter.handle(params)

        assert isinstance(result, Message)
        assert result.role == Role.agent
        assert result.context_id == "e2e-ctx"
        assert result.parts[0].root.text == "processed response"


class TestCrewAIAdapterAsyncMode:
    """Tests for async task mode in CrewAIAgentAdapter."""

    def test_supports_async_tasks_false_by_default(self):
        """Test that async mode is disabled by default."""
        mock_crew = MagicMock()
        adapter = CrewAIAgentAdapter(crew=mock_crew)
        assert adapter.supports_async_tasks() is False

    def test_supports_async_tasks_true_when_enabled(self):
        """Test that async mode can be enabled."""
        mock_crew = MagicMock()
        adapter = CrewAIAgentAdapter(crew=mock_crew, async_mode=True)
        assert adapter.supports_async_tasks() is True

    def test_async_mode_creates_task_store(self):
        """Test that async mode creates an InMemoryTaskStore by default."""
        mock_crew = MagicMock()
        adapter = CrewAIAgentAdapter(crew=mock_crew, async_mode=True)
        assert adapter.task_store is not None
        assert isinstance(adapter.task_store, InMemoryTaskStore)

    @pytest.mark.asyncio
    async def test_get_task_raises_when_not_async_mode(self):
        """Test that get_task raises RuntimeError when not in async mode."""
        mock_crew = MagicMock()
        adapter = CrewAIAgentAdapter(crew=mock_crew)

        with pytest.raises(RuntimeError, match="only available in async mode"):
            await adapter.get_task("some-task-id")

    @pytest.mark.asyncio
    async def test_cancel_task_raises_when_not_async_mode(self):
        """Test that cancel_task raises RuntimeError when not in async mode."""
        mock_crew = MagicMock()
        adapter = CrewAIAgentAdapter(crew=mock_crew)

        with pytest.raises(RuntimeError, match="only available in async mode"):
            await adapter.cancel_task("some-task-id")

    @pytest.mark.asyncio
    async def test_handle_async_returns_task_immediately(self):
        """Test that handle() returns a Task immediately in async mode."""
        mock_crew = MagicMock()
        mock_crew.kickoff_async = AsyncMock(return_value="delayed response")

        adapter = CrewAIAgentAdapter(crew=mock_crew, async_mode=True)
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
        mock_crew = MagicMock()
        mock_crew.kickoff_async = AsyncMock(return_value="crew result")

        adapter = CrewAIAgentAdapter(crew=mock_crew, async_mode=True)
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
        assert "crew result" in completed_task.status.message.parts[0].root.text

        await adapter.close()

    @pytest.mark.asyncio
    async def test_cancel_task_marks_task_as_canceled(self):
        """Test that cancel_task() marks the task as canceled."""
        mock_crew = MagicMock()

        # Mock a slow kickoff
        async def slow_kickoff(inputs):
            await asyncio.sleep(10)
            return "result"

        mock_crew.kickoff_async = slow_kickoff

        adapter = CrewAIAgentAdapter(crew=mock_crew, async_mode=True)
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
        mock_crew = MagicMock()
        adapter = CrewAIAgentAdapter(crew=mock_crew, async_mode=True)

        result = await adapter.get_task("nonexistent-task-id")
        assert result is None

        await adapter.close()


class TestCrewAIAdapterStreaming:
    """Tests for streaming (not supported)."""

    def test_supports_streaming_returns_false(self):
        """Test that CrewAI adapter does not support streaming."""
        mock_crew = MagicMock()
        adapter = CrewAIAgentAdapter(crew=mock_crew)
        assert adapter.supports_streaming() is False


class TestCrewAIAdapterLegacySupport:
    """Tests for legacy params.messages format support."""

    @pytest.mark.asyncio
    async def test_to_framework_handles_legacy_messages_format(self):
        """Test that to_framework handles legacy messages array format."""
        mock_crew = MagicMock()
        adapter = CrewAIAgentAdapter(crew=mock_crew)

        # Create legacy-style params
        legacy_message = MagicMock()
        legacy_message.content = "legacy content"

        params = MagicMock(spec=MessageSendParams)
        params.message = None
        params.messages = [legacy_message]

        payload = await adapter.to_framework(params)

        assert payload["message"] == "legacy content"
