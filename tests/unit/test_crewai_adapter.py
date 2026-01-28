"""
Unit tests for CrewAIAgentAdapter.
"""

import asyncio
import json
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


def make_message_send_params_with_dict_text(data: dict, context_id: str | None = None) -> MessageSendParams:
    """Helper to create MessageSendParams where text is a dict (edge case)."""
    # Create a mock Part where root.text returns a dict
    mock_part = MagicMock()
    mock_part.root.text = data  # This is the edge case: text is dict, not str

    mock_message = MagicMock()
    mock_message.parts = [mock_part]
    mock_message.context_id = context_id

    params = MagicMock(spec=MessageSendParams)
    params.message = mock_message
    return params


class TestCrewAIAdapterBasic:
    """Basic functionality tests for CrewAIAgentAdapter."""

    @pytest.mark.asyncio
    async def test_to_framework_extracts_message_text(self):
        """Test that to_framework extracts message text correctly (plain text fallback)."""
        mock_crew = MagicMock()
        # Disable JSON parsing to test plain text fallback
        adapter = CrewAIAgentAdapter(crew=mock_crew, parse_json_input=False)

        params = make_message_send_params("hello world")
        payload = await adapter.to_framework(params)

        assert payload["message"] == "hello world"
        assert payload["inputs"] == "hello world"

    @pytest.mark.asyncio
    async def test_to_framework_custom_inputs_key(self):
        """Test that custom inputs_key is used for plain text."""
        mock_crew = MagicMock()
        adapter = CrewAIAgentAdapter(crew=mock_crew, inputs_key="query", parse_json_input=False)

        params = make_message_send_params("test query")
        payload = await adapter.to_framework(params)

        assert payload["query"] == "test query"
        assert payload["message"] == "test query"

    @pytest.mark.asyncio
    async def test_to_framework_handles_multiple_parts(self):
        """Test that multiple text parts are joined correctly."""
        mock_crew = MagicMock()
        adapter = CrewAIAgentAdapter(crew=mock_crew, parse_json_input=False)

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


class TestCrewAIAdapterDictTextHandling:
    """Tests for handling dict type in part.root.text (edge case fix)."""

    @pytest.mark.asyncio
    async def test_to_framework_handles_dict_text(self):
        """Test that dict type in part.root.text is handled correctly."""
        mock_crew = MagicMock()
        adapter = CrewAIAgentAdapter(crew=mock_crew)

        # Create params where text is a dict (edge case)
        params = make_message_send_params_with_dict_text(
            {"customer_domain": "example.com", "project_description": "Test project"},
            context_id="ctx-123"
        )
        payload = await adapter.to_framework(params)

        # Should parse the dict directly as JSON input
        assert payload["customer_domain"] == "example.com"
        assert payload["project_description"] == "Test project"

    @pytest.mark.asyncio
    async def test_extract_raw_input_converts_dict_to_json(self):
        """Test that _extract_raw_input converts dict to JSON string."""
        mock_crew = MagicMock()
        adapter = CrewAIAgentAdapter(crew=mock_crew, parse_json_input=False)

        # Create params where text is a dict
        params = make_message_send_params_with_dict_text(
            {"key": "value"},
            context_id="ctx-123"
        )
        raw_input = adapter._extract_raw_input(params)

        # Should be a JSON string
        assert isinstance(raw_input, str)
        parsed = json.loads(raw_input)
        assert parsed["key"] == "value"


class TestCrewAIAdapterJSONParsing:
    """Tests for automatic JSON input parsing."""

    @pytest.mark.asyncio
    async def test_to_framework_parses_json_input(self):
        """Test that JSON input is automatically parsed."""
        mock_crew = MagicMock()
        adapter = CrewAIAgentAdapter(crew=mock_crew)  # parse_json_input=True by default

        json_input = json.dumps({
            "customer_domain": "example.com",
            "project_description": "Build a marketing strategy"
        })
        params = make_message_send_params(json_input, context_id="ctx-456")
        payload = await adapter.to_framework(params)

        # Should parse JSON and use keys directly
        assert payload["customer_domain"] == "example.com"
        assert payload["project_description"] == "Build a marketing strategy"
        assert payload["context_id"] == "ctx-456"

    @pytest.mark.asyncio
    async def test_to_framework_falls_back_for_plain_text(self):
        """Test fallback to inputs_key for non-JSON input."""
        mock_crew = MagicMock()
        adapter = CrewAIAgentAdapter(crew=mock_crew)

        # Plain text that's not valid JSON
        params = make_message_send_params("Create a marketing campaign", context_id="ctx-789")
        payload = await adapter.to_framework(params)

        # Should fallback to inputs_key mode
        assert payload["inputs"] == "Create a marketing campaign"
        assert payload["message"] == "Create a marketing campaign"

    @pytest.mark.asyncio
    async def test_to_framework_disables_json_parsing(self):
        """Test that JSON parsing can be disabled."""
        mock_crew = MagicMock()
        adapter = CrewAIAgentAdapter(crew=mock_crew, parse_json_input=False)

        json_input = json.dumps({"customer_domain": "example.com"})
        params = make_message_send_params(json_input)
        payload = await adapter.to_framework(params)

        # Should NOT parse JSON, use inputs_key instead
        assert "customer_domain" not in payload
        assert payload["inputs"] == json_input


class TestCrewAIAdapterInputMapper:
    """Tests for custom input_mapper function."""

    @pytest.mark.asyncio
    async def test_to_framework_uses_input_mapper(self):
        """Test that custom input_mapper is used when provided."""
        mock_crew = MagicMock()

        def custom_mapper(raw_input: str, context_id: str | None) -> dict:
            # Custom parsing logic
            return {
                "customer_domain": "mapped.com",
                "raw_query": raw_input,
            }

        adapter = CrewAIAgentAdapter(crew=mock_crew, input_mapper=custom_mapper)
        params = make_message_send_params("any input", context_id="ctx-mapper")
        payload = await adapter.to_framework(params)

        assert payload["customer_domain"] == "mapped.com"
        assert payload["raw_query"] == "any input"
        assert payload["context_id"] == "ctx-mapper"

    @pytest.mark.asyncio
    async def test_input_mapper_takes_priority_over_json(self):
        """Test that input_mapper takes priority over JSON parsing."""
        mock_crew = MagicMock()

        def override_mapper(raw_input: str, context_id: str | None) -> dict:
            return {"overridden": True}

        adapter = CrewAIAgentAdapter(
            crew=mock_crew,
            input_mapper=override_mapper,
            parse_json_input=True
        )

        json_input = json.dumps({"customer_domain": "example.com"})
        params = make_message_send_params(json_input)
        payload = await adapter.to_framework(params)

        # input_mapper should override JSON parsing
        assert payload["overridden"] is True
        assert "customer_domain" not in payload

    @pytest.mark.asyncio
    async def test_input_mapper_failure_falls_back_to_json(self):
        """Test that failed input_mapper falls back to JSON parsing."""
        mock_crew = MagicMock()

        def failing_mapper(raw_input: str, context_id: str | None) -> dict:
            raise ValueError("Intentional failure")

        adapter = CrewAIAgentAdapter(crew=mock_crew, input_mapper=failing_mapper)

        json_input = json.dumps({"fallback_key": "fallback_value"})
        params = make_message_send_params(json_input)
        payload = await adapter.to_framework(params)

        # Should fall back to JSON parsing
        assert payload["fallback_key"] == "fallback_value"


class TestCrewAIAdapterDefaultInputs:
    """Tests for default_inputs parameter."""

    @pytest.mark.asyncio
    async def test_default_inputs_merged_with_json(self):
        """Test that default_inputs are merged with parsed JSON."""
        mock_crew = MagicMock()
        adapter = CrewAIAgentAdapter(
            crew=mock_crew,
            default_inputs={"default_key": "default_value", "customer_domain": "default.com"}
        )

        # JSON input overrides customer_domain but default_key should remain
        json_input = json.dumps({"customer_domain": "user.com"})
        params = make_message_send_params(json_input)
        payload = await adapter.to_framework(params)

        assert payload["customer_domain"] == "user.com"  # Overridden by user
        assert payload["default_key"] == "default_value"  # Default preserved

    @pytest.mark.asyncio
    async def test_default_inputs_used_with_plain_text(self):
        """Test that default_inputs are used with plain text input."""
        mock_crew = MagicMock()
        adapter = CrewAIAgentAdapter(
            crew=mock_crew,
            default_inputs={"customer_domain": "default.com"},
            parse_json_input=False
        )

        params = make_message_send_params("plain text query")
        payload = await adapter.to_framework(params)

        assert payload["customer_domain"] == "default.com"
        assert payload["inputs"] == "plain text query"


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
