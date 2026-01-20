"""
Unit tests for LangChainAgentAdapter.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from a2a_adapter.integrations.langchain import LangChainAgentAdapter
from a2a.types import Message, MessageSendParams, TextPart, Role, Part


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


class TestLangChainAdapterBasic:
    """Basic functionality tests for LangChainAgentAdapter."""

    @pytest.mark.asyncio
    async def test_to_framework_extracts_message_text(self):
        """Test that to_framework extracts message text correctly."""
        mock_runnable = MagicMock()
        adapter = LangChainAgentAdapter(runnable=mock_runnable, input_key="input")

        params = make_message_send_params("hello world")
        payload = await adapter.to_framework(params)

        assert payload["input"] == "hello world"

    @pytest.mark.asyncio
    async def test_to_framework_custom_input_key(self):
        """Test that custom input_key is used."""
        mock_runnable = MagicMock()
        adapter = LangChainAgentAdapter(runnable=mock_runnable, input_key="query")

        params = make_message_send_params("test query")
        payload = await adapter.to_framework(params)

        assert payload["query"] == "test query"
        assert "input" not in payload

    @pytest.mark.asyncio
    async def test_to_framework_handles_multiple_parts(self):
        """Test that multiple text parts are joined correctly."""
        mock_runnable = MagicMock()
        adapter = LangChainAgentAdapter(runnable=mock_runnable)

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


class TestLangChainAdapterContextId:
    """Tests for context_id handling in LangChainAgentAdapter."""

    @pytest.mark.asyncio
    async def test_from_framework_preserves_context_id(self):
        """Test that from_framework preserves context_id on the response Message."""
        mock_runnable = MagicMock()
        adapter = LangChainAgentAdapter(runnable=mock_runnable)

        params = make_message_send_params("hello", context_id="ctx-123")
        framework_output = "response text"

        result = await adapter.from_framework(framework_output, params)

        assert isinstance(result, Message)
        assert result.context_id == "ctx-123"
        assert result.role == Role.agent

    @pytest.mark.asyncio
    async def test_from_framework_handles_missing_context_id(self):
        """Test that from_framework handles missing context_id gracefully."""
        mock_runnable = MagicMock()
        adapter = LangChainAgentAdapter(runnable=mock_runnable)

        params = make_message_send_params("hello", context_id=None)
        framework_output = "response text"

        result = await adapter.from_framework(framework_output, params)

        assert isinstance(result, Message)
        assert result.context_id is None


class TestLangChainAdapterOutputExtraction:
    """Tests for output extraction from various LangChain output types."""

    @pytest.mark.asyncio
    async def test_from_framework_extracts_string_output(self):
        """Test extraction from simple string output."""
        mock_runnable = MagicMock()
        adapter = LangChainAgentAdapter(runnable=mock_runnable)

        params = make_message_send_params("hello")
        framework_output = "simple string response"

        result = await adapter.from_framework(framework_output, params)

        assert result.parts[0].root.text == "simple string response"

    @pytest.mark.asyncio
    async def test_from_framework_extracts_ai_message_content(self):
        """Test extraction from AIMessage-like object with content attribute."""
        mock_runnable = MagicMock()
        adapter = LangChainAgentAdapter(runnable=mock_runnable)

        # Mock AIMessage-like object
        mock_ai_message = MagicMock()
        mock_ai_message.content = "AI response content"

        params = make_message_send_params("hello")
        result = await adapter.from_framework(mock_ai_message, params)

        assert result.parts[0].root.text == "AI response content"

    @pytest.mark.asyncio
    async def test_from_framework_extracts_dict_with_output_key(self):
        """Test extraction from dict using configured output_key."""
        mock_runnable = MagicMock()
        adapter = LangChainAgentAdapter(
            runnable=mock_runnable,
            output_key="custom_output"
        )

        params = make_message_send_params("hello")
        framework_output = {"custom_output": "extracted value", "other": "ignored"}

        result = await adapter.from_framework(framework_output, params)

        assert result.parts[0].root.text == "extracted value"

    @pytest.mark.asyncio
    async def test_from_framework_extracts_dict_common_keys(self):
        """Test extraction from dict using common output keys."""
        mock_runnable = MagicMock()
        adapter = LangChainAgentAdapter(runnable=mock_runnable)

        params = make_message_send_params("hello")

        # Test 'output' key
        result = await adapter.from_framework({"output": "from output"}, params)
        assert result.parts[0].root.text == "from output"

        # Test 'result' key
        result = await adapter.from_framework({"result": "from result"}, params)
        assert result.parts[0].root.text == "from result"

        # Test 'answer' key
        result = await adapter.from_framework({"answer": "from answer"}, params)
        assert result.parts[0].root.text == "from answer"


class TestLangChainAdapterCallFramework:
    """Tests for call_framework execution."""

    @pytest.mark.asyncio
    async def test_call_framework_invokes_runnable(self):
        """Test that call_framework calls runnable.ainvoke()."""
        mock_runnable = MagicMock()
        mock_runnable.ainvoke = AsyncMock(return_value="runnable result")

        adapter = LangChainAgentAdapter(runnable=mock_runnable)
        params = make_message_send_params("test")
        framework_input = {"input": "test message"}

        result = await adapter.call_framework(framework_input, params)

        mock_runnable.ainvoke.assert_called_once_with(framework_input)
        assert result == "runnable result"


class TestLangChainAdapterHandle:
    """Tests for the complete handle flow."""

    @pytest.mark.asyncio
    async def test_handle_end_to_end(self):
        """Test complete handle() flow from params to message."""
        mock_runnable = MagicMock()
        mock_runnable.ainvoke = AsyncMock(return_value="processed response")

        adapter = LangChainAgentAdapter(runnable=mock_runnable, input_key="input")
        params = make_message_send_params("test message", context_id="e2e-ctx")

        result = await adapter.handle(params)

        assert isinstance(result, Message)
        assert result.role == Role.agent
        assert result.context_id == "e2e-ctx"
        assert result.parts[0].root.text == "processed response"

        # Verify runnable was called with correct input
        call_args = mock_runnable.ainvoke.call_args
        assert call_args[0][0]["input"] == "test message"


class TestLangChainAdapterStreaming:
    """Tests for streaming functionality."""

    def test_supports_streaming_with_astream(self):
        """Test that supports_streaming returns True when runnable has astream."""
        mock_runnable = MagicMock()
        mock_runnable.astream = MagicMock()

        adapter = LangChainAgentAdapter(runnable=mock_runnable)
        assert adapter.supports_streaming() is True

    def test_supports_streaming_without_astream(self):
        """Test that supports_streaming returns False when runnable lacks astream."""
        mock_runnable = MagicMock(spec=[])  # No astream method

        adapter = LangChainAgentAdapter(runnable=mock_runnable)
        assert adapter.supports_streaming() is False

    @pytest.mark.asyncio
    async def test_handle_stream_yields_events(self):
        """Test that handle_stream yields proper SSE events."""
        # Create mock runnable with async generator
        async def mock_astream(input_dict):
            chunks = ["Hello", " ", "World"]
            for chunk in chunks:
                mock_chunk = MagicMock()
                mock_chunk.content = chunk
                yield mock_chunk

        mock_runnable = MagicMock()
        mock_runnable.astream = mock_astream

        adapter = LangChainAgentAdapter(runnable=mock_runnable)
        params = make_message_send_params("test", context_id="stream-ctx")

        events = []
        async for event in adapter.handle_stream(params):
            events.append(event)

        # Should have message events for each chunk plus a done event
        message_events = [e for e in events if e["event"] == "message"]
        done_events = [e for e in events if e["event"] == "done"]

        assert len(message_events) == 3
        assert len(done_events) == 1


class TestLangChainAdapterLegacySupport:
    """Tests for legacy params.messages format support."""

    @pytest.mark.asyncio
    async def test_to_framework_handles_legacy_messages_format(self):
        """Test that to_framework handles legacy messages array format."""
        mock_runnable = MagicMock()
        adapter = LangChainAgentAdapter(runnable=mock_runnable)

        # Create legacy-style params
        legacy_message = MagicMock()
        legacy_message.content = "legacy content"

        params = MagicMock(spec=MessageSendParams)
        params.message = None
        params.messages = [legacy_message]

        payload = await adapter.to_framework(params)

        assert payload["input"] == "legacy content"
