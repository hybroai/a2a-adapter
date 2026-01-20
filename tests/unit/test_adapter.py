"""
Unit tests for BaseAgentAdapter.
"""

import pytest
from a2a_adapter.adapter import BaseAgentAdapter
from a2a.types import Message, MessageSendParams, TextPart, Role, Part


class MockAdapter(BaseAgentAdapter):
    """Mock adapter for testing."""

    async def to_framework(self, params: MessageSendParams):
        return {"input": "test"}

    async def call_framework(self, framework_input, params):
        return {"output": "response"}

    async def from_framework(self, framework_output, params):
        return Message(
            role=Role.agent,
            message_id="test-response-id",
            parts=[Part(root=TextPart(text=framework_output["output"]))]
        )


def make_message_send_params(text: str) -> MessageSendParams:
    """Helper to create MessageSendParams with correct A2A types."""
    return MessageSendParams(
        message=Message(
            message_id="test-msg-id",
            role=Role.user,
            parts=[Part(root=TextPart(text=text))],
        )
    )


@pytest.mark.asyncio
async def test_adapter_handle():
    """Test basic adapter handle method."""
    adapter = MockAdapter()
    
    params = make_message_send_params("test message")
    
    result = await adapter.handle(params)
    
    assert isinstance(result, Message)
    assert result.role == Role.agent
    assert result.parts[0].root.text == "response"


def test_adapter_supports_streaming_default():
    """Test that adapters don't support streaming by default."""
    adapter = MockAdapter()
    assert adapter.supports_streaming() is False


@pytest.mark.asyncio
async def test_adapter_handle_stream_not_implemented():
    """Test that handle_stream raises NotImplementedError by default."""
    adapter = MockAdapter()
    
    params = make_message_send_params("test message")
    
    with pytest.raises(NotImplementedError):
        await adapter.handle_stream(params)


def test_adapter_supports_async_tasks_default():
    """Test that adapters don't support async tasks by default."""
    adapter = MockAdapter()
    assert adapter.supports_async_tasks() is False


@pytest.mark.asyncio
async def test_adapter_get_task_not_implemented():
    """Test that get_task raises NotImplementedError by default."""
    adapter = MockAdapter()
    
    with pytest.raises(NotImplementedError):
        await adapter.get_task("some-task-id")


@pytest.mark.asyncio
async def test_adapter_cancel_task_not_implemented():
    """Test that cancel_task raises NotImplementedError by default."""
    adapter = MockAdapter()
    
    with pytest.raises(NotImplementedError):
        await adapter.cancel_task("some-task-id")
