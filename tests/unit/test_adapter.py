"""
Unit tests for BaseAgentAdapter.
"""

import pytest
from a2a_adapter.adapter import BaseAgentAdapter
from a2a.types import Message, MessageSendParams, TextContent


class MockAdapter(BaseAgentAdapter):
    """Mock adapter for testing."""

    async def to_framework(self, params: MessageSendParams):
        return {"input": "test"}

    async def call_framework(self, framework_input, params):
        return {"output": "response"}

    async def from_framework(self, framework_output, params):
        return Message(
            role="assistant",
            content=[TextContent(type="text", text=framework_output["output"])]
        )


@pytest.mark.asyncio
async def test_adapter_handle():
    """Test basic adapter handle method."""
    adapter = MockAdapter()
    
    params = MessageSendParams(
        messages=[
            Message(
                role="user",
                content=[TextContent(type="text", text="test message")]
            )
        ]
    )
    
    result = await adapter.handle(params)
    
    assert isinstance(result, Message)
    assert result.role == "assistant"
    assert result.content[0].text == "response"


def test_adapter_supports_streaming_default():
    """Test that adapters don't support streaming by default."""
    adapter = MockAdapter()
    assert adapter.supports_streaming() is False


@pytest.mark.asyncio
async def test_adapter_handle_stream_not_implemented():
    """Test that handle_stream raises NotImplementedError by default."""
    adapter = MockAdapter()
    
    params = MessageSendParams(
        messages=[
            Message(
                role="user",
                content=[TextContent(type="text", text="test message")]
            )
        ]
    )
    
    with pytest.raises(NotImplementedError):
        async for _ in adapter.handle_stream(params):
            pass

