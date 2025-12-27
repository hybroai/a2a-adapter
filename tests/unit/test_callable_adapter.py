"""
Unit tests for CallableAgentAdapter.
"""

import pytest
from a2a_adapter.integrations.callable import CallableAgentAdapter
from a2a.types import Message, MessageSendParams, TextContent


async def simple_echo(inputs: dict) -> str:
    """Simple echo function for testing."""
    return f"Echo: {inputs['message']}"


async def dict_response(inputs: dict) -> dict:
    """Function that returns a dict."""
    return {"response": f"Processed: {inputs['message']}"}


@pytest.mark.asyncio
async def test_callable_adapter_string_response():
    """Test callable adapter with string response."""
    adapter = CallableAgentAdapter(func=simple_echo)
    
    params = MessageSendParams(
        messages=[
            Message(
                role="user",
                content=[TextContent(type="text", text="hello")]
            )
        ]
    )
    
    result = await adapter.handle(params)
    
    assert isinstance(result, Message)
    assert result.role == "assistant"
    assert "Echo: hello" in result.content[0].text


@pytest.mark.asyncio
async def test_callable_adapter_dict_response():
    """Test callable adapter with dict response."""
    adapter = CallableAgentAdapter(func=dict_response)
    
    params = MessageSendParams(
        messages=[
            Message(
                role="user",
                content=[TextContent(type="text", text="test")]
            )
        ]
    )
    
    result = await adapter.handle(params)
    
    assert isinstance(result, Message)
    assert "Processed: test" in result.content[0].text


def test_callable_adapter_no_streaming_by_default():
    """Test that callable adapter doesn't support streaming by default."""
    adapter = CallableAgentAdapter(func=simple_echo)
    assert adapter.supports_streaming() is False


def test_callable_adapter_streaming_enabled():
    """Test that callable adapter can be configured for streaming."""
    adapter = CallableAgentAdapter(func=simple_echo, supports_streaming=True)
    assert adapter.supports_streaming() is True

