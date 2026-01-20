"""
Unit tests for CallableAgentAdapter.
"""

import pytest
from a2a_adapter.integrations.callable import CallableAgentAdapter
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
    
    params = make_message_send_params("hello")
    
    result = await adapter.handle(params)
    
    assert isinstance(result, Message)
    assert result.role == Role.agent
    assert "Echo: hello" in result.parts[0].root.text


@pytest.mark.asyncio
async def test_callable_adapter_dict_response():
    """Test callable adapter with dict response."""
    adapter = CallableAgentAdapter(func=dict_response)
    
    params = make_message_send_params("test")
    
    result = await adapter.handle(params)
    
    assert isinstance(result, Message)
    assert "Processed: test" in result.parts[0].root.text


def test_callable_adapter_no_streaming_by_default():
    """Test that callable adapter doesn't support streaming by default."""
    adapter = CallableAgentAdapter(func=simple_echo)
    assert adapter.supports_streaming() is False


def test_callable_adapter_streaming_enabled():
    """Test that callable adapter can be configured for streaming."""
    adapter = CallableAgentAdapter(func=simple_echo, supports_streaming=True)
    assert adapter.supports_streaming() is True


@pytest.mark.asyncio
async def test_callable_adapter_context_id():
    """Test that callable adapter preserves context_id."""
    adapter = CallableAgentAdapter(func=simple_echo)
    
    params = make_message_send_params("hello", context_id="ctx-123")
    
    result = await adapter.handle(params)
    
    assert isinstance(result, Message)
    assert result.context_id == "ctx-123"


@pytest.mark.asyncio
async def test_callable_adapter_streaming():
    """Test callable adapter streaming mode."""
    async def streaming_func(inputs: dict):
        chunks = ["Hello", " ", "World"]
        for chunk in chunks:
            yield chunk
    
    adapter = CallableAgentAdapter(func=streaming_func, supports_streaming=True)
    params = make_message_send_params("test", context_id="stream-ctx")
    
    events = []
    async for event in adapter.handle_stream(params):
        events.append(event)
    
    # Should have message events plus a done event
    message_events = [e for e in events if e["event"] == "message"]
    done_events = [e for e in events if e["event"] == "done"]
    
    assert len(message_events) == 3
    assert len(done_events) == 1


@pytest.mark.asyncio
async def test_callable_adapter_streaming_not_enabled():
    """Test that handle_stream raises when streaming not enabled."""
    adapter = CallableAgentAdapter(func=simple_echo, supports_streaming=False)
    params = make_message_send_params("test")
    
    with pytest.raises(NotImplementedError):
        async for _ in adapter.handle_stream(params):
            pass
