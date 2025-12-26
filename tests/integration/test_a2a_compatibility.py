#!/usr/bin/env python3
"""
Integration tests to verify A2A compatibility of the n8n adapter.

These tests verify that the n8n adapter properly creates A2A-compliant
Message objects that other A2A agents can accept.
"""

import pytest
from a2a_adapters.integrations.n8n import N8nAgentAdapter
from a2a.types import MessageSendParams, Message, Role, Part, TextPart


@pytest.fixture
def adapter():
    """Create an N8nAgentAdapter instance for testing."""
    return N8nAgentAdapter('https://n8n.example.com/webhook')


@pytest.fixture
def user_params():
    """Create mock input parameters (like from another A2A agent)."""
    user_message = Message(
        role=Role.user,
        message_id='incoming-msg-123',
        parts=[Part(root=TextPart(text='What is AI?'))]
    )
    return MessageSendParams(message=user_message)


@pytest.mark.asyncio
@pytest.mark.parametrize("n8n_response", [
    pytest.param({'output': 'Hello from n8n workflow!'}, id='direct_output_field'),
    pytest.param({'result': 'Calculation result: 42'}, id='result_field'),
    pytest.param({'message': 'Custom message response'}, id='message_field'),
    pytest.param({'data': {'key': 'value'}}, id='fallback_json_serialization'),
])
async def test_n8n_adapter_creates_a2a_compatible_messages(
    adapter, user_params, n8n_response
):
    """Test that the n8n adapter creates A2A-compatible messages for various response formats."""
    # Test the from_framework method
    result = await adapter.from_framework(n8n_response, user_params)

    # Verify A2A compliance
    assert result.role == Role.agent, f"Role should be Role.agent, got {result.role}"
    assert result.message_id, "Message should have an ID"
    assert len(result.parts) == 1, "Message should have exactly 1 part"
    assert isinstance(result.parts[0].root, TextPart), "Part should contain TextPart"


@pytest.mark.asyncio
@pytest.mark.parametrize("n8n_response", [
    {'output': 'Hello from n8n workflow!'},
    {'result': 'Calculation result: 42'},
    {'message': 'Custom message response'},
    {'data': {'key': 'value'}},
])
async def test_n8n_adapter_json_serialization(adapter, user_params, n8n_response):
    """Test that n8n adapter messages serialize to valid JSON-RPC format."""
    result = await adapter.from_framework(n8n_response, user_params)

    # Test JSON serialization (critical for A2A protocol)
    json_data = result.model_dump()
    assert json_data['role'] == 'agent', "JSON role should be 'agent'"
    assert 'messageId' in json_data, "JSON should have camelCase messageId"
    assert 'parts' in json_data, "JSON should have parts array"


@pytest.mark.asyncio
async def test_n8n_adapter_message_has_unique_id(adapter, user_params):
    """Test that each message gets a unique ID."""
    response = {'output': 'Test response'}
    
    result1 = await adapter.from_framework(response, user_params)
    result2 = await adapter.from_framework(response, user_params)
    
    assert result1.message_id != result2.message_id, "Each message should have a unique ID"
