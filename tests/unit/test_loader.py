"""
Unit tests for adapter loader.
"""

import pytest
from a2a_adapter.loader import load_a2a_agent
from a2a_adapter.integrations.callable import CallableAgentAdapter


async def mock_callable(inputs: dict) -> str:
    """Mock callable for testing."""
    return "test response"


@pytest.mark.asyncio
async def test_load_callable_adapter():
    """Test loading a callable adapter."""
    adapter = await load_a2a_agent({
        "adapter": "callable",
        "callable": mock_callable
    })
    
    assert isinstance(adapter, CallableAgentAdapter)


@pytest.mark.asyncio
async def test_load_adapter_missing_type():
    """Test that loader raises error when adapter type is missing."""
    with pytest.raises(ValueError, match="Config must include 'adapter' key"):
        await load_a2a_agent({})


@pytest.mark.asyncio
async def test_load_adapter_unknown_type():
    """Test that loader raises error for unknown adapter type."""
    with pytest.raises(ValueError, match="Unknown adapter type"):
        await load_a2a_agent({"adapter": "unknown_adapter"})


@pytest.mark.asyncio
async def test_load_n8n_adapter_missing_url():
    """Test that n8n adapter requires webhook_url."""
    with pytest.raises(ValueError, match="n8n adapter requires 'webhook_url'"):
        await load_a2a_agent({"adapter": "n8n"})


@pytest.mark.asyncio
async def test_load_callable_adapter_missing_function():
    """Test that callable adapter requires callable function."""
    with pytest.raises(ValueError, match="callable adapter requires 'callable'"):
        await load_a2a_agent({"adapter": "callable"})

