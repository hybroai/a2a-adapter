"""Tests for the v0.2 CrewAIAdapter.

Covers: invoke (async + sync fallback), input building, output extraction,
        metadata, and integration with to_a2a / build_agent_card / load_adapter.
"""

import json
import pytest

from a2a_adapter.integrations.crewai import CrewAIAdapter
from a2a_adapter.base_adapter import AdapterMetadata
from a2a_adapter.server import build_agent_card, to_a2a
from a2a_adapter.loader import load_adapter


# ──── Helpers: Mock Crews ────


class MockCrewOutput:
    """Simulates a CrewAI CrewOutput object with .raw attribute."""

    def __init__(self, raw):
        self.raw = raw


class MockCrewOutputResult:
    """Simulates a CrewOutput with .result (older API)."""

    def __init__(self, result):
        self.result = result


class MockCrew:
    """Mock CrewAI Crew with kickoff_async."""

    def __init__(self, response="crew result"):
        self._response = response
        self.last_inputs = None

    async def kickoff_async(self, inputs=None):
        self.last_inputs = inputs
        return MockCrewOutput(self._response)


class MockCrewSyncOnly:
    """Mock crew without kickoff_async (older CrewAI version)."""

    def __init__(self, response="sync result"):
        self._response = response
        self.last_inputs = None

    def kickoff(self, inputs=None):
        self.last_inputs = inputs
        return MockCrewOutput(self._response)


class MockCrewReturnsDict:
    """Mock crew that returns a dict."""

    async def kickoff_async(self, inputs=None):
        return {"output": "from dict", "extra": "ignored"}


class MockCrewReturnsResult:
    """Mock crew that returns CrewOutput with .result."""

    async def kickoff_async(self, inputs=None):
        return MockCrewOutputResult("from result attr")


# ──── Test: Invoke ────


@pytest.mark.asyncio
async def test_invoke_basic():
    adapter = CrewAIAdapter(crew=MockCrew("hello world"))
    result = await adapter.invoke("test input")
    assert result == "hello world"


@pytest.mark.asyncio
async def test_invoke_passes_inputs():
    crew = MockCrew()
    adapter = CrewAIAdapter(crew=crew)
    await adapter.invoke("run this")
    assert crew.last_inputs == {"inputs": "run this"}


@pytest.mark.asyncio
async def test_invoke_sync_fallback():
    """Falls back to sync kickoff() when kickoff_async is missing."""
    crew = MockCrewSyncOnly("sync response")
    adapter = CrewAIAdapter(crew=crew)
    result = await adapter.invoke("test")
    assert result == "sync response"
    assert crew.last_inputs == {"inputs": "test"}


@pytest.mark.asyncio
async def test_invoke_dict_output():
    adapter = CrewAIAdapter(crew=MockCrewReturnsDict())
    result = await adapter.invoke("test")
    assert result == "from dict"


@pytest.mark.asyncio
async def test_invoke_result_attr():
    adapter = CrewAIAdapter(crew=MockCrewReturnsResult())
    result = await adapter.invoke("test")
    assert result == "from result attr"


# ──── Test: Input Building ────


@pytest.mark.asyncio
async def test_input_key_custom():
    crew = MockCrew()
    adapter = CrewAIAdapter(crew=crew, inputs_key="topic")
    await adapter.invoke("machine learning")
    assert crew.last_inputs == {"topic": "machine learning"}


@pytest.mark.asyncio
async def test_input_json_parsing():
    crew = MockCrew()
    adapter = CrewAIAdapter(crew=crew, parse_json_input=True)
    await adapter.invoke('{"customer_domain": "example.com", "topic": "AI"}')
    assert crew.last_inputs == {"customer_domain": "example.com", "topic": "AI"}


@pytest.mark.asyncio
async def test_input_json_disabled():
    crew = MockCrew()
    adapter = CrewAIAdapter(crew=crew, parse_json_input=False)
    json_str = '{"key": "value"}'
    await adapter.invoke(json_str)
    assert crew.last_inputs == {"inputs": json_str}


@pytest.mark.asyncio
async def test_input_mapper():
    crew = MockCrew()

    def mapper(raw_input, context_id):
        return {"customer_domain": raw_input, "ctx": context_id or "none"}

    adapter = CrewAIAdapter(crew=crew, input_mapper=mapper)
    await adapter.invoke("example.com", context_id="ctx-1")
    assert crew.last_inputs == {"customer_domain": "example.com", "ctx": "ctx-1"}


@pytest.mark.asyncio
async def test_input_mapper_fallback():
    crew = MockCrew()

    def bad_mapper(raw_input, context_id):
        raise ValueError("broken")

    adapter = CrewAIAdapter(crew=crew, input_mapper=bad_mapper)
    await adapter.invoke("fallback test")
    assert crew.last_inputs == {"inputs": "fallback test"}


@pytest.mark.asyncio
async def test_default_inputs_merged():
    crew = MockCrew()
    adapter = CrewAIAdapter(
        crew=crew,
        default_inputs={"language": "en"},
    )
    await adapter.invoke("test")
    assert crew.last_inputs == {"language": "en", "inputs": "test"}


@pytest.mark.asyncio
async def test_default_inputs_overridden_by_json():
    crew = MockCrew()
    adapter = CrewAIAdapter(
        crew=crew,
        default_inputs={"language": "en"},
    )
    await adapter.invoke('{"language": "zh", "topic": "AI"}')
    assert crew.last_inputs == {"language": "zh", "topic": "AI"}


# ──── Test: Streaming NOT supported ────


def test_no_streaming():
    adapter = CrewAIAdapter(crew=MockCrew())
    assert adapter.supports_streaming() is False


# ──── Test: Metadata ────


def test_metadata_defaults():
    adapter = CrewAIAdapter(crew=MockCrew())
    meta = adapter.get_metadata()
    assert meta.name == "CrewAIAdapter"
    assert meta.streaming is False


def test_metadata_custom():
    adapter = CrewAIAdapter(
        crew=MockCrew(),
        name="Research Crew",
        description="Researches topics using multiple agents",
    )
    meta = adapter.get_metadata()
    assert meta.name == "Research Crew"
    assert meta.description == "Researches topics using multiple agents"


# ──── Test: Output Edge Cases ────


@pytest.mark.asyncio
async def test_output_string():
    class StringCrew:
        async def kickoff_async(self, inputs=None):
            return "plain string"

    adapter = CrewAIAdapter(crew=StringCrew())
    result = await adapter.invoke("test")
    assert result == "plain string"


@pytest.mark.asyncio
async def test_output_dict_fallback_json():
    class WeirdDictCrew:
        async def kickoff_async(self, inputs=None):
            return {"custom_key": "custom_value"}

    adapter = CrewAIAdapter(crew=WeirdDictCrew())
    result = await adapter.invoke("test")
    assert json.loads(result) == {"custom_key": "custom_value"}


@pytest.mark.asyncio
async def test_output_dict_common_keys():
    """Test extraction via common output keys."""
    for key in ("output", "result", "response", "answer", "text"):

        class DictCrew:
            _key = key

            async def kickoff_async(self, inputs=None):
                return {self._key: f"from_{self._key}"}

        adapter = CrewAIAdapter(crew=DictCrew())
        result = await adapter.invoke("test")
        assert result == f"from_{key}"


# ──── Test: Integration with server layer ────


def test_build_agent_card():
    adapter = CrewAIAdapter(
        crew=MockCrew(),
        name="Test Crew",
        description="A test crew",
    )
    card = build_agent_card(adapter, url="http://localhost:9004")
    assert card.name == "Test Crew"
    assert card.description == "A test crew"
    assert card.capabilities.streaming is False
    assert card.url == "http://localhost:9004"


def test_to_a2a_builds_app():
    adapter = CrewAIAdapter(crew=MockCrew())
    app = to_a2a(adapter)
    assert app is not None


def test_load_adapter_crewai():
    adapter = load_adapter({
        "adapter": "crewai",
        "crew": MockCrew(),
        "inputs_key": "topic",
        "timeout": 600,
    })
    assert isinstance(adapter, CrewAIAdapter)
    assert adapter.inputs_key == "topic"
    assert adapter.timeout == 600


# ──── Test: Flat import ────


def test_flat_import():
    from a2a_adapter import CrewAIAdapter as CA
    assert CA is CrewAIAdapter
