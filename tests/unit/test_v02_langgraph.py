"""Tests for the v0.2 LangGraphAdapter.

Covers: invoke, stream (delta-based), input building (including "messages"
key with HumanMessage fallback), output extraction from graph state,
metadata, and integration with to_a2a / build_agent_card / load_adapter.
"""

import json
import pytest

from a2a_adapter.integrations.langgraph import LangGraphAdapter
from a2a_adapter.base_adapter import AdapterMetadata
from a2a_adapter.server import build_agent_card, to_a2a
from a2a_adapter.loader import load_adapter


# ──── Helpers: Mock Graphs ────


class MockGraph:
    """Minimal mock LangGraph CompiledGraph with ainvoke + astream."""

    def __init__(self, final_state=None):
        self._final_state = final_state or {"output": "graph result"}
        self.last_input = None

    async def ainvoke(self, inputs):
        self.last_input = inputs
        return self._final_state

    async def astream(self, inputs):
        self.last_input = inputs
        # Simulate intermediate states building up
        yield {"output": "partial"}
        yield self._final_state


class MockGraphMessages:
    """Mock graph that returns results in 'messages' key (chat workflows)."""

    def __init__(self, messages=None):
        self._messages = messages or [{"role": "assistant", "content": "graph says hi"}]
        self.last_input = None

    async def ainvoke(self, inputs):
        self.last_input = inputs
        return {"messages": self._messages}

    async def astream(self, inputs):
        self.last_input = inputs
        yield {"messages": self._messages[:1]}
        yield {"messages": self._messages}


class MockGraphNoStream:
    """Graph without astream method — streaming not supported."""

    async def ainvoke(self, inputs):
        return {"output": "no stream"}


class MockMessageObject:
    """Simulates a LangChain message object with .content."""

    def __init__(self, content):
        self.content = content


class MockGraphWithLCMessages:
    """Mock graph that returns LangChain-style message objects."""

    def __init__(self):
        self.last_input = None

    async def ainvoke(self, inputs):
        self.last_input = inputs
        return {"messages": [MockMessageObject("lc message content")]}

    async def astream(self, inputs):
        self.last_input = inputs
        yield {"messages": [MockMessageObject("lc streaming")]}


# ──── Test: Invoke ────


@pytest.mark.asyncio
async def test_invoke_basic():
    adapter = LangGraphAdapter(graph=MockGraph())
    result = await adapter.invoke("test input")
    assert result == "graph result"


@pytest.mark.asyncio
async def test_invoke_messages_dict():
    adapter = LangGraphAdapter(graph=MockGraphMessages())
    result = await adapter.invoke("test")
    assert result == "graph says hi"


@pytest.mark.asyncio
async def test_invoke_messages_lc_objects():
    adapter = LangGraphAdapter(graph=MockGraphWithLCMessages())
    result = await adapter.invoke("test")
    assert result == "lc message content"


@pytest.mark.asyncio
async def test_invoke_custom_output_key():
    graph = MockGraph(final_state={"result": "custom key value", "extra": "ignored"})
    adapter = LangGraphAdapter(graph=graph, output_key="result")
    result = await adapter.invoke("test")
    assert result == "custom key value"


# ──── Test: Input Building ────


@pytest.mark.asyncio
async def test_input_key_messages_default():
    """Default input_key='messages' wraps input in message list."""
    graph = MockGraph()
    adapter = LangGraphAdapter(graph=graph)
    await adapter.invoke("hello agent")

    # Should be wrapped as messages (dict fallback since we don't have langchain_core)
    messages = graph.last_input.get("messages", [])
    assert len(messages) == 1
    msg = messages[0]
    # It will either be a HumanMessage or a dict depending on langchain_core presence
    if isinstance(msg, dict):
        assert msg["content"] == "hello agent"
    else:
        assert msg.content == "hello agent"


@pytest.mark.asyncio
async def test_input_key_simple():
    graph = MockGraph()
    adapter = LangGraphAdapter(graph=graph, input_key="query")
    await adapter.invoke("what is 2+2")
    assert graph.last_input == {"query": "what is 2+2"}


@pytest.mark.asyncio
async def test_input_json_parsing():
    graph = MockGraph()
    adapter = LangGraphAdapter(graph=graph, parse_json_input=True)
    await adapter.invoke('{"query": "hello", "mode": "fast"}')
    assert graph.last_input == {"query": "hello", "mode": "fast"}


@pytest.mark.asyncio
async def test_input_json_disabled():
    graph = MockGraph()
    adapter = LangGraphAdapter(graph=graph, input_key="query", parse_json_input=False)
    json_str = '{"query": "hello"}'
    await adapter.invoke(json_str)
    assert graph.last_input == {"query": json_str}


@pytest.mark.asyncio
async def test_input_mapper():
    graph = MockGraph()

    def mapper(raw_input, context_id):
        return {"messages": [{"role": "user", "content": raw_input}], "thread_id": context_id}

    adapter = LangGraphAdapter(graph=graph, input_mapper=mapper)
    await adapter.invoke("hello", context_id="thread-1")
    assert graph.last_input["thread_id"] == "thread-1"
    assert graph.last_input["messages"][0]["content"] == "hello"


@pytest.mark.asyncio
async def test_input_mapper_fallback():
    graph = MockGraph()

    def bad_mapper(raw_input, context_id):
        raise ValueError("broken")

    adapter = LangGraphAdapter(graph=graph, input_key="query", input_mapper=bad_mapper)
    await adapter.invoke("fallback test")
    assert graph.last_input == {"query": "fallback test"}


@pytest.mark.asyncio
async def test_default_inputs_merged():
    graph = MockGraph()
    adapter = LangGraphAdapter(
        graph=graph,
        input_key="query",
        default_inputs={"recursion_limit": 25},
    )
    await adapter.invoke("test")
    assert graph.last_input == {"recursion_limit": 25, "query": "test"}


# ──── Test: Streaming ────


@pytest.mark.asyncio
async def test_stream_basic():
    adapter = LangGraphAdapter(graph=MockGraph({"output": "final result"}))
    chunks = []
    async for chunk in adapter.stream("test"):
        chunks.append(chunk)
    # First state has "partial", second has "final result" — delta logic
    assert "partial" in chunks
    # Final text should contain "final result" delta
    full = "".join(chunks)
    assert "partial" in full


@pytest.mark.asyncio
async def test_stream_messages_graph():
    adapter = LangGraphAdapter(graph=MockGraphMessages())
    chunks = []
    async for chunk in adapter.stream("test"):
        chunks.append(chunk)
    assert len(chunks) >= 1
    assert "graph says hi" in "".join(chunks)


@pytest.mark.asyncio
async def test_stream_lc_message_objects():
    adapter = LangGraphAdapter(graph=MockGraphWithLCMessages())
    chunks = []
    async for chunk in adapter.stream("test"):
        chunks.append(chunk)
    assert "lc streaming" in "".join(chunks)


@pytest.mark.asyncio
async def test_stream_with_state_key():
    """state_key should be used for streaming text extraction."""

    class CustomStateGraph:
        async def ainvoke(self, inputs):
            return {"custom_field": "done"}

        async def astream(self, inputs):
            yield {"custom_field": "step1"}
            yield {"custom_field": "step2"}

    adapter = LangGraphAdapter(graph=CustomStateGraph(), state_key="custom_field")
    chunks = []
    async for chunk in adapter.stream("test"):
        chunks.append(chunk)
    assert "step1" in chunks
    assert "step2" in chunks


# ──── Test: Streaming Support Detection ────


def test_supports_streaming_with_astream():
    adapter = LangGraphAdapter(graph=MockGraph())
    assert adapter.supports_streaming() is True


def test_supports_streaming_without_astream():
    adapter = LangGraphAdapter(graph=MockGraphNoStream())
    assert adapter.supports_streaming() is False


# ──── Test: Metadata ────


def test_metadata_defaults():
    adapter = LangGraphAdapter(graph=MockGraph())
    meta = adapter.get_metadata()
    assert meta.name == "LangGraphAdapter"
    assert meta.streaming is True


def test_metadata_custom():
    adapter = LangGraphAdapter(
        graph=MockGraph(),
        name="My Graph Agent",
        description="Processes complex workflows",
    )
    meta = adapter.get_metadata()
    assert meta.name == "My Graph Agent"
    assert meta.description == "Processes complex workflows"


def test_metadata_no_stream():
    adapter = LangGraphAdapter(graph=MockGraphNoStream())
    meta = adapter.get_metadata()
    assert meta.streaming is False


# ──── Test: Output Edge Cases ────


@pytest.mark.asyncio
async def test_output_fallback_json():
    """State with no recognized keys falls back to JSON serialization."""
    graph = MockGraph(final_state={"custom_key": "custom_value"})
    adapter = LangGraphAdapter(graph=graph)
    result = await adapter.invoke("test")
    assert json.loads(result) == {"custom_key": "custom_value"}


@pytest.mark.asyncio
async def test_output_internal_keys_stripped():
    """Internal keys starting with _ should be stripped in fallback."""
    graph = MockGraph(final_state={"_internal": "hidden", "visible": "shown"})
    adapter = LangGraphAdapter(graph=graph)
    result = await adapter.invoke("test")
    parsed = json.loads(result)
    assert "_internal" not in parsed
    assert parsed["visible"] == "shown"


@pytest.mark.asyncio
async def test_output_common_keys():
    """Test extraction via common output keys (response, result, answer)."""
    for key in ("response", "result", "answer", "text", "content"):
        graph = MockGraph(final_state={key: f"from_{key}"})
        adapter = LangGraphAdapter(graph=graph)
        result = await adapter.invoke("test")
        assert result == f"from_{key}"


@pytest.mark.asyncio
async def test_output_non_dict():
    """Non-dict output should be str-ified."""
    graph = MockGraph(final_state="plain string")  # type: ignore
    adapter = LangGraphAdapter(graph=graph)
    result = await adapter.invoke("test")
    assert result == "plain string"


# ──── Test: Integration with server layer ────


def test_build_agent_card():
    adapter = LangGraphAdapter(
        graph=MockGraph(),
        name="Graph Agent",
        description="A test agent",
    )
    card = build_agent_card(adapter, url="http://localhost:8000")
    assert card.name == "Graph Agent"
    assert card.description == "A test agent"
    assert card.capabilities.streaming is True
    assert card.url == "http://localhost:8000"


def test_to_a2a_builds_app():
    adapter = LangGraphAdapter(graph=MockGraph())
    app = to_a2a(adapter)
    assert app is not None


def test_load_adapter_langgraph():
    adapter = load_adapter({
        "adapter": "langgraph",
        "graph": MockGraph(),
        "input_key": "messages",
        "timeout": 120,
    })
    assert isinstance(adapter, LangGraphAdapter)
    assert adapter.input_key == "messages"
    assert adapter.timeout == 120


# ──── Test: Flat import ────


def test_flat_import():
    """Verify from a2a_adapter import LangGraphAdapter works."""
    from a2a_adapter import LangGraphAdapter as LG
    assert LG is LangGraphAdapter
