"""Tests for the v0.2 LangChainAdapter.

Covers: invoke, stream, input building, output extraction, metadata,
        and integration with to_a2a / build_agent_card / load_adapter.
"""

import asyncio
import json
import pytest

from a2a_adapter.integrations.langchain import LangChainAdapter
from a2a_adapter.server import build_agent_card, to_a2a
from a2a_adapter.loader import load_adapter


# ──── Helpers: Mock Runnables ────


class MockRunnable:
    """Minimal mock LangChain Runnable with ainvoke + astream."""

    def __init__(self, response="mock response"):
        self._response = response
        self.last_input = None

    async def ainvoke(self, inputs):
        self.last_input = inputs
        return self._response

    async def astream(self, inputs):
        self.last_input = inputs
        for word in self._response.split():
            yield word


class MockAIMessage:
    """Simulates a LangChain AIMessage with .content attribute."""

    def __init__(self, content):
        self.content = content


class MockRunnableReturnsAIMessage(MockRunnable):
    """Returns an AIMessage from ainvoke."""

    async def ainvoke(self, inputs):
        self.last_input = inputs
        return MockAIMessage(self._response)


class MockAIMessageChunk:
    """Simulates a streaming AIMessageChunk."""

    def __init__(self, content):
        self.content = content


class MockRunnableStreamsChunks(MockRunnable):
    """Returns AIMessageChunk objects from astream."""

    async def astream(self, inputs):
        self.last_input = inputs
        for word in self._response.split():
            yield MockAIMessageChunk(word)


class MockRunnableReturnsDict(MockRunnable):
    """Returns a dict from ainvoke."""

    async def ainvoke(self, inputs):
        self.last_input = inputs
        return {"output": self._response, "extra_key": "value"}


class MockRunnableNoStream:
    """Runnable without astream — streaming not supported."""

    async def ainvoke(self, inputs):
        return "no stream"


class MockRunnableReturnsDictWithCustomKey(MockRunnable):
    """Returns a dict with a custom key."""

    async def ainvoke(self, inputs):
        self.last_input = inputs
        return {"result": self._response}


# ──── Test: Invoke ────


@pytest.mark.asyncio
async def test_invoke_basic():
    adapter = LangChainAdapter(runnable=MockRunnable("hello world"))
    result = await adapter.invoke("test input")
    assert result == "hello world"


@pytest.mark.asyncio
async def test_invoke_ai_message():
    adapter = LangChainAdapter(runnable=MockRunnableReturnsAIMessage("from ai"))
    result = await adapter.invoke("test")
    assert result == "from ai"


@pytest.mark.asyncio
async def test_invoke_dict_output():
    adapter = LangChainAdapter(runnable=MockRunnableReturnsDict("dict result"))
    result = await adapter.invoke("test")
    assert result == "dict result"  # Extracts from "output" key


@pytest.mark.asyncio
async def test_invoke_dict_custom_output_key():
    adapter = LangChainAdapter(
        runnable=MockRunnableReturnsDictWithCustomKey("custom"),
        output_key="result",
    )
    result = await adapter.invoke("test")
    assert result == "custom"


# ──── Test: Input Building ────


@pytest.mark.asyncio
async def test_input_key_default():
    runnable = MockRunnable()
    adapter = LangChainAdapter(runnable=runnable, input_key="question")
    await adapter.invoke("what is 2+2")
    assert runnable.last_input == {"question": "what is 2+2"}


@pytest.mark.asyncio
async def test_input_json_parsing():
    runnable = MockRunnable()
    adapter = LangChainAdapter(runnable=runnable, parse_json_input=True)
    await adapter.invoke('{"question": "hello", "context": "greet"}')
    assert runnable.last_input == {"question": "hello", "context": "greet"}


@pytest.mark.asyncio
async def test_input_json_disabled():
    runnable = MockRunnable()
    adapter = LangChainAdapter(runnable=runnable, parse_json_input=False)
    json_input = '{"question": "hello"}'
    await adapter.invoke(json_input)
    assert runnable.last_input == {"input": json_input}


@pytest.mark.asyncio
async def test_input_mapper():
    runnable = MockRunnable()

    def mapper(raw_input, context_id):
        return {"custom_field": raw_input, "ctx": context_id or "none"}

    adapter = LangChainAdapter(runnable=runnable, input_mapper=mapper)
    await adapter.invoke("hello", context_id="ctx123")
    assert runnable.last_input == {"custom_field": "hello", "ctx": "ctx123"}


@pytest.mark.asyncio
async def test_input_mapper_error_fails_the_task():
    """A bug in input_mapper must surface, not silently fall back."""
    runnable = MockRunnable()

    def bad_mapper(raw_input, context_id):
        raise ValueError("broken")

    adapter = LangChainAdapter(runnable=runnable, input_mapper=bad_mapper)
    with pytest.raises(RuntimeError, match="input_mapper failed: ValueError: broken") as exc:
        await adapter.invoke("hello")
    assert isinstance(exc.value.__cause__, ValueError)
    assert runnable.last_input is None


@pytest.mark.asyncio
async def test_input_mapper_must_return_a_dict():
    runnable = MockRunnable()
    adapter = LangChainAdapter(runnable=runnable, input_mapper=lambda raw, ctx: ["not", "a", "dict"])
    with pytest.raises(TypeError, match="input_mapper must return a dict, got list"):
        await adapter.invoke("hello")


@pytest.mark.asyncio
async def test_default_inputs_merged():
    runnable = MockRunnable()
    adapter = LangChainAdapter(
        runnable=runnable,
        default_inputs={"temperature": 0.7},
    )
    await adapter.invoke("test")
    assert runnable.last_input == {"temperature": 0.7, "input": "test"}


@pytest.mark.asyncio
async def test_json_context_id_stripped():
    """context_id in JSON input should be stripped (LangChain doesn't use it)."""
    runnable = MockRunnable()
    adapter = LangChainAdapter(runnable=runnable)
    await adapter.invoke('{"question": "hi", "context_id": "abc"}')
    assert "context_id" not in runnable.last_input
    assert runnable.last_input == {"question": "hi"}


# ──── Test: Streaming ────


@pytest.mark.asyncio
async def test_stream_basic():
    adapter = LangChainAdapter(runnable=MockRunnable("hello world"))
    chunks = []
    async for chunk in adapter.stream("test"):
        chunks.append(chunk)
    assert chunks == ["hello", "world"]


@pytest.mark.asyncio
async def test_stream_ai_message_chunks():
    adapter = LangChainAdapter(runnable=MockRunnableStreamsChunks("foo bar baz"))
    chunks = []
    async for chunk in adapter.stream("test"):
        chunks.append(chunk)
    assert chunks == ["foo", "bar", "baz"]


# ──── Test: Streaming Support Detection ────


def test_supports_streaming_with_astream():
    adapter = LangChainAdapter(runnable=MockRunnable())
    assert adapter.supports_streaming() is True


def test_supports_streaming_without_astream():
    adapter = LangChainAdapter(runnable=MockRunnableNoStream())
    assert adapter.supports_streaming() is False


# ──── Test: Metadata ────


def test_metadata_defaults():
    adapter = LangChainAdapter(runnable=MockRunnable())
    meta = adapter.get_metadata()
    assert meta.name == "LangChainAdapter"
    assert meta.streaming is True  # MockRunnable has astream


def test_metadata_custom():
    adapter = LangChainAdapter(
        runnable=MockRunnable(),
        name="My LangChain Agent",
        description="A test agent",
    )
    meta = adapter.get_metadata()
    assert meta.name == "My LangChain Agent"
    assert meta.description == "A test agent"


def test_metadata_no_stream():
    adapter = LangChainAdapter(runnable=MockRunnableNoStream())
    meta = adapter.get_metadata()
    assert meta.streaming is False


# ──── Test: Output Edge Cases ────


@pytest.mark.asyncio
async def test_output_list_content():
    """AIMessage with list content (multimodal)."""

    class ListContentMessage:
        content = ["hello", "world"]

    class ListRunnable:
        async def ainvoke(self, inputs):
            return ListContentMessage()

    adapter = LangChainAdapter(runnable=ListRunnable())
    result = await adapter.invoke("test")
    assert result == "hello world"


@pytest.mark.asyncio
async def test_output_dict_fallback_json():
    """Dict output with no recognized keys falls back to JSON."""

    class DictRunnable:
        async def ainvoke(self, inputs):
            return {"custom_key": "custom_value"}

    adapter = LangChainAdapter(runnable=DictRunnable())
    result = await adapter.invoke("test")
    assert json.loads(result) == {"custom_key": "custom_value"}


# ──── Test: Integration with server layer ────


def test_build_agent_card():
    adapter = LangChainAdapter(
        runnable=MockRunnable(),
        name="Test Agent",
        description="A test agent",
    )
    card = build_agent_card(adapter, url="http://localhost:9999")
    assert card.name == "Test Agent"
    assert card.description == "A test agent"
    assert card.capabilities.streaming is True
    assert card.supported_interfaces[0].url == "http://localhost:9999"


def test_to_a2a_builds_app():
    adapter = LangChainAdapter(runnable=MockRunnable())
    app = to_a2a(adapter)
    assert app is not None


def test_load_adapter_langchain():
    adapter = load_adapter({
        "adapter": "langchain",
        "runnable": MockRunnable(),
        "input_key": "query",
        "timeout": 30,
    })
    assert isinstance(adapter, LangChainAdapter)
    assert adapter.input_key == "query"
    assert adapter.timeout == 30


# ──── Test: Chunk extraction edge cases ────


@pytest.mark.asyncio
async def test_stream_dict_chunks():
    """Stream yielding dict chunks with 'content' key."""

    class DictStreamRunnable:
        async def ainvoke(self, inputs):
            return "done"

        async def astream(self, inputs):
            yield {"content": "hello"}
            yield {"content": " world"}

    adapter = LangChainAdapter(runnable=DictStreamRunnable())
    chunks = []
    async for chunk in adapter.stream("test"):
        chunks.append(chunk)
    assert chunks == ["hello", " world"]


@pytest.mark.asyncio
async def test_stream_string_chunks():
    """Stream yielding raw strings."""

    class StringStreamRunnable:
        async def ainvoke(self, inputs):
            return "done"

        async def astream(self, inputs):
            yield "chunk1"
            yield "chunk2"

    adapter = LangChainAdapter(runnable=StringStreamRunnable())
    chunks = []
    async for chunk in adapter.stream("test"):
        chunks.append(chunk)
    assert chunks == ["chunk1", "chunk2"]


@pytest.mark.asyncio
async def test_stream_empty_chunks_filtered():
    """Empty chunks should be filtered out."""

    class EmptyChunkRunnable:
        async def ainvoke(self, inputs):
            return "done"

        async def astream(self, inputs):
            yield ""
            yield "real"
            yield ""

    adapter = LangChainAdapter(runnable=EmptyChunkRunnable())
    chunks = []
    async for chunk in adapter.stream("test"):
        chunks.append(chunk)
    assert chunks == ["real"]


# ──── Test: Flat import ────


def test_flat_import():
    """Verify from a2a_adapter import LangChainAdapter works."""
    from a2a_adapter import LangChainAdapter as LC
    assert LC is LangChainAdapter


# ──── Test: Structured content blocks ────


BLOCKS = [
    {"type": "text", "text": "Hello"},
    {"type": "tool_use", "id": "t1", "name": "search", "input": {}},
    {"type": "text", "text": "world"},
]


@pytest.mark.asyncio
async def test_invoke_content_blocks_text_only():
    class BlockRunnable:
        async def ainvoke(self, inputs):
            return MockAIMessage(BLOCKS)

    adapter = LangChainAdapter(runnable=BlockRunnable())
    assert await adapter.invoke("test") == "Hello world"


@pytest.mark.asyncio
async def test_stream_content_block_chunks():
    """Chunks whose content is a list of text blocks yield their text."""

    class BlockStreamRunnable:
        async def ainvoke(self, inputs):
            return "done"

        async def astream(self, inputs):
            yield MockAIMessageChunk([{"type": "text", "text": "Hel", "index": 0}])
            yield MockAIMessageChunk([{"type": "text", "text": "lo", "index": 0}])
            yield MockAIMessageChunk([{"type": "tool_use", "id": "t1", "index": 1}])

    adapter = LangChainAdapter(runnable=BlockStreamRunnable())
    chunks = [c async for c in adapter.stream("test")]
    assert chunks == ["Hel", "lo"]


# ──── Test: Timeouts and explicit streaming flag ────


class SlowFramework:
    """ainvoke sleeps; astream yields every 30ms and records closing."""

    def __init__(self, items=10, raise_inside=None):
        self.items = items
        self.raise_inside = raise_inside
        self.closed = False

    async def ainvoke(self, inputs):
        await asyncio.sleep(1)
        return "late"

    async def astream(self, inputs):
        try:
            if self.raise_inside is not None:
                raise self.raise_inside
            for i in range(self.items):
                await asyncio.sleep(0.03)
                yield {"n": {"output": f"c{i}"}} if "langchain" == "langgraph" else f"c{i}"
        finally:
            self.closed = True


@pytest.mark.asyncio
async def test_invoke_timeout_has_clear_message():
    adapter = LangChainAdapter(runnable=SlowFramework(), timeout=0.05)
    with pytest.raises(TimeoutError, match="timed out after 0.05 seconds"):
        await adapter.invoke("x")


@pytest.mark.asyncio
async def test_stream_timeout_bounds_whole_stream():
    """The timeout bounds the whole stream, not each chunk."""
    framework = SlowFramework(items=10)
    adapter = LangChainAdapter(runnable=framework, timeout=0.1)
    chunks = []
    with pytest.raises(TimeoutError, match="timed out after 0.1 seconds"):
        async for chunk in adapter.stream("x"):
            chunks.append(chunk)
    assert 0 < len(chunks) < 10
    assert framework.closed


@pytest.mark.asyncio
async def test_stream_completes_within_timeout():
    adapter = LangChainAdapter(runnable=SlowFramework(items=2), timeout=5)
    assert len([c async for c in adapter.stream("x")]) == 2


@pytest.mark.asyncio
async def test_stream_inner_timeout_error_not_relabelled():
    framework = SlowFramework(raise_inside=TimeoutError("upstream API"))
    adapter = LangChainAdapter(runnable=framework, timeout=5)
    with pytest.raises(TimeoutError, match="^upstream API$"):
        _ = [c async for c in adapter.stream("x")]


def test_streaming_flag_disables_streaming():
    adapter = LangChainAdapter(runnable=MockRunnable(), streaming=False)
    assert adapter.supports_streaming() is False
    assert adapter.get_metadata().streaming is False


def test_streaming_flag_default_auto_detects():
    assert LangChainAdapter(runnable=MockRunnable()).supports_streaming() is True
    assert LangChainAdapter(runnable=MockRunnableNoStream()).supports_streaming() is False


@pytest.mark.asyncio
async def test_output_dict_with_non_json_values():
    from datetime import datetime

    when = datetime(2026, 1, 2, 3, 4, 5)

    class DictRunnable:
        async def ainvoke(self, inputs):
            return {"when": when, "msg": MockAIMessage("hi")}

    result = await LangChainAdapter(runnable=DictRunnable()).invoke("test")
    assert json.loads(result)["when"] == str(when)
