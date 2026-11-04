"""Tests for the v0.2 LangGraphAdapter.

Covers: invoke, stream (node updates as deltas), input building (including "messages"
key with HumanMessage fallback), output extraction from graph state,
metadata, and integration with to_a2a / build_agent_card / load_adapter.
"""

import asyncio
import json
import pytest

from a2a_adapter.integrations.langgraph import LangGraphAdapter
from a2a_adapter.server import build_agent_card, to_a2a
from a2a_adapter.loader import load_adapter


# ──── Helpers: Mock Graphs ────


class MockGraph:
    """Minimal mock LangGraph CompiledGraph with ainvoke + astream."""

    def __init__(self, final_state=None):
        self._final_state = final_state or {"output": "graph result"}
        self.last_input = None

    async def ainvoke(self, inputs, config=None):
        self.last_input = inputs
        return self._final_state

    async def astream(self, inputs, config=None, stream_mode=None):
        self.last_input = inputs
        # stream_mode="updates": one {node: update} item per node
        yield {"draft": {"output": "partial"}}
        yield {"final": self._final_state}


class MockGraphMessages:
    """Mock graph that returns results in 'messages' key (chat workflows)."""

    def __init__(self, messages=None):
        self._messages = messages or [{"role": "assistant", "content": "graph says hi"}]
        self.last_input = None

    async def ainvoke(self, inputs, config=None):
        self.last_input = inputs
        return {"messages": self._messages}

    async def astream(self, inputs, config=None, stream_mode=None):
        self.last_input = inputs
        yield {"agent": {"messages": self._messages}}


class MockGraphNoStream:
    """Graph without astream method — streaming not supported."""

    async def ainvoke(self, inputs, config=None):
        return {"output": "no stream"}


class MockMessageObject:
    """Simulates a LangChain message object with .content."""

    def __init__(self, content):
        self.content = content


class MockGraphWithLCMessages:
    """Mock graph that returns LangChain-style message objects."""

    def __init__(self):
        self.last_input = None

    async def ainvoke(self, inputs, config=None):
        self.last_input = inputs
        return {"messages": [MockMessageObject("lc message content")]}

    async def astream(self, inputs, config=None, stream_mode=None):
        self.last_input = inputs
        yield {"agent": {"messages": [MockMessageObject("lc streaming")]}}


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
async def test_input_mapper_error_fails_the_task():
    """A bug in input_mapper must surface, not silently fall back."""
    graph = MockGraph()

    def bad_mapper(raw_input, context_id):
        raise ValueError("broken")

    adapter = LangGraphAdapter(graph=graph, input_mapper=bad_mapper)
    with pytest.raises(RuntimeError, match="input_mapper failed: ValueError: broken") as exc:
        await adapter.invoke("hello")
    assert isinstance(exc.value.__cause__, ValueError)
    assert graph.last_input is None


@pytest.mark.asyncio
async def test_input_mapper_must_return_a_dict():
    graph = MockGraph()
    adapter = LangGraphAdapter(graph=graph, input_mapper=lambda raw, ctx: ["not", "a", "dict"])
    with pytest.raises(TypeError, match="input_mapper must return a dict, got list"):
        await adapter.invoke("hello")


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
    # Each node's output is emitted once, separated by a blank line
    assert chunks == ["partial", "\n\nfinal result"]


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

        async def astream(self, inputs, config=None, stream_mode=None):
            yield {"n1": {"custom_field": "step1"}}
            yield {"n2": {"custom_field": "step2"}}

    adapter = LangGraphAdapter(graph=CustomStateGraph(), state_key="custom_field")
    chunks = []
    async for chunk in adapter.stream("test"):
        chunks.append(chunk)
    assert chunks == ["step1", "\n\nstep2"]


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
    assert card.supported_interfaces[0].url == "http://localhost:8000"


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


# ──── Test: context_id → checkpointer thread_id ────


class ConfigRecordingGraph:
    """Graph that records the config passed to ainvoke/astream."""

    def __init__(self):
        self.configs = []

    async def ainvoke(self, inputs, config=None):
        self.configs.append(config)
        return {"output": "ok"}

    async def astream(self, inputs, config=None, stream_mode=None):
        self.configs.append(config)
        yield {"agent": {"output": "ok"}}


@pytest.mark.asyncio
async def test_invoke_maps_context_id_to_thread_id():
    graph = ConfigRecordingGraph()
    adapter = LangGraphAdapter(graph=graph, input_key="query")
    await adapter.invoke("hi", context_id="ctx-1")
    assert graph.configs == [{"configurable": {"thread_id": "ctx-1"}}]


@pytest.mark.asyncio
async def test_stream_maps_context_id_to_thread_id():
    graph = ConfigRecordingGraph()
    adapter = LangGraphAdapter(graph=graph, input_key="query")
    _ = [c async for c in adapter.stream("hi", context_id="ctx-1")]
    assert graph.configs == [{"configurable": {"thread_id": "ctx-1"}}]


@pytest.mark.asyncio
async def test_thread_id_key_renamed():
    graph = ConfigRecordingGraph()
    adapter = LangGraphAdapter(graph=graph, input_key="query", thread_id_key="session")
    await adapter.invoke("hi", context_id="ctx-1")
    assert graph.configs == [{"configurable": {"session": "ctx-1"}}]


@pytest.mark.asyncio
async def test_thread_id_mapping_disabled():
    graph = ConfigRecordingGraph()
    adapter = LangGraphAdapter(graph=graph, input_key="query", thread_id_key=None)
    await adapter.invoke("hi", context_id="ctx-1")
    assert graph.configs == [None]


@pytest.mark.asyncio
async def test_user_config_kept_and_merged():
    graph = ConfigRecordingGraph()
    user_config = {"recursion_limit": 5, "configurable": {"user_id": "u1"}}
    adapter = LangGraphAdapter(graph=graph, input_key="query", config=user_config)
    await adapter.invoke("hi", context_id="ctx-1")
    assert graph.configs == [
        {"recursion_limit": 5, "configurable": {"user_id": "u1", "thread_id": "ctx-1"}}
    ]
    # The caller's dict is not mutated across requests.
    assert user_config == {"recursion_limit": 5, "configurable": {"user_id": "u1"}}


@pytest.mark.asyncio
async def test_user_supplied_thread_id_wins():
    graph = ConfigRecordingGraph()
    adapter = LangGraphAdapter(
        graph=graph, input_key="query", config={"configurable": {"thread_id": "fixed"}},
    )
    await adapter.invoke("hi", context_id="ctx-1")
    assert graph.configs == [{"configurable": {"thread_id": "fixed"}}]


@pytest.mark.asyncio
async def test_no_context_id_passes_user_config_only():
    graph = ConfigRecordingGraph()
    adapter = LangGraphAdapter(graph=graph, input_key="query", config={"recursion_limit": 5})
    await adapter.invoke("hi")
    assert graph.configs == [{"recursion_limit": 5}]


# ──── Test: Structured content blocks ────


BLOCKS = [
    {"type": "text", "text": "Hello"},
    {"type": "tool_use", "id": "t1", "name": "search", "input": {}},
    {"type": "text", "text": "world"},
]


@pytest.mark.asyncio
async def test_invoke_message_object_content_blocks():
    graph = MockGraph(final_state={"messages": [MockMessageObject(BLOCKS)]})
    adapter = LangGraphAdapter(graph=graph)
    assert await adapter.invoke("test") == "Hello world"


@pytest.mark.asyncio
async def test_invoke_dict_message_content_blocks():
    graph = MockGraph(final_state={"messages": [{"role": "assistant", "content": BLOCKS}]})
    adapter = LangGraphAdapter(graph=graph)
    assert await adapter.invoke("test") == "Hello world"


# ──── Test: Streaming node updates as deltas ────


class TypedMessage:
    """LangChain-style message with .type ("ai", "human", "tool", ...)."""

    def __init__(self, content, type="ai"):
        self.content = content
        self.type = type


class UpdatesGraph:
    """Graph whose astream yields {node: update} dicts, like a compiled
    StateGraph with stream_mode="updates"."""

    def __init__(self, updates):
        self._updates = updates
        self.stream_kwargs = None

    async def ainvoke(self, inputs, config=None):
        return {}

    async def astream(self, inputs, config=None, stream_mode=None):
        self.stream_kwargs = {"config": config, "stream_mode": stream_mode}
        for update in self._updates:
            yield update


async def _collect(adapter, text="hi"):
    return [c async for c in adapter.stream(text)]


@pytest.mark.asyncio
async def test_stream_requests_updates_mode():
    graph = UpdatesGraph([{"agent": {"output": "x"}}])
    await _collect(LangGraphAdapter(graph=graph))
    assert graph.stream_kwargs["stream_mode"] == "updates"


@pytest.mark.asyncio
async def test_stream_emits_each_ai_message_once():
    """A tool-using agent: AI messages stream once each, tool output and
    the user's input are not echoed as the reply."""
    graph = UpdatesGraph([
        {"agent": {"messages": [TypedMessage("Let me check.")]}},
        {"tools": {"messages": [TypedMessage("42", type="tool")]}},
        {"agent": {"messages": [TypedMessage("The answer is 42.")]}},
    ])
    chunks = await _collect(LangGraphAdapter(graph=graph))
    assert "".join(chunks) == "Let me check.\n\nThe answer is 42."


@pytest.mark.asyncio
async def test_stream_skips_human_and_empty_ai_messages():
    graph = UpdatesGraph([
        {"intake": {"messages": [TypedMessage("hi", type="human")]}},
        {"agent": {"messages": [TypedMessage("")]}},  # tool-call-only message
        {"agent": {"messages": [{"role": "assistant", "content": "done"}]}},
    ])
    assert await _collect(LangGraphAdapter(graph=graph)) == ["done"]


@pytest.mark.asyncio
async def test_stream_skips_interrupts_and_empty_updates():
    graph = UpdatesGraph([
        {"noop": None},
        {"__interrupt__": ("waiting",)},
        {"writer": {"output": "draft"}},
    ])
    assert await _collect(LangGraphAdapter(graph=graph)) == ["draft"]


@pytest.mark.asyncio
async def test_stream_state_key_on_node_updates():
    graph = UpdatesGraph([
        {"a": {"report": "part one", "output": "ignored"}},
        {"b": {"report": "part two"}},
    ])
    chunks = await _collect(LangGraphAdapter(graph=graph, state_key="report"))
    assert "".join(chunks) == "part one\n\npart two"


# ──── Test: Timeouts and explicit streaming flag ────


class SlowFramework:
    """ainvoke sleeps; astream yields every 30ms and records closing."""

    def __init__(self, items=10, raise_inside=None):
        self.items = items
        self.raise_inside = raise_inside
        self.closed = False

    async def ainvoke(self, inputs, config=None):
        await asyncio.sleep(1)
        return "late"

    async def astream(self, inputs, config=None, stream_mode=None):
        try:
            if self.raise_inside is not None:
                raise self.raise_inside
            for i in range(self.items):
                await asyncio.sleep(0.03)
                yield {"n": {"output": f"c{i}"}} if "langgraph" == "langgraph" else f"c{i}"
        finally:
            self.closed = True


@pytest.mark.asyncio
async def test_invoke_timeout_has_clear_message():
    adapter = LangGraphAdapter(graph=SlowFramework(), timeout=0.05)
    with pytest.raises(TimeoutError, match="timed out after 0.05 seconds"):
        await adapter.invoke("x")


@pytest.mark.asyncio
async def test_stream_timeout_bounds_whole_stream():
    """The timeout bounds the whole stream, not each chunk."""
    framework = SlowFramework(items=10)
    adapter = LangGraphAdapter(graph=framework, timeout=0.1)
    chunks = []
    with pytest.raises(TimeoutError, match="timed out after 0.1 seconds"):
        async for chunk in adapter.stream("x"):
            chunks.append(chunk)
    assert 0 < len(chunks) < 10
    assert framework.closed


@pytest.mark.asyncio
async def test_stream_completes_within_timeout():
    adapter = LangGraphAdapter(graph=SlowFramework(items=2), timeout=5)
    assert len([c async for c in adapter.stream("x")]) == 2


@pytest.mark.asyncio
async def test_stream_inner_timeout_error_not_relabelled():
    framework = SlowFramework(raise_inside=TimeoutError("upstream API"))
    adapter = LangGraphAdapter(graph=framework, timeout=5)
    with pytest.raises(TimeoutError, match="^upstream API$"):
        _ = [c async for c in adapter.stream("x")]


def test_streaming_flag_disables_streaming():
    adapter = LangGraphAdapter(graph=MockGraph(), streaming=False)
    assert adapter.supports_streaming() is False
    assert adapter.get_metadata().streaming is False


def test_streaming_flag_default_auto_detects():
    assert LangGraphAdapter(graph=MockGraph()).supports_streaming() is True
    assert LangGraphAdapter(graph=MockGraphNoStream()).supports_streaming() is False


@pytest.mark.asyncio
async def test_stream_fan_out_list_updates():
    """Send/map-reduce fan-out groups same-named node writes in a list."""
    graph = UpdatesGraph([
        {"worker": [{"output": "a"}, {"output": "b"}]},
    ])
    assert "".join(await _collect(LangGraphAdapter(graph=graph))) == "a\n\nb"


@pytest.mark.asyncio
async def test_stream_no_reducer_full_history_emits_only_latest_reply():
    """Without a reducer a node returns the whole message list; only its
    latest AI message is the new reply (as invoke() reads messages[-1])."""
    history = [
        TypedMessage("turn 1", type="human"),
        TypedMessage("reply 1"),
        TypedMessage("turn 2", type="human"),
        TypedMessage("reply 2"),
    ]
    graph = UpdatesGraph([{"agent": {"messages": history}}])
    assert await _collect(LangGraphAdapter(graph=graph)) == ["reply 2"]


@pytest.mark.asyncio
async def test_stream_message_tuples_and_strings():
    """add_messages accepts (role, content) tuples and bare strings (human)."""
    graph = UpdatesGraph([
        {"intake": {"messages": [("user", "question"), "also human"]}},
        {"agent": {"messages": [("assistant", "hi")]}},
    ])
    assert await _collect(LangGraphAdapter(graph=graph)) == ["hi"]


@pytest.mark.asyncio
async def test_stream_string_node_write():
    """Functional API entrypoints can write a plain string."""
    graph = UpdatesGraph([{"workflow": "answer"}])
    assert await _collect(LangGraphAdapter(graph=graph)) == ["answer"]


@pytest.mark.asyncio
async def test_stream_reads_same_output_keys_as_invoke():
    for key in ("result", "answer"):
        graph = UpdatesGraph([{"solve": {key: "42"}}])
        assert await _collect(LangGraphAdapter(graph=graph)) == ["42"]


class RoleMessage:
    """LangChain ChatMessage: type "chat", speaker in .role."""

    def __init__(self, content, role):
        self.content = content
        self.type = "chat"
        self.role = role


@pytest.mark.asyncio
async def test_stream_skips_non_ai_chunks_and_chat_messages():
    graph = UpdatesGraph([
        {"agent": {"messages": [TypedMessage("reply"), TypedMessage("raw", type="ToolMessageChunk")]}},
        {"agent": {"messages": [TypedMessage("more", type="AIMessageChunk"), RoleMessage("q", "user")]}},
        {"agent": {"messages": [RoleMessage("chat reply", "assistant")]}},
    ])
    chunks = await _collect(LangGraphAdapter(graph=graph))
    assert "".join(chunks) == "reply\n\nmore\n\nchat reply"


@pytest.mark.asyncio
async def test_stream_falls_through_to_output_keys_without_ai_message():
    """Like invoke(), a write whose messages hold no AI reply falls through
    to the common output keys."""
    for messages in ([], [TypedMessage("tool result", type="tool")]):
        graph = UpdatesGraph([{"agent": {"messages": messages, "output": "final"}}])
        assert await _collect(LangGraphAdapter(graph=graph)) == ["final"]


@pytest.mark.asyncio
async def test_stream_tool_only_messages_write_emits_nothing():
    graph = UpdatesGraph([{"tools": {"messages": [TypedMessage("raw", type="tool")]}}])
    assert await _collect(LangGraphAdapter(graph=graph)) == []


@pytest.mark.asyncio
async def test_stream_ai_reply_wins_over_output_key():
    graph = UpdatesGraph([{"agent": {"messages": [TypedMessage("reply")], "output": "x"}}])
    assert await _collect(LangGraphAdapter(graph=graph)) == ["reply"]
