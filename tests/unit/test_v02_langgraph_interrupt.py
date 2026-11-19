"""LangGraphAdapter with graphs that pause on ``interrupt()``.

Uses real compiled LangGraph graphs with an in-memory checkpointer.
"""

import json

import pytest

pytest.importorskip("langgraph")

from a2a_adapter.base_adapter import InputRequired  # noqa: E402
from a2a_adapter.exceptions import UserFacingError  # noqa: E402
from a2a_adapter.integrations.langgraph import LangGraphAdapter  # noqa: E402
from tests.langgraph_graphs import approval_graph  # noqa: E402


async def _collect(adapter, text="go", context_id="ctx-1", **kwargs):
    return [item async for item in adapter.stream(text, context_id, **kwargs)]


# ──── Reporting the pause ────


async def test_invoke_returns_input_required_with_interrupt_value():
    adapter = LangGraphAdapter(graph=approval_graph())

    result = await adapter.invoke("go", "ctx-1")

    assert isinstance(result, InputRequired)
    assert result.message == "Approve the draft?"


async def test_stream_yields_input_required_last():
    adapter = LangGraphAdapter(graph=approval_graph())

    items = await _collect(adapter)

    assert isinstance(items[-1], InputRequired)
    assert items[-1].message == "Approve the draft?"
    assert not any(isinstance(item, InputRequired) for item in items[:-1])


async def test_structured_interrupt_value_is_sent_as_json():
    value = {"question": "Approve?", "options": ["yes", "no"]}
    adapter = LangGraphAdapter(graph=approval_graph(prompt=value))

    result = await adapter.invoke("go", "ctx-1")
    streamed = (await _collect(adapter, context_id="ctx-2"))[-1]

    assert json.loads(result.message) == value
    assert json.loads(streamed.message) == value


async def test_graph_without_checkpointer_fails_clearly():
    adapter = LangGraphAdapter(graph=approval_graph(checkpointer=None))

    with pytest.raises(UserFacingError, match="checkpointer"):
        await adapter.invoke("go", "ctx-1")
    with pytest.raises(UserFacingError, match="checkpointer"):
        await _collect(adapter)


async def test_multiple_interrupts_fail_clearly():
    adapter = LangGraphAdapter(graph=approval_graph(parallel=True))

    with pytest.raises(UserFacingError, match="2 interrupts"):
        await adapter.invoke("go", "ctx-1")
    with pytest.raises(UserFacingError, match="2 interrupts"):
        await _collect(adapter, context_id="ctx-2")
