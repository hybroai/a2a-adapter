"""LangGraphAdapter with graphs that pause on ``interrupt()``.

Uses real compiled LangGraph graphs with an in-memory checkpointer.
"""

import json
from unittest.mock import MagicMock

import pytest
from a2a.types import Message, Role, Task, TaskState, TaskStatus

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


# ──── Resuming on the client's reply ────


def _reply_context(state=TaskState.TASK_STATE_INPUT_REQUIRED, paused=None):
    """A RequestContext for a message sent to an existing task.

    ``paused`` is the InputRequired the task ended with; its metadata is on
    the task's status message, as the executor stores it.
    """
    status = TaskStatus(state=state)
    if paused is not None:
        status.message.CopyFrom(
            Message(message_id="m", role=Role.ROLE_AGENT, metadata=paused.metadata)
        )
    ctx = MagicMock()
    ctx.current_task = Task(id="task-1", context_id="ctx-1", status=status)
    return ctx


async def test_invoke_reply_resumes_the_graph():
    adapter = LangGraphAdapter(graph=approval_graph())
    await adapter.invoke("go", "ctx-1")

    result = await adapter.invoke("yes", "ctx-1", context=_reply_context())

    assert result == "approved='yes'"


async def test_stream_reply_resumes_the_graph():
    adapter = LangGraphAdapter(graph=approval_graph())
    await _collect(adapter)

    items = await _collect(adapter, "yes", context=_reply_context())

    assert items == ["approved='yes'"]


async def test_json_object_reply_is_parsed():
    adapter = LangGraphAdapter(graph=approval_graph())
    await adapter.invoke("go", "ctx-1")

    result = await adapter.invoke('{"ok": true}', "ctx-1", context=_reply_context())

    assert result == "approved={'ok': True}"


async def test_json_reply_kept_as_text_without_parse_json_input():
    adapter = LangGraphAdapter(graph=approval_graph(), parse_json_input=False)
    await adapter.invoke("go", "ctx-1")

    result = await adapter.invoke('{"ok": true}', "ctx-1", context=_reply_context())

    assert result == """approved='{"ok": true}'"""


async def test_resume_mapper_builds_the_resume_value():
    calls = []

    def resume_mapper(text, context_id):
        calls.append((text, context_id))
        return {"decision": text.upper()}

    adapter = LangGraphAdapter(graph=approval_graph(), resume_mapper=resume_mapper)
    await adapter.invoke("go", "ctx-1")

    result = await adapter.invoke("yes", "ctx-1", context=_reply_context())

    assert calls == [("yes", "ctx-1")]
    assert result == "approved={'decision': 'YES'}"


async def test_graph_can_pause_again_after_resuming():
    adapter = LangGraphAdapter(graph=approval_graph(publish_step=True))
    await adapter.invoke("go", "ctx-1")

    second = await adapter.invoke("yes", "ctx-1", context=_reply_context())
    final = await adapter.invoke("now", "ctx-1", context=_reply_context())

    assert second.message == "Publish now?"
    assert final == "approved='yes'\npublished='now'"


async def test_new_task_in_a_paused_context_starts_fresh():
    graph = approval_graph()
    adapter = LangGraphAdapter(graph=graph)
    await adapter.invoke("go", "ctx-1")

    # No current task: a new task, not a reply. The graph reruns from the
    # start (LangGraph drops the old pause) and pauses again.
    result = await adapter.invoke("go again", "ctx-1", context=MagicMock(current_task=None))

    assert result.message == "Approve the draft?"
    state = await graph.aget_state({"configurable": {"thread_id": "ctx-1"}})
    assert len(state.interrupts) == 1


async def test_follow_up_to_a_working_task_is_not_a_resume():
    adapter = LangGraphAdapter(graph=approval_graph())
    await adapter.invoke("go", "ctx-1")

    result = await adapter.invoke(
        "again", "ctx-1", context=_reply_context(TaskState.TASK_STATE_WORKING)
    )

    assert result.message == "Approve the draft?"


async def test_reply_with_nothing_to_resume_fails_clearly():
    adapter = LangGraphAdapter(graph=approval_graph())

    with pytest.raises(UserFacingError, match="nothing to resume"):
        await adapter.invoke("yes", "ctx-unknown", context=_reply_context())
    with pytest.raises(UserFacingError, match="nothing to resume"):
        await _collect(adapter, "yes", context_id="ctx-unknown", context=_reply_context())


async def test_pause_records_the_interrupt_id():
    graph = approval_graph()
    adapter = LangGraphAdapter(graph=graph)

    paused = await adapter.invoke("go", "ctx-1")

    state = await graph.aget_state({"configurable": {"thread_id": "ctx-1"}})
    assert paused.metadata == {"langgraph_interrupt_id": state.interrupts[0].id}


async def test_reply_resumes_the_recorded_interrupt():
    adapter = LangGraphAdapter(graph=approval_graph())
    paused = await adapter.invoke("go", "ctx-1")

    result = await adapter.invoke("yes", "ctx-1", context=_reply_context(paused=paused))

    assert result == "approved='yes'"


@pytest.mark.parametrize("streaming", [False, True])
async def test_reply_to_a_superseded_task_fails_and_keeps_the_newer_pause(streaming):
    adapter = LangGraphAdapter(graph=approval_graph())
    first = await adapter.invoke("draft A", "ctx-1")
    second = await adapter.invoke("draft B", "ctx-1")  # a new task, same context

    with pytest.raises(UserFacingError, match="superseded"):
        if streaming:
            await _collect(adapter, "yes", context=_reply_context(paused=first))
        else:
            await adapter.invoke("yes", "ctx-1", context=_reply_context(paused=first))

    result = await adapter.invoke("yes", "ctx-1", context=_reply_context(paused=second))
    assert result == "approved='yes'"


async def test_async_resume_mapper_is_awaited():
    async def resume_mapper(text, context_id):
        return {"decision": text}

    adapter = LangGraphAdapter(graph=approval_graph(), resume_mapper=resume_mapper)
    paused = await adapter.invoke("go", "ctx-1")

    result = await adapter.invoke("yes", "ctx-1", context=_reply_context(paused=paused))

    assert result == "approved={'decision': 'yes'}"
