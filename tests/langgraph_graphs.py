"""Small real LangGraph graphs that pause with ``interrupt()``, shared by tests.

Importing this module needs ``langgraph``; callers skip with
``pytest.importorskip("langgraph")`` first.
"""

import operator
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt


class ApprovalState(TypedDict, total=False):
    messages: list
    draft: str
    output: Annotated[list, operator.add]


def approval_graph(
    *,
    checkpointer: Any = "memory",
    prompt: Any = "Approve the draft?",
    parallel: bool = False,
):
    """draft -> approve (interrupts with ``prompt``) -> end.

    The draft node writes ``{"draft": "Draft v1"}``; once resumed, approve
    writes ``{"output": ["approved=<resume value>"]}``. With ``parallel``, a
    second node interrupts at the same time as approve.
    """

    def draft(state: ApprovalState) -> dict:
        return {"draft": "Draft v1"}

    def approve(state: ApprovalState) -> dict:
        answer = interrupt(prompt)
        return {"output": [f"approved={answer!r}"]}

    def review(state: ApprovalState) -> dict:
        answer = interrupt("Second reviewer?")
        return {"output": [f"reviewed={answer!r}"]}

    builder = StateGraph(ApprovalState)
    builder.add_node("draft", draft)
    builder.add_node("approve", approve)
    builder.add_edge(START, "draft")
    builder.add_edge("draft", "approve")
    builder.add_edge("approve", END)
    if parallel:
        builder.add_node("review", review)
        builder.add_edge("draft", "review")
        builder.add_edge("review", END)
    saver = InMemorySaver() if checkpointer == "memory" else checkpointer
    return builder.compile(checkpointer=saver)
