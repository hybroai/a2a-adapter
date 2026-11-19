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
    draft: str
    output: Annotated[list, operator.add]


def approval_graph(
    *,
    checkpointer: Any = "memory",
    prompt: Any = "Approve the draft?",
    parallel: bool = False,
    publish_step: bool = False,
):
    """draft -> approve (interrupts with ``prompt``) -> end.

    The draft node writes ``{"draft": "Draft v1"}``; once resumed, approve
    writes ``{"output": ["approved=<resume value>"]}``. With ``parallel``, a
    second node interrupts at the same time as approve. With
    ``publish_step``, a publish node after approve interrupts again
    ("Publish now?") and writes ``"published=<resume value>"``.
    """

    def draft(state: ApprovalState) -> dict:
        return {"draft": "Draft v1"}

    def approve(state: ApprovalState) -> dict:
        answer = interrupt(prompt)
        return {"output": [f"approved={answer!r}"]}

    def publish(state: ApprovalState) -> dict:
        answer = interrupt("Publish now?")
        return {"output": [f"published={answer!r}"]}

    def review(state: ApprovalState) -> dict:
        answer = interrupt("Second reviewer?")
        return {"output": [f"reviewed={answer!r}"]}

    builder = StateGraph(ApprovalState)
    builder.add_node("draft", draft)
    builder.add_node("approve", approve)
    builder.add_edge(START, "draft")
    builder.add_edge("draft", "approve")
    if publish_step:
        builder.add_node("publish", publish)
        builder.add_edge("approve", "publish")
        builder.add_edge("publish", END)
    else:
        builder.add_edge("approve", END)
    if parallel:
        builder.add_node("review", review)
        builder.add_edge("draft", "review")
        builder.add_edge("review", END)
    saver = InMemorySaver() if checkpointer == "memory" else checkpointer
    return builder.compile(checkpointer=saver)
