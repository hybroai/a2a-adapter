"""
Example: LangGraph Workflow as A2A Agent (with streaming)

Expose a LangGraph compiled graph as an A2A-compatible agent in 3 lines.
Streaming is auto-detected.

Prerequisites:
- langgraph installed: pip install a2a-adapter[langgraph]
- For LLM version: OPENAI_API_KEY set in environment

Usage:
    python examples/langgraph_server.py              # Demo (no LLM)
    USE_LLM=true python examples/langgraph_server.py  # With OpenAI
"""

import os
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, END
from a2a_adapter import LangGraphAdapter, serve_agent


# ── Demo version: no LLM required ──

class ResearchState(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]
    research_topic: str
    search_results: str
    final_answer: str


def create_demo_graph():
    """Simple research workflow (no LLM needed)."""

    def extract_topic(state: ResearchState) -> ResearchState:
        messages = state.get("messages", [])
        if messages:
            last = messages[-1]
            topic = last.content if hasattr(last, "content") else str(last)
        else:
            topic = "unknown"
        return {"research_topic": topic}

    def search(state: ResearchState) -> ResearchState:
        topic = state.get("research_topic", "")
        return {"search_results": f"Found 3 key insights about: {topic}"}

    def synthesize(state: ResearchState) -> ResearchState:
        topic = state.get("research_topic", "")
        results = state.get("search_results", "")
        return {"final_answer": f"Research on '{topic}':\n{results}\n\nThis is a demo — use USE_LLM=true for real LLM responses."}

    wf = StateGraph(ResearchState)
    wf.add_node("extract", extract_topic)
    wf.add_node("search", search)
    wf.add_node("synthesize", synthesize)
    wf.set_entry_point("extract")
    wf.add_edge("extract", "search")
    wf.add_edge("search", "synthesize")
    wf.add_edge("synthesize", END)
    return wf.compile()


# ── LLM version: requires OpenAI ──

def create_llm_graph():
    """LLM-powered chat workflow."""
    from langchain_openai import ChatOpenAI
    from langgraph.graph import StateGraph, END

    class ChatState(TypedDict):
        messages: list
        response: str

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    async def process(state: ChatState) -> ChatState:
        resp = await llm.ainvoke(state.get("messages", []))
        return {"messages": state.get("messages", []) + [resp], "response": resp.content}

    wf = StateGraph(ChatState)
    wf.add_node("llm", process)
    wf.set_entry_point("llm")
    wf.add_edge("llm", END)
    return wf.compile()


# ── Main ──

if __name__ == "__main__":
    use_llm = os.getenv("USE_LLM", "false").lower() == "true"

    if use_llm:
        graph = create_llm_graph()
        name = "LLM Chat Agent"
        desc = "LangGraph chat agent powered by GPT-4o-mini"
        output_key = "response"
    else:
        graph = create_demo_graph()
        name = "Research Agent"
        desc = "LangGraph research workflow (demo mode)"
        output_key = "final_answer"

    # --- A2A: 3 lines ---

    adapter = LangGraphAdapter(
        graph=graph,
        input_key="messages",
        output_key=output_key,
        name=name,
        description=desc,
    )

    serve_agent(adapter, port=9002)  # Streaming auto-detected!
