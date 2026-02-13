"""
Example: Call A2A Agents from LangGraph

This is a CLIENT example — it shows how to call a remote A2A agent
from within a LangGraph workflow (not hosting an A2A server).

Prerequisites:
- langgraph and langchain-openai installed
- An A2A agent running on port 9000 (e.g., python examples/01_single_n8n_agent.py)
- OPENAI_API_KEY set in environment

Usage:
    # Terminal 1: start an agent
    python examples/01_single_n8n_agent.py

    # Terminal 2: run this workflow
    python examples/06_langgraph_single_agent.py
"""

import asyncio
import json
from typing import Annotated, TypedDict

import httpx
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    result: str


async def call_a2a_agent(url: str, text: str) -> str:
    """Call an A2A agent via message/send."""
    payload = {
        "jsonrpc": "2.0",
        "id": "1",
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "messageId": "msg-1",
                "parts": [{"kind": "text", "text": text}],
            }
        },
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, json=payload)
        data = resp.json()
        # Extract text from status message
        status_msg = data.get("result", {}).get("status", {}).get("message", {})
        parts = status_msg.get("parts", [])
        return " ".join(p["text"] for p in parts if p.get("kind") == "text") or json.dumps(data)


async def main():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    async def analyze(state: AgentState) -> AgentState:
        last = state["messages"][-1]
        query = last.content
        math_keywords = ["calculate", "compute", "solve", "math", "+", "-", "*", "/"]
        if any(k in query.lower() for k in math_keywords):
            print(f"  -> Routing to A2A Math Agent: {query}")
            result = await call_a2a_agent("http://localhost:9000", query)
        else:
            print(f"  -> Answering locally with LLM")
            resp = await llm.ainvoke([HumanMessage(content=query)])
            result = resp.content
        return {"result": result}

    wf = StateGraph(AgentState)
    wf.add_node("analyze", analyze)
    wf.set_entry_point("analyze")
    wf.add_edge("analyze", END)
    app = wf.compile()

    queries = [
        "What is 25 * 37 + 18?",
        "Calculate the area of a circle with radius 5",
        "What is the weather today?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        result = await app.ainvoke({"messages": [HumanMessage(content=query)], "result": ""})
        print(f"Result: {result['result']}")


if __name__ == "__main__":
    print("LangGraph + A2A Agent Integration")
    print("Make sure an A2A agent is running on port 9000")
    print()
    asyncio.run(main())
