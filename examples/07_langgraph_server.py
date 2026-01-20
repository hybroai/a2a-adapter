"""
Example: LangGraph Workflow as A2A Server

This example demonstrates exposing a LangGraph workflow as an A2A-compliant
agent server. Any A2A client can interact with this workflow.

This is different from example 06 which shows calling an A2A agent FROM
LangGraph. This example shows wrapping a LangGraph workflow AS an A2A server.

Prerequisites:
- langgraph and langchain packages installed
- OpenAI API key set in environment: OPENAI_API_KEY

Usage:
    # Start the LangGraph A2A server
    python examples/07_langgraph_server.py

    # In another terminal, test with the A2A client
    python examples/04_single_agent_client.py
"""

import asyncio
import os
from typing import Annotated, TypedDict

from a2a.types import AgentCard, AgentSkill, AgentCapabilities
from a2a_adapter import load_a2a_agent, serve_agent


# Define the LangGraph state
class AgentState(TypedDict):
    """State for the research agent workflow."""
    messages: Annotated[list, lambda x, y: x + y]  # Append messages
    research_topic: str
    search_results: str
    final_answer: str


def create_research_graph():
    """
    Create a simple research workflow using LangGraph.

    This workflow:
    1. Extracts the research topic from user input
    2. Simulates searching for information
    3. Synthesizes a response

    In a real application, you would integrate with actual search APIs
    and use LLMs for synthesis.
    """
    from langgraph.graph import StateGraph, END

    # Node functions
    def extract_topic(state: AgentState) -> AgentState:
        """Extract the research topic from the user message."""
        messages = state.get("messages", [])
        if messages:
            last_msg = messages[-1]
            # Extract content from various message formats
            if hasattr(last_msg, "content"):
                topic = last_msg.content
            elif isinstance(last_msg, dict):
                topic = last_msg.get("content", str(last_msg))
            else:
                topic = str(last_msg)
        else:
            topic = "unknown topic"

        return {"research_topic": topic}

    def search_information(state: AgentState) -> AgentState:
        """Simulate searching for information about the topic."""
        topic = state.get("research_topic", "")

        # Simulated search results
        results = f"""
Research findings for: {topic}

1. Key Overview:
   This topic involves important concepts that are widely studied.

2. Main Points:
   - Point A: Fundamental understanding is essential
   - Point B: Multiple perspectives exist on this subject
   - Point C: Recent developments have advanced the field

3. Related Topics:
   - Similar concepts in related domains
   - Historical context and evolution
   - Future directions and applications
"""
        return {"search_results": results}

    def synthesize_answer(state: AgentState) -> AgentState:
        """Synthesize a final answer from the search results."""
        topic = state.get("research_topic", "")
        results = state.get("search_results", "")

        answer = f"""Based on my research about "{topic}":

{results}

In summary, this is a comprehensive topic with multiple facets worth exploring.
I recommend further investigation into the specific aspects most relevant to your needs.

Is there a particular aspect you'd like me to elaborate on?
"""
        return {"final_answer": answer}

    # Build the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("extract_topic", extract_topic)
    workflow.add_node("search", search_information)
    workflow.add_node("synthesize", synthesize_answer)

    # Add edges
    workflow.set_entry_point("extract_topic")
    workflow.add_edge("extract_topic", "search")
    workflow.add_edge("search", "synthesize")
    workflow.add_edge("synthesize", END)

    # Compile
    return workflow.compile()


async def main():
    """Start the LangGraph A2A server."""

    print("=" * 60)
    print("LangGraph A2A Server Example")
    print("=" * 60)
    print()

    # Create the LangGraph workflow
    print("Creating LangGraph research workflow...")
    graph = create_research_graph()

    # Load the adapter
    print("Initializing A2A adapter...")
    adapter = await load_a2a_agent({
        "adapter": "langgraph",
        "graph": graph,
        "input_key": "messages",  # Use messages format for chat-like input
        "output_key": "final_answer",  # Extract final_answer from state
    })

    # Define agent card
    agent_card = AgentCard(
        name="Research Agent",
        description="A LangGraph-powered research assistant that can help investigate topics and provide summaries.",
        url="http://localhost:9002",
        version="1.0.0",
        skills=[
            AgentSkill(
                id="research",
                name="Research Topics",
                description="Research any topic and provide a comprehensive summary",
                tags=["research", "summary", "information"],
                examples=[
                    "Research quantum computing",
                    "Tell me about machine learning",
                    "What is the A2A protocol?",
                ],
            ),
        ],
        capabilities=AgentCapabilities(
            streaming=True,  # LangGraph supports streaming
            pushNotifications=False,
        ),
    )

    print()
    print(f"Agent: {agent_card.name}")
    print(f"Description: {agent_card.description}")
    print(f"Skills: {[s.name for s in agent_card.skills]}")
    print()
    print("Starting A2A server on http://localhost:9002")
    print()
    print("Test with:")
    print("  curl http://localhost:9002/.well-known/agent.json")
    print()
    print("Or run: python examples/04_single_agent_client.py")
    print("(Update the URL to http://localhost:9002 in the client)")
    print()
    print("-" * 60)

    # Start serving
    serve_agent(
        agent_card=agent_card,
        adapter=adapter,
        host="0.0.0.0",
        port=9002,
    )


async def main_with_llm():
    """
    Alternative: Start a LangGraph A2A server with real LLM integration.

    This version uses OpenAI for actual language understanding and generation.
    Requires: OPENAI_API_KEY environment variable.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, AIMessage
    from langgraph.graph import StateGraph, END

    print("=" * 60)
    print("LangGraph A2A Server (with OpenAI)")
    print("=" * 60)
    print()

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # Define state for LLM-powered workflow
    class LLMAgentState(TypedDict):
        messages: list
        response: str

    # Node functions
    async def process_with_llm(state: LLMAgentState) -> LLMAgentState:
        """Process the user message with the LLM."""
        messages = state.get("messages", [])

        # Call the LLM
        response = await llm.ainvoke(messages)

        return {
            "messages": messages + [response],
            "response": response.content,
        }

    # Build the graph
    workflow = StateGraph(LLMAgentState)
    workflow.add_node("llm", process_with_llm)
    workflow.set_entry_point("llm")
    workflow.add_edge("llm", END)
    graph = workflow.compile()

    # Load the adapter
    adapter = await load_a2a_agent({
        "adapter": "langgraph",
        "graph": graph,
        "input_key": "messages",
        "output_key": "response",
    })

    # Define agent card
    agent_card = AgentCard(
        name="LLM Chat Agent",
        description="A LangGraph-powered chat agent using OpenAI GPT-4o-mini",
        url="http://localhost:9002",
        version="1.0.0",
        skills=[
            AgentSkill(
                id="chat",
                name="Chat",
                description="General conversation and question answering",
                tags=["chat", "qa", "general"],
                examples=[
                    "Hello, how are you?",
                    "Explain quantum computing in simple terms",
                    "Write a haiku about programming",
                ],
            ),
        ],
        capabilities=AgentCapabilities(
            streaming=True,
            pushNotifications=False,
        ),
    )

    print(f"Agent: {agent_card.name}")
    print(f"Model: gpt-4o-mini")
    print()
    print("Starting A2A server on http://localhost:9002")
    print("-" * 60)

    serve_agent(
        agent_card=agent_card,
        adapter=adapter,
        host="0.0.0.0",
        port=9002,
    )


if __name__ == "__main__":
    # Choose which version to run
    use_llm = os.getenv("USE_LLM", "false").lower() == "true"

    if use_llm:
        if not os.getenv("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY environment variable not set")
            print("Either set it or run without USE_LLM=true for the demo version")
            exit(1)
        asyncio.run(main_with_llm())
    else:
        asyncio.run(main())
