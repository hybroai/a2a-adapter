"""
Example: Single LangChain Agent Server with Streaming

This example demonstrates how to expose a LangChain chain as an A2A-compliant
agent with streaming support.

Prerequisites:
- langchain and langchain-openai packages installed
- OpenAI API key set in environment: OPENAI_API_KEY

Usage:
    python examples/03_single_langchain_agent.py
"""

import asyncio
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from a2a_adapter import load_a2a_agent, serve_agent
from a2a.types import AgentCard, AgentCapabilities, AgentSkill


async def main():
    """Start serving a LangChain chain as an A2A agent."""
    
    # Create a LangChain chain with streaming enabled
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Answer questions clearly and concisely."),
        ("user", "{input}"),
    ])
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        streaming=True,
        temperature=0.7,
    )
    
    chain = prompt | llm
    
    # Load the adapter
    adapter = await load_a2a_agent({
        "adapter": "langchain",
        "runnable": chain,
        "input_key": "input",
    })
    
    # Define the agent card
    agent_card = AgentCard(
        name="LangChain Chat Agent",
        description="AI chat assistant powered by LangChain and GPT-4. Supports streaming responses for real-time interaction.",
        url="http://localhost:8002",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(
            streaming=True,  # LangChain supports streaming
        ),
        skills=[
            AgentSkill(
                id="chat",
                name="chat",
                description="Have a conversation and answer questions",
                tags=["chat", "conversation"]
            ),
            AgentSkill(
                id="analyze",
                name="analyze",
                description="Analyze text and provide insights",
                tags=["analysis", "text"]
            ),
        ]
    )
    
    # Start serving the agent
    print("Starting LangChain Chat Agent on port 8002...")
    print("Streaming enabled: Yes")
    serve_agent(agent_card=agent_card, adapter=adapter, port=8002)


if __name__ == "__main__":
    # Ensure OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        exit(1)
    
    asyncio.run(main())

