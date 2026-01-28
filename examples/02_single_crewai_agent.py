"""
Example: Single CrewAI Agent Server

This example demonstrates how to expose a CrewAI crew as an A2A-compliant agent.
The crew performs research tasks using multiple specialized agents.

Demonstrates three input handling modes:
1. JSON Input (default): Send structured JSON matching tasks.yaml variables
2. Custom input_mapper: Full control over input transformation
3. Plain text with default_inputs: Fallback for simple text inputs

Prerequisites:
- crewai package installed: pip install crewai
- OpenAI API key set in environment: OPENAI_API_KEY

Usage:
    python examples/02_single_crewai_agent.py
"""

import asyncio
import json
import os

from crewai import Agent, Crew, Process
from a2a_adapter import load_a2a_agent, serve_agent
from a2a.types import AgentCard, AgentCapabilities, AgentSkill


def create_research_crew():
    """Create a research crew with specialized agents."""
    
    # Define agents
    researcher = Agent(
        role="Senior Research Analyst",
        goal="Uncover cutting-edge developments and insights",
        backstory="You're a seasoned researcher with a knack for finding the most relevant information.",
        verbose=True,
        allow_delegation=False,
    )
    
    writer = Agent(
        role="Content Writer",
        goal="Craft compelling content based on research findings",
        backstory="You're a skilled writer who can transform complex information into clear, engaging content.",
        verbose=True,
        allow_delegation=False,
    )
    
    # Note: Tasks will be created dynamically based on user input
    # For the adapter, we'll use a generic task structure
    
    crew = Crew(
        agents=[researcher, writer],
        tasks=[],  # Tasks will be added dynamically
        process=Process.sequential,
        verbose=True,
    )
    
    return crew


# Custom input mapper example (for complex parsing logic)
def custom_input_mapper(raw_input: str, context_id: str | None) -> dict:
    """
    Custom input mapper that handles both JSON and text inputs.
    
    This is useful when you need complex validation or transformation logic.
    """
    try:
        # Try to parse as JSON first
        data = json.loads(raw_input)
        return {
            "customer_domain": data.get("domain", "unknown"),
            "project_description": data.get("description", raw_input),
            "research_topic": data.get("topic", "general research"),
        }
    except json.JSONDecodeError:
        # Fallback for plain text
        return {
            "customer_domain": "general",
            "project_description": raw_input,
            "research_topic": raw_input,
        }


async def main():
    """Start serving a CrewAI crew as an A2A agent."""
    
    # Create the research crew
    crew = create_research_crew()
    
    # =================================================================
    # Example 1: Default JSON parsing (recommended for structured input)
    # =================================================================
    # Client sends: {"customer_domain": "example.com", "project_description": "..."}
    # Adapter automatically parses JSON and passes to crew
    adapter_json_mode = await load_a2a_agent({
        "adapter": "crewai",
        "crew": crew,
        # parse_json_input=True by default
    })
    
    # =================================================================
    # Example 2: Custom input_mapper (for complex transformation)
    # =================================================================
    # Use when you need custom validation/transformation logic
    adapter_custom_mapper = await load_a2a_agent({
        "adapter": "crewai",
        "crew": crew,
        "input_mapper": custom_input_mapper,
    })
    
    # =================================================================
    # Example 3: Plain text with default values
    # =================================================================
    # Client sends: "Research the AI market trends"
    # Adapter maps to inputs_key and merges default_inputs
    adapter_plain_text = await load_a2a_agent({
        "adapter": "crewai",
        "crew": crew,
        "parse_json_input": False,  # Disable JSON parsing
        "inputs_key": "research_topic",
        "default_inputs": {
            "customer_domain": "general",
            "project_description": "User research request",
        }
    })
    
    # Use the JSON mode adapter for this example
    adapter = adapter_json_mode
    
    # Define the agent card
    agent_card = AgentCard(
        name="Research Crew",
        description="Multi-agent research crew powered by CrewAI. Conducts in-depth research and produces comprehensive reports on any topic.",
        url="http://localhost:8001",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(
            streaming=False,  # CrewAI doesn't natively support streaming
        ),
        skills=[
            AgentSkill(
                id="research",
                name="research",
                description="Conduct comprehensive research on a topic",
                tags=["research", "analysis"]
            ),
            AgentSkill(
                id="analyze",
                name="analyze",
                description="Analyze information and provide insights",
                tags=["analysis", "insights"]
            ),
        ]
    )
    
    # Start serving the agent
    print("Starting Research Crew Agent on port 8001...")
    print("\nInput modes supported:")
    print('  1. JSON: {"customer_domain": "example.com", "project_description": "..."}')
    print('  2. Plain text: "Research the AI market trends"')
    serve_agent(agent_card=agent_card, adapter=adapter, port=8001)


if __name__ == "__main__":
    # Ensure OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        exit(1)
    
    asyncio.run(main())

