"""
Example: Single CrewAI Agent Server

This example demonstrates how to expose a CrewAI crew as an A2A-compliant agent.
The crew performs research tasks using multiple specialized agents.

Prerequisites:
- crewai package installed: pip install crewai
- OpenAI API key set in environment: OPENAI_API_KEY

Usage:
    python examples/02_single_crewai_agent.py
"""

import asyncio
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


async def main():
    """Start serving a CrewAI crew as an A2A agent."""
    
    # Create the research crew
    crew = create_research_crew()
    
    # Load the adapter
    adapter = await load_a2a_agent({
        "adapter": "crewai",
        "crew": crew,
        "inputs_key": "inputs",
    })
    
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
    serve_agent(agent_card=agent_card, adapter=adapter, port=8001)


if __name__ == "__main__":
    # Ensure OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        exit(1)
    
    asyncio.run(main())

