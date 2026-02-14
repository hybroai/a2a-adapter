"""
Example: CrewAI Crew as A2A Agent

Expose a CrewAI multi-agent crew as an A2A-compatible agent in 3 lines.

Prerequisites:
- crewai installed: pip install a2a-adapter[crewai]
- OPENAI_API_KEY set in environment

Usage:
    python examples/crewai_agent.py
"""

import os

from crewai import Agent, Crew, Process
from a2a_adapter import CrewAIAdapter, serve_agent

# --- Set up your CrewAI crew (this is framework code, not adapter code) ---

researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge developments and insights",
    backstory="You're a seasoned researcher with a knack for finding relevant information.",
    verbose=True,
    allow_delegation=False,
)

writer = Agent(
    role="Content Writer",
    goal="Craft compelling content based on research findings",
    backstory="You're a skilled writer who transforms complex info into clear content.",
    verbose=True,
    allow_delegation=False,
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[],
    process=Process.sequential,
    verbose=True,
)

# --- A2A: 3 lines ---

adapter = CrewAIAdapter(
    crew=crew,
    name="Research Crew",
    description="Multi-agent research crew that conducts research and writes reports",
    timeout=600,
)

serve_agent(adapter, port=8001)
