"""
Example: Single n8n Agent Server

This example demonstrates how to expose an n8n workflow as an A2A-compliant agent.
The n8n workflow receives A2A messages via webhook and returns responses.

Prerequisites:
- A running n8n instance with a webhook configured
- Configure the adapter to match your n8n workflow's expected payload format

Usage:
    python examples/01_single_n8n_agent.py
"""

import asyncio
import os

from a2a_adapter import load_a2a_agent, serve_agent
from a2a.types import AgentCard, AgentCapabilities, AgentSkill


async def setup_agent():
    """Setup and return the agent configuration."""

    # Configuration for the n8n adapter
    # Replace with your actual n8n webhook URL
    webhook_url = os.getenv(
        "N8N_WEBHOOK_URL",
        "http://localhost:5678/webhook/my-webhook"
    )

    # Load the adapter with custom payload mapping
    # This example configures the adapter to send:
    #   {"name": "A2A Agent", "event": "<user message>"}
    # Instead of the default:
    #   {"message": "<user message>", "metadata": {...}}
    adapter = await load_a2a_agent({
        "adapter": "n8n",
        "webhook_url": webhook_url,
        "timeout": 30,
        # Custom payload template - add static fields your n8n workflow expects
        "payload_template": {
            "name": "A2A Agent",
        },
        # Custom message field name - use "event" instead of default "message"
        "message_field": "event",
        "headers": {
            # Optional: Add custom headers if your n8n webhook requires authentication
            # "Authorization": "Bearer YOUR_TOKEN"
        }
    })

    # Define the agent card describing this agent's capabilities
    agent_card = AgentCard(
        name="N8n Math Agent",
        description="Math operations agent powered by an n8n workflow. Can perform calculations, solve equations, and provide mathematical insights.",
        url="http://localhost:9000",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(
            streaming=False,  # n8n webhooks don't support streaming
        ),
        skills=[
            AgentSkill(
                id="calculate",
                name="calculate",
                description="Perform mathematical calculations",
                tags=["math", "calculation"]
            ),
            AgentSkill(
                id="solve_equation",
                name="solve_equation",
                description="Solve mathematical equations",
                tags=["math", "equation"]
            ),
        ]
    )

    return adapter, agent_card, webhook_url


def main():
    """Main entry point - setup agent and start server."""

    # Run async setup
    adapter, agent_card, webhook_url = asyncio.run(setup_agent())

    # Start serving the agent (this will block)
    print(f"Starting N8n Math Agent on port 9000...")
    print(f"Webhook URL: {webhook_url}")
    serve_agent(agent_card=agent_card, adapter=adapter, port=9000)


if __name__ == "__main__":
    main()

