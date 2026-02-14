"""
Example: v0.2 Quick Start — All Adapters

Demonstrates the new v0.2 API for each adapter type.
Each adapter can be started with a single `serve_agent()` call.
No manual AgentCard construction needed — it's auto-generated.

Usage:
    python examples/v02_quickstart.py callable   # Echo agent
    python examples/v02_quickstart.py n8n        # n8n webhook
    python examples/v02_quickstart.py custom     # Custom adapter class
"""

import sys


def start_callable_agent():
    """Simplest possible agent — wraps an async function."""
    from a2a_adapter import CallableAdapter, serve_agent

    async def echo(inputs):
        return f"Echo: {inputs['message']}"

    adapter = CallableAdapter(
        func=echo,
        name="Echo Agent",
        description="Echoes back whatever you send",
    )
    serve_agent(adapter, port=9000)


def start_n8n_agent():
    """n8n webhook agent — forwards requests to an n8n workflow."""
    from a2a_adapter import N8nAdapter, serve_agent

    adapter = N8nAdapter(
        webhook_url="http://localhost:5678/webhook/my-agent",
        timeout=30,
        name="N8n Math Agent",
        description="Math operations via n8n workflow",
    )
    serve_agent(adapter, port=9000)


def start_custom_agent():
    """Custom adapter class — full control with minimal code."""
    from a2a_adapter import BaseA2AAdapter, AdapterMetadata, serve_agent

    class SentimentAdapter(BaseA2AAdapter):
        async def invoke(self, user_input: str, context_id: str | None = None, **kwargs) -> str:
            positive = ["good", "great", "happy", "love", "excellent"]
            negative = ["bad", "terrible", "sad", "hate", "awful"]
            text = user_input.lower()
            pos = sum(1 for w in positive if w in text)
            neg = sum(1 for w in negative if w in text)
            if pos > neg:
                return f"Positive sentiment detected in: {user_input}"
            elif neg > pos:
                return f"Negative sentiment detected in: {user_input}"
            return f"Neutral sentiment detected in: {user_input}"

        def get_metadata(self) -> AdapterMetadata:
            return AdapterMetadata(
                name="Sentiment Analyzer",
                description="Analyzes the sentiment of text messages",
            )

    serve_agent(SentimentAdapter(), port=9000)


if __name__ == "__main__":
    agents = {
        "callable": start_callable_agent,
        "n8n": start_n8n_agent,
        "custom": start_custom_agent,
    }

    agent_type = sys.argv[1] if len(sys.argv) > 1 else "callable"

    if agent_type not in agents:
        print(f"Unknown type: {agent_type}. Available: {', '.join(agents)}")
        sys.exit(1)

    print(f"Starting {agent_type} agent on port 9000...")
    print("AgentCard auto-served at http://localhost:9000/.well-known/agent-card.json")
    agents[agent_type]()
