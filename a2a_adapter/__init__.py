"""
A2A Adapters - Python SDK for A2A Protocol Integration.

This package provides adapters to integrate various agent frameworks with the
A2A (Agent-to-Agent) Protocol, enabling seamless inter-agent communication.

Main exports:
- load_a2a_agent: Factory function to create adapters from configuration
- build_agent_app: Build an ASGI app for serving an A2A agent
- serve_agent: Convenience function to start serving an A2A agent
- BaseAgentAdapter: Abstract base class for creating custom adapters

Example:
    >>> import asyncio
    >>> from a2a_adapter import load_a2a_agent, serve_agent
    >>> from a2a.types import AgentCard
    >>> 
    >>> async def main():
    ...     adapter = await load_a2a_agent({
    ...         "adapter": "n8n",
    ...         "webhook_url": "https://n8n.example.com/webhook"
    ...     })
    ...     card = AgentCard(name="My Agent", description="...")
    ...     serve_agent(card, adapter, port=9000)
    >>> 
    >>> asyncio.run(main())
"""

__version__ = "0.1.0"

from .adapter import BaseAgentAdapter
from .client import build_agent_app, serve_agent
from .loader import load_a2a_agent

__all__ = [
    "__version__",
    "BaseAgentAdapter",
    "load_a2a_agent",
    "build_agent_app",
    "serve_agent",
]

