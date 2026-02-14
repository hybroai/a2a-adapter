"""
Server layer: to_a2a(), serve_agent(), build_agent_card().

This module provides user-facing entry points for converting any
BaseA2AAdapter into a fully compliant A2A Protocol server.

Design rationale:
    This layer wires together the adapter, bridge (AdapterAgentExecutor),
    SDK handler (DefaultRequestHandler), and ASGI app (A2AStarletteApplication).
    The old AdapterRequestHandler in client.py is completely replaced by
    DefaultRequestHandler, which handles all A2A protocol details.

Replaces: client.py (deprecated in v0.2, removed in v0.3)
"""

import logging
from typing import Any

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.tasks.task_store import TaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from .base_adapter import BaseA2AAdapter
from .executor import AdapterAgentExecutor

logger = logging.getLogger(__name__)


def to_a2a(
    adapter: BaseA2AAdapter,
    agent_card: AgentCard | None = None,
    task_store: TaskStore | None = None,
    **card_overrides: Any,
) -> Any:
    """Convert any adapter into an A2A Protocol ASGI application.

    This is the "magic function" that wires everything together.
    Give it an adapter, get back a standards-compliant A2A server.

    Aligned with Google ADK's ``to_a2a(root_agent)`` pattern.

    Args:
        adapter: Any BaseA2AAdapter implementation.
        agent_card: Optional pre-built AgentCard. If None, one is
            auto-generated from adapter.get_metadata().
        task_store: Optional TaskStore for task persistence.
            Defaults to InMemoryTaskStore.
        **card_overrides: Override individual AgentCard fields
            (name, description, url, version, streaming).

    Returns:
        A Starlette ASGI application ready to be served.

    Example:
        >>> from a2a_adapter import N8nAdapter, to_a2a
        >>> adapter = N8nAdapter(webhook_url="http://localhost:5678/webhook/agent")
        >>> app = to_a2a(adapter)
        >>> # Deploy with any ASGI server: uvicorn, hypercorn, daphne, etc.
    """
    if agent_card is None:
        agent_card = build_agent_card(adapter, **card_overrides)

    executor = AdapterAgentExecutor(adapter)
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store or InMemoryTaskStore(),
    )

    app_builder = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=handler,
    )
    return app_builder.build()


def serve_agent(
    adapter: BaseA2AAdapter,
    agent_card: AgentCard | None = None,
    host: str = "0.0.0.0",
    port: int = 9000,
    log_level: str = "info",
    **kwargs: Any,
) -> None:
    """One-line A2A server startup.

    Combines to_a2a() + uvicorn.run() for maximum convenience.
    This is the recommended way to start an A2A server for development
    and simple deployments.

    Args:
        adapter: Any BaseA2AAdapter implementation.
        agent_card: Optional pre-built AgentCard.
        host: Host address to bind to (default: "0.0.0.0").
        port: Port to listen on (default: 9000).
        log_level: Logging level (default: "info").
        **kwargs: Additional arguments passed to uvicorn.run()
            (e.g., workers, ssl_keyfile, reload).

    Example:
        >>> from a2a_adapter import N8nAdapter, serve_agent
        >>> adapter = N8nAdapter(webhook_url="http://localhost:5678/webhook/agent")
        >>> serve_agent(adapter, port=9000)
    """
    app = to_a2a(adapter, agent_card)
    uvicorn.run(app, host=host, port=port, log_level=log_level, **kwargs)


def build_agent_card(
    adapter: BaseA2AAdapter,
    **overrides: Any,
) -> AgentCard:
    """Auto-generate an AgentCard from adapter metadata.

    Most developers don't want to manually construct AgentCard objects
    with 10+ fields. This function reads the adapter's get_metadata()
    and produces a reasonable default card.

    The generated card is served at ``/.well-known/agent.json`` by the
    A2A Starlette application.

    Args:
        adapter: The adapter to generate a card for.
        **overrides: Override any auto-generated field.
            Supported keys: name, description, url, version, streaming.

    Returns:
        A fully populated AgentCard.

    Example:
        >>> from a2a_adapter import N8nAdapter, build_agent_card
        >>> adapter = N8nAdapter(webhook_url="...")
        >>> card = build_agent_card(adapter, name="Math Agent", url="http://myserver:9000")
    """
    meta = adapter.get_metadata()

    # Auto-detect streaming if not explicitly set in metadata
    streaming = overrides.get("streaming", meta.streaming)
    if not streaming:
        streaming = adapter.supports_streaming()

    return AgentCard(
        name=overrides.get("name", meta.name or type(adapter).__name__),
        description=overrides.get("description", meta.description or ""),
        url=overrides.get("url", "http://localhost:9000"),
        version=overrides.get("version", meta.version),
        capabilities=AgentCapabilities(
            streaming=streaming,
        ),
        skills=[
            AgentSkill(
                id=s.get("id", f"skill-{i}"),
                name=s.get("name", ""),
                description=s.get("description", ""),
                tags=s.get("tags", []),
                examples=s.get("examples"),
                input_modes=s.get("input_modes"),
                output_modes=s.get("output_modes"),
            )
            for i, s in enumerate(meta.skills)
        ] if meta.skills else [],
        default_input_modes=meta.input_modes,
        default_output_modes=meta.output_modes,
    )
