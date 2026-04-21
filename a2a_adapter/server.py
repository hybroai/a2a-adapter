"""
Server layer: to_a2a(), serve_agent(), build_agent_card().

This module provides user-facing entry points for converting any
BaseA2AAdapter into a fully compliant A2A Protocol server.

Design rationale:
    This layer wires together the adapter, bridge (AdapterAgentExecutor),
    the SDK request handler, and a Starlette ASGI app built from the
    a2a-sdk 1.x route helpers.

Replaces: client.py (deprecated in v0.2, removed in v0.3)
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

import httpx
import uvicorn
from a2a.server.context import ServerCallContext
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes, create_rest_routes
from a2a.server.request_handlers.response_helpers import agent_card_to_dict
from a2a.server.tasks import (
    BasePushNotificationSender,
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
)
from a2a.server.tasks.task_store import TaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
)
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH, PROTOCOL_VERSION_CURRENT, TransportProtocol
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from .base_adapter import BaseA2AAdapter
from .executor import AdapterAgentExecutor

logger = logging.getLogger(__name__)
LEGACY_AGENT_CARD_PATH = "/.well-known/agent.json"


class _SafeInMemoryPushNotificationConfigStore(InMemoryPushNotificationConfigStore):
    """Ignore malformed empty-URL configs emitted by v0.3 compatibility clients."""

    async def set_info(self, task_id, notification_config, context) -> None:
        if not getattr(notification_config, "url", ""):
            return
        await super().set_info(task_id, notification_config, context)


def _legacy_card_url(agent_card: AgentCard) -> str | None:
    if agent_card.supported_interfaces:
        primary = agent_card.supported_interfaces[0]
        return primary.url or None
    return None


def _agent_card_json(agent_card: AgentCard) -> dict[str, Any]:
    """Return a superset JSON card: A2A 1.0 + legacy top-level url."""
    data = agent_card_to_dict(agent_card)
    # Legacy consumers in Hybro backend still require `skills` to exist even
    # when the A2A 1.0 proto omits empty repeated fields from JSON output.
    data.setdefault("skills", [])
    legacy_url = _legacy_card_url(agent_card)
    if legacy_url:
        data.setdefault("url", legacy_url)
        data.setdefault("preferredTransport", TransportProtocol.JSONRPC)
        data.setdefault("protocolVersion", PROTOCOL_VERSION_CURRENT)
    return data


def _create_compat_agent_card_routes(agent_card: AgentCard) -> list[Route]:
    async def _get_agent_card(_request) -> JSONResponse:
        return JSONResponse(_agent_card_json(agent_card))

    paths = [AGENT_CARD_WELL_KNOWN_PATH]
    if LEGACY_AGENT_CARD_PATH != AGENT_CARD_WELL_KNOWN_PATH:
        paths.append(LEGACY_AGENT_CARD_PATH)
    return [Route(path=path, endpoint=_get_agent_card, methods=["GET"]) for path in paths]


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

    task_store = task_store or InMemoryTaskStore()
    push_config_store = _SafeInMemoryPushNotificationConfigStore()
    push_httpx_client = httpx.AsyncClient()
    push_sender = BasePushNotificationSender(
        httpx_client=push_httpx_client,
        config_store=push_config_store,
        context=ServerCallContext(),
    )

    executor = AdapterAgentExecutor(adapter)
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
        agent_card=agent_card,
        push_config_store=push_config_store,
        push_sender=push_sender,
    )

    @asynccontextmanager
    async def _lifespan(app):
        yield
        logger.info("Shutting down: closing adapter and HTTP clients")
        try:
            await adapter.close()
        finally:
            await push_httpx_client.aclose()

    routes = []
    routes.extend(_create_compat_agent_card_routes(agent_card))
    routes.extend(
        create_jsonrpc_routes(handler, rpc_url="/", enable_v0_3_compat=True)
    )
    routes.extend(create_rest_routes(handler, enable_v0_3_compat=True))

    return Starlette(routes=routes, lifespan=_lifespan)


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
    if agent_card is None:
        display_host = "localhost" if host in ("0.0.0.0", "::") else host
        agent_card = build_agent_card(
            adapter,
            url=f"http://{display_host}:{port}/",
        )

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

    The generated card is served at both the SDK 1.x well-known path and
    the legacy ``/.well-known/agent.json`` alias for compatibility.

    Args:
        adapter: The adapter to generate a card for.
        **overrides: Override any auto-generated field.
            Supported keys: name, description, version, streaming, url,
            provider, documentation_url, icon_url.

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

    # Build provider from metadata or override
    provider_data = overrides.get("provider", meta.provider)
    provider = None
    if provider_data:
        if isinstance(provider_data, dict):
            provider = AgentProvider(**provider_data)
        else:
            provider = provider_data

    return AgentCard(
        name=overrides.get("name", meta.name or type(adapter).__name__),
        description=overrides.get("description", meta.description or ""),
        version=overrides.get("version", meta.version),
        capabilities=AgentCapabilities(
            streaming=streaming,
            push_notifications=True,
        ),
        supported_interfaces=[
            AgentInterface(
                url=overrides.get("url", "http://localhost:9000/"),
                protocol_binding=TransportProtocol.JSONRPC,
                protocol_version=PROTOCOL_VERSION_CURRENT,
            )
        ],
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
        provider=provider,
        documentation_url=overrides.get("documentation_url", meta.documentation_url),
        icon_url=overrides.get("icon_url", meta.icon_url),
    )
