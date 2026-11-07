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

import contextlib
import hmac
import inspect
import ipaddress
import logging
import os
import signal
import sys
import threading
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import httpx
import uvicorn
from a2a.server.context import ServerCallContext
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_jsonrpc_routes, create_rest_routes
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
    HTTPAuthSecurityScheme,
    SecurityRequirement,
    SecurityScheme,
    StringList,
)
from a2a.utils.constants import (
    AGENT_CARD_WELL_KNOWN_PATH,
    PROTOCOL_VERSION_CURRENT,
    TransportProtocol,
)
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import JSONResponse
from starlette.routing import Route

from .base_adapter import BaseA2AAdapter
from .executor import AdapterAgentExecutor

if TYPE_CHECKING:
    from .terminal import ActivityLog, TerminalDisplay

logger = logging.getLogger(__name__)
_STARTUP_FAILURE = 3  # uvicorn's exit code when startup fails
LEGACY_AGENT_CARD_PATH = "/.well-known/agent.json"
# Signals that stop running CLI agents at once (SIGTERM drains first).
_INTERRUPT_SIGNALS = frozenset(
    sig for sig in (signal.SIGINT, getattr(signal, "SIGHUP", None)) if sig is not None
)


class _SafeInMemoryPushNotificationConfigStore(InMemoryPushNotificationConfigStore):
    """Ignore malformed empty-URL configs emitted by v0.3 compatibility clients."""

    async def set_info(self, task_id, notification_config, context) -> None:
        if not getattr(notification_config, "url", ""):
            return
        await super().set_info(task_id, notification_config, context)


_PUBLIC_CARD_PATHS = frozenset({AGENT_CARD_WELL_KNOWN_PATH, LEGACY_AGENT_CARD_PATH})


class _BearerAuthMiddleware:
    """Require ``Authorization: Bearer <token>`` on every HTTP request.

    Only GET/HEAD on the two agent-card paths stay public, so routes the SDK
    adds later are protected by default. Card paths are matched exactly:
    behind a proxy prefix or uvicorn ``root_path`` they need the token too.
    Pure ASGI, and it never sets ``scope["user"]``: the SDK keys push configs
    by user, and the push sender looks them up anonymously.
    """

    def __init__(self, app: Any, token: str, on_reject=None) -> None:
        self.app = app
        self._token = token.encode("utf-8")
        self._on_reject = on_reject

    async def __call__(self, scope: dict, receive, send) -> None:
        if scope["type"] == "websocket":
            if self._authorized(scope):
                await self.app(scope, receive, send)
            else:
                await send({"type": "websocket.close", "code": 1008})
            return
        if scope["type"] != "http":  # lifespan
            await self.app(scope, receive, send)
            return
        method = scope.get("method", "")
        path = scope.get("path", "")
        public = method in ("GET", "HEAD") and path in _PUBLIC_CARD_PATHS
        if public or self._authorized(scope):
            await self.app(scope, receive, send)
            return
        if self._on_reject is not None:
            try:
                self._on_reject(method, path)
            except Exception:
                logger.debug("Recording a rejected request failed", exc_info=True)
        response = JSONResponse(
            {"error": "unauthorized"}, status_code=401, headers={"WWW-Authenticate": "Bearer"}
        )
        await response(scope, receive, send)

    def _authorized(self, scope: dict) -> bool:
        for name, value in scope.get("headers", []):
            if name.lower() == b"authorization":
                scheme, _, credentials = value.strip().partition(b" ")
                return scheme.lower() == b"bearer" and hmac.compare_digest(
                    credentials.strip(), self._token
                )
        return False


def _is_loopback_host(host: str) -> bool:
    """Whether binding ``host`` only accepts connections from this machine."""
    host = host.strip("[]")
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _check_auth_token(auth_token: str | None) -> None:
    if auth_token is None:
        return
    if not auth_token.strip():
        raise ValueError("auth_token must not be empty; pass None to disable auth")
    if auth_token != auth_token.strip():
        # Clients send the token without it, so every request would get 401.
        raise ValueError("auth_token must not start or end with whitespace")
    if not all(" " <= char <= "~" for char in auth_token):
        # HTTP clients send header values as ASCII; anything else never matches.
        raise ValueError("auth_token must be printable ASCII")


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


def _push_sender(
    httpx_client: httpx.AsyncClient, config_store: InMemoryPushNotificationConfigStore
) -> BasePushNotificationSender:
    """Build the push sender for whichever a2a-sdk 1.x is installed.

    1.0 requires a ``context``; from 1.3 it is ignored and logs a
    deprecation warning unless it is None, so pass it only when required.
    """
    kwargs: dict[str, Any] = {"httpx_client": httpx_client, "config_store": config_store}
    try:
        context = inspect.signature(BasePushNotificationSender).parameters.get("context")
    except (TypeError, ValueError):  # uninspectable: only 1.0 needs a context, and it's not
        context = None
    if context is not None and context.default is inspect.Parameter.empty:
        kwargs["context"] = ServerCallContext()
    return BasePushNotificationSender(**kwargs)


def to_a2a(
    adapter: BaseA2AAdapter,
    agent_card: AgentCard | None = None,
    task_store: TaskStore | None = None,
    *,
    activity: "ActivityLog | None" = None,
    auth_token: str | None = None,
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
        activity: Optional ActivityLog that prints one row per A2A
            operation and task state change (used by the CLI).
        auth_token: Optional bearer token. When set, every request except
            GET/HEAD of the agent card needs ``Authorization: Bearer <token>``
            (401 otherwise), and an auto-generated card declares the bearer
            scheme. A card passed in is served as given. None (default)
            leaves the server open.
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
    _check_auth_token(auth_token)
    if agent_card is None:
        card_overrides.setdefault("bearer_auth", auth_token is not None)
        agent_card = build_agent_card(adapter, **card_overrides)
    elif auth_token is not None and not agent_card.security_schemes:
        logger.debug("auth_token is set but the given agent card declares no security scheme")

    task_store = task_store or InMemoryTaskStore()
    push_config_store = _SafeInMemoryPushNotificationConfigStore()
    push_httpx_client = httpx.AsyncClient()
    push_sender = _push_sender(push_httpx_client, push_config_store)

    executor = AdapterAgentExecutor(adapter, observer=activity)
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
    routes.extend(create_jsonrpc_routes(handler, rpc_url="/", enable_v0_3_compat=True))
    routes.extend(create_rest_routes(handler, enable_v0_3_compat=True))

    middleware = []
    if auth_token is not None:
        # Outermost: the activity middleware buffers and (at --debug) logs
        # request bodies, so it must never see an unauthenticated payload.
        on_reject = activity.request_rejected if activity is not None else None
        middleware.append(Middleware(_BearerAuthMiddleware, token=auth_token, on_reject=on_reject))
    if activity is not None:
        from .terminal import ActivityMiddleware

        middleware.append(Middleware(ActivityMiddleware, log=activity))

    return Starlette(routes=routes, lifespan=_lifespan, middleware=middleware)


class _ReadyServer(uvicorn.Server):
    """Uvicorn server with two hooks.

    ``on_ready`` runs once the socket is listening. ``on_interrupt`` (optional,
    synchronous) runs inside the signal handler on Ctrl+C (SIGINT), on SIGHUP
    or on a second signal of any kind, at most once. CLI agents run in their own
    process group, so the terminal's Ctrl+C no longer reaches them; this hook
    kills them so in-flight requests end and uvicorn can finish. A single
    SIGTERM keeps uvicorn's graceful drain; the lifespan's ``adapter.close()``
    stops the children afterwards.

    The work must be synchronous: uvicorn re-raises captured signals after
    ``serve()`` returns, so after SIGTERM (or a forced exit) nothing scheduled
    on the event loop is guaranteed to run.
    """

    def __init__(self, config: uvicorn.Config, on_ready, on_interrupt=None) -> None:
        super().__init__(config)
        self._on_ready = on_ready
        self._on_interrupt = on_interrupt
        self._signal_count = 0
        self._interrupt_done = False

    async def startup(self, sockets=None) -> None:
        await super().startup(sockets=sockets)
        if self.started and not self.should_exit:
            self._on_ready()

    @contextlib.contextmanager
    def capture_signals(self):
        """Also treat SIGHUP (terminal closed, SSH dropped) like Ctrl+C.

        The CLI agents run in their own session, so the hangup no longer
        reaches them; without this the server would die by SIGHUP and leave
        them running. uvicorn re-raises captured signals once the original
        handlers are back, so the process still exits by SIGHUP afterwards.
        """
        with super().capture_signals():
            sighup = getattr(signal, "SIGHUP", None)
            if (
                sighup is None
                or threading.current_thread() is not threading.main_thread()
                or signal.getsignal(sighup) is signal.SIG_IGN  # e.g. under nohup: keep it
            ):
                yield
                return
            original = signal.signal(sighup, self.handle_exit)
            try:
                yield
            finally:
                signal.signal(sighup, original)

    def handle_exit(self, sig: int, frame) -> None:
        super().handle_exit(sig, frame)
        # Our own counter: uvicorn sets force_exit only on a second SIGINT.
        self._signal_count += 1
        if self._on_interrupt is None or self._interrupt_done:
            return
        if sig in _INTERRUPT_SIGNALS or self._signal_count >= 2:
            self._interrupt_done = True
            try:
                self._on_interrupt()
            except Exception:
                logger.exception("Stopping running agents on interrupt failed")


def serve_agent(
    adapter: BaseA2AAdapter,
    agent_card: AgentCard | None = None,
    host: str = "127.0.0.1",
    port: int = 9000,
    log_level: str = "info",
    *,
    display: "TerminalDisplay | None" = None,
    auth_token: str | None = None,
    **kwargs: Any,
) -> None:
    """One-line A2A server startup.

    Combines to_a2a() with an in-process uvicorn server.
    This is the recommended way to start an A2A server for development
    and simple deployments.

    Args:
        adapter: Any BaseA2AAdapter implementation.
        agent_card: Optional pre-built AgentCard.
        host: Host address to bind to (default: "127.0.0.1", this machine
            only). Use "0.0.0.0" to accept remote connections, with
            ``auth_token``.
        port: Port to listen on (default: 9000).
        log_level: Logging level (default: "info").
        auth_token: Optional bearer token required on every request except
            the agent card (see :func:`to_a2a`). Serving a non-loopback host
            without one logs a warning.
        display: Optional TerminalDisplay. When given, Uvicorn's own
            startup and access lines are replaced by the display's card
            (shown once the socket is listening) and activity rail.
        **kwargs: Additional arguments passed to ``uvicorn.Config``
            (e.g., ssl_keyfile, uds; ``app_dir`` as in uvicorn.run()).
            ``workers`` and ``reload`` raise ValueError: they need an
            import string, not an app object.

    Ctrl+C or a terminal hangup stops running CLI agents (and everything they started)
    immediately; SIGTERM lets in-flight requests finish within uvicorn's
    graceful timeout, then stops them.

    Example:
        >>> from a2a_adapter import N8nAdapter, serve_agent
        >>> adapter = N8nAdapter(webhook_url="http://localhost:5678/webhook/agent")
        >>> serve_agent(adapter, port=9000)
    """
    _check_auth_token(auth_token)
    if agent_card is None:
        display_host = "localhost" if host in ("0.0.0.0", "::") else host
        agent_card = build_agent_card(
            adapter,
            url=f"http://{display_host}:{port}/",
            bearer_auth=auth_token is not None,
        )

    if kwargs.get("workers", 1) not in (None, 1) or kwargs.get("reload"):
        raise ValueError("serve_agent runs one in-process server; drop workers/reload")
    on_interrupt = getattr(adapter, "kill_children_now", None)

    if display is None:
        app = to_a2a(adapter, agent_card, auth_token=auth_token)
        _warn_if_exposed(host, auth_token, kwargs)
        # What uvicorn.run() does around Server.run(): app_dir on sys.path,
        # Ctrl+C swallowed, the uds socket removed, exit 3 on failed startup.
        app_dir = kwargs.pop("app_dir", None)
        if app_dir is not None:
            sys.path.insert(0, app_dir)
        config = uvicorn.Config(app, host=host, port=port, log_level=log_level, **kwargs)
        server = _ReadyServer(config, on_ready=lambda: None, on_interrupt=on_interrupt)
        try:
            server.run()
        except KeyboardInterrupt:
            pass
        finally:
            if config.uds and os.path.exists(config.uds):
                os.remove(config.uds)
        if not server.started:
            sys.exit(_STARTUP_FAILURE)
        return

    owned = sorted({"access_log", "log_config"} & kwargs.keys())
    if owned:
        raise ValueError(
            f"serve_agent(display=...) sets Uvicorn logging itself; drop {', '.join(owned)}"
        )
    app = to_a2a(adapter, agent_card, activity=display.activity, auth_token=auth_token)
    display.install_logging(log_level)
    try:
        _warn_if_exposed(host, auth_token, kwargs)
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level=log_level,
            log_config=None,
            access_log=False,
            **kwargs,
        )
        server = _ReadyServer(
            config, on_ready=lambda: display.ready(agent_card), on_interrupt=on_interrupt
        )
        try:
            server.run()
        except KeyboardInterrupt:
            pass  # Ctrl+C after a graceful shutdown, as in uvicorn.run()
        finally:
            display.close()
    finally:
        display.uninstall_logging()
    if not server.started:
        # Startup failed (e.g. the lifespan raised); exit as uvicorn.run() does.
        sys.exit(_STARTUP_FAILURE)


def _warn_if_exposed(host: str, auth_token: str | None, kwargs: dict) -> None:
    if auth_token is None and not _is_loopback_host(host) and not {"uds", "fd"} & kwargs.keys():
        logger.warning(
            "Serving on %s without auth: anyone who can reach this port can run the agent. "
            "Set auth_token / --auth-token or bind 127.0.0.1.",
            host,
        )


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
            provider, documentation_url, icon_url, and ``bearer_auth``
            (bool, default False) to declare a bearer-token security scheme.

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

    security: dict[str, Any] = {}
    if overrides.get("bearer_auth", False):
        security = {
            "security_schemes": {
                "bearer": SecurityScheme(
                    http_auth_security_scheme=HTTPAuthSecurityScheme(scheme="bearer")
                )
            },
            "security_requirements": [SecurityRequirement(schemes={"bearer": StringList()})],
        }

    return AgentCard(
        **security,
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
        skills=(
            [
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
            ]
            if meta.skills
            else []
        ),
        default_input_modes=meta.input_modes,
        default_output_modes=meta.output_modes,
        provider=provider,
        documentation_url=overrides.get("documentation_url", meta.documentation_url),
        icon_url=overrides.get("icon_url", meta.icon_url),
    )
