"""
a2a-adapter: Convert any AI agent into an A2A Protocol server.

Usage (v0.2 — recommended):
    >>> from a2a_adapter import N8nAdapter, serve_agent
    >>>
    >>> adapter = N8nAdapter(webhook_url="http://localhost:5678/webhook/agent")
    >>> serve_agent(adapter, port=9000)

Usage (v0.1 — deprecated, still works):
    >>> from a2a_adapter import load_a2a_agent, serve_agent
    >>> # See migration guide for upgrading to v0.2 API
"""

__version__ = "0.2.0"

# ──── v0.2 Core Exports (eager, no optional deps) ────
from .base_adapter import AdapterMetadata, BaseA2AAdapter
from .server import build_agent_card, serve_agent, to_a2a

# ──── v0.2 Loader Exports ────
from .loader import load_adapter, register_adapter

# ──── v0.1 Backwards Compatibility (deprecated) ────
from .adapter import BaseAgentAdapter  # deprecated: use BaseA2AAdapter
from .loader import load_a2a_agent  # deprecated: use load_adapter
from .client import build_agent_app  # deprecated: use to_a2a

# Note: v0.1 serve_agent(card, adapter) is replaced by v0.2
# serve_agent(adapter). The v0.2 version is imported above from .server.
# The old signature from .client is NOT re-exported to avoid conflicts.
# Users of the old API should use build_agent_app() + uvicorn directly,
# or migrate to the new serve_agent(adapter) API.

# ──── Lazy Imports (framework adapters with optional deps) ────
_ADAPTER_LAZY_MAP = {
    "N8nAdapter": (".integrations.n8n", "N8nAdapter"),
    "CrewAIAdapter": (".integrations.crewai", "CrewAIAdapter"),
    "LangChainAdapter": (".integrations.langchain", "LangChainAdapter"),
    "LangGraphAdapter": (".integrations.langgraph", "LangGraphAdapter"),
    "CallableAdapter": (".integrations.callable", "CallableAdapter"),
    "OpenClawAdapter": (".integrations.openclaw", "OpenClawAdapter"),
}


def __getattr__(name: str):
    """Lazy-load adapter classes on first access.

    This avoids importing optional framework dependencies (crewai,
    langchain, langgraph, etc.) until the user actually requests a
    specific adapter class.
    """
    if name in _ADAPTER_LAZY_MAP:
        import importlib

        module_path, class_name = _ADAPTER_LAZY_MAP[name]
        module = importlib.import_module(module_path, package="a2a_adapter")
        value = getattr(module, class_name)
        globals()[name] = value  # Cache for subsequent accesses
        return value
    raise AttributeError(f"module 'a2a_adapter' has no attribute {name!r}")


__all__ = [
    "__version__",
    # v0.2 Core
    "BaseA2AAdapter",
    "AdapterMetadata",
    # v0.2 Server
    "serve_agent",
    "to_a2a",
    "build_agent_card",
    # v0.2 Loader
    "load_adapter",
    "register_adapter",
    # v0.1 Deprecated (still exported for backwards compat)
    "BaseAgentAdapter",
    "load_a2a_agent",
    "build_agent_app",
    # Adapters (lazy-loaded)
    "N8nAdapter",
    "CrewAIAdapter",
    "LangChainAdapter",
    "LangGraphAdapter",
    "CallableAdapter",
    "OpenClawAdapter",
]
