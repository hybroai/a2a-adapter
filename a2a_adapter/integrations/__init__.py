"""
Framework-specific adapter implementations.

This package contains concrete adapter implementations for various agent frameworks:
- n8n: HTTP webhook-based workflows
- CrewAI: Multi-agent collaboration framework
- LangChain: LLM application framework with LCEL support
- LangGraph: Stateful workflow orchestration framework
- Callable: Generic Python async function adapter
- OpenClaw: Personal AI super agent CLI wrapper

Each module exports both:
- v0.2 adapter class (e.g., N8nAdapter) — new simplified interface
- v0.1 adapter class (e.g., N8nAgentAdapter) — deprecated, for backwards compat
"""

__all__ = [
    # v0.2 (new)
    "N8nAdapter",
    "CrewAIAdapter",
    "LangChainAdapter",
    "LangGraphAdapter",
    "CallableAdapter",
    "OpenClawAdapter",
    # v0.1 (deprecated)
    "N8nAgentAdapter",
    "CrewAIAgentAdapter",
    "LangChainAgentAdapter",
    "LangGraphAgentAdapter",
    "CallableAgentAdapter",
    "OpenClawAgentAdapter",
]


# Lazy imports to avoid requiring all optional dependencies
def __getattr__(name: str):
    # v0.2 adapters
    if name == "N8nAdapter":
        from .n8n import N8nAdapter
        return N8nAdapter
    elif name == "CallableAdapter":
        from .callable import CallableAdapter
        return CallableAdapter
    elif name == "LangChainAdapter":
        from .langchain import LangChainAdapter
        return LangChainAdapter
    elif name == "LangGraphAdapter":
        from .langgraph import LangGraphAdapter
        return LangGraphAdapter
    # v0.2 adapters not yet migrated — will be added in PR 4
    elif name in ("CrewAIAdapter", "OpenClawAdapter"):
        raise AttributeError(
            f"{name} is not yet available. It will be added in a future PR. "
            f"Use the v0.1 class name for now (e.g., {name.replace('Adapter', 'AgentAdapter')})."
        )
    # v0.1 adapters (deprecated)
    elif name == "N8nAgentAdapter":
        from .n8n import N8nAgentAdapter
        return N8nAgentAdapter
    elif name == "CrewAIAgentAdapter":
        from .crewai import CrewAIAgentAdapter
        return CrewAIAgentAdapter
    elif name == "LangChainAgentAdapter":
        from .langchain import LangChainAgentAdapter
        return LangChainAgentAdapter
    elif name == "LangGraphAgentAdapter":
        from .langgraph import LangGraphAgentAdapter
        return LangGraphAgentAdapter
    elif name == "CallableAgentAdapter":
        from .callable import CallableAgentAdapter
        return CallableAgentAdapter
    elif name == "OpenClawAgentAdapter":
        from .openclaw import OpenClawAgentAdapter
        return OpenClawAgentAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
