"""
Framework-specific adapter implementations.

This package contains concrete adapter implementations for various agent frameworks:
- n8n: HTTP webhook-based workflows
- CrewAI: Multi-agent collaboration framework
- LangChain: LLM application framework with LCEL support
- LangGraph: Stateful workflow orchestration framework
- Callable: Generic Python async function adapter
- OpenClaw: Personal AI super agent CLI wrapper
"""

__all__ = [
    "N8nAgentAdapter",
    "CrewAIAgentAdapter",
    "LangChainAgentAdapter",
    "LangGraphAgentAdapter",
    "CallableAgentAdapter",
    "OpenClawAgentAdapter",
]

# Lazy imports to avoid requiring all optional dependencies
def __getattr__(name: str):
    if name == "N8nAgentAdapter":
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
