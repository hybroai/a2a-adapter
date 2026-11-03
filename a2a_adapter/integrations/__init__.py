"""
Framework-specific adapter implementations.

This package contains concrete adapter implementations for various agent frameworks:
- n8n: HTTP webhook-based workflows
- CrewAI: Multi-agent collaboration framework
- LangChain: LLM application framework with LCEL support
- LangGraph: Stateful workflow orchestration framework
- Callable: Generic Python async function adapter
- Claude Code: Streaming coding agent CLI wrapper
- Codex: Coding agent CLI wrapper
- OpenClaw: Personal AI super agent CLI wrapper
- Ollama: Local model HTTP integration
- Hermes: Multi-purpose AI agent with tool use and persistent memory
- Pi: Persistent coding agent RPC wrapper
- DeepSeek Harness: dsh coding agent via its Python SDK runtime

Migrated framework modules also retain deprecated v0.1 `*AgentAdapter`
classes for backwards compatibility. Newer CLI-backed integrations expose
only the current adapter interface.
"""

__all__ = [
    # Current adapters
    "N8nAdapter",
    "CrewAIAdapter",
    "LangChainAdapter",
    "LangGraphAdapter",
    "CallableAdapter",
    "OpenClawAdapter",
    "ClaudeCodeAdapter",
    "CodexAdapter",
    "OllamaAdapter",
    "OllamaClient",
    "HermesAdapter",
    "PiAdapter",
    "DeepSeekHarnessAdapter",
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
    # Current adapters
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
    elif name == "CrewAIAdapter":
        from .crewai import CrewAIAdapter
        return CrewAIAdapter
    elif name == "OpenClawAdapter":
        from .openclaw import OpenClawAdapter
        return OpenClawAdapter
    elif name == "ClaudeCodeAdapter":
        from .claude_code import ClaudeCodeAdapter
        return ClaudeCodeAdapter
    elif name == "CodexAdapter":
        from .codex import CodexAdapter
        return CodexAdapter
    elif name == "OllamaAdapter":
        from .ollama import OllamaAdapter
        return OllamaAdapter
    elif name == "OllamaClient":
        from .ollama import OllamaClient
        return OllamaClient
    elif name == "HermesAdapter":
        from .hermes import HermesAdapter
        return HermesAdapter
    elif name == "PiAdapter":
        from .pi import PiAdapter
        return PiAdapter
    elif name == "DeepSeekHarnessAdapter":
        from .deepseek_harness import DeepSeekHarnessAdapter
        return DeepSeekHarnessAdapter
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
