---
name: a2a-adapter
description: Use when building A2A Protocol agents, converting AI agents from any framework (LangChain, CrewAI, n8n, LangGraph, Ollama, or custom) into A2A-compatible servers, or working with the a2a-adapter Python SDK
---

# a2a-adapter SDK

Convert any AI agent into an A2A Protocol server. Adapters only answer "given text, return text" -- all protocol handling (JSON-RPC, task management, SSE streaming, push notifications) is delegated to the A2A SDK automatically.

## Install

```bash
pip install a2a-adapter                # Core (n8n, callable, ollama, openclaw)
pip install a2a-adapter[crewai]        # + CrewAI
pip install a2a-adapter[langchain]     # + LangChain
pip install a2a-adapter[langgraph]     # + LangGraph
pip install a2a-adapter[all]           # Everything
```

## Core Pattern (3 lines)

```python
from a2a_adapter import XxxAdapter, serve_agent

adapter = XxxAdapter(...)       # Create adapter
serve_agent(adapter, port=9000) # Start A2A server
```

`serve_agent()` starts uvicorn with auto-generated AgentCard at `/.well-known/agent.json`.

## Decision Guide

| Scenario | Use |
|---|---|
| Wrap a function | `CallableAdapter(func=fn)` |
| n8n workflow | `N8nAdapter(webhook_url=...)` |
| LangChain chain | `LangChainAdapter(runnable=chain)` |
| LangGraph workflow | `LangGraphAdapter(graph=graph)` |
| CrewAI crew | `CrewAIAdapter(crew=crew)` |
| OpenClaw agent | `OpenClawAdapter(...)` |
| Local Ollama model | `OllamaAdapter(model="llama3.2:8b")` |
| Any other framework | Subclass `BaseA2AAdapter`, implement `invoke()` |
| Need streaming | Implement `stream()` or use LangChain/LangGraph/Ollama (auto) |
| Need multimodal output | Return `list[Part]` from `invoke()` |
| Production deploy | `to_a2a(adapter)` -> ASGI server |
| Config-driven | `load_adapter({"adapter": "n8n", ...})` |

## Custom Adapter Template

```python
from a2a_adapter import BaseA2AAdapter, AdapterMetadata, serve_agent

class MyAdapter(BaseA2AAdapter):
    # REQUIRED: the only method you must implement
    async def invoke(self, user_input: str, context_id: str | None = None, **kwargs) -> str:
        # kwargs['context'] provides the full A2A RequestContext
        return await call_my_framework(user_input)

    # OPTIONAL: implement for streaming support (auto-detected)
    async def stream(self, user_input: str, context_id: str | None = None, **kwargs):
        async for chunk in my_framework_stream(user_input):
            yield str(chunk)

    # OPTIONAL: cancellation support
    async def cancel(self, context_id: str | None = None, **kwargs) -> None:
        pass

    # OPTIONAL: resource cleanup (called by async with)
    async def close(self) -> None:
        pass

    # OPTIONAL: metadata for auto AgentCard generation
    def get_metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name="My Agent", description="Does something useful",
            version="1.0.0", streaming=True,
            skills=[{"id": "main", "name": "Main Skill", "description": "..."}],
        )

serve_agent(MyAdapter(), port=9000)
```

## Reference

See `api-reference.md` in this directory for full adapter parameters, server function signatures, multimodal response patterns, and architecture details.
