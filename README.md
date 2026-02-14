# A2A Adapter

[![PyPI version](https://badge.fury.io/py/a2a-adapter.svg)](https://badge.fury.io/py/a2a-adapter)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**Convert any AI agent into an A2A Protocol server in 3 lines.**

A Python SDK that makes any agent framework (n8n, LangGraph, CrewAI, LangChain, [OpenClaw](https://openclaw.ai/), or a plain function) compatible with the [A2A (Agent-to-Agent) Protocol](https://github.com/a2aproject/A2A).

```python
from a2a_adapter import N8nAdapter, serve_agent

adapter = N8nAdapter(webhook_url="http://localhost:5678/webhook/agent")
serve_agent(adapter, port=9000)
```

That's it. Your agent is now A2A-compatible with auto-generated AgentCard, task management, and streaming support — all handled by the A2A SDK.

## Features

- **3-line setup** — `import`, `create`, `serve`
- **6 built-in adapters** — n8n, LangChain, LangGraph, CrewAI, OpenClaw, Callable
- **Streaming** — auto-detected for LangChain and LangGraph
- **Auto AgentCard** — generated from adapter metadata, served at `/.well-known/agent.json`
- **SDK-First** — delegates task management, SSE, push notifications to the A2A SDK
- **Extensible** — `register_adapter()` for third-party frameworks
- **Minimal surface** — implement `invoke()`, get a full A2A server

## Installation

```bash
pip install a2a-adapter                # Core (includes n8n, callable)
pip install a2a-adapter[crewai]        # + CrewAI
pip install a2a-adapter[langchain]     # + LangChain
pip install a2a-adapter[langgraph]     # + LangGraph
pip install a2a-adapter[all]           # Everything
```

## Quick Start

### n8n Workflow

```python
from a2a_adapter import N8nAdapter, serve_agent

adapter = N8nAdapter(webhook_url="http://localhost:5678/webhook/agent")
serve_agent(adapter, port=9000)
```

### LangChain (with streaming)

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from a2a_adapter import LangChainAdapter, serve_agent

chain = ChatPromptTemplate.from_template("Answer: {input}") | ChatOpenAI(model="gpt-4o-mini")
adapter = LangChainAdapter(runnable=chain, input_key="input")
serve_agent(adapter, port=8002)  # Streaming auto-detected!
```

### LangGraph (with streaming)

```python
from a2a_adapter import LangGraphAdapter, serve_agent

graph = builder.compile()  # Your LangGraph workflow
adapter = LangGraphAdapter(graph=graph)
serve_agent(adapter, port=9002)
```

### CrewAI

```python
from a2a_adapter import CrewAIAdapter, serve_agent

adapter = CrewAIAdapter(crew=your_crew, timeout=600)
serve_agent(adapter, port=8001)
```

### OpenClaw

```python
from a2a_adapter import OpenClawAdapter, serve_agent

adapter = OpenClawAdapter(thinking="low", agent_id="main")
serve_agent(adapter, port=9008)
```

### Custom Function

```python
from a2a_adapter import CallableAdapter, serve_agent

async def my_agent(inputs):
    return f"Echo: {inputs['message']}"

adapter = CallableAdapter(func=my_agent, name="Echo Agent")
serve_agent(adapter, port=9005)
```

### Custom Adapter Class

For full control, subclass `BaseA2AAdapter`:

```python
from a2a_adapter import BaseA2AAdapter, serve_agent

class MyAdapter(BaseA2AAdapter):
    async def invoke(self, user_input: str, context_id: str | None = None, **kwargs) -> str:
        return f"You said: {user_input}"

serve_agent(MyAdapter(), port=8003)
```

## Architecture

```
A2A Caller (other agents)
    │  A2A Protocol (HTTP + JSON-RPC 2.0 / SSE)
    ▼
┌──────────────────────────────────────────────┐
│  A2A SDK (DefaultRequestHandler, TaskStore)  │  ← handles protocol
├──────────────────────────────────────────────┤
│  AdapterAgentExecutor (bridge layer)         │  ← adapts interface
├──────────────────────────────────────────────┤
│  Your Adapter (invoke / stream)              │  ← YOUR CODE HERE
├──────────────────────────────────────────────┤
│  Framework (n8n / LangChain / CrewAI / ...)  │
└──────────────────────────────────────────────┘
```

**Design principle:** Adapters answer ONE question — "given text, return text." Everything else (task management, SSE streaming, push notifications, AgentCard serving) is handled by the A2A SDK.

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design documentation, and [DESIGN_V0.2.md](DESIGN_V0.2.md) for the v0.2 design rationale.

## API Reference

### Core

| Function | Description |
|---|---|
| `serve_agent(adapter, port=9000)` | One-line server startup |
| `to_a2a(adapter)` | Convert adapter to ASGI app |
| `build_agent_card(adapter)` | Auto-generate AgentCard from metadata |
| `load_adapter(config)` | Factory: create adapter from config dict |
| `register_adapter(name)` | Decorator: register third-party adapters |

### BaseA2AAdapter (implement this)

| Method | Required | Description |
|---|---|---|
| `invoke(user_input, context_id, **kwargs)` | **Yes** | Execute agent, return text |
| `stream(user_input, context_id, **kwargs)` | No | Yield text chunks (streaming) |
| `cancel()` | No | Cancel current execution |
| `close()` | No | Release resources |
| `get_metadata()` | No | Return `AdapterMetadata` for AgentCard |

### Adapter Support

| Framework | Adapter | Streaming | Auto-detected |
|---|---|---|---|
| **n8n** | `N8nAdapter` | - | - |
| **LangChain** | `LangChainAdapter` | Yes | `hasattr(runnable, "astream")` |
| **LangGraph** | `LangGraphAdapter` | Yes | `hasattr(graph, "astream")` |
| **CrewAI** | `CrewAIAdapter` | - | - |
| **OpenClaw** | `OpenClawAdapter` | - | - |
| **Callable** | `CallableAdapter` | Optional | `streaming=True` param |

### Input Handling

All adapters support a 3-priority input pipeline:

1. **`input_mapper`** (highest) — custom function `(raw_input, context_id) -> dict`
2. **`parse_json_input`** — auto-parse JSON strings to dict
3. **`input_key`** (fallback) — map text to `{input_key: text}`

### Config-driven Loading

```python
from a2a_adapter import load_adapter

adapter = load_adapter({
    "adapter": "n8n",
    "webhook_url": "http://localhost:5678/webhook/agent",
    "timeout": 60,
})
```

### Third-party Adapters

```python
from a2a_adapter import register_adapter, BaseA2AAdapter

@register_adapter("my_framework")
class MyFrameworkAdapter(BaseA2AAdapter):
    async def invoke(self, user_input, context_id=None, **kwargs):
        return "Hello from my framework!"

# Now loadable via config:
adapter = load_adapter({"adapter": "my_framework"})
```

## Advanced: ASGI Deployment

For production deployments with Gunicorn/Hypercorn:

```python
from a2a_adapter import N8nAdapter, to_a2a

adapter = N8nAdapter(webhook_url="http://localhost:5678/webhook/agent")
app = to_a2a(adapter)  # Returns Starlette ASGI app

# Deploy with: gunicorn app:app -k uvicorn.workers.UvicornWorker
```

## Migration from v0.1

v0.2 is backwards compatible — v0.1 code still works but emits deprecation warnings.

| v0.1 (deprecated) | v0.2 (recommended) |
|---|---|
| `BaseAgentAdapter` | `BaseA2AAdapter` |
| `load_a2a_agent(config)` | `load_adapter(config)` |
| `build_agent_app(card, adapter)` | `to_a2a(adapter)` |
| `serve_agent(card, adapter)` | `serve_agent(adapter)` |
| `N8nAgentAdapter` | `N8nAdapter` |
| 3-method override (`to_framework` + `call_framework` + `from_framework`) | Single `invoke()` method |

## Examples

The [`examples/`](examples/) directory contains working examples for each adapter:

```bash
python examples/n8n_agent.py          # n8n
python examples/langchain_agent.py    # LangChain (streaming)
python examples/langgraph_server.py   # LangGraph (streaming)
python examples/crewai_agent.py       # CrewAI
python examples/openclaw_agent.py     # OpenClaw
python examples/custom_adapter.py     # Custom BaseA2AAdapter
python examples/single_agent_client.py  # Test any running agent
```

See [examples/README.md](examples/README.md) for details.

## Testing

```bash
pip install a2a-adapter[dev]
pytest                    # All tests
pytest tests/unit/        # Unit tests only
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Quick start:**
1. Fork & clone
2. `pip install -e ".[dev]"`
3. Make changes + add tests
4. `pytest` to verify
5. Submit a PR

## License

Apache-2.0 — see [LICENSE](LICENSE).

Built with care by [HYBRO AI](https://hybro.ai). Powered by the [A2A Protocol](https://github.com/a2aproject/A2A).
