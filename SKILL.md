# a2a-adapter SDK — Agent Skill

> This document teaches coding agents how to use the `a2a-adapter` SDK.
> Feed this file to any AI coding assistant so it knows how to convert
> AI agents into A2A Protocol servers using this library.

## Install

```bash
pip install a2a-adapter                # Core (n8n, callable)
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

`serve_agent()` starts a uvicorn server with auto-generated AgentCard at `/.well-known/agent-card.json`. All A2A protocol handling (JSON-RPC, task management, SSE streaming, push notifications) is done by the A2A SDK automatically.

## Public API

All imports come from `a2a_adapter`:

```python
from a2a_adapter import (
    # Core
    BaseA2AAdapter,    # Abstract base — subclass for custom adapters
    AdapterMetadata,   # Dataclass for AgentCard auto-generation
    # Server
    serve_agent,       # One-line server: adapter → uvicorn
    to_a2a,            # Adapter → ASGI app (for production deployment)
    build_agent_card,  # Adapter → AgentCard
    # Loader
    load_adapter,      # Factory: config dict → adapter instance
    register_adapter,  # Decorator: register third-party adapters
    # Built-in adapters (lazy-loaded)
    N8nAdapter,
    LangChainAdapter,
    LangGraphAdapter,
    CrewAIAdapter,
    OpenClawAdapter,
    CallableAdapter,
)
```

## Built-in Adapters

### N8nAdapter — n8n webhook

```python
from a2a_adapter import N8nAdapter, serve_agent

adapter = N8nAdapter(
    webhook_url="http://localhost:5678/webhook/agent",  # Required
    timeout=30,               # HTTP timeout (seconds)
    name="My Agent",          # For AgentCard
    description="Does X",     # For AgentCard
    headers={"Authorization": "Bearer xxx"},  # Extra headers
    message_field="message",  # Payload field name (default: "message")
    input_mapper=None,        # Optional: (raw_input, context_id) -> dict
    parse_json_input=True,    # Auto-parse JSON strings
    default_inputs=None,      # Merge into every request
    max_retries=3,            # Retry count
    retry_delay=1.0,          # Seconds between retries
)
serve_agent(adapter, port=9000)
```

### LangChainAdapter — LangChain Runnable (auto-streaming)

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from a2a_adapter import LangChainAdapter, serve_agent

chain = ChatPromptTemplate.from_template("Answer: {input}") | ChatOpenAI(model="gpt-4o-mini")
adapter = LangChainAdapter(
    runnable=chain,           # Required: any LangChain Runnable
    input_key="input",        # Key to wrap user text (default: "input")
    output_key=None,          # Extract specific output key
    name="LangChain Agent",
    description="Answers questions",
)
serve_agent(adapter, port=9001)  # Streaming auto-detected via hasattr(runnable, "astream")
```

### LangGraphAdapter — LangGraph CompiledGraph (auto-streaming)

```python
from a2a_adapter import LangGraphAdapter, serve_agent

graph = builder.compile()  # Your LangGraph compiled graph
adapter = LangGraphAdapter(
    graph=graph,              # Required: CompiledGraph
    input_key="messages",     # Input key (default: "messages")
    output_key=None,          # Extract specific output key
    name="LangGraph Agent",
    description="Workflow agent",
)
serve_agent(adapter, port=9002)  # Streaming auto-detected
```

When `input_key="messages"`, user text is auto-wrapped in `HumanMessage` if `langchain_core` is available.

### CrewAIAdapter — CrewAI Crew

```python
from a2a_adapter import CrewAIAdapter, serve_agent

adapter = CrewAIAdapter(
    crew=your_crew,           # Required: CrewAI Crew instance
    timeout=600,              # Execution timeout (seconds)
    input_key="inputs",       # Key for crew inputs (default: "inputs")
    name="CrewAI Agent",
    description="Research crew",
)
serve_agent(adapter, port=9003)  # No streaming support
```

### OpenClawAdapter — OpenClaw CLI

```python
from a2a_adapter import OpenClawAdapter, serve_agent

adapter = OpenClawAdapter(
    thinking="low",           # Thinking level: "none", "low", "medium", "high"
    agent_id="main",          # Agent ID
    session_id=None,          # Auto-generated if None
    timeout=600,              # Subprocess timeout (seconds)
    openclaw_path="openclaw", # Binary path
    working_directory=None,   # CWD for subprocess
    env_vars=None,            # Extra environment variables
)
serve_agent(adapter, port=9004)  # Supports cancel() via process kill
```

### CallableAdapter — Any async/sync function

```python
from a2a_adapter import CallableAdapter, serve_agent

async def my_agent(inputs: dict) -> str:
    return f"Echo: {inputs.get('message', '')}"

adapter = CallableAdapter(
    func=my_agent,            # Required: callable(dict) -> str
    streaming=False,          # Set True if func is async generator
    name="Echo Agent",
    description="Echoes input",
)
serve_agent(adapter, port=9005)
```

For streaming, `func` should be an async generator yielding str chunks.

## Custom Adapter (subclass BaseA2AAdapter)

For any framework not covered by built-in adapters:

```python
from a2a_adapter import BaseA2AAdapter, AdapterMetadata, serve_agent

class MyAdapter(BaseA2AAdapter):
    # REQUIRED: the only method you must implement
    async def invoke(self, user_input: str, context_id: str | None = None, **kwargs) -> str:
        # kwargs['context'] provides the full A2A RequestContext if needed
        result = await call_my_framework(user_input)
        return str(result)

    # OPTIONAL: streaming support
    async def stream(self, user_input: str, context_id: str | None = None, **kwargs):
        async for chunk in my_framework_stream(user_input):
            yield str(chunk)

    # OPTIONAL: cancellation support
    async def cancel(self) -> None:
        self._process.kill()

    # OPTIONAL: resource cleanup
    async def close(self) -> None:
        await self._client.aclose()

    # OPTIONAL: metadata for auto AgentCard generation
    def get_metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name="My Agent",
            description="Does something useful",
            version="1.0.0",
            skills=[{"id": "main", "name": "Main Skill", "description": "..."}],
            streaming=True,  # Set True if stream() is implemented
        )

serve_agent(MyAdapter(), port=9000)
```

**Key rules:**
- `invoke()` is the ONLY required method. Signature: `(str, str|None) -> str`
- `stream()` is auto-detected. If overridden, `supports_streaming()` returns True
- Exceptions in `invoke()`/`stream()` are caught and converted to failed tasks
- `context_id` enables multi-turn conversations (same ID = same conversation)
- Adapter supports `async with` for resource management

## Server Functions

### `serve_agent()` — development server

```python
serve_agent(
    adapter: BaseA2AAdapter,
    agent_card: AgentCard | None = None,   # Override auto-generated card
    host: str = "0.0.0.0",
    port: int = 9000,
    log_level: str = "info",
    **kwargs,                              # Passed to uvicorn.run()
)
```

### `to_a2a()` — production ASGI app

```python
app = to_a2a(
    adapter: BaseA2AAdapter,
    agent_card: AgentCard | None = None,
    task_store: TaskStore | None = None,   # Default: InMemoryTaskStore
    **card_overrides,                      # name=, description=, url=, version=, streaming=
)
# Deploy: gunicorn app:app -k uvicorn.workers.UvicornWorker
```

### `build_agent_card()` — generate AgentCard

```python
card = build_agent_card(
    adapter: BaseA2AAdapter,
    **overrides,     # name=, description=, url=, version=, streaming=
)
```

Default url is `http://localhost:9000`. Override with `url="http://prod:8080"`.

## Config-Driven Loading

```python
from a2a_adapter import load_adapter, serve_agent

adapter = load_adapter({
    "adapter": "n8n",
    "webhook_url": "http://localhost:5678/webhook/agent",
    "timeout": 60,
})
serve_agent(adapter)
```

Valid adapter values: `"n8n"`, `"langchain"`, `"langgraph"`, `"crewai"`, `"openclaw"`, `"callable"`, or any registered name.

## Third-Party Adapter Registration

```python
from a2a_adapter import register_adapter, BaseA2AAdapter, load_adapter

@register_adapter("my_framework")
class MyFrameworkAdapter(BaseA2AAdapter):
    async def invoke(self, user_input, context_id=None, **kwargs):
        return "result"

# Now loadable via config:
adapter = load_adapter({"adapter": "my_framework"})
```

Registered adapters take priority over built-ins with the same name.

## Input Handling (all built-in adapters)

All adapters use a 3-priority input pipeline:
1. **`input_mapper`** (highest priority): `Callable[[str, str|None], dict]`
2. **JSON parse**: auto-parse if input looks like JSON
3. **`input_key`** (fallback): wraps text as `{input_key: text}`

## Architecture (for understanding, not for user code)

```
Client → A2AStarletteApplication → DefaultRequestHandler → AdapterAgentExecutor → YourAdapter.invoke()
```

The `AdapterAgentExecutor` bridge handles:
- Extracting user text from `RequestContext`
- Routing to `invoke()` or `stream()` based on adapter capabilities
- Converting text responses to A2A events via `TaskUpdater`
- Error handling (exception → failed task)
- Cancellation (`adapter.cancel()` → canceled task)

Users never interact with the bridge layer directly.

## API Quick Reference

### BaseA2AAdapter

| Method | Required | Signature | Description |
|---|---|---|---|
| `invoke` | **Yes** | `async (str, str\|None) -> str` | Execute agent, return text |
| `stream` | No | `async (str, str\|None) -> AsyncIterator[str]` | Yield text chunks |
| `cancel` | No | `async () -> None` | Cancel current execution |
| `close` | No | `async () -> None` | Release resources |
| `get_metadata` | No | `() -> AdapterMetadata` | Metadata for AgentCard |
| `supports_streaming` | No | `() -> bool` | Auto-detects from stream() override |

### AdapterMetadata

```python
AdapterMetadata(
    name="",                          # Agent name (defaults to class name in card)
    description="",                   # What the agent does
    version="1.0.0",                  # Semantic version
    skills=[],                        # List of skill dicts: [{"id", "name", "description", "tags"}]
    input_modes=["text"],             # Supported input MIME types
    output_modes=["text"],            # Supported output MIME types
    streaming=False,                  # Whether adapter supports streaming
)
```

## Decision Guide

| Scenario | Use |
|---|---|
| Wrap a function | `CallableAdapter(func=fn)` |
| n8n workflow | `N8nAdapter(webhook_url=...)` |
| LangChain chain | `LangChainAdapter(runnable=chain)` |
| LangGraph workflow | `LangGraphAdapter(graph=graph)` |
| CrewAI crew | `CrewAIAdapter(crew=crew)` |
| OpenClaw agent | `OpenClawAdapter(...)` |
| Any other framework | Subclass `BaseA2AAdapter`, implement `invoke()` |
| Need streaming | Implement `stream()` or use LangChain/LangGraph (auto) |
| Production deploy | `to_a2a(adapter)` → ASGI server |
| Config-driven | `load_adapter({"adapter": "n8n", ...})` |
