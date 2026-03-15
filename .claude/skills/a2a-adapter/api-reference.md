# a2a-adapter API Reference

**Version**: 0.2.0 | **Python**: >=3.11 | **License**: Apache-2.0

For installation and quickstart, see SKILL.md.

## Public API

All imports come from `a2a_adapter`:

```python
from a2a_adapter import (
    # Core
    BaseA2AAdapter,    # Abstract base -- subclass for custom adapters
    AdapterMetadata,   # Dataclass for AgentCard auto-generation
    # Server
    serve_agent,       # One-line server: adapter -> uvicorn
    to_a2a,            # Adapter -> ASGI app (for production deployment)
    build_agent_card,  # Adapter -> AgentCard
    # Loader
    load_adapter,      # Factory: config dict -> adapter instance
    register_adapter,  # Decorator: register third-party adapters
    # Built-in adapters (lazy-loaded)
    N8nAdapter,
    LangChainAdapter,
    LangGraphAdapter,
    CrewAIAdapter,
    OpenClawAdapter,
    CallableAdapter,
    OllamaAdapter,
    OllamaClient,
)
```

## Built-in Adapters

### N8nAdapter -- n8n webhook

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

Supports multimodal responses when the n8n workflow returns structured content.

### LangChainAdapter -- LangChain Runnable (auto-streaming)

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

### LangGraphAdapter -- LangGraph CompiledGraph (auto-streaming)

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

### CrewAIAdapter -- CrewAI Crew

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

### OpenClawAdapter -- OpenClaw CLI

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

### OllamaAdapter -- Local Ollama LLM (streaming)

```python
from a2a_adapter import OllamaAdapter, serve_agent
from a2a_adapter.integrations.ollama import OllamaClient

client = OllamaClient(
    model="llama3.2",          # Required: Ollama model name
    base_url="http://localhost:11434",  # Ollama server URL
    system_prompt=None,           # Optional system prompt
    temperature=None,             # Sampling temperature
    timeout=120,                  # HTTP timeout (seconds)
    keep_alive=None,              # Model keep-alive duration (e.g. "5m")
)
adapter = OllamaAdapter(
    client=client,                # Required: OllamaClient instance
    name="Local LLM",
    description="Ollama-powered agent",
)
serve_agent(adapter, port=10010)  # Streaming always supported
```

Convenience shorthand (creates `OllamaClient` internally):

```python
adapter = OllamaAdapter(model="llama3.2", name="Local LLM")
serve_agent(adapter, port=10010)
```

### CallableAdapter -- Any async/sync function

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

## Server Functions

### `serve_agent()` -- development server

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

### `to_a2a()` -- production ASGI app

```python
app = to_a2a(
    adapter: BaseA2AAdapter,
    agent_card: AgentCard | None = None,
    task_store: TaskStore | None = None,   # Default: InMemoryTaskStore
    **card_overrides,                      # name=, description=, url=, version=, streaming=,
                                           # provider=, documentation_url=, icon_url=
)
# Deploy with any ASGI server:
# gunicorn app:app -k uvicorn.workers.UvicornWorker
# hypercorn app:app
# daphne app:app
```

Auto-enabled capabilities on the generated AgentCard: `streaming` (auto-detected), `push_notifications=True`, `state_transition_history=True`.

### `build_agent_card()` -- generate AgentCard

```python
card = build_agent_card(
    adapter: BaseA2AAdapter,
    **overrides,     # name=, description=, url=, version=, streaming=,
                     # provider=, documentation_url=, icon_url=
)
```

Default url is `http://localhost:9000`. Override with `url="http://prod:8080"`.

## BaseA2AAdapter Method Reference

| Method | Required | Signature | Description |
|---|---|---|---|
| `invoke` | **Yes** | `async (str, str\|None, **kwargs) -> str \| list[Part]` | Execute agent, return text or multimodal |
| `stream` | No | `async (str, str\|None, **kwargs) -> AsyncIterator[str \| Part]` | Yield text/multimodal chunks |
| `cancel` | No | `async (str\|None, **kwargs) -> None` | Cancel current execution |
| `close` | No | `async () -> None` | Release resources |
| `get_metadata` | No | `() -> AdapterMetadata` | Metadata for AgentCard |
| `supports_streaming` | No | `() -> bool` | Auto-detects from stream() override |

All methods receiving `**kwargs` get `context=RequestContext` from the bridge layer.

**Behavioral notes:**
- `invoke()` is the ONLY required method; all others are optional
- `stream()` is auto-detected: if overridden, `supports_streaming()` returns `True`
- Exceptions in `invoke()`/`stream()` are caught and converted to failed tasks automatically
- `context_id` enables multi-turn conversations (same ID = same conversation)
- Adapter supports `async with` for resource management (`close()` called on exit)

## Custom Adapter Example

Full example showing all optional methods working together:

```python
from a2a_adapter import BaseA2AAdapter, AdapterMetadata, serve_agent

class MyAdapter(BaseA2AAdapter):
    # REQUIRED: the only method you must implement
    async def invoke(self, user_input: str, context_id: str | None = None, **kwargs) -> str:
        # kwargs['context'] provides the full A2A RequestContext (see below)
        result = await call_my_framework(user_input)
        return str(result)

    # OPTIONAL: streaming support
    async def stream(self, user_input: str, context_id: str | None = None, **kwargs):
        async for chunk in my_framework_stream(user_input):
            yield str(chunk)

    # OPTIONAL: cancellation support
    async def cancel(self, context_id: str | None = None, **kwargs) -> None:
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

## AdapterMetadata

```python
AdapterMetadata(
    name="",                          # Agent name (defaults to class name in card)
    description="",                   # What the agent does
    version="1.0.0",                  # Semantic version
    skills=[],                        # List of skill dicts (see below)
    input_modes=["text"],             # Supported input MIME types
    output_modes=["text"],            # Supported output MIME types
    streaming=False,                  # Whether adapter supports streaming
    provider=None,                    # Dict with "organization" and "url" keys
    documentation_url=None,           # URL to agent documentation
    icon_url=None,                    # URL to agent icon
)
```

**Skill dict format:**
```python
{
    "id": "skill-0",            # Unique skill ID
    "name": "Main Skill",       # Human-readable name
    "description": "...",       # What this skill does
    "tags": ["tag1", "tag2"],   # Optional tags
    "examples": ["..."],        # Optional example queries
    "input_modes": ["text"],    # Optional per-skill input modes
    "output_modes": ["text"],   # Optional per-skill output modes
}
```

## Multimodal Responses

Adapters can return multimodal content using A2A `Part` types:

```python
from a2a.types import Part, TextPart, FilePart, FileWithUri

class MultimodalAdapter(BaseA2AAdapter):
    # Return list[Part] instead of str for multimodal output
    async def invoke(self, user_input, context_id=None, **kwargs) -> list[Part]:
        return [
            Part(root=TextPart(text="Here's your report")),
            Part(root=FilePart(file=FileWithUri(
                uri="http://example.com/report.pdf",
                name="report.pdf",
                mimeType="application/pdf"
            )))
        ]

    # stream() can also yield Part objects alongside str chunks
    async def stream(self, user_input, context_id=None, **kwargs):
        yield "Generating chart..."
        yield Part(root=FilePart(file=FileWithUri(
            uri="http://example.com/chart.png",
            name="chart.png",
            mimeType="image/png"
        )))
```

**Return types:**
- `invoke()` -> `str` (text only) or `list[Part]` (multimodal)
- `stream()` yields `str` (text chunks) or `Part` (multimodal chunks)

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

Valid adapter values: `"n8n"`, `"langchain"`, `"langgraph"`, `"crewai"`, `"openclaw"`, `"ollama"`, `"callable"`, or any registered name.

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

## Input Handling Pipeline

All built-in adapters use a 3-priority input pipeline:

1. **`input_mapper`** (highest priority): `Callable[[str, str|None], dict]` -- custom function to transform raw input
2. **JSON parse**: auto-parse if input looks like JSON
3. **`input_key`** (fallback): wraps text as `{input_key: text}`

## Accessing RequestContext

The bridge layer passes `context=RequestContext` via `**kwargs`. Use it when you need more than just the text input:

```python
async def invoke(self, user_input: str, context_id: str | None = None, **kwargs) -> str:
    context = kwargs.get('context')  # A2A SDK RequestContext
    if context:
        # Access full message parts (text, files, data, etc.)
        for part in context.message.parts:
            ...
        # Access task ID
        task_id = context.task_id
    return "response"
```

## Resource Management

Adapters support `async with` for automatic cleanup:

```python
async with MyAdapter() as adapter:
    result = await adapter.invoke("hello")
# adapter.close() called automatically
```

## Adapter Capabilities Summary

| Adapter | Streaming | Cancel | Multimodal | Extra Deps |
|---|---|---|---|---|
| `N8nAdapter` | No | No | Yes | None |
| `LangChainAdapter` | Auto-detected | No | No | `langchain`, `langchain-core` |
| `LangGraphAdapter` | Auto-detected | No | No | `langgraph` |
| `CrewAIAdapter` | No | No | No | `crewai` |
| `OpenClawAdapter` | No | Yes (kill) | No | None |
| `OllamaAdapter` | Always | No | No | None |
| `CallableAdapter` | Optional | No | No | None |

## Architecture (Request Flow)

```
Client -> A2AStarletteApplication -> DefaultRequestHandler -> AdapterAgentExecutor -> YourAdapter.invoke()
```

The `AdapterAgentExecutor` bridge handles:
- Extracting user text from `RequestContext` via `context.get_user_input()`
- Routing to `invoke()` or `stream()` based on `adapter.supports_streaming()`
- Converting text/multimodal responses to A2A events via `TaskUpdater`
- Error handling (exception -> Task with state=failed, error message preserved)
- Cancellation (`adapter.cancel()` -> Task with state=canceled)

Users never interact with the bridge layer directly.

### Non-streaming (`message/send`)

```
HTTP POST / (JSON-RPC) -> DefaultRequestHandler.on_message_send()
  -> asyncio.create_task(executor.execute(ctx, queue))
    -> TaskUpdater.start_work()
    -> adapter.invoke(user_input, context_id, context=ctx)
    -> TaskUpdater.add_artifact(result)
    -> TaskUpdater.complete()
  -> ResultAggregator collects events -> returns Task
```

### Streaming (`message/stream`)

```
HTTP POST / (JSON-RPC) -> DefaultRequestHandler.on_message_send_stream()
  -> adapter.stream(user_input, context_id, context=ctx)
    -> each chunk -> TaskUpdater.add_artifact(chunk, append=True)
  -> events SSE-streamed to client
  -> TaskUpdater.complete()
```

### Cancellation (`tasks/cancel`)

```
HTTP POST / (JSON-RPC) -> DefaultRequestHandler.on_cancel_task()
  -> cancels asyncio.Task
    -> adapter.cancel(context=ctx)
    -> TaskUpdater.cancel()
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
| Local Ollama model | `OllamaAdapter(model="llama3.2")` |
| Any other framework | Subclass `BaseA2AAdapter`, implement `invoke()` |
| Need streaming | Implement `stream()` or use LangChain/LangGraph/Ollama (auto) |
| Need multimodal output | Return `list[Part]` from `invoke()` |
| Production deploy | `to_a2a(adapter)` -> ASGI server |
| Config-driven | `load_adapter({"adapter": "n8n", ...})` |
