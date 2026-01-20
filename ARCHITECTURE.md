# A2A Adapter Python SDK — Architecture

## Overview

The A2A Adapter SDK provides a clean abstraction layer for integrating various agent frameworks with the A2A (Agent-to-Agent) Protocol. This document describes the architecture, design decisions, and implementation details.

## Design Philosophy

### Single-Agent per Process

Each server instance hosts **exactly one agent**. Multi-agent orchestration is handled by:
- External orchestrators (LangGraph, custom workflows)
- Multiple A2A server instances communicating via the A2A protocol
- Higher-level orchestration frameworks

This design keeps the adapter SDK focused and simple.

### Protocol Implementation

We leverage the **official A2A SDK** for all protocol-level concerns:
- Message serialization/deserialization
- HTTP/SSE transport
- JSON-RPC 2.0 handling
- Type definitions (Message, Task, AgentCard, etc.)

Our SDK focuses solely on **framework adaptation** — translating between A2A messages and framework-specific APIs.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    A2A Caller (other A2A Agents)            │
└────────────────────────┬────────────────────────────────────┘
                         │  A2A Protocol (HTTP + JSON-RPC 2.0 / SSE)
                         │  (via official A2A Python SDK)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   A2A Adapter SDK (this SDK)                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │          A2A Server (official A2A SDK)               │   │
│  │   • Receives A2A messages                            │   │
│  │   • Uses official A2A types                          │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         │                                   │
│  ┌──────────────────────▼───────────────────────────────┐   │
│  │        Core Adapter Interface (abstract layer)       │   │
│  │   • BaseAgentAdapter (abstract base class)           │   │
│  │   • to_framework() / from_framework()               │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         │                                   │
│  ┌──────────────────────▼───────────────────────────────┐   │
│  │            Framework Adapters (concrete)             │   │
│  │   ┌────────────┐  ┌──────────────┐  ┌─────────────┐ │   │
│  │   │ N8nAdapter │  │LangGraphAdapt│  │ CrewAIAdapt │ │   │
│  │   └────────────┘  └──────────────┘  └─────────────┘ │   │
│  │   ┌────────────────┐  ┌────────────────┐            │   │
│  │   │LangChainAdapter│  │CallableAdapter │            │   │
│  │   └────────────────┘  └────────────────┘            │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Underlying Agent Implementations               │
│  (n8n workflows / LangGraph graphs / CrewAI / LangChain)    │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. BaseAgentAdapter (`adapter.py`)

Abstract base class defining the adapter interface:

```python
class BaseAgentAdapter(ABC):
    async def handle(self, params: MessageSendParams) -> Message | Task
    async def handle_stream(self, params: MessageSendParams) -> AsyncIterator
    
    @abstractmethod
    async def to_framework(self, params: MessageSendParams) -> Any
    
    @abstractmethod
    async def call_framework(self, framework_input: Any, params: MessageSendParams) -> Any
    
    @abstractmethod
    async def from_framework(self, framework_output: Any, params: MessageSendParams) -> Message | Task
```

**Three-Step Translation Process:**

1. **`to_framework()`**: Convert A2A `MessageSendParams` to framework-specific input
2. **`call_framework()`**: Execute the framework (n8n webhook, CrewAI crew, etc.)
3. **`from_framework()`**: Convert framework output back to A2A `Message` or `Task`

**Streaming Support:**

- Optional `handle_stream()` for frameworks supporting streaming
- LangChain adapter uses `runnable.astream()` for streaming responses
- N8n and CrewAI don't support streaming natively

### 2. Adapter Loader (`loader.py`)

Factory function for creating adapters from configuration:

```python
async def load_a2a_agent(config: Dict[str, Any]) -> BaseAgentAdapter
```

**Supported Adapter Types:**

| Adapter | Required Config | Optional Config |
|---------|----------------|-----------------|
| `n8n` | `webhook_url` | `timeout`, `headers`, `async_mode`, `task_store` |
| `langgraph` | `graph` | `input_key`, `output_key`, `async_mode`, `task_store` |
| `crewai` | `crew` | `inputs_key`, `async_mode`, `task_store` |
| `langchain` | `runnable` | `input_key`, `output_key` |
| `callable` | `callable` | `supports_streaming` |

**Lazy Imports:**

Framework-specific packages are only imported when the corresponding adapter is instantiated, avoiding unnecessary dependencies.

### 3. Server Helpers (`client.py`)

Utilities for creating and serving A2A agents:

```python
def build_agent_app(agent_card: AgentCard, adapter: BaseAgentAdapter) -> ASGIApp
def serve_agent(agent_card: AgentCard, adapter: BaseAgentAdapter, host, port)
```

**`AdapterRequestHandler`:**

Bridges our `BaseAgentAdapter` with the official A2A SDK's handler interface:
- Wraps `adapter.handle()` for non-streaming requests
- Wraps `adapter.handle_stream()` for streaming requests
- Automatically detects streaming support via `adapter.supports_streaming()`

### 4. Framework Adapters (`integrations/`)

#### N8n Adapter (`n8n.py`)

**Purpose:** Expose n8n workflows as A2A agents

**Execution Modes:**

1. **Synchronous Mode** (default):
   - Blocks until n8n workflow completes
   - Returns A2A `Message` with the response
   - Best for quick workflows (< 30 seconds)

2. **Async Task Mode** (`async_mode=True`):
   - Returns A2A `Task` immediately with `state=working`
   - Executes workflow in background
   - Clients poll `get_task()` for status updates
   - Best for long-running workflows

**Flow (Sync Mode):**
1. Extract user message text from A2A message
2. POST to n8n webhook URL with JSON payload
3. Parse n8n response and wrap in A2A Message

**Flow (Async Task Mode):**
1. Create Task with `state=working`, save to TaskStore
2. Start background coroutine for workflow execution
3. Return Task immediately to client
4. Background: POST to n8n webhook, wait for response
5. Background: Update Task to `state=completed` or `state=failed`
6. Client polls `get_task(task_id)` to check status

**Key Features:**
- Configurable timeout and custom headers
- Async HTTP client with connection pooling
- Supports common n8n response formats (`output`, `result`, `message`)
- **Async Task Mode** for long-running workflows
- Integrates with A2A SDK's `TaskStore` (InMemoryTaskStore or DatabaseTaskStore)
- Task cancellation support via `cancel_task()`

#### LangGraph Adapter (`langgraph.py`)

**Purpose:** Expose LangGraph compiled workflows as A2A agents

**Execution Modes:**

1. **Synchronous Mode** (default):
   - Blocks until workflow completes
   - Returns A2A `Message` with final state output
   - Best for quick workflows

2. **Streaming Mode**:
   - Uses `graph.astream()` to yield intermediate states
   - Streams tokens/content as workflow progresses
   - Best for real-time feedback

3. **Async Task Mode** (`async_mode=True`):
   - Returns A2A `Task` immediately with `state=working`
   - Executes workflow in background
   - Clients poll `get_task()` for status updates
   - Best for long-running workflows

**Flow (Sync/Streaming Mode):**
1. Extract user message text from A2A message
2. Build input state (messages format or simple input)
3. Call `graph.ainvoke()` or `graph.astream()`
4. Extract output from final state (configurable output_key)
5. Return A2A Message

**Key Features:**
- Works with any LangGraph `CompiledGraph`
- Full streaming support via `astream()`
- Flexible input handling (messages or simple keys)
- Intelligent output extraction from state
- Async Task Mode for long-running workflows
- Task cancellation support

#### CrewAI Adapter (`crewai.py`)

**Purpose:** Expose CrewAI crews as A2A agents

**Execution Modes:**

1. **Synchronous Mode** (default):
   - Blocks until crew completes
   - Returns A2A `Message` with crew output
   - Best for quick crews

2. **Async Task Mode** (`async_mode=True`):
   - Returns A2A `Task` immediately
   - Executes crew in background
   - Clients poll `get_task()` for status
   - Best for long-running crews (10+ minutes)

**Flow:**
1. Extract user message and format as crew inputs
2. Call `crew.kickoff_async(inputs=...)` or fallback to sync
3. Extract result from `CrewOutput` or raw response

**Key Features:**
- Async execution via `kickoff_async`
- Fallback to sync execution for older CrewAI versions
- Configurable input key name
- Async Task Mode for long-running crews
- Task cancellation support

#### LangChain Adapter (`langchain.py`)

**Purpose:** Expose LangChain runnables as A2A agents

**Flow:**
1. Extract user message and format as runnable input
2. Call `runnable.ainvoke(...)` for non-streaming
3. Call `runnable.astream(...)` for streaming
4. Extract output based on runnable type (AIMessage, dict, etc.)

**Key Features:**
- **Full streaming support** via `astream()`
- Works with any LangChain Runnable (chains, agents, RAG pipelines)
- Configurable input/output keys
- Automatic detection of streaming capability

#### Callable Adapter (`callable.py`)

**Purpose:** Wrap custom async functions as A2A agents

**Flow:**
1. Format A2A message as dictionary
2. Call the provided async function
3. Convert result to A2A Message

**Key Features:**
- Maximum flexibility for custom logic
- Optional streaming support (function must be async generator)
- Simple integration for custom agents

## Message Flow Example

### Non-Streaming Request

```
A2A Client
    │
    │ POST /messages (JSON-RPC 2.0)
    ▼
A2AStarletteApplication (official SDK)
    │
    │ Parse & validate
    ▼
AdapterRequestHandler.handle_message()
    │
    ▼
BaseAgentAdapter.handle()
    │
    ├─▶ to_framework()       → {"message": "What is 2+2?"}
    │
    ├─▶ call_framework()     → {"output": "4"}
    │
    └─▶ from_framework()     → Message(role="assistant", content="4")
    │
    ▼
Return Message to A2A Client
```

### Streaming Request

```
A2A Client
    │
    │ GET /messages/stream?... (SSE)
    ▼
A2AStarletteApplication (official SDK)
    │
    ▼
AdapterRequestHandler.handle_message_stream()
    │
    ▼
BaseAgentAdapter.handle_stream()
    │
    ├─▶ to_framework()
    │
    ├─▶ runnable.astream()
    │       │
    │       ├─▶ yield {"event": "message", "data": "The"}
    │       ├─▶ yield {"event": "message", "data": " answer"}
    │       ├─▶ yield {"event": "message", "data": " is 4"}
    │       └─▶ yield {"event": "done", "data": {...}}
    │
    ▼
Stream SSE events to A2A Client
```

### Async Task Request (N8n Adapter)

For long-running workflows, the N8n adapter supports async task mode:

```
A2A Client                              N8n Adapter                         n8n Workflow
    │                                        │                                   │
    │ POST /messages                         │                                   │
    ├───────────────────────────────────────▶│                                   │
    │                                        │                                   │
    │                                        │ Create Task(state=working)        │
    │                                        │ Save to TaskStore                 │
    │                                        │ Start background coroutine        │
    │                                        │                                   │
    │◀─── Task(id=xyz, state=working) ───────│                                   │
    │                                        │                                   │
    │                                        │──── POST webhook ────────────────▶│
    │                                        │                                   │
    │ GET /tasks/xyz (polling)               │                     (processing)  │
    ├───────────────────────────────────────▶│                                   │
    │◀─── Task(state=working) ───────────────│                                   │
    │                                        │                                   │
    │                                        │◀─── {"output": "..."} ────────────│
    │                                        │                                   │
    │                                        │ Update Task(state=completed)      │
    │                                        │ Save to TaskStore                 │
    │                                        │                                   │
    │ GET /tasks/xyz (polling)               │                                   │
    ├───────────────────────────────────────▶│                                   │
    │◀─── Task(state=completed, result) ─────│                                   │
    │                                        │                                   │
```

**Key Points:**
- Client receives `Task` immediately (non-blocking)
- Workflow executes in background coroutine
- Client polls `get_task()` to check status
- Task transitions: `working` → `completed` (or `failed`)
- Uses A2A SDK's `TaskStore` for state persistence

## Extension Points

### Custom Adapters

Users can create custom adapters in two ways:

#### 1. Subclass BaseAgentAdapter

For full control over the adapter lifecycle:

```python
from a2a_adapter import BaseAgentAdapter

class MyCustomAdapter(BaseAgentAdapter):
    async def to_framework(self, params): ...
    async def call_framework(self, framework_input, params): ...
    async def from_framework(self, framework_output, params): ...
```

#### 2. Use CallableAgentAdapter

For simple custom logic:

```python
async def my_agent_function(inputs: dict) -> str:
    return f"Processed: {inputs['message']}"

adapter = await load_a2a_agent({
    "adapter": "callable",
    "callable": my_agent_function
})
```

### Adding New Framework Adapters

To add support for a new framework:

1. Create `a2a_adapter/integrations/{framework}.py`
2. Implement a class extending `BaseAgentAdapter`
3. Add to `loader.py` factory
4. Update `integrations/__init__.py`
5. Add optional dependency to `pyproject.toml`
6. Create example in `examples/`

## Design Decisions

### Why Separate Adapters per Framework?

- **Clean separation of concerns**: Each adapter handles one framework
- **Optional dependencies**: Users only install what they need
- **Easier testing**: Each adapter can be tested independently
- **Simpler maintenance**: Changes to one framework don't affect others

### Why Not Multi-Agent Server?

- **Simplicity**: Single-agent servers are easier to understand and debug
- **Composability**: Multiple servers can be orchestrated externally
- **Scalability**: Each agent can scale independently
- **Alignment with A2A protocol**: A2A is designed for agent-to-agent communication, not internal orchestration

### Why Use Official A2A SDK?

- **Protocol compliance**: Automatic updates to A2A spec
- **Reduced maintenance**: No need to reimplement protocol details
- **Community support**: Official SDK benefits from broader ecosystem
- **Type safety**: Leverages official type definitions

## Testing Strategy

### Unit Tests (`tests/unit/`)

- Test each adapter's `to_framework()`, `call_framework()`, `from_framework()` in isolation
- Mock framework calls (HTTP requests, crew execution, etc.)
- Test error handling and edge cases

### Integration Tests (`tests/integration/`)

- Test complete A2A request/response cycle
- Requires framework dependencies installed
- May use real services (n8n webhook) or mocks

## Performance Considerations

### Async Throughout

All adapters use `async/await` for maximum concurrency:
- Non-blocking HTTP requests (n8n)
- Async crew execution (CrewAI)
- Async runnable invocation (LangChain)

### Connection Pooling

- N8n adapter reuses HTTP client (`httpx.AsyncClient`)
- Proper cleanup via context managers

### Streaming for LLMs

- LangChain adapter streams tokens as they're generated
- Reduces perceived latency for long responses
- Uses Server-Sent Events (SSE) per A2A spec

## Security Considerations

### Input Validation

- A2A SDK handles message validation
- Adapters should sanitize framework-specific inputs

### Authentication

- N8n adapter supports custom headers for webhook auth
- Server-level auth handled by A2A SDK (out of scope for adapters)

### Rate Limiting

- Out of scope for adapters
- Should be handled at infrastructure level (reverse proxy, API gateway)

## Future Enhancements

### Implemented Features

1. ✅ **Task Support** (N8n Adapter): Async task execution with background polling
   - `async_mode=True` enables async task execution
   - Integrates with A2A SDK's `TaskStore` (InMemoryTaskStore or DatabaseTaskStore)
   - Supports task cancellation

### Planned Features

1. **Artifact Support**: Handle file uploads/downloads
2. **More Adapters**: AutoGen, Semantic Kernel, Haystack, etc.
3. **Middleware**: Logging, metrics, rate limiting hooks
4. **Configuration Validation**: Pydantic schemas for adapter configs
5. **Task Support for Other Adapters**: Extend async task pattern to CrewAI, LangChain

### Breaking Changes

We follow semantic versioning:
- **v0.x.x**: Alpha/Beta, breaking changes allowed
- **v1.x.x**: Stable API, breaking changes only in major versions

## Conclusion

The A2A Adapter SDK provides a clean, extensible architecture for integrating agent frameworks with the A2A Protocol. By focusing on single-agent adaptation and leveraging the official A2A SDK, we deliver a simple yet powerful toolkit for building interoperable AI agent systems.

