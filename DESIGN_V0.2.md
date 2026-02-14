# a2a-adapter v0.2 Architecture Design

## 1. Executive Summary

### 1.1 What is a2a-adapter

a2a-adapter is a Python SDK that converts any AI agent framework (n8n, CrewAI, LangChain, LangGraph, OpenClaw, or custom Python functions) into a fully compliant [A2A Protocol](https://a2a-protocol.org) server with a single function call.

### 1.2 Design Goal

```python
from a2a_adapter import N8nAdapter, serve_agent

adapter = N8nAdapter(webhook_url="http://localhost:5678/webhook/my-agent")
serve_agent(adapter, port=9000)
```

Three lines of code. One import source. A complete A2A server with task management, streaming, cancellation, push notifications, and agent discovery — all handled automatically.

### 1.3 Core Insight

The A2A SDK (`a2a-sdk`) already provides a production-grade implementation of the A2A protocol server — including `DefaultRequestHandler`, `TaskManager`, `TaskUpdater`, `EventQueue`, `ResultAggregator`, and push notification infrastructure. Our job is **not** to rebuild these, but to provide the thinnest possible bridge between framework-specific agent logic and the SDK's execution model.

| OpenAI Reference Layer (6-layer ideal) | Who Provides It | What We Do |
|---|---|---|
| Framework Driver Layer | **We implement** | `BaseA2AAdapter` subclasses |
| Unified Agent Runtime | **We implement** | `AdapterAgentExecutor` (thin bridge, ~80 lines) |
| A2A Protocol Facade | **A2A SDK** | `DefaultRequestHandler` (we don't touch) |
| Transport Layer | **A2A SDK** | `A2AStarletteApplication` (we don't touch) |
| AgentCard & Discovery | **A2A SDK + We** | SDK serves `/.well-known/agent.json`; we auto-generate the card |
| Ops Layer | **Not yet** | Future: auth, rate limiting, tracing |

---

## 2. Design Principles

### 2.1 SDK-First

> **If the A2A SDK already provides it, we don't write it.**

The A2A SDK's `DefaultRequestHandler` handles all of the following, so we delegate entirely:

- `message/send` (sync and non-blocking)
- `message/stream` (SSE streaming)
- `tasks/get` (task status query)
- `tasks/cancel` (task cancellation with terminal state checks)
- `tasks/resubscribe` (SSE reconnection)
- Push notification config CRUD and delivery
- Task lifecycle state machine (submitted → working → completed/failed/canceled/rejected)
- Task persistence via `TaskStore`
- Event queuing and consumption via `EventQueue` / `QueueManager`

**Design rationale:** The A2A protocol evolves. By delegating protocol compliance to the SDK, we automatically stay up-to-date with spec changes without modifying our code. We also avoid subtle compliance bugs — the SDK team has already handled edge cases like terminal state re-entry, history length truncation, and non-blocking mode.

### 2.2 Minimal Surface

> **`invoke(user_input: str) -> str` is the only method a developer must implement.**

Every agent framework, no matter how complex internally, can ultimately answer one question: "Given a user's text, what's your text response?" This is `invoke()`.

**Design rationale:** The more we require, the more we alienate. A CrewAI developer shouldn't need to learn `EventQueue`, `TaskUpdater`, or `TaskStatusUpdateEvent` to expose their crew as an A2A agent. They should just tell us how to call their crew and extract the result.

### 2.3 Layered Escape Hatch

> **Simple things should be simple. Complex things should be possible.**

Three levels of control, progressively more powerful:

| Level | Interface | Who It's For |
|---|---|---|
| Level 1 | `invoke(user_input) -> str` | 90% of use cases |
| Level 2 | `stream(user_input) -> AsyncIterator[str]` | Frameworks with streaming (LangChain, LangGraph) |
| Level 3 | Implement `AgentExecutor` directly | Full SDK access (multi-artifact, custom metadata, multi-turn) |

**Design rationale:** The bridge layer (`AdapterAgentExecutor`) handles the Level 1/2 → SDK translation. Developers who need Level 3 bypass our adapter entirely and talk to the SDK directly. No artificial ceiling.

### 2.4 Open-Closed

> **Adding a new framework adapter should not require modifying any core file.**

New adapters are added by:
1. Creating a new file in `integrations/`
2. Registering it in the `__init__.py` lazy import map

No changes to `adapter.py`, `executor.py`, `server.py`, or `loader.py`.

**Design rationale:** Open-source contributors should be able to add framework support without understanding or risking the core codebase.

### 2.5 Single Import Source

> **Everything the user needs comes from `from a2a_adapter import ...`.**

All adapters and utilities are exported from the package root via lazy imports. Users never need to know about `a2a_adapter.integrations.n8n` or `a2a_adapter.server`.

**Design rationale:** This is the standard pattern for well-designed Python SDKs (`stripe`, `openai`, `fastapi`, `pydantic`). It provides:
- One mental model (one import source)
- IDE autocomplete via `__all__`
- Freedom to refactor internal structure without breaking public API
- Clean separation between "what users see" and "how it's built"

---

## 3. Architecture

### 3.1 Layer Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│  Layer 5: Public API                                             │
│                                                                  │
│  from a2a_adapter import N8nAdapter, serve_agent                 │
│  serve_agent(adapter, port=9000)                                 │
│  # or: app = to_a2a(adapter)                                     │
│  # or: card = build_agent_card(adapter)                          │
├──────────────────────────────────────────────────────────────────┤
│  Layer 4: A2A SDK (we don't write, we delegate)                  │
│                                                                  │
│  ┌─────────────────────────┐  ┌────────────────────────────┐    │
│  │ DefaultRequestHandler   │  │ A2AStarletteApplication    │    │
│  │  on_message_send()      │  │  JSON-RPC routing          │    │
│  │  on_message_send_stream │  │  SSE transport             │    │
│  │  on_get_task()          │  │  /.well-known/agent.json   │    │
│  │  on_cancel_task()       │  └────────────────────────────┘    │
│  │  on_resubscribe()       │                                     │
│  │  on_*_push_config()     │  ┌────────────────────────────┐    │
│  └────────────┬────────────┘  │ TaskManager                │    │
│               │               │ ResultAggregator           │    │
│               │               │ EventQueue / QueueManager  │    │
│               │               │ TaskStore (InMemory/custom) │    │
│               ▼               │ PushNotification*          │    │
│  ┌─────────────────────────┐  └────────────────────────────┘    │
│  │ AgentExecutor (ABC)     │                                     │
│  │  execute(ctx, queue)    │                                     │
│  │  cancel(ctx, queue)     │                                     │
│  └────────────┬────────────┘                                     │
├───────────────┼──────────────────────────────────────────────────┤
│  Layer 3: Bridge (we write, ~80 lines)                           │
│               │                                                  │
│  ┌────────────▼────────────┐                                     │
│  │ AdapterAgentExecutor    │  implements AgentExecutor            │
│  │                         │                                     │
│  │  execute():             │                                     │
│  │    input = ctx.get_user_input()                               │
│  │    updater.start_work()                                       │
│  │    if streaming:                                              │
│  │      for chunk in adapter.stream(input):                      │
│  │        updater.add_artifact(chunk)                            │
│  │    else:                                                      │
│  │      result = adapter.invoke(input)                           │
│  │      updater.add_artifact(result)                             │
│  │    updater.complete()                                         │
│  │                         │                                     │
│  │  cancel():              │                                     │
│  │    adapter.cancel()                                           │
│  │    updater.cancel()                                           │
│  └────────────┬────────────┘                                     │
├───────────────┼──────────────────────────────────────────────────┤
│  Layer 2: Adapter Interface (we write, ~60 lines)                │
│               │                                                  │
│  ┌────────────▼────────────┐                                     │
│  │ BaseA2AAdapter (ABC)    │                                     │
│  │                         │                                     │
│  │  invoke(input, ctx_id)  │  ← REQUIRED (Level 1)              │
│  │    -> str               │                                     │
│  │                         │                                     │
│  │  stream(input, ctx_id)  │  ← optional (Level 2)              │
│  │    -> AsyncIterator     │                                     │
│  │                         │                                     │
│  │  cancel()               │  ← optional                        │
│  │  close()                │  ← optional (resource cleanup)      │
│  │  get_metadata()         │  ← optional (AgentCard generation)  │
│  └────────────┬────────────┘                                     │
├───────────────┼──────────────────────────────────────────────────┤
│  Layer 1: Framework Drivers (we write, ~50-150 lines each)       │
│               │                                                  │
│  ┌────────────▼──────┐ ┌──────────┐ ┌───────────┐ ┌───────────┐│
│  │ N8nAdapter        │ │CrewAI    │ │LangChain  │ │LangGraph  ││
│  │  invoke():        │ │Adapter   │ │Adapter    │ │Adapter    ││
│  │   build payload   │ │ invoke():│ │ invoke(): │ │ invoke(): ││
│  │   POST webhook    │ │  kickoff │ │  ainvoke  │ │  ainvoke  ││
│  │   extract text    │ │  extract │ │  extract  │ │  extract  ││
│  └──────────────────┘ └──────────┘ │ stream(): │ │ stream(): ││
│  ┌──────────────────┐ ┌──────────┐ │  astream  │ │  astream  ││
│  │ OpenClawAdapter   │ │Callable  │ └───────────┘ └───────────┘│
│  │  invoke():        │ │Adapter   │                             │
│  │   spawn CLI       │ │ invoke():│                             │
│  │   parse JSON      │ │  call fn │                             │
│  │  cancel():        │ └──────────┘                             │
│  │   kill process    │                                          │
│  └──────────────────┘                                           │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 Request Flow: message/send (Synchronous)

```
Client POST /  {"jsonrpc":"2.0","method":"message/send","params":{...}}
    │
    ▼
A2AStarletteApplication                      [Layer 4: SDK]
    │  parse JSON-RPC, route to handler
    ▼
DefaultRequestHandler.on_message_send()      [Layer 4: SDK]
    │  create TaskManager, EventQueue
    │  asyncio.create_task(executor.execute(context, queue))
    ▼
AdapterAgentExecutor.execute()               [Layer 3: Bridge]
    │  user_input = context.get_user_input()
    │  TaskUpdater.start_work()
    │  result = adapter.invoke(user_input)
    │  TaskUpdater.add_artifact(result)
    │  TaskUpdater.complete()
    ▼
N8nAdapter.invoke()                          [Layer 1: Driver]
    │  payload = build_payload(user_input)
    │  response = POST webhook_url
    │  return extract_text(response)
    ▼
DefaultRequestHandler                        [Layer 4: SDK]
    │  ResultAggregator.consume_all()
    │  return Task(status=completed, artifacts=[...])
    ▼
Client receives JSON-RPC response
```

### 3.3 Request Flow: message/stream (Streaming)

```
Client POST /  {"jsonrpc":"2.0","method":"message/stream","params":{...}}
    │
    ▼
DefaultRequestHandler.on_message_send_stream()
    │  same setup as above
    ▼
AdapterAgentExecutor.execute()
    │  if adapter.supports_streaming():
    │    async for chunk in adapter.stream(user_input):
    │      TaskUpdater.add_artifact(chunk, append=True)
    │    TaskUpdater.complete()
    ▼
DefaultRequestHandler
    │  ResultAggregator.consume_and_emit()
    │  yield SSE events to client in real-time
    ▼
Client receives SSE stream:
  data: {"task_id":"...","status":{"state":"working"}}
  data: {"artifact":{"parts":[{"text":"chunk1"}],"append":true}}
  data: {"artifact":{"parts":[{"text":"chunk2"}],"append":true}}
  data: {"status":{"state":"completed","message":{...}},"final":true}
```

### 3.4 Request Flow: tasks/cancel

```
Client POST /  {"jsonrpc":"2.0","method":"tasks/cancel","params":{"id":"task-123"}}
    │
    ▼
DefaultRequestHandler.on_cancel_task()       [Layer 4: SDK]
    │  check task exists, check not in terminal state
    │  call executor.cancel(context, queue)
    │  cancel the asyncio producer task
    ▼
AdapterAgentExecutor.cancel()                [Layer 3: Bridge]
    │  adapter.cancel()   ← e.g. OpenClaw kills subprocess
    │  TaskUpdater.cancel()
    ▼
DefaultRequestHandler
    │  return Task(status=canceled)
```

### 3.5 Request Flow: tasks/resubscribe (Reconnection)

```
Client POST /  {"jsonrpc":"2.0","method":"tasks/resubscribe","params":{"id":"task-123"}}
    │
    ▼
DefaultRequestHandler.on_resubscribe_to_task()    [Layer 4: SDK]
    │  check task exists, check not in terminal state
    │  QueueManager.tap(task_id) → get existing EventQueue
    │  ResultAggregator.consume_and_emit()
    │  yield pending + live SSE events to client
    ▼
Client receives SSE stream (resumed)
```

---

## 4. Core Interfaces

### 4.1 BaseA2AAdapter

**File:** `a2a_adapter/adapter.py`

**Design rationale:** This is the only class framework developers interact with. Every method except `invoke()` has a sensible default, so the minimum implementation is just one method.

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator
from dataclasses import dataclass, field


@dataclass
class AdapterMetadata:
    """
    Self-describing metadata for automatic AgentCard generation.

    Design rationale: Instead of forcing users to manually construct
    AgentCard objects, adapters can declare their capabilities here.
    The server layer uses this to auto-generate a well-known card.
    """
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    skills: list[dict] = field(default_factory=list)
    input_modes: list[str] = field(default_factory=lambda: ["text"])
    output_modes: list[str] = field(default_factory=lambda: ["text"])
    streaming: bool = False


class BaseA2AAdapter(ABC):
    """
    The only interface framework developers need to implement.

    Design philosophy:
    - invoke() is the single required method — answers "given text, return text"
    - stream() is optional — for frameworks that support token-by-token output
    - cancel() is optional — for frameworks where execution can be interrupted
    - close() is optional — for resource cleanup (HTTP clients, subprocesses)
    - get_metadata() is optional — for automatic AgentCard generation

    Everything else (task management, SSE streaming, push notifications,
    resubscription, state persistence) is handled by the A2A SDK via
    the AdapterAgentExecutor bridge layer.
    """

    @abstractmethod
    async def invoke(
        self,
        user_input: str,
        context_id: str | None = None,
    ) -> str:
        """
        Execute the agent and return a text response.

        This is the ONLY method you must implement.

        Args:
            user_input: The user's message as plain text.
                        Extracted from A2A MessageSendParams by the bridge layer
                        using SDK's RequestContext.get_user_input().
            context_id: Conversation context ID for multi-turn support.
                        Same context_id = same conversation. None for single-turn.

        Returns:
            The agent's text response.

        Raises:
            Any exception will be caught by the bridge layer and converted
            to a Task with state=failed.
        """
        ...

    async def stream(
        self,
        user_input: str,
        context_id: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream the agent response, yielding text chunks.

        Optional. If not implemented, the bridge layer falls back to invoke()
        and delivers the full result as a single event.

        When implemented, each yielded chunk becomes a TaskArtifactUpdateEvent
        in the A2A SSE stream, giving clients real-time token-by-token output.
        """
        raise NotImplementedError

    def supports_streaming(self) -> bool:
        """Whether this adapter supports streaming responses."""
        return type(self).stream is not BaseA2AAdapter.stream

    async def cancel(self) -> None:
        """
        Cancel the current execution. Optional.

        Override for frameworks where execution can be interrupted
        (e.g., OpenClaw can kill a subprocess).
        """
        pass

    async def close(self) -> None:
        """Release resources held by this adapter. Optional."""
        pass

    def get_metadata(self) -> AdapterMetadata:
        """Return adapter metadata for automatic AgentCard generation. Optional."""
        return AdapterMetadata()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
```

### 4.2 AdapterAgentExecutor (Bridge Layer)

**File:** `a2a_adapter/executor.py`

**Design rationale:** This is the key innovation — a thin bridge (~80 lines) that translates our simple `invoke()/stream()` interface into the SDK's event-driven `AgentExecutor` interface. Users never see or interact with this class.

```python
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, TextPart
from a2a.utils.message import new_agent_text_message

from .adapter import BaseA2AAdapter


class AdapterAgentExecutor(AgentExecutor):
    """
    Bridge: translates BaseA2AAdapter's simple interface into the
    SDK's AgentExecutor event-driven model.

    Design rationale:
    - Adapters should never import or use EventQueue, TaskUpdater, etc.
    - This bridge handles all the "protocol plumbing" in one place.
    - If the adapter raises an exception, we catch it and emit failed state.
    - If the adapter supports streaming, we emit artifact chunks incrementally.
    - If not, we call invoke() and emit a single artifact + completion.
    """

    def __init__(self, adapter: BaseA2AAdapter):
        self.adapter = adapter

    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        updater = TaskUpdater(
            event_queue, context.task_id, context.context_id
        )
        user_input = context.get_user_input()

        try:
            await updater.start_work()

            if self.adapter.supports_streaming():
                await self._execute_streaming(updater, user_input, context)
            else:
                await self._execute_sync(updater, user_input, context)

        except Exception as e:
            error_msg = new_agent_text_message(
                f"Error: {e}", context.context_id, context.task_id
            )
            await updater.failed(message=error_msg)

    async def _execute_sync(self, updater, user_input, context):
        result_text = await self.adapter.invoke(
            user_input, context.context_id
        )
        await updater.add_artifact(
            [Part(root=TextPart(text=result_text))],
            name="response",
        )
        message = new_agent_text_message(
            result_text, context.context_id, context.task_id
        )
        await updater.complete(message=message)

    async def _execute_streaming(self, updater, user_input, context):
        chunks = []
        async for chunk in self.adapter.stream(
            user_input, context.context_id
        ):
            chunks.append(chunk)
            await updater.add_artifact(
                [Part(root=TextPart(text=chunk))],
                append=len(chunks) > 1,
                last_chunk=False,
            )

        full_text = "".join(chunks)
        message = new_agent_text_message(
            full_text, context.context_id, context.task_id
        )
        await updater.complete(message=message)

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        updater = TaskUpdater(
            event_queue, context.task_id, context.context_id
        )
        await self.adapter.cancel()
        await updater.cancel()
```

### 4.3 Server Layer

**File:** `a2a_adapter/server.py` (renamed from `client.py`)

**Design rationale:** This layer provides the user-facing entry points (`serve_agent`, `to_a2a`, `build_agent_card`). It wires together the adapter, bridge, SDK handler, and Starlette app. The old `AdapterRequestHandler` is completely deleted — `DefaultRequestHandler` replaces it.

```python
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskStore
from a2a.types import AgentCard, AgentCapabilities, AgentSkill

from .adapter import BaseA2AAdapter
from .executor import AdapterAgentExecutor


def to_a2a(
    adapter: BaseA2AAdapter,
    agent_card: AgentCard | None = None,
    task_store: TaskStore | None = None,
    **card_overrides,
):
    """
    Convert any adapter into an A2A Server ASGI application.

    Aligned with Google ADK's to_a2a(root_agent) pattern.

    Design rationale: This is the "magic function" that wires everything
    together. User gives us an adapter, we return a standards-compliant
    A2A server.
    """
    if agent_card is None:
        agent_card = build_agent_card(adapter, **card_overrides)

    executor = AdapterAgentExecutor(adapter)
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store or InMemoryTaskStore(),
    )

    return A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=handler,
    ).build()


def serve_agent(
    adapter: BaseA2AAdapter,
    agent_card: AgentCard | None = None,
    host: str = "0.0.0.0",
    port: int = 9000,
    log_level: str = "info",
    **uvicorn_kwargs,
) -> None:
    """
    One-line A2A server startup.

    Design rationale: The most common use case is "start serving now".
    This function combines to_a2a() + uvicorn.run() for maximum convenience.
    """
    app = to_a2a(adapter, agent_card)
    uvicorn.run(
        app, host=host, port=port, log_level=log_level, **uvicorn_kwargs
    )


def build_agent_card(adapter: BaseA2AAdapter, **overrides) -> AgentCard:
    """
    Auto-generate AgentCard from adapter metadata.

    Design rationale: Most developers don't want to manually construct
    AgentCard objects with 10+ fields. This function reads the adapter's
    get_metadata() and produces a reasonable default.
    """
    meta = adapter.get_metadata()
    return AgentCard(
        name=overrides.get("name", meta.name or type(adapter).__name__),
        description=overrides.get("description", meta.description or ""),
        url=overrides.get("url", "http://localhost:9000"),
        version=overrides.get("version", meta.version),
        capabilities=AgentCapabilities(
            streaming=overrides.get("streaming", meta.streaming),
        ),
        skills=[
            AgentSkill(**s) for s in meta.skills
        ] if meta.skills else [],
        default_input_modes=meta.input_modes,
        default_output_modes=meta.output_modes,
    )
```

### 4.4 Loader (Registry Pattern)

**File:** `a2a_adapter/loader.py`

**Design rationale:** For config-driven deployments (YAML/JSON configs, orchestration systems), the loader creates adapters from dictionaries. The registry pattern allows third-party adapters to register themselves without modifying core code.

```python
from typing import Dict, Type
import importlib

from .adapter import BaseA2AAdapter

_REGISTRY: Dict[str, Type[BaseA2AAdapter]] = {}

_BUILTIN_MAP = {
    "n8n":       ("a2a_adapter.integrations.n8n",       "N8nAdapter"),
    "crewai":    ("a2a_adapter.integrations.crewai",    "CrewAIAdapter"),
    "langchain": ("a2a_adapter.integrations.langchain", "LangChainAdapter"),
    "langgraph": ("a2a_adapter.integrations.langgraph", "LangGraphAdapter"),
    "callable":  ("a2a_adapter.integrations.callable",  "CallableAdapter"),
    "openclaw":  ("a2a_adapter.integrations.openclaw",  "OpenClawAdapter"),
}


def register_adapter(name: str):
    """
    Decorator for third-party adapters to register themselves.

    Usage:
        @register_adapter("my_framework")
        class MyFrameworkAdapter(BaseA2AAdapter):
            ...
    """
    def decorator(cls: Type[BaseA2AAdapter]):
        _REGISTRY[name] = cls
        return cls
    return decorator


def load_adapter(config: dict) -> BaseA2AAdapter:
    """
    Factory: create an adapter from a configuration dictionary.

    Design rationale: Enables config-driven deployments where agent
    configurations come from YAML/JSON files, databases, or APIs.

    Example:
        adapter = load_adapter({
            "adapter": "n8n",
            "webhook_url": "http://localhost:5678/webhook/my-agent",
            "timeout": 60,
        })
    """
    config = dict(config)
    adapter_type = config.pop("adapter", None)
    if not adapter_type:
        raise ValueError("Config must include 'adapter' key")

    cls = _REGISTRY.get(adapter_type)

    if cls is None and adapter_type in _BUILTIN_MAP:
        module_path, class_name = _BUILTIN_MAP[adapter_type]
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)

    if cls is None:
        registered = sorted(set(_REGISTRY) | set(_BUILTIN_MAP))
        raise ValueError(
            f"Unknown adapter: {adapter_type!r}. "
            f"Available: {', '.join(registered)}"
        )

    return cls(**config)
```

---

## 5. Public API Design

### 5.1 Package `__init__.py`

**File:** `a2a_adapter/__init__.py`

**Design rationale:** All public symbols are exported from the package root. Core utilities are imported eagerly (no optional deps). Adapter classes use `__getattr__` for lazy loading to avoid requiring all framework packages.

```python
"""
a2a-adapter: Convert any AI agent into an A2A Protocol server.

Usage:
    from a2a_adapter import N8nAdapter, serve_agent

    adapter = N8nAdapter(webhook_url="http://localhost:5678/webhook/agent")
    serve_agent(adapter, port=9000)
"""

__version__ = "0.2.0"

# Eager imports (no optional dependencies)
from .adapter import BaseA2AAdapter, AdapterMetadata
from .server import serve_agent, to_a2a, build_agent_card
from .loader import load_adapter, register_adapter

# Lazy imports (framework adapters with optional deps)
_ADAPTER_LAZY_MAP = {
    "N8nAdapter":       (".integrations.n8n",       "N8nAdapter"),
    "CrewAIAdapter":    (".integrations.crewai",    "CrewAIAdapter"),
    "LangChainAdapter": (".integrations.langchain", "LangChainAdapter"),
    "LangGraphAdapter": (".integrations.langgraph", "LangGraphAdapter"),
    "CallableAdapter":  (".integrations.callable",  "CallableAdapter"),
    "OpenClawAdapter":  (".integrations.openclaw",  "OpenClawAdapter"),
}


def __getattr__(name: str):
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
    # Core
    "BaseA2AAdapter",
    "AdapterMetadata",
    # Server
    "serve_agent",
    "to_a2a",
    "build_agent_card",
    # Loader
    "load_adapter",
    "register_adapter",
    # Adapters (lazy)
    "N8nAdapter",
    "CrewAIAdapter",
    "LangChainAdapter",
    "LangGraphAdapter",
    "CallableAdapter",
    "OpenClawAdapter",
]
```

### 5.2 User-Facing Import Patterns

```python
# Pattern 1: Direct adapter import (recommended)
from a2a_adapter import N8nAdapter, serve_agent

# Pattern 2: Config-driven
from a2a_adapter import load_adapter, serve_agent

# Pattern 3: Custom adapter development
from a2a_adapter import BaseA2AAdapter, serve_agent

# Pattern 4: Advanced (ASGI app for custom deployment)
from a2a_adapter import N8nAdapter, to_a2a
app = to_a2a(N8nAdapter(webhook_url="..."))

# Pattern 5: Third-party adapter registration
from a2a_adapter import register_adapter, BaseA2AAdapter

@register_adapter("my_framework")
class MyAdapter(BaseA2AAdapter):
    async def invoke(self, user_input, context_id=None):
        return "Hello from my framework!"
```

---

## 6. Framework Drivers (Adapter Implementations)

### 6.1 Design Philosophy for All Adapters

Each adapter answers ONE question: **"Given a user's text, how do I call my framework and extract a text response?"**

Adapters do NOT:
- Manage asyncio background tasks
- Construct Task or TaskStatus objects
- Create SSE events
- Handle push notifications
- Implement get_task / cancel_task
- Manage TaskStore

All of the above is handled by the SDK + bridge layer.

### 6.2 N8nAdapter

**Framework:** n8n webhook-triggered workflows
**Unique aspects:** HTTP POST to webhook, configurable payload format, retry with backoff
**Streaming:** No

```python
class N8nAdapter(BaseA2AAdapter):
    """~80 lines (was 800 lines)"""

    def __init__(self, webhook_url, timeout=30, headers=None,
                 payload_template=None, message_field="message",
                 input_mapper=None, parse_json_input=True,
                 default_inputs=None, max_retries=2, backoff=0.25):
        # Store config, create httpx client lazily

    async def invoke(self, user_input, context_id=None) -> str:
        payload = self._build_payload(user_input, context_id)
        response = await self._post_with_retry(payload)
        return self._extract_response_text(response)

    # Internal: _build_payload, _post_with_retry, _extract_response_text
    # Lifecycle: close() to close httpx client
```

**Usage:**

```python
from a2a_adapter import N8nAdapter, serve_agent

adapter = N8nAdapter(webhook_url="http://localhost:5678/webhook/math-agent")
serve_agent(adapter, port=9000)
```

### 6.3 CrewAIAdapter

**Framework:** CrewAI multi-agent crews
**Unique aspects:** `crew.kickoff_async()` with sync fallback, long timeout
**Streaming:** No

```python
class CrewAIAdapter(BaseA2AAdapter):
    """~50 lines (was 631 lines)"""

    def __init__(self, crew, inputs_key="topic", timeout=300):
        ...

    async def invoke(self, user_input, context_id=None) -> str:
        inputs = self._build_inputs(user_input)
        result = await asyncio.wait_for(
            self.crew.kickoff_async(inputs=inputs), timeout=self.timeout
        )
        return self._extract_output(result)

    # Internal: _build_inputs (JSON parse or fallback), _extract_output (.raw/.result)
```

**Usage:**

```python
from crewai import Agent, Crew, Process, Task as CrewTask
from a2a_adapter import CrewAIAdapter, serve_agent

researcher = Agent(role="Researcher", goal="...", backstory="...")
task = CrewTask(description="Research {topic}", agent=researcher, expected_output="...")
crew = Crew(agents=[researcher], tasks=[task], process=Process.sequential)

adapter = CrewAIAdapter(crew=crew)
serve_agent(adapter, port=9001)
```

### 6.4 LangChainAdapter

**Framework:** LangChain LCEL runnables (chains, agents, RAG)
**Unique aspects:** `runnable.ainvoke()` and `runnable.astream()`
**Streaming:** Yes (automatic via `astream()`)

```python
class LangChainAdapter(BaseA2AAdapter):
    """~60 lines (was 366 lines)"""

    def __init__(self, runnable, input_key="input", output_key=None, timeout=60):
        ...

    async def invoke(self, user_input, context_id=None) -> str:
        inputs = self._build_inputs(user_input)
        result = await asyncio.wait_for(
            self.runnable.ainvoke(inputs), timeout=self.timeout
        )
        return self._extract_output(result)

    async def stream(self, user_input, context_id=None):
        inputs = self._build_inputs(user_input)
        async for chunk in self.runnable.astream(inputs):
            text = self._extract_chunk_text(chunk)
            if text:
                yield text

    def supports_streaming(self):
        return hasattr(self.runnable, "astream")
```

**Usage:**

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from a2a_adapter import LangChainAdapter, serve_agent

llm = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_template("Answer: {input}")
chain = prompt | llm

adapter = LangChainAdapter(runnable=chain, input_key="input")
serve_agent(adapter, port=9002)
# Automatically supports both message/send AND message/stream!
```

### 6.5 LangGraphAdapter

**Framework:** LangGraph compiled state graphs
**Unique aspects:** State-based I/O, `graph.ainvoke()` and `graph.astream()`
**Streaming:** Yes (yields intermediate states)

```python
class LangGraphAdapter(BaseA2AAdapter):
    """~70 lines (was 796 lines)"""

    def __init__(self, graph, input_key="messages", output_key=None, timeout=60):
        ...

    async def invoke(self, user_input, context_id=None) -> str:
        inputs = self._build_inputs(user_input)
        result = await asyncio.wait_for(
            self.graph.ainvoke(inputs), timeout=self.timeout
        )
        return self._extract_output(result)

    async def stream(self, user_input, context_id=None):
        inputs = self._build_inputs(user_input)
        last_text = ""
        async for state in self.graph.astream(inputs):
            text = self._extract_output(state)
            if text and text != last_text:
                delta = text[len(last_text):] if text.startswith(last_text) else text
                last_text = text
                if delta:
                    yield delta
```

**Usage:**

```python
from langgraph.graph import StateGraph
from a2a_adapter import LangGraphAdapter, serve_agent

graph = builder.compile()

adapter = LangGraphAdapter(graph=graph)
serve_agent(adapter, port=9003)
```

### 6.6 CallableAdapter

**Framework:** Any async Python function
**Unique aspects:** Maximum flexibility
**Streaming:** Optional (function can be async generator)

```python
class CallableAdapter(BaseA2AAdapter):
    """~30 lines (was 287 lines)"""

    def __init__(self, func, supports_streaming=False):
        self.func = func
        self._streaming = supports_streaming

    async def invoke(self, user_input, context_id=None) -> str:
        result = await self.func({"message": user_input, "context_id": context_id})
        return str(result) if not isinstance(result, str) else result

    async def stream(self, user_input, context_id=None):
        async for chunk in self.func({"message": user_input, "context_id": context_id}):
            yield str(chunk) if not isinstance(chunk, str) else chunk

    def supports_streaming(self):
        return self._streaming
```

**Usage:**

```python
from a2a_adapter import CallableAdapter, serve_agent

async def my_agent(inputs: dict) -> str:
    return f"You said: {inputs['message']}"

adapter = CallableAdapter(func=my_agent)
serve_agent(adapter, port=9004)
```

### 6.7 OpenClawAdapter

**Framework:** OpenClaw CLI (`openclaw agent --local --json`)
**Unique aspects:** Subprocess management, session ID mapping, cancel = kill process
**Streaming:** No

```python
class OpenClawAdapter(BaseA2AAdapter):
    """~150 lines (was 1298 lines)"""

    def __init__(self, session_id=None, agent_id=None, thinking="low",
                 timeout=600, openclaw_path="openclaw",
                 working_directory=None, env_vars=None):
        ...
        self._current_process = None

    async def invoke(self, user_input, context_id=None) -> str:
        session_id = self._context_to_session(context_id)
        cmd = self._build_command(user_input, session_id)
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=PIPE, stderr=PIPE, ...
        )
        self._current_process = proc
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )
        finally:
            self._current_process = None
        return self._extract_text(json.loads(stdout.decode()))

    async def cancel(self):
        if self._current_process and self._current_process.returncode is None:
            self._current_process.kill()

    # Internal: _build_command, _context_to_session, _extract_text
```

**Usage:**

```python
from a2a_adapter import OpenClawAdapter, serve_agent

adapter = OpenClawAdapter(
    session_id="my-session",
    agent_id="main",
    thinking="low",
)
serve_agent(adapter, port=9005)
```

---

## 7. SDK Delegation Table

What we **delete** from our codebase and **where the SDK handles it**:

| Our Current Code | Lines | Replaced By (A2A SDK) |
|---|---|---|
| `AdapterRequestHandler` class (client.py) | 196 | `DefaultRequestHandler` |
| `extract_raw_input()` (adapter.py) | 50 | `RequestContext.get_user_input()` |
| `extract_context_id()` (adapter.py) | 12 | `RequestContext.context_id` |
| `try_parse_json()` + `apply_input_mapping()` (adapter.py) | 50 | Moved into each adapter's internal `_build_inputs()` |
| `handle()` + `handle_stream()` (adapter.py) | 40 | `AdapterAgentExecutor.execute()` |
| `to_framework/call_framework/from_framework` (adapter.py) | 30 | Merged into `invoke()` |
| `_handle_async()` per adapter (n8n/crewai/langgraph/openclaw) | 4x70=280 | `DefaultRequestHandler._setup_message_execution()` |
| `_execute_*_background()` per adapter | 4x80=320 | `DefaultRequestHandler` + `ResultAggregator` |
| `_execute_*_with_timeout()` per adapter | 4x30=120 | Adapter's own `invoke()` with `asyncio.wait_for` |
| `get_task()` per adapter | 4x20=80 | `DefaultRequestHandler.on_get_task()` |
| `cancel_task()` per adapter | 4x40=160 | `DefaultRequestHandler.on_cancel_task()` |
| `delete_task()` per adapter | 2x30=60 | TaskStore lifecycle |
| `_background_tasks` / `_cancelled_tasks` management | 4x30=120 | SDK internal task tracking |
| Manual `Task()` / `TaskStatus()` construction | ~200 | `TaskUpdater` + `TaskManager` |
| Manual `Message()` construction | ~100 | `new_agent_text_message()` + `TaskUpdater.new_agent_message()` |
| Manual SSE dict construction (callable/langchain/langgraph) | ~100 | `ResultAggregator.consume_and_emit()` |
| Push notification impl (openclaw) | ~120 | `PushNotificationSender` + `PushNotificationConfigStore` |
| TTL cleanup loop (openclaw) | ~70 | TaskStore lifecycle |
| `InMemoryTaskStore` lazy import (n8n/crewai/langgraph) | ~30 | Direct SDK import in `server.py` |
| **Total deleted** | **~2140** | |

---

## 8. Directory Structure

```
a2a_adapter/
├── __init__.py              # Public API: all exports, lazy adapter imports
│                            # (~50 lines)
│
├── adapter.py               # BaseA2AAdapter + AdapterMetadata
│                            # (~60 lines)
│
├── executor.py              # AdapterAgentExecutor bridge
│                            # (~80 lines)
│
├── server.py                # to_a2a(), serve_agent(), build_agent_card()
│                            # (~80 lines, replaces client.py)
│
├── loader.py                # register_adapter(), load_adapter()
│                            # (~60 lines)
│
└── integrations/
    ├── __init__.py           # Lazy imports + __all__
    ├── n8n.py               # N8nAdapter (~80 lines)
    ├── crewai.py            # CrewAIAdapter (~50 lines)
    ├── langchain.py         # LangChainAdapter (~60 lines)
    ├── langgraph.py         # LangGraphAdapter (~70 lines)
    ├── callable.py          # CallableAdapter (~30 lines)
    └── openclaw.py          # OpenClawAdapter (~150 lines)
```

**Code volume comparison:**

| Component | v0.1 (current) | v0.2 (proposed) | Change |
|---|---|---|---|
| Core (`__init__` + `adapter` + `executor` + `server` + `loader`) | 896 lines | ~330 lines | -63% |
| Integrations (all 6 adapters) | 4,270 lines | ~440 lines | -90% |
| **Total** | **~5,166 lines** | **~770 lines** | **-85%** |

---

## 9. Migration Strategy

### PR 1: Foundation

**Goal:** Introduce new core files without breaking existing code.

**Changes:**
- Create `executor.py` (AdapterAgentExecutor)
- Create `server.py` (to_a2a, serve_agent, build_agent_card)
- Create new `adapter.py` with `BaseA2AAdapter` + `AdapterMetadata`
  - Keep old `BaseAgentAdapter` as deprecated alias
- Update `__init__.py` with new public API + lazy imports
- Update `loader.py` with registry pattern
- `client.py` remains but is marked deprecated

**Tests:** New unit tests for executor, server, loader.

### PR 2: Adapter Migration

**Goal:** Migrate all adapters to new interface, one by one.

For each adapter:
1. Implement `invoke()` (and `stream()` where applicable)
2. Remove `to_framework` / `call_framework` / `from_framework`
3. Remove `_handle_async` / `_execute_*_background` / `_execute_*_with_timeout`
4. Remove `get_task` / `cancel_task` / `delete_task`
5. Remove `_background_tasks` / `_cancelled_tasks` management
6. Remove manual Task/Message/SSE construction

**Order:** callable (simplest) → n8n → langchain → crewai → langgraph → openclaw (most complex)

### PR 3: Cleanup

**Goal:** Remove all deprecated code and update examples.

**Changes:**
- Delete `client.py` (fully replaced by `server.py`)
- Delete old `BaseAgentAdapter` alias
- Update all examples to new API
- Update README and ARCHITECTURE docs
- Bump version to `0.2.0`

### PR 4: Tests & Docs

**Goal:** Comprehensive test coverage and documentation.

- Integration tests with mock adapters
- Example for each framework
- Migration guide for v0.1 → v0.2

---

## 10. Open Questions & Future Work

### 10.1 Open Questions for v0.2

| Question | Current Decision | Rationale |
|---|---|---|
| `invoke()` returns `str` only — what about multi-modal? | Text only in v0.2 | Keep minimal. OpenClaw can use Level 3 escape hatch. v0.3 can extend to `str \| list[Part]`. |
| Should `build_agent_card` auto-detect streaming? | Yes, from `supports_streaming()` | Less manual config for users. |
| Keep `input_mapper` / `parse_json_input` on base class? | No, move into each adapter's internal logic | Each adapter knows its framework's input format best. |
| `client.py` rename to `server.py`? | Yes | It's a server, not a client. |

### 10.2 Future Work (v0.3+)

| Feature | Priority | Description |
|---|---|---|
| Multi-modal `invoke()` | High | `invoke() -> str \| list[Part]` for images, files |
| RemoteA2aAgent client | Medium | Call remote A2A agents as local sub-agents (ADK consuming pattern) |
| AgentCard from docstring | Low | Auto-extract description/skills from adapter class docstring |
| Ops middleware | Low | Auth, rate limiting, tracing middleware for Starlette app |
| `to_a2a` CLI | Low | `a2a-adapter serve --config agent.yaml` command-line tool |

---

*Document version: v0.2-draft-1*
*Last updated: 2026-02-09*
*Authors: HYBRO AI*
