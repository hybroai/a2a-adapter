# A2A Adapter SDK — Architecture (v0.2)

## Overview

The A2A Adapter SDK converts any AI agent into an A2A Protocol server. The key insight: adapters should only answer "given text, return text" — everything else is delegated to the official A2A SDK.

## Design Principles

1. **SDK-First** — use A2A SDK's `DefaultRequestHandler`, `TaskStore`, `TaskUpdater` for all protocol handling
2. **Minimal Surface** — single required method: `invoke(user_input) -> str`
3. **Layered Escape Hatch** — Level 1: `invoke()` / Level 2: `stream()` / Level 3: implement `AgentExecutor` directly
4. **Open-Closed** — extend via `register_adapter()` without modifying core
5. **Single Import Source** — `from a2a_adapter import XxxAdapter, serve_agent`

## Architecture Layers

```
┌──────────────────────────────────────────────────────────────┐
│  Layer 1: User-Facing API                                    │
│  serve_agent() / to_a2a() / build_agent_card()               │
│  (a2a_adapter/server.py)                                     │
├──────────────────────────────────────────────────────────────┤
│  Layer 2: A2A SDK Protocol Handling                          │
│  DefaultRequestHandler + InMemoryTaskStore                   │
│  (a2a-sdk — handles JSON-RPC, SSE, task lifecycle)           │
├──────────────────────────────────────────────────────────────┤
│  Layer 3: Bridge                                             │
│  AdapterAgentExecutor (implements AgentExecutor ABC)          │
│  (a2a_adapter/executor.py)                                   │
├──────────────────────────────────────────────────────────────┤
│  Layer 4: Adapter Interface                                  │
│  BaseA2AAdapter (invoke / stream / cancel / close)           │
│  (a2a_adapter/base_adapter.py)                               │
├──────────────────────────────────────────────────────────────┤
│  Layer 5: Framework Drivers                                  │
│  N8nAdapter / LangChainAdapter / LangGraphAdapter / ...      │
│  (a2a_adapter/integrations/*.py)                             │
└──────────────────────────────────────────────────────────────┘
```

## Request Flow

### message/send (non-streaming)

```
A2A Client → HTTP POST /
  → A2AStarletteApplication (routes JSON-RPC)
    → DefaultRequestHandler.on_message_send()
      → creates RequestContext + EventQueue
      → asyncio.create_task(executor.execute(ctx, queue))
        → AdapterAgentExecutor.execute()
          → TaskUpdater.start_work()
          → adapter.invoke(user_input, context_id)
          → TaskUpdater.add_artifact(text)
          → TaskUpdater.complete()
      → ResultAggregator collects events → returns Task
```

### message/stream (SSE streaming)

```
A2A Client → HTTP POST /
  → DefaultRequestHandler.on_message_send_stream()
    → same flow, but events are SSE-streamed
      → adapter.stream(user_input, context_id)
        → each yielded chunk → TaskUpdater.update_status()
      → TaskUpdater.complete()
```

### tasks/cancel

```
A2A Client → HTTP POST / (tasks/cancel)
  → DefaultRequestHandler.on_cancel_task()
    → cancels asyncio.Task (raises CancelledError)
      → AdapterAgentExecutor catches CancelledError
        → adapter.cancel()
        → TaskUpdater.cancel()
```

## Core Components

### BaseA2AAdapter (`base_adapter.py`)

The only interface framework developers implement:

```python
class BaseA2AAdapter(ABC):
    @abstractmethod
    async def invoke(self, user_input: str, context_id: str | None = None, **kwargs) -> str: ...

    async def stream(self, user_input: str, context_id: str | None = None, **kwargs) -> AsyncIterator[str]: ...
    async def cancel(self, **kwargs) -> None: ...
    async def close(self) -> None: ...
    def get_metadata(self) -> AdapterMetadata: ...
```

### AdapterAgentExecutor (`executor.py`)

Bridge between `BaseA2AAdapter` and A2A SDK's `AgentExecutor`:

- Extracts `user_input` from `RequestContext.get_user_input()`
- Calls `adapter.invoke()` or `adapter.stream()`
- Converts results to A2A events via `TaskUpdater`
- Handles errors → `TaskUpdater.failed()`
- Handles cancellation → `adapter.cancel()` + `TaskUpdater.cancel()`

### Server Layer (`server.py`)

Three public functions:

- **`to_a2a(adapter)`** — adapter → ASGI app (wires executor + handler + Starlette)
- **`serve_agent(adapter, port)`** — `to_a2a()` + `uvicorn.run()`
- **`build_agent_card(adapter)`** — `adapter.get_metadata()` → `AgentCard`

### Loader (`loader.py`)

Registry pattern for config-driven deployments:

- **`load_adapter(config)`** — factory: `{"adapter": "n8n", ...}` → `N8nAdapter(...)`
- **`register_adapter(name)`** — decorator for third-party adapters
- Built-in map + user registry, with lazy imports

## Framework Adapters

Each adapter extends `BaseA2AAdapter` with framework-specific logic:

| Adapter | Core Logic | Streaming |
|---|---|---|
| `N8nAdapter` | HTTP POST to webhook + retry | No |
| `LangChainAdapter` | `ainvoke()` / `astream()` with output extraction | Yes (auto-detected) |
| `LangGraphAdapter` | `ainvoke()` / `astream()` with state delta streaming | Yes (auto-detected) |
| `CrewAIAdapter` | `kickoff_async()` with sync fallback | No |
| `OpenClawAdapter` | Subprocess exec + JSON parse + cancel (kill) | No |
| `CallableAdapter` | Direct function call | Optional |

**Input pipeline (all adapters):** `input_mapper` > JSON parse > `input_key` fallback.

## What the A2A SDK Handles (not us)

| Concern | SDK Component |
|---|---|
| JSON-RPC 2.0 parsing | `DefaultRequestHandler` |
| Task lifecycle (working → completed/failed/canceled) | `TaskUpdater` + `TaskStore` |
| SSE streaming transport | `DefaultRequestHandler` + `EventQueue` |
| Concurrent request handling | `asyncio.create_task()` per request |
| Task persistence | `InMemoryTaskStore` (pluggable) |
| AgentCard serving | `A2AStarletteApplication` at `/.well-known/agent-card.json` |
| Push notifications | `PushNotificationSender` |
| Task resubscription | `DefaultRequestHandler.on_resubscribe` |

## Directory Structure

```
a2a_adapter/
├── __init__.py          # Public API + lazy imports
├── base_adapter.py      # BaseA2AAdapter + AdapterMetadata
├── executor.py          # AdapterAgentExecutor (bridge)
├── server.py            # to_a2a() / serve_agent() / build_agent_card()
├── loader.py            # load_adapter() / register_adapter()
├── adapter.py           # [deprecated] v0.1 BaseAgentAdapter
├── client.py            # [deprecated] v0.1 server helpers
└── integrations/
    ├── __init__.py      # Lazy exports for all adapters
    ├── n8n.py           # N8nAdapter (v0.2) + N8nAgentAdapter (v0.1)
    ├── callable.py      # CallableAdapter + CallableAgentAdapter
    ├── langchain.py     # LangChainAdapter + LangChainAgentAdapter
    ├── langgraph.py     # LangGraphAdapter + LangGraphAgentAdapter
    ├── crewai.py        # CrewAIAdapter + CrewAIAgentAdapter
    └── openclaw.py      # OpenClawAdapter + OpenClawAgentAdapter
```

## Testing Strategy

- **Unit tests** (`tests/unit/test_v02_*.py`) — test each v0.2 adapter in isolation with mocks
- **Unit tests** (`tests/unit/test_*.py`) — existing v0.1 tests (still passing for backwards compat)
- **Integration tests** (`tests/integration/`) — full A2A request/response cycle

## Migration Path

- **v0.1 → v0.2**: All v0.1 classes/functions still work with deprecation warnings
- **v0.2 → v0.3**: v0.1 deprecated code will be removed
- The v0.1 adapter classes (`*AgentAdapter`) coexist in each integration file alongside v0.2 classes
