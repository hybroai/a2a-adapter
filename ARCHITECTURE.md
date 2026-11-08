# A2A Adapter SDK — Architecture (v0.2)

## Overview

The A2A Adapter SDK converts any AI agent into an A2A Protocol server. The key insight: adapters should only answer "given text, return text" — everything else is delegated to the official A2A SDK. A thin CLI constructs selected CLI-backed adapters and then enters the same server layer as Python callers.

## Design Principles

1. **SDK-First** — use A2A SDK's `DefaultRequestHandler`, `TaskStore`, `TaskUpdater` for all protocol handling
2. **Minimal Surface** — single required method: `invoke(user_input) -> str`
3. **Layered Escape Hatch** — Level 1: `invoke()` / Level 2: `stream()` / Level 3: implement `AgentExecutor` directly
4. **Open-Closed** — extend via `register_adapter()` without modifying core
5. **Single Import Source** — `from a2a_adapter import XxxAdapter, serve_agent`
6. **Thin CLI** — parse process/server options, construct an existing adapter,
   and delegate to `serve_agent()` without duplicating protocol behavior

## Architecture Layers

```
┌──────────────────────────────────────────────────────────────┐
│  Layer 1: User-Facing Interface                              │
│  CLI → existing adapter → serve_agent()                      │
│  Python → serve_agent() / to_a2a() / build_agent_card()      │
│  (a2a_adapter/cli.py, a2a_adapter/server.py)                 │
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
    → cancels the running execute() (CancelledError)
      → execute() waits for the cancel below before it returns,
        so the event queue stays open for the terminal state; until
        then it sends no terminal state or late artifact of its own
        (an open streaming artifact is left open)
    → AdapterAgentExecutor.cancel()
      → adapter.cancel()          (may await; CLI adapters kill the CLI's
                                   process group on POSIX)
      → TaskUpdater.cancel()
        if adapter.cancel() raised: wait (≤ _CANCEL_SETTLE_TIMEOUT, 10 s)
        for the interrupted run to unwind; canceled if it stopped, else
        one failed state ("Cancel failed: …"), which the RPC returns as
        the task, not an error. The wait holds the SDK's per-task lock,
        so a resubscribe or second cancel of that task waits too
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

CLI adapters built on the subprocess runtime (`_run_subprocess` /
`_run_subprocess_streaming`) start the CLI in its own session on POSIX, so
cancel, timeout, `close()` and shutdown kill the CLI's whole process group
(shells, MCP servers, tool processes), not only the direct child. A helper
that moves itself into a new session escapes the group; Windows still kills
only the direct child.
A cancel that finds no running CLI queues a per-task marker; the executor
calls `_release_task(task_id)` once the task's `execute()` and `cancel()`
have both finished, so unconsumed markers do not pile up. Adapters that keep
their own per-task cancel markers (Pi, DeepSeek Harness) override it.
`timeout` bounds the whole run for both invoke and stream (a streaming CLI
that keeps printing still stops at the deadline). The stale-session retry
gets a fresh budget.

A resumed run whose session no longer exists is retried once without resume
and the stored session is dropped. By default that is detected only when the
run failed with no output at all; an adapter can override
`_is_stale_session_error(stdout_text, stderr_text)` to recognise its CLI's
own message (Claude Code and Codex do). A streaming run is never retried after
output reached the client.

### AdapterAgentExecutor (`executor.py`)

Bridge between `BaseA2AAdapter` and A2A SDK's `AgentExecutor`:

- Extracts `user_input` from `RequestContext.get_user_input()`
- Calls `adapter.invoke()` or `adapter.stream()`
- Converts results to A2A events via `TaskUpdater`
- Handles errors → `TaskUpdater.failed()`
- Handles cancellation → `adapter.cancel()` + `TaskUpdater.cancel()`; if `adapter.cancel()` raises, the executor waits, bounded, for the interrupted run to unwind: if it stopped, the task is canceled (the adapter error is logged); if not, the executor sends a single failed state with a "Cancel failed: …" message and does not re-raise, so `tasks/cancel` returns that failed task rather than a JSON-RPC error. "Stopped" means the adapter's `invoke()` returned or raised, or its `stream()` generator was closed (generators nested inside it are left to the event loop); an adapter that starts remote work must stop it in `cancel()` or on `CancelledError`. The interrupted `execute()` waits for the cancel to finish, so this holds when `adapter.cancel()` awaits. While a cancel is in progress, `execute()` sends no terminal state and no late artifact (an error, or a result the adapter returns after `CancelledError`, is dropped); if `execute()` already sent its terminal state, the cancel sends none. Each run sends at most one terminal state

### Server Layer (`server.py`)

Three public functions:

- **`to_a2a(adapter)`** — adapter → ASGI app (wires executor + handler + Starlette)
- **`serve_agent(adapter, port)`** — `to_a2a()` + an in-process `uvicorn.Server`.
  Ctrl+C (or a terminal hangup, SIGHUP) stops running CLI agents and everything they started immediately
  (`adapter.kill_children_now()` from the signal handler); SIGTERM lets
  in-flight requests finish within uvicorn's graceful timeout, then the
  lifespan's `adapter.close()` stops them. Apps served by another ASGI server
  get only the lifespan `close()` and an `atexit` fallback.
- **`build_agent_card(adapter)`** — `adapter.get_metadata()` → `AgentCard`

`serve_agent()` binds `127.0.0.1` by default. `to_a2a(auth_token=...)` /
`serve_agent(auth_token=...)` add a pure-ASGI bearer-token middleware,
outermost (before the activity middleware, which buffers and may log request
bodies). Only GET/HEAD of the two agent-card paths stay public; every other
route, including ones the SDK adds later, needs `Authorization: Bearer
<token>` and gets 401 otherwise. The generated card declares the bearer
scheme. Serving a non-loopback host without a token logs a warning. The
token never reaches CLI children: `A2A_AUTH_TOKEN` is removed from their
environment unless the adapter's own `env_vars` set it.

Push notifications are opt-in (`push_notifications=True`). The card's
`capabilities.push_notifications` is the single switch: when it is off,
`to_a2a` passes the SDK no push store and no sender (the SDK stores inline
configs whenever a store exists). When it is on, the adapter's config store
rejects URLs that are not http(s), point at `localhost`, or resolve to any
non-public address (loopback, private, link-local/metadata, CGNAT,
IPv4-embedding IPv6 forms such as mapped and NAT64, site-local, multicast)
unless `allow_private_push_urls=True`. The client gets one generic
"Rejected push notification URL" error (`-32602` on A2A 1.0 JSON-RPC, 400 on
REST, `-32603` on the v0.3 names, which a2a-sdk 1.0 maps that way); the reason
and any resolved address go only to the server log. Against DNS rebinding, the
push client's network backend resolves each new connection's host once,
rejects it if any address is non-public, and connects to the checked IP while
keeping the hostname for the `Host` header, SNI and certificate checks.
Redirects are not followed (and a followed hop would open a new, re-checked
connection). The push client does not use `HTTP(S)_PROXY` env vars unless
private URLs are allowed. Checked addresses are tried one at a time in resolver
order within one connect timeout (no happy-eyeballs racing), so a host whose
first address silently drops packets can time out instead of falling back.

### CLI Entry Point (`cli.py`)

The `a2a-adapter` console script is intentionally thin:

- Supports `pi`, `codex`, `claude`, `openclaw`, and `dsh` (DeepSeek Harness)
- Resolves the working directory and A2A server settings
- Passes native agent arguments after `--` into the selected adapter
- Rejects protocol/session flags owned by the adapter
- Calls the same `serve_agent()` used by Python SDK consumers
- `--auth-token` (or `$A2A_AUTH_TOKEN`, which the CLI then removes from its
  own environment) turns on bearer-token auth
- `--push-notifications` / `--allow-private-push-urls` turn on push and allow
  private receivers; unset flags keep the SDK defaults

Hermes is not a v1 CLI target because `HermesAdapter` is an in-process gateway
adapter rather than a local CLI subprocess adapter. `dsh` is a target because
`DeepSeekHarnessAdapter` drives a local `dsh` subprocess like Pi; its SDK takes
no native argv, so the subcommand exposes explicit options and rejects
arguments after `--`.

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
| `ClaudeCodeAdapter` | `claude` subprocess + stream-json parsing and resume | Yes |
| `CodexAdapter` | `codex exec --json` subprocess + thread resume | No |
| `OpenClawAdapter` | Subprocess exec + JSON parse + cancel (kill) | No |
| `HermesAdapter` | Gateway pattern: `SessionDB` + `AIAgent.run_conversation()` in a thread pool; requires Hermes on `PYTHONPATH` | Yes (callback → queue) |
| `OllamaAdapter` | Wraps `OllamaClient` HTTP client (`/api/chat`) | Yes (NDJSON streaming) |
| `PiAdapter` | One persistent `pi --mode rpc` process and session | Yes (JSONL events) |
| `DeepSeekHarnessAdapter` | One `deepseek-harness-sdk` runtime (`dsh` over stdio JSON-RPC) on a single worker thread; one durable session per `context_id`; streams each assistant message | Yes |
| `DeepSeekHarnessHeadlessAdapter` | `dsh --profile headless` subprocess, task on stdin; stateless by design, every turn is a fresh dsh session | No |
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
| Push notifications (opt-in; URL checks are ours, in the config store) | `PushNotificationSender` |
| Task resubscription | `DefaultRequestHandler.on_resubscribe` |

## Directory Structure

```
a2a_adapter/
├── __init__.py          # Public API + lazy imports
├── base_adapter.py      # BaseA2AAdapter + AdapterMetadata
├── executor.py          # AdapterAgentExecutor (bridge)
├── cli.py               # Thin console entry point for CLI-backed agents
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
    ├── ollama.py        # OllamaClient + OllamaAdapter
    ├── claude_code.py   # ClaudeCodeAdapter
    ├── codex.py         # CodexAdapter
    ├── openclaw.py      # OpenClawAdapter + OpenClawAgentAdapter
    ├── hermes.py        # HermesAdapter (Hermes Agent gateway pattern)
    ├── deepseek_harness.py  # DeepSeekHarnessAdapter (DeepSeek Harness SDK runtime)
    ├── deepseek_harness_headless.py  # DeepSeekHarnessHeadlessAdapter (stateless dsh CLI)
    └── pi.py            # PiAdapter and internal persistent JSONL RPC runtime
```

## Testing Strategy

- **Unit tests** (`tests/unit/test_v02_*.py`) — test each v0.2 adapter in isolation with mocks
- **Unit tests** (`tests/unit/test_*.py`) — existing v0.1 tests (still passing for backwards compat)
- **CLI tests** (`tests/unit/test_cli.py`) — parsing, help, adapter construction,
  native argument placement, and managed-option rejection
- **Integration tests** (`tests/integration/`) — full A2A request/response cycle

## Migration Path

- **v0.1 → v0.2**: All v0.1 classes/functions still work with deprecation warnings
- **v0.2 → v0.3**: v0.1 deprecated code will be removed
- The v0.1 adapter classes (`*AgentAdapter`) coexist in each integration file alongside v0.2 classes
