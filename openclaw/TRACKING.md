# OpenClaw A2A Adapter - Design and Implementation Plan

## Overview

This document tracks the design and implementation of an A2A adapter for [OpenClaw](https://github.com/openclaw/openclaw), a personal AI super agent. The adapter wraps the OpenClaw CLI as an A2A-compliant agent with async task support.

## Feature Summary

The OpenClaw adapter exposes OpenClaw agents via the A2A protocol, supporting:

1. **Async Task Mode** (default): Returns `Task` immediately, processes in background, supports polling
2. **Synchronous Mode** (`async_mode=False`): Blocks until command completes, returns `Message`

### Key Design Decisions

- **CLI-based integration**: Uses `openclaw agent --local --json` subprocess (no changes to OpenClaw required)
- **Async-first**: Default to async tasks for UI responsiveness
- **Subprocess management**: Track and kill processes on cancellation/timeout

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        A2A Client                                   │
└─────────────────────────────────────────────────────────────────────┘
         │                                    ▲
         │ message/send                       │ Task(state=working)
         ▼                                    │
┌─────────────────────────────────────────────────────────────────────┐
│                    OpenClawAgentAdapter                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  1. Create Task(id=xyz, state=working)                      │   │
│  │  2. Save to TaskStore                                       │   │
│  │  3. Spawn background subprocess                             │   │
│  │  4. Return Task immediately                                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Background:                                                        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  openclaw agent --local --message "..." --json              │   │
│  │                                                             │   │
│  │  On success: Task(state=completed, message=response)        │   │
│  │  On error:   Task(state=failed, message=error)              │   │
│  │  On timeout: Task(state=failed, message="timed out")        │   │
│  │  On cancel:  Task(state=canceled)                           │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Task State Flow

```
send_message() ──► Task(state=working)
                        │
                        ▼
              ┌─────────────────┐
              │  Background     │
              │  Subprocess     │
              └────────┬────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
    completed       failed       canceled
         │             │             │
         ▼             ▼             ▼
  task.status.   task.status.   (no response)
  message has    message has
  response       error
```

---

## Requirements

### Host Requirements

- OpenClaw CLI installed and in PATH (or provide custom path)
- OpenClaw configured with API keys (`ANTHROPIC_API_KEY`, etc.)
- Valid OpenClaw configuration at `~/.openclaw/config.yaml`

### Runtime Requirements

- Python 3.11+
- A2A SDK with task support (`pip install a2a-sdk`)

---

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session_id` | str | auto-generated | Session ID for conversation continuity |
| `agent_id` | str | None | OpenClaw agent ID (from `openclaw agents list`) |
| `thinking` | str | "low" | Thinking level: off\|minimal\|low\|medium\|high\|xhigh |
| `timeout` | int | 300 | Command timeout in seconds |
| `openclaw_path` | str | "openclaw" | Path to openclaw binary |
| `working_directory` | str | None | Working directory for subprocess |
| `env_vars` | dict | {} | Additional environment variables |
| `async_mode` | bool | True | Enable async task mode |
| `task_store` | TaskStore | InMemoryTaskStore | Task state storage |

---

## Implementation Plan

### Phase 1: Core Implementation ✅

- [x] Create `a2a_adapter/integrations/openclaw.py`
- [x] Implement `OpenClawAgentAdapter` class
- [x] Implement sync mode (`to_framework`, `call_framework`, `from_framework`)
- [x] Implement async task mode (`_handle_async`, background execution)
- [x] Implement subprocess management (spawn, kill, track)
- [x] Implement task lifecycle (`get_task`, `cancel_task`, `delete_task`)

### Phase 2: Integration ✅

- [x] Update `a2a_adapter/loader.py` to register openclaw adapter
- [x] Update `a2a_adapter/integrations/__init__.py` exports
- [x] Add optional httpx dependency in `pyproject.toml`

### Phase 3: Testing ✅

- [x] Create `tests/unit/test_openclaw_adapter.py`
- [x] Test sync mode execution
- [x] Test async task creation and polling
- [x] Test timeout handling
- [x] Test cancellation
- [x] Test error handling (binary not found, invalid config, etc.)
- [x] Test context_id to session_id mapping

### Phase 4: Documentation and Examples ✅

- [x] Create `examples/08_openclaw_agent.py`
- [ ] Update `README.md` with OpenClaw adapter docs
- [ ] Update `ARCHITECTURE.md` with OpenClaw section

### Phase 5: Push Notification Support ✅

- [x] Add `supports_push_notifications()` method to adapter
- [x] Add `_push_configs` dict to track push configs per task
- [x] Add `_send_push_notification()` method to POST to webhook
- [x] Update `_handle_async()` to extract push config from params
- [x] Update `_execute_command_background()` to send notifications on completion/failure
- [x] Update `cancel_task()` to send notification on cancellation
- [x] Add `set_push_notification_config()`, `get_push_notification_config()`, `delete_push_notification_config()` methods
- [x] Update `AdapterRequestHandler` to support push notification operations
- [x] Update example to use `pushNotifications=True` in agent card

---

## Critical Design Decisions

### Decision 1: CLI with `--local` Flag (Critical)

**Problem**: The `openclaw agent` command by default calls the Gateway via WebSocket RPC, not the embedded agent directly.

**Solution**: Always use `--local` flag to run the embedded agent directly.

```python
cmd = [
    self.openclaw_path,
    "agent",
    "--local",  # CRITICAL: Run embedded, not via gateway
    "--message", message,
    "--json",
    ...
]
```

**Rationale**:
- Avoids Gateway dependency
- Consistent JSON output format
- Works standalone without Gateway running

### Decision 2: Session Identification

**Problem**: CLI requires at least one of `--to`, `--session-id`, or `--agent`.

**Solution**: Always provide `--session-id` (auto-generate if not provided).

```python
self.session_id = session_id or f"a2a-{uuid.uuid4().hex[:12]}"
cmd.extend(["--session-id", self.session_id])
```

### Decision 3: Subprocess Tracking for Cancellation

**Problem**: When cancelling a task, the subprocess may continue running.

**Solution**: Track subprocess references and kill them on cancellation.

```python
self._background_processes: Dict[str, asyncio.subprocess.Process] = {}

# On cancel:
proc = self._background_processes.get(task_id)
if proc and proc.returncode is None:
    proc.kill()
```

### Decision 4: Async-First Default

**Problem**: OpenClaw agents can run for minutes (tool execution, complex reasoning).

**Solution**: Default to `async_mode=True` for UI responsiveness.

**Rationale**:
- UI clients need immediate feedback
- Polling allows progress updates
- Cancellation support for long operations

### Decision 5: Context ID to Session ID Mapping

**Problem**: A2A's `context_id` format (typically UUID) may not be compatible with OpenClaw's session ID format (`^[a-z0-9][a-z0-9_-]{0,63}$`).

**Solution**: Sanitize and map `context_id` to a valid OpenClaw session ID.

```python
# In to_framework():
context_id = self._extract_context_id(params)
effective_session_id = self._context_id_to_session_id(context_id)
```

**Rationale**:
- **Stateless**: No mapping table to manage, survives adapter restarts
- **Multi-tenant**: Each A2A context gets its own OpenClaw session
- **Safe**: Sanitization handles any input format (UUIDs, custom strings, special chars)
- **Backward compatible**: Falls back to default session when no context_id

---

## Known Issues and Risks

### Issue 1: Environment Dependencies (CLI mode)

**Risk**: OpenClaw must be installed and configured on the A2A server host.

**Mitigation**: Document requirements; consider Docker deployment with OpenClaw pre-installed.

### Issue 2: No Streaming Support

**Risk**: CLI mode cannot stream partial responses.

**Status**: Accepted limitation. CLI returns only after completion.

**Alternative**: HTTP mode could support streaming via OpenAI-compatible endpoint (future enhancement).

### Issue 3: Long-Running Operations

**Risk**: OpenClaw agents can run for minutes.

**Mitigation**:
- Configurable timeout (default 300s)
- Async task mode with polling
- Cancellation support

### Issue 4: Session State Management ✅ RESOLVED

**Risk**: A2A's `contextId` doesn't map 1:1 to OpenClaw's session system.

**Solution Implemented**: The adapter now maps A2A `context_id` to OpenClaw `session_id` with sanitization:

```python
def _context_id_to_session_id(self, context_id: str | None) -> str:
    """
    Convert A2A context_id to a valid OpenClaw session_id.
    
    OpenClaw session IDs must match: ^[a-z0-9][a-z0-9_-]{0,63}$
    """
    if not context_id:
        return self.session_id  # fallback to default
    
    # Sanitize: lowercase, replace invalid chars, remove leading/trailing hyphens
    sanitized = re.sub(r'[^a-z0-9_-]+', '-', context_id.lower())
    sanitized = re.sub(r'^-+|-+$', '', sanitized)
    
    if not sanitized:
        return self.session_id
    
    # Prefix with 'a2a-' and truncate to 64 chars
    return f"a2a-{sanitized[:60]}"
```

**Behavior**:
- Each A2A `context_id` gets its own OpenClaw session (true multi-tenancy)
- Conversation continuity within the same context
- Falls back to adapter's default session when `context_id` is None/empty

**Example Mappings**:

| A2A `context_id` | OpenClaw `session_id` |
|------------------|----------------------|
| `550e8400-e29b-41d4-a716-446655440000` | `a2a-550e8400-e29b-41d4-a716-446655440000` |
| `user:123/session:456` | `a2a-user-123-session-456` |
| `my_context_123` | `a2a-my_context_123` |
| `None` | (adapter's default session) |

### Issue 5: Media/File Handling

**Risk**: OpenClaw can return `mediaUrls` (images, files).

**Mitigation**: Convert `mediaUrls` to A2A `FilePart` objects with MIME type detection.

### Issue 6: Concurrent Session Access

**Risk**: Multiple A2A requests with same `session_id` may conflict.

**Mitigation**: Use unique session IDs per A2A context; document recommendation.

### Issue 7: Push Notification Delivery ✅ RESOLVED

**Risk**: Webhook delivery may fail due to network issues, invalid URLs, or authentication problems.

**Solution Implemented**: The adapter handles push notification failures gracefully:

```python
async def _send_push_notification(self, task_id: str, task: Task) -> bool:
    """Send push notification with error handling."""
    push_config = self._push_configs.get(task_id)
    if not push_config or not push_config.url:
        return False

    try:
        # Build headers with Bearer token if provided
        headers = {"Content-Type": "application/json"}
        if push_config.token:
            headers["Authorization"] = f"Bearer {push_config.token}"

        # Send TaskStatusUpdateEvent payload
        response = await client.post(push_config.url, json=payload, headers=headers)
        
        if response.status_code in (200, 201, 202, 204):
            logger.info("Push notification sent for task %s", task_id)
            return True
        else:
            logger.warning("Push notification failed: HTTP %s", response.status_code)
            return False
    except Exception as e:
        logger.error("Failed to send push notification: %s", e)
        return False
```

**Behavior**:
- Push notifications are sent on task completion, failure, timeout, and cancellation
- Bearer token authentication is supported via `PushNotificationConfig.token`
- Failures are logged but don't affect task state (task still completes/fails normally)
- HTTP client is reused for efficiency and properly closed on adapter shutdown

---

## API Reference

### OpenClawAgentAdapter

```python
class OpenClawAgentAdapter(BaseAgentAdapter):
    def __init__(
        self,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        thinking: str = "low",
        timeout: int = 300,
        openclaw_path: str = "openclaw",
        working_directory: Optional[str] = None,
        env_vars: Optional[Dict[str, str]] = None,
        async_mode: bool = True,
        task_store: "TaskStore | None" = None,
    ): ...
    
    # Core A2A interface
    async def handle(self, params: MessageSendParams) -> Message | Task: ...
    
    # Framework interface (for sync mode)
    async def to_framework(self, params: MessageSendParams) -> Dict[str, Any]: ...
    async def call_framework(self, framework_input, params) -> Dict[str, Any]: ...
    async def from_framework(self, framework_output, params) -> Message: ...
    
    # Context/Session mapping
    def _context_id_to_session_id(self, context_id: str | None) -> str: ...
    
    # Async task support
    def supports_async_tasks(self) -> bool: ...
    async def get_task(self, task_id: str) -> Task | None: ...
    async def cancel_task(self, task_id: str) -> Task | None: ...
    async def delete_task(self, task_id: str) -> bool: ...
    
    # Push notification support
    def supports_push_notifications(self) -> bool: ...
    async def set_push_notification_config(self, task_id: str, config: PushNotificationConfig) -> bool: ...
    async def get_push_notification_config(self, task_id: str) -> PushNotificationConfig | None: ...
    async def delete_push_notification_config(self, task_id: str) -> bool: ...
    
    # Lifecycle
    async def close(self) -> None: ...
```

### Loader Configuration

```python
adapter = await load_a2a_agent({
    "adapter": "openclaw",
    "session_id": "my-session",
    "agent_id": "main",
    "thinking": "low",
    "timeout": 300,
    "async_mode": True,
})
```

---

## Usage Examples

### Basic Async Mode (Default)

```python
from a2a_adapter import load_a2a_agent, serve_agent
from a2a.types import AgentCard

async def main():
    adapter = await load_a2a_agent({
        "adapter": "openclaw",
        "session_id": "a2a-demo-session",
        "agent_id": "main",
        "thinking": "low",
    })
    
    agent_card = AgentCard(
        name="OpenClaw Agent",
        description="Personal AI super agent powered by OpenClaw",
        url="http://localhost:9000",
    )
    
    serve_agent(agent_card=agent_card, adapter=adapter, port=9000)
```

### Client Polling for Response

```python
import asyncio
from a2a.client import A2AClient

async def call_agent():
    client = A2AClient(base_url="http://localhost:9000")
    
    # Send message - returns Task immediately
    task = await client.send_message(
        message="Write a hello world function"
    )
    print(f"Task {task.id} created, state: {task.status.state}")
    
    # Poll for completion
    while task.status.state in ("submitted", "working"):
        await asyncio.sleep(2)
        task = await client.get_task(task.id)
        print(f"Polling... state: {task.status.state}")
    
    # Extract response
    if task.status.state == "completed":
        response = task.status.message
        for part in response.parts:
            if hasattr(part, "root") and hasattr(part.root, "text"):
                print("Response:", part.root.text)
    
    # Clean up
    await client.delete_task(task.id)
```

### Sync Mode (Simple Integration)

```python
adapter = await load_a2a_agent({
    "adapter": "openclaw",
    "async_mode": False,  # Blocks until complete
    "session_id": "sync-session",
})

# This blocks until OpenClaw completes
message = await adapter.handle(params)
print(message.parts[0].root.text)
```

---

## OpenClaw CLI Reference

### Command Format

```bash
openclaw agent \
    --local \
    --message "Your message here" \
    --json \
    --session-id "session-123" \
    --agent "main" \
    --thinking "low"
```

### JSON Output Format

```json
{
  "payloads": [
    {
      "text": "Response text from agent",
      "mediaUrl": null,
      "mediaUrls": ["https://example.com/image.png"]
    }
  ],
  "meta": {
    "durationMs": 1234,
    "agentMeta": {
      "sessionId": "session-123",
      "provider": "anthropic",
      "model": "claude-opus-4-5",
      "usage": {
        "input": 100,
        "output": 50
      }
    }
  }
}
```

### Valid Thinking Levels

| Level | Description |
|-------|-------------|
| `off` | No extended thinking |
| `minimal` | Minimal thinking |
| `low` | Low thinking (default) |
| `medium` | Medium thinking |
| `high` | High thinking |
| `xhigh` | Extended high thinking |

---

## Files Created/Modified

| File | Status | Description |
|------|--------|-------------|
| `a2a_adapter/integrations/openclaw.py` | ✅ Created | Main adapter implementation (~1100 lines) |
| `a2a_adapter/client.py` | ✅ Modified | Added push notification and task support to RequestHandler |
| `a2a_adapter/loader.py` | ✅ Modified | Register openclaw adapter |
| `a2a_adapter/integrations/__init__.py` | ✅ Modified | Export OpenClawAgentAdapter |
| `tests/unit/test_openclaw_adapter.py` | ✅ Created | 53 unit tests |
| `examples/08_openclaw_agent.py` | ✅ Created | Usage example with push notifications |
| `README.md` | ⏳ Pending | Add OpenClaw documentation |
| `ARCHITECTURE.md` | ⏳ Pending | Add OpenClaw section |

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2025-01-30 | 0.1.0 | Initial design and implementation plan |
| 2025-01-31 | 0.2.0 | Core implementation complete (async/sync modes, task lifecycle) |
| 2025-01-31 | 0.3.0 | Added context_id to session_id mapping for multi-tenancy |
| 2025-02-01 | 0.4.0 | Added push notification support for webhook callbacks |

---

## References

- [OpenClaw Repository](https://github.com/openclaw/openclaw)
- [A2A Protocol Specification](https://google.github.io/A2A/specification/)
- [A2A Adapter README](../README.md)
- [A2A Adapter Architecture](../ARCHITECTURE.md)
