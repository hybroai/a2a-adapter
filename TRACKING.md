# N8N Adapter Async Task Support - Implementation Tracking

## Overview

This document tracks the implementation of async task support for the N8N adapter, including design decisions, issues found during code review, and their resolutions.

## Feature Summary

The N8N adapter now supports two execution modes:

1. **Synchronous Mode** (default): Blocks until the n8n workflow completes, returns `Message`
2. **Async Task Mode** (`async_mode=True`): Returns `Task` immediately, processes in background, supports polling

### Key Components

- `async_mode` parameter to enable async execution
- Integration with A2A SDK's `TaskStore` (InMemoryTaskStore or DatabaseTaskStore)
- Background task execution with timeout
- Task lifecycle management (get, cancel, delete)

---

## Code Review Findings

### Round 1: Initial Implementation Review

#### 🔴 Critical Issues (Fixed)

| Issue | Description | Resolution |
|-------|-------------|------------|
| Race condition in `cancel_task()` | After cancelling asyncio task, background task could still save `failed` state, overwriting `canceled` state | Added `_cancelled_tasks` set to track cancelled task IDs; background task checks this before saving state |
| `asyncio.CancelledError` not handled | `CancelledError` is a `BaseException`, not `Exception`, so it wasn't being caught | Added explicit `except asyncio.CancelledError` handler that re-raises properly |

#### 🟡 Medium Issues (Fixed)

| Issue | Description | Resolution |
|-------|-------------|------------|
| Memory leak with InMemoryTaskStore | Completed tasks never cleaned up from memory | Added `delete_task()` method; documented memory considerations in class docstring |
| Unhandled exceptions in done callback | Lambda callback didn't check for unhandled exceptions | Replaced with named function that logs exceptions via `t.exception()` |
| No await for cancelled tasks in `close()` | Tasks cancelled but not awaited, could leave dangling coroutines | Now uses `asyncio.gather(*tasks, return_exceptions=True)` |
| No task timeout | Long-running workflows could hang indefinitely | Added `_execute_workflow_with_timeout()` using `asyncio.wait_for()` |

#### 🟢 Minor Issues (Fixed)

| Issue | Description | Resolution |
|-------|-------------|------------|
| f-strings in logger calls | Performance cost when log level disabled | Changed to lazy formatting: `logger.debug("msg %s", var)` |
| Missing `cancel_task` in base class | API inconsistency with `get_task()` | Added `cancel_task()` to `BaseAgentAdapter` |

---

### Round 2: Post-Fix Review

#### 🟡 Remaining Medium Issues

| Issue | Description | Recommendation | Status |
|-------|-------------|----------------|--------|
| TaskStore failure handling | If `task_store.save()` fails, task state becomes inconsistent | Consider retry logic or fallback for critical state transitions | Documented |
| `_cancelled_tasks` potential leak | If `cancel_task()` called for non-existent task, entry never removed | Add `self._cancelled_tasks.discard(task_id)` at end of `cancel_task()` | Minor edge case |

#### 🟢 Remaining Minor Issues

| Issue | Description | Recommendation | Status |
|-------|-------------|----------------|--------|
| No `async_timeout` validation | Could accept 0 or negative values | Add `self.async_timeout = max(1, int(async_timeout))` | Low priority |
| `delete_task()` not in base class | API inconsistency | Add to `BaseAgentAdapter` for consistency | Low priority |
| HTTP client state | Client created once, could enter bad state after network issues | httpx handles this well, low risk | Acceptable |

---

## Test Coverage

### New Tests Added (6 tests)

| Test | Purpose |
|------|---------|
| `test_task_timeout` | Verifies tasks fail after `async_timeout` seconds |
| `test_cancel_task_prevents_race_condition` | Verifies cancellation tracking prevents state overwrites |
| `test_delete_task_removes_completed_task` | Verifies cleanup functionality |
| `test_delete_task_raises_for_running_task` | Verifies safety check for non-terminal states |
| `test_delete_task_returns_false_for_unknown_id` | Verifies edge case handling |
| `test_delete_task_raises_when_not_async_mode` | Verifies mode check |

### Total Test Count

- **Before**: 25 tests
- **After**: 31 tests
- All tests passing ✅

---

## Usage Examples

### Basic Async Mode

```python
from a2a_adapter.integrations.n8n import N8nAgentAdapter

adapter = N8nAgentAdapter(
    webhook_url="http://localhost:5678/webhook/my-workflow",
    async_mode=True,
    async_timeout=300,  # 5 minutes
)

# Handle request - returns Task immediately
task = await adapter.handle(params)
print(f"Task ID: {task.id}, State: {task.status.state}")  # state=working

# Poll for completion
while True:
    task = await adapter.get_task(task.id)
    if task.status.state in ("completed", "failed", "canceled"):
        break
    await asyncio.sleep(2)

# Clean up to prevent memory leak
await adapter.delete_task(task.id)
```

### With Database TaskStore

```python
from a2a.server.tasks import DatabaseTaskStore
from sqlalchemy.ext.asyncio import create_async_engine

engine = create_async_engine("postgresql+asyncpg://...")
task_store = DatabaseTaskStore(engine)

adapter = N8nAgentAdapter(
    webhook_url="http://localhost:5678/webhook/my-workflow",
    async_mode=True,
    task_store=task_store,
)
```

---

## Architecture Decision Records

### ADR-1: Background Polling vs Callback

**Decision**: Use background polling (adapter runs workflow in background coroutine)

**Rationale**:
- Works with any n8n workflow without modifications
- No need for n8n API access or callback endpoints
- Simpler implementation and deployment

**Trade-offs**:
- Requires client to poll for status
- Task state lost on adapter restart (with InMemoryTaskStore)

### ADR-2: TaskStore Integration

**Decision**: Integrate with A2A SDK's `TaskStore` abstraction

**Rationale**:
- Consistent with A2A SDK patterns
- Allows swapping InMemoryTaskStore for DatabaseTaskStore
- Future-proof for additional TaskStore implementations

### ADR-3: Cancellation Tracking

**Decision**: Use `_cancelled_tasks` set to prevent race conditions

**Rationale**:
- Simple and effective solution
- Avoids need for locks or complex synchronization
- Background task checks set before saving state

---

## Production Considerations

### Memory Management

When using `InMemoryTaskStore` (default):
1. Call `delete_task()` after processing completed tasks
2. Implement periodic cleanup for abandoned tasks
3. Consider using `DatabaseTaskStore` for production

### Reliability

- TaskStore failures can leave tasks in inconsistent state
- Consider implementing health checks for database connectivity
- Monitor for tasks stuck in `working` state

### Scalability

- Each adapter instance has its own task state
- For multi-instance deployments, use `DatabaseTaskStore`
- Consider task ID namespacing if multiple adapters share a TaskStore

---

## Files Modified

| File | Changes |
|------|---------|
| `a2a_adapter/adapter.py` | Added `supports_async_tasks()`, `get_task()`, `cancel_task()` to base class |
| `a2a_adapter/integrations/n8n.py` | Full async task implementation |
| `tests/unit/test_n8n_adapter.py` | Added 6 new tests for async functionality |
| `ARCHITECTURE.md` | Updated with async task documentation |

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2025-12-28 | 1.0.0 | Initial async task implementation |
| 2025-12-28 | 1.0.1 | Fixed critical race condition and CancelledError handling |
| 2025-12-28 | 1.0.2 | Added task timeout, improved cleanup, added delete_task() |
