# v0.2.9 A2A SDK V1.0 Compatibility Sweep

**Date:** 2026-04-21
**Scope:** Make all adapters loadable and functional on `a2a-sdk>=1,<2`. Preserve V0.1 deprecated classes without formal removal (deferred to 0.3.0).
**Version:** 0.2.8 -> 0.2.9

## A2A SDK V1.0 Breaking Changes

The SDK moved from Pydantic models to protobuf-generated types. Key removals and renames:

| Old (V0.x SDK) | V1.0 Status | Replacement |
|---|---|---|
| `TextPart` | **Removed** | `Part(text="...")` |
| `FilePart`, `FileWithUri`, `FileWithBytes` | **Removed** | `Part(url="...", filename="...", media_type="...")` or `Part(raw=b"...", ...)` |
| `Part(root=TextPart(text="..."))` | **Breaks** | `Part(text="...")` |
| `Part(root=FilePart(file=FileWithUri(...)))` | **Breaks** | `Part(url=..., media_type=..., filename=...)` |
| `part.root.text` accessor | **Breaks** | `part.text` (proto field) |
| `MessageSendParams` | **Removed** | `SendMessageRequest` |
| `PushNotificationConfig` | **Removed** | `TaskPushNotificationConfig` |
| `Role.agent` / `Role.user` | **Removed** | `Role.ROLE_AGENT` / `Role.ROLE_USER` |
| `TaskState.working` / `.completed` / `.failed` / `.canceled` | **Removed** | `TaskState.TASK_STATE_WORKING` / `TASK_STATE_COMPLETED` / `TASK_STATE_FAILED` / `TASK_STATE_CANCELED` |
| `A2AStarletteApplication` (`a2a.server.apps`) | **Removed** | Module deleted; no drop-in replacement |
| `ServerError` (`a2a.utils.errors`) | **Removed** | No drop-in replacement |
| Various `*Response`, `*Params` RPC types | **Removed** | Protobuf request/response messages |

## Section 1: Core Infrastructure

### `executor.py` - No code changes needed

Lines 19-22 import `Part, Task, TaskState, TaskStatus` — all still valid in V1.0. No `TextPart` import. The code body already uses `Part(text=...)` (lines 94, 155, 232). Confirmed clean.

### `base_adapter.py` - Docstring-only update

The `from a2a.types import Part` import on line 25 is V1.0 clean. No code changes needed.

**Docstring fixes:**
- `invoke()` docstring (lines 166-181): Update examples from `Part(root=TextPart(text=...))` / `Part(root=FilePart(file=FileWithUri(...)))` to V1.0 `Part(text=...)` / `Part(url=..., filename=..., media_type=...)`. Remove references to `TextPart`, `FilePart`, `FileWithUri` imports.
- `stream()` docstring (lines 220-230): Same pattern.
- `invoke()` kwargs docstring (line 159): Replace "``FilePart``, ``DataPart``" with V1.0 equivalents ("parts with `url`, `raw`, or `data` fields").
- Line 150 docstring: "Extracted from A2A MessageSendParams" -> "Extracted from A2A SendMessageRequest".

### `server.py` - No changes needed

Already V1.0 compatible. All imports resolve.

## Section 2: V0.2 Adapter Fixes

### Module-level import unblock (crewai, langchain, langgraph, callable, openclaw)

**Problem:** These files have V0.2 and V0.1 classes in the same module. Module-level imports pull in `TextPart`, `MessageSendParams`, `Role`, etc. for V0.1 classes, which prevents the V0.2 class from loading at all.

**Solution:** Add `from __future__ import annotations` at the top of each mixed module. This defers annotation evaluation, so `MessageSendParams` in V0.1 method signatures becomes a string literal at class-definition time and won't trigger an `ImportError`. Then move the removed types (`TextPart`, `MessageSendParams`, `Role`, `TaskState`, `TaskStatus`, `Message`, `Task`) out of the module-level import and into a conditional block that's evaluated at V0.1 class call time only.

Pattern for each mixed module:
```python
from __future__ import annotations  # NEW: defer annotation evaluation

# V1.0-safe imports (used by V0.2 adapter)
from a2a.types import Part

# V0.1-only imports — deferred to avoid ImportError at module load.
# These types are removed in a2a-sdk V1.0; they're only needed when
# V0.1 classes are actually instantiated, not at module parse time.
try:
    from a2a.types import (
        Message, SendMessageRequest as MessageSendParams,
        Role, Task, TaskState, TaskStatus,
    )
except ImportError:
    Message = None  # type: ignore
    MessageSendParams = None  # type: ignore
    # ... etc.
```

**Note on `from __future__ import annotations`:** This is necessary because the V0.1 class method signatures use `MessageSendParams`, `Message`, `Task` as type annotations. Without postponed evaluation, Python evaluates these at class definition time, which crashes before any code runs. `from __future__ import annotations` makes all annotations strings by default, so the module loads cleanly. The renamed/aliased imports (`SendMessageRequest as MessageSendParams`) then only need to resolve when V0.1 code is actually called.

**Files:**
- `integrations/crewai.py` (lines 24-33)
- `integrations/langchain.py` (lines 23-30)
- `integrations/langgraph.py` (lines 24-33)
- `integrations/callable.py` (lines 17-24)
- `integrations/openclaw.py` (lines 24-37)

### `integrations/n8n.py` - Substantive V0.2 code changes

N8n is the only adapter where the V0.2 class body directly depends on removed types — not just a module-level import issue.

**Import restructure (lines 28-43):**
Same `from __future__ import annotations` + conditional pattern. But `Part` stays at module level (V1.0 safe). `RequestContext` stays (still valid). Remove `FilePart`, `FileWithUri`, `FileWithBytes`, `TextPart` entirely from V0.2 usage.

**`_build_multimodal_payload()` rewrite (lines 277-335):**
- `isinstance(part.root, FilePart)` -> detect file parts via proto Part oneof: `part.HasField('url') or part.HasField('raw')`
- `part.root.file.name` -> `part.filename`
- `part.root.file.mimeType` -> `part.media_type`
- `isinstance(file_part, FileWithUri) and file_part.uri` -> `bool(part.url)`
- Categorize by `part.media_type.startswith("image/")`

**`_fetch_file_content()` rewrite (lines 337-357):**
- Signature: accept `Part` instead of `FileWithUri | FileWithBytes`
- `part.url` -> HTTP fetch; `part.raw` -> return directly

**`_extract_response()` rewrite (lines 455-514):**
- `Part(root=TextPart(text=...))` -> `Part(text=...)`
- `Part(root=FilePart(file=FileWithUri(uri=..., name=..., mimeType=...)))` -> `Part(url=..., filename=..., media_type=...)`

### `integrations/openclaw.py` - V0.2 import unblock only

V0.2 `OpenClawAdapter.invoke()` returns `str` only — no multimodal Part construction in V0.2 path. Only needs the `from __future__ import annotations` + conditional import pattern.

### No changes needed

- `integrations/ollama.py` — no A2A type imports beyond `BaseA2AAdapter`
- `integrations/hermes.py` — same
- `integrations/claude_code.py` — V1.0 native
- `integrations/codex.py` — V1.0 native

## Section 3: V0.1 Legacy Compatibility

**Goal:** Keep V0.1 classes alive and callable in 0.2.9. Fix their internal usage of removed types so they work on V1.0. Formal removal deferred to 0.3.0.

### `adapter.py` (BaseAgentAdapter)

- Replace `from a2a.types import Message, MessageSendParams, Task` with V1.0 equivalents
- `MessageSendParams` -> `SendMessageRequest` (import alias: `from a2a.types import SendMessageRequest as MessageSendParams` for minimal churn in method signatures, or rename all signatures)
- **Decision: use import alias** `SendMessageRequest as MessageSendParams` to minimize body changes. Add deprecation comment.

### `client.py` - Graceful degradation, not compatibility fix

This module depends on `A2AStarletteApplication` (removed), `ServerError` (removed), and numerous `*Response`/`*Params` types (removed). It cannot be made functional on V1.0.

**Strategy:**
- Wrap the entire module body in a try/except at the top level
- On import failure, define `build_agent_app` and `serve_agent` as stubs that raise `RuntimeError` with a clear message: "client.py is deprecated and incompatible with a2a-sdk>=1.0. Use to_a2a(adapter) instead."
- This preserves `import a2a_adapter` (which lazy-loads `build_agent_app` via `__getattr__`), while giving a clear runtime error if someone actually tries to use it

### V0.1 adapter classes (in each mixed module)

For each V0.1 class body, update:
- `Part(root=TextPart(text=...))` -> `Part(text=...)`
- `Part(root=FilePart(file=FileWithUri(...)))` -> `Part(url=..., filename=..., media_type=...)`
- `Role.agent` -> `Role.ROLE_AGENT`; `Role.user` -> `Role.ROLE_USER`
- `TaskState.working` -> `TaskState.TASK_STATE_WORKING`; `.completed` -> `TASK_STATE_COMPLETED`; `.failed` -> `TASK_STATE_FAILED`; `.canceled` -> `TASK_STATE_CANCELED`
- `PushNotificationConfig` -> `TaskPushNotificationConfig` (openclaw V0.1 only)
- `isinstance(result.parts[0].root, TextPart)` -> check `result.parts[0].text`

**Files:**
- `integrations/crewai.py` — `CrewAIAgentAdapter`
- `integrations/langchain.py` — `LangChainAgentAdapter`
- `integrations/langgraph.py` — `LangGraphAgentAdapter`
- `integrations/callable.py` — `CallableAgentAdapter`
- `integrations/n8n.py` — `N8nAgentAdapter`
- `integrations/openclaw.py` — `OpenClawAgentAdapter`

### `loader.py`

- `_V1_BUILTIN_MAP` and `load_a2a_agent()` — no changes needed. They use lazy `importlib.import_module`, so broken types in target modules won't surface until instantiation. The V0.1 class fixes (above) handle the runtime.

## Section 4: Tests

### Group A: Shared fixtures

**`tests/conftest.py`:**
- Line 37: `getattr(c.root, "text", "")` -> `c.text if isinstance(c, str) else (c.text or "")` — Part is now proto, no `.root`
- Remove any `TextPart` import if present

### Group B: V0.2 core tests

**`tests/test_base_adapter.py`:**
- Remove `TextPart` import (line 6)
- `Part(root=TextPart(text="multi"))` -> `Part(text="multi")` (line 110)
- `result[0].root.text` -> `result[0].text` (line 119)

**`tests/test_executor.py`:**
- Remove `TextPart` import (line 11)
- All `Part(root=TextPart(text="..."))` -> `Part(text="...")` throughout
- All `part.root.text` accessors -> `part.text`
- TaskState `.value` comparison: V1.0 enum values are integers, not strings. `"failed" in states` (line 314) needs update to compare against `TaskState.TASK_STATE_FAILED` or its integer value.

**`tests/unit/test_v02_executor.py`:**
- Same pattern as test_executor.py — update Part construction and accessor

**`tests/unit/test_v02_core.py`, `test_v02_crewai.py`, `test_v02_langchain.py`, `test_v02_langgraph.py`, `test_v02_openclaw.py`, `test_v02_hermes.py`:**
- Update any residual `Part(root=TextPart(...))` patterns
- Should be lighter since V0.2 adapters mostly return `str`

**`tests/integration/test_v02_integration.py`:**
- Verify no stale type usage (likely clean since V0.2 adapters return strings)

### Group C: V0.1 legacy tests

Every V0.1 unit test file is built on the removed types. Each needs a full migration pass:

**Shared pattern across all files:**
- `from a2a.types import MessageSendParams` -> `from a2a.types import SendMessageRequest`
- `from a2a.types import TextPart` -> remove
- `from a2a.types import Role` -> keep (but update values)
- Helper `make_message_send_params()` -> rewrite: `SendMessageRequest(message=Message(role=Role.ROLE_USER, parts=[Part(text=...)]))` 
- All `Part(root=TextPart(text=...))` -> `Part(text=...)`
- All `Role.agent` -> `Role.ROLE_AGENT`, `Role.user` -> `Role.ROLE_USER`
- All `TaskState.working` -> `TaskState.TASK_STATE_WORKING`, etc.
- All `isinstance(x.root, TextPart)` assertions -> check `x.text`
- All `x.root.text` accessors -> `x.text`

**Files:**
- `tests/unit/test_adapter.py` — BaseAgentAdapter tests
- `tests/unit/test_n8n_adapter.py` — N8nAgentAdapter tests
- `tests/unit/test_crewai_adapter.py` — CrewAIAgentAdapter tests
- `tests/unit/test_langchain_adapter.py` — LangChainAgentAdapter tests
- `tests/unit/test_langgraph_adapter.py` — LangGraphAgentAdapter tests
- `tests/unit/test_openclaw_adapter.py` — OpenClawAgentAdapter tests
- `tests/unit/test_callable_adapter.py` — CallableAgentAdapter tests

**`tests/integration/test_a2a_compatibility.py`:**
- Full rewrite: `MessageSendParams` -> `SendMessageRequest`, Part construction, Role values, TextPart assertions

## Section 5: Version & Packaging

**Fix version drift:**
- `a2a_adapter/__init__.py` line 15: `__version__ = "0.2.0"` -> `"0.2.9"` (currently drifted from pyproject.toml which is `0.2.8`)
- `pyproject.toml` line 7: `version = "0.2.8"` -> `"0.2.9"`

**Keep V0.1 exports alive in `__init__.py`:**
- `BaseAgentAdapter` stays in `__all__` and `__getattr__` lazy map
- `build_agent_app` stays in `__getattr__` (will raise RuntimeError on use, per Section 3)
- `load_a2a_agent` stays as deprecated re-export

## Validation Criteria

1. `import a2a_adapter` succeeds
2. `from a2a_adapter import N8nAdapter, CrewAIAdapter, LangChainAdapter, ...` — all V0.2 adapters load
3. `from a2a_adapter import BaseAgentAdapter` — loads (deprecated, but alive)
4. `from a2a_adapter import build_agent_app` — loads but raises RuntimeError on call
5. All V0.2 adapter `invoke()` / `stream()` work end-to-end
6. N8n multimodal mode works with V1.0 Part proto fields
7. `uv run pytest` passes all tests
8. No `TextPart`, `FilePart`, `FileWithUri`, `FileWithBytes`, `MessageSendParams` appear in any non-test, non-deprecated code path
