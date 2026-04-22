# A2A SDK V1.0 Compatibility Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all adapters loadable and functional on `a2a-sdk>=1,<2`. Preserve V0.1 deprecated classes. Bump version to 0.2.9.

**Architecture:** The SDK moved from Pydantic models to protobuf-generated types. We fix imports, enum values, Part construction, and Part accessors across source and test files. V0.1 legacy classes stay alive via `from __future__ import annotations` + alias imports for truly removed types; V1.0-surviving types stay at module level.

**Tech Stack:** Python 3.11+, a2a-sdk>=1.0, pytest, pytest-asyncio

**Spec:** `docs/superpowers/specs/2026-04-21-a2a-sdk-v1-compat-design.md`

---

## File Map

### Source files to modify:
- `a2a_adapter/__init__.py` — version bump
- `pyproject.toml` — version bump
- `a2a_adapter/base_adapter.py` — docstring-only
- `a2a_adapter/adapter.py` — import alias for V0.1 base class
- `a2a_adapter/client.py` — graceful degradation wrapper
- `a2a_adapter/integrations/callable.py` — import restructure + V0.1 body fixes
- `a2a_adapter/integrations/crewai.py` — import restructure + V0.1 body fixes
- `a2a_adapter/integrations/langchain.py` — import restructure + V0.1 body fixes
- `a2a_adapter/integrations/langgraph.py` — import restructure + V0.1 body fixes
- `a2a_adapter/integrations/n8n.py` — import restructure + V0.2 code rewrite + V0.1 body fixes
- `a2a_adapter/integrations/openclaw.py` — import restructure + V0.1 body fixes

### Test files to modify:
- `tests/conftest.py` — fix `.root` accessor
- `tests/test_base_adapter.py` — remove TextPart, fix Part construction/accessor
- `tests/test_callable.py` — remove unused TextPart import
- `tests/test_executor.py` — remove TextPart, fix Part construction/accessors/enums
- `tests/unit/test_v02_n8n_multimodal.py` — **NEW**: targeted tests for V0.2 N8n multimodal rewrite
- `tests/unit/test_adapter.py` — full V0.1 test migration
- `tests/unit/test_callable_adapter.py` — full V0.1 test migration
- `tests/unit/test_crewai_adapter.py` — full V0.1 test migration
- `tests/unit/test_langchain_adapter.py` — full V0.1 test migration
- `tests/unit/test_langgraph_adapter.py` — full V0.1 test migration
- `tests/unit/test_n8n_adapter.py` — full V0.1 test migration
- `tests/unit/test_openclaw_adapter.py` — full V0.1 test migration
- `tests/integration/test_a2a_compatibility.py` — full migration

### No code changes needed (docstring cleanup only):
- `a2a_adapter/executor.py` — runtime code already V1.0 clean; docstring on line 162 mentions "TextPart, FilePart" — cleaned up in Task 2
- `a2a_adapter/server.py` — already V1.0 clean
- `a2a_adapter/loader.py` — uses lazy importlib, no changes needed
- `a2a_adapter/integrations/ollama.py` — no A2A type imports
- `a2a_adapter/integrations/hermes.py` — no A2A type imports
- `a2a_adapter/integrations/claude_code.py` — V1.0 native
- `a2a_adapter/integrations/codex.py` — V1.0 native

---

## Task 1: Version bump

**Files:**
- Modify: `pyproject.toml:7`
- Modify: `a2a_adapter/__init__.py:15`

- [ ] **Step 1: Bump pyproject.toml version**

```python
# In pyproject.toml, line 7
# Old:
version = "0.2.8"
# New:
version = "0.2.9"
```

- [ ] **Step 2: Fix __init__.py version drift**

```python
# In a2a_adapter/__init__.py, line 15
# Old:
__version__ = "0.2.0"
# New:
__version__ = "0.2.9"
```

- [ ] **Step 3: Verify import**

Run: `cd "/Users/caijiangnan/Desktop/Hybro/hybro open source/a2a-adapters" && uv run python -c "import a2a_adapter; print(a2a_adapter.__version__)"`
Expected: `0.2.9`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml a2a_adapter/__init__.py
git commit -m "chore: bump version to 0.2.9 and fix __init__.py version drift"
```

---

## Task 2: Core infrastructure — adapter.py, client.py, base_adapter.py

**Files:**
- Modify: `a2a_adapter/adapter.py:18`
- Modify: `a2a_adapter/client.py` (full file rewrite)
- Modify: `a2a_adapter/base_adapter.py` (docstrings only)

- [ ] **Step 1: Fix adapter.py import**

In `a2a_adapter/adapter.py`, line 18, replace the import:

```python
# Old:
from a2a.types import Message, MessageSendParams, Task

# New:
from a2a.types import Message, Task
from a2a.types import SendMessageRequest as MessageSendParams  # V1.0 alias; removed in 0.3.0
```

- [ ] **Step 2: Fix adapter.py extract_raw_input — docstring + code**

**2a. Docstring cleanup** — In `a2a_adapter/adapter.py`, lines 66-81, the `extract_raw_input()` docstring references removed types. Replace:
```python
        """
        Extract raw text content from A2A message parameters.
        
        This utility method handles:
        - New format: message.parts with Part(root=TextPart(...)) structure
        - Legacy format: messages array (deprecated)
        - Edge case: part.root.text returning dict instead of str
        
        All adapters can use this method to extract user input consistently.
        
        Args:
            params: A2A message parameters
            
        Returns:
            Extracted text as string
        """
```
with:
```python
        """
        Extract raw text content from A2A message parameters.
        
        This utility method handles:
        - New format: message.parts with Part(text=...) structure
        - Legacy format: messages array (deprecated)
        - Edge case: part.text returning dict instead of str
        
        All adapters can use this method to extract user input consistently.
        
        Args:
            params: A2A message parameters
            
        Returns:
            Extracted text as string
        """
```

**2b. Code cleanup** — In the same method, lines 89-108, replace the part iteration block:

```python
                for part in msg.parts:
                    # Handle Part(root=TextPart(...)) structure
                    if hasattr(part, "root") and hasattr(part.root, "text"):
                        text_value = part.root.text
                        # Handle dict type - convert to JSON string
                        if isinstance(text_value, dict):
                            text_parts.append(json.dumps(text_value, ensure_ascii=False))
                        elif isinstance(text_value, str):
                            text_parts.append(text_value)
                        else:
                            text_parts.append(str(text_value))
                    # Handle direct TextPart
                    elif hasattr(part, "text"):
                        text_value = part.text
                        if isinstance(text_value, dict):
                            text_parts.append(json.dumps(text_value, ensure_ascii=False))
                        elif isinstance(text_value, str):
                            text_parts.append(text_value)
                        else:
                            text_parts.append(str(text_value))
```

with:

```python
                for part in msg.parts:
                    text_value = getattr(part, "text", None)
                    if text_value is not None:
                        if isinstance(text_value, dict):
                            text_parts.append(json.dumps(text_value, ensure_ascii=False))
                        elif isinstance(text_value, str):
                            text_parts.append(text_value)
                        else:
                            text_parts.append(str(text_value))
```

- [ ] **Step 3: Wrap client.py in try/except for graceful degradation**

This is a structural edit, not a full rewrite. The original module body (classes and functions) stays verbatim — only the wrapping changes.

**3a.** At the very top of the file, keep the module docstring (lines 1-12) unchanged.

**3b.** After the docstring, wrap everything from `import warnings` (line 14) through the end of the file in a `try:` block. Indent the entire existing body by 4 spaces. This means lines 14-284 (all imports, `AdapterRequestHandler` class, `build_agent_app()`, and `serve_agent()`) become the body of `try:`.

**3c.** After the `try:` block, add an `except ImportError:` with stub functions:

```python
except ImportError:

    def build_agent_app(*args, **kwargs):
        raise RuntimeError(
            "client.py is deprecated and incompatible with a2a-sdk>=1.0. "
            "Use to_a2a(adapter) instead. See: "
            "https://github.com/hybro-ai/a2a-adapter/blob/main/docs/migration-v0.2.md"
        )

    def serve_agent(*args, **kwargs):
        raise RuntimeError(
            "client.py serve_agent() is deprecated and incompatible with a2a-sdk>=1.0. "
            "Use a2a_adapter.server.serve_agent(adapter) instead. See: "
            "https://github.com/hybro-ai/a2a-adapter/blob/main/docs/migration-v0.2.md"
        )
```

**The result:** module docstring at top level, `try:` containing the entire original body (unchanged except indentation), `except ImportError:` with two stubs. No class or function definitions are removed or modified.

- [ ] **Step 4: Update base_adapter.py docstrings**

In `a2a_adapter/base_adapter.py`, update the `invoke()` docstring (around lines 150-181):

Replace:
```python
                Extracted from A2A MessageSendParams by the bridge layer
```
with:
```python
                Extracted from A2A SendMessageRequest by the bridge layer
```

Replace:
```python
                to the full message including non-text parts (``FilePart``,
                ``DataPart``, etc.). Access via ``kwargs.get('context')``.
```
with:
```python
                to the full message including non-text parts (parts with
                ``url``, ``raw``, or ``data`` fields). Access via
                ``kwargs.get('context')``.
```

Replace the multimodal example:
```python
            # Multimodal response with text and file
            from a2a.types import Part, TextPart, FilePart, FileWithUri
            async def invoke(self, user_input, context_id=None, **kwargs):
                return [
                    Part(root=TextPart(text="Generated report")),
                    Part(root=FilePart(file=FileWithUri(
                        uri="http://example.com/report.pdf",
                        name="report.pdf",
                        mimeType="application/pdf"
                    )))
                ]
```
with:
```python
            # Multimodal response with text and file
            from a2a.types import Part
            async def invoke(self, user_input, context_id=None, **kwargs):
                return [
                    Part(text="Generated report"),
                    Part(url="http://example.com/report.pdf",
                         filename="report.pdf",
                         media_type="application/pdf")
                ]
```

Similarly update the `stream()` docstring (around lines 223-231):
```python
            # Multimodal streaming
            async def stream(self, user_input, context_id=None, **kwargs):
                from a2a.types import Part, TextPart, FilePart, FileWithUri
                yield "Generating chart..."
                yield Part(root=FilePart(file=FileWithUri(
                    uri="http://example.com/chart.png",
                    name="chart.png",
                    mimeType="image/png"
                )))
```
with:
```python
            # Multimodal streaming
            async def stream(self, user_input, context_id=None, **kwargs):
                from a2a.types import Part
                yield "Generating chart..."
                yield Part(url="http://example.com/chart.png",
                           filename="chart.png",
                           media_type="image/png")
```

- [ ] **Step 5: Clean up executor.py docstring**

In `a2a_adapter/executor.py`, line 162, replace:
```python
        """Extract text content from multimodal parts for completion message.

        Args:
            parts: List of Part objects (may contain TextPart, FilePart, etc.)

        Returns:
            Concatenated text from all TextPart objects, or a placeholder
            if no text parts are found.
        """
```
with:
```python
        """Extract text content from multimodal parts for completion message.

        Args:
            parts: List of Part objects (may contain text, file, or data parts).

        Returns:
            Concatenated text from all text parts, or a placeholder
            if no text parts are found.
        """
```

- [ ] **Step 6: Verify imports**

Run: `uv run python -c "from a2a_adapter.adapter import BaseAgentAdapter; print('adapter.py OK')"`
Run: `uv run python -c "from a2a_adapter.client import build_agent_app; print('client.py OK')"`
Expected: Both print OK (client.py may print OK via the stub that raises on call, or via the original if SDK happens to have the old types).

- [ ] **Step 7: Commit**

```bash
git add a2a_adapter/adapter.py a2a_adapter/client.py a2a_adapter/base_adapter.py a2a_adapter/executor.py
git commit -m "fix: V1.0 compat for core infrastructure (adapter.py, client.py, base_adapter.py, executor.py docstring)"
```

---

## Task 3: Import restructure — callable.py

**Files:**
- Modify: `a2a_adapter/integrations/callable.py:1-26` (imports)
- Modify: `a2a_adapter/integrations/callable.py` (V0.1 class body)

- [ ] **Step 1: Add `from __future__ import annotations` and restructure imports**

Replace lines 1-26 of `a2a_adapter/integrations/callable.py`:

```python
"""
Generic callable adapter for A2A Protocol.

This adapter allows any async Python function to be exposed as an A2A-compliant
agent, providing maximum flexibility for custom implementations.

Contains:
    - CallableAdapter (v0.2): New simplified interface based on BaseA2AAdapter
    - CallableAgentAdapter (v0.1): Legacy interface, deprecated
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, AsyncIterator, Callable, Dict

from a2a.types import (
    Message,
    Part,
    Role,
    Task,
)

try:
    from a2a.types import TextPart
except ImportError:
    TextPart = None  # type: ignore

try:
    from a2a.types import SendMessageRequest as MessageSendParams
except ImportError:
    MessageSendParams = None  # type: ignore

from ..adapter import BaseAgentAdapter
from ..base_adapter import BaseA2AAdapter, AdapterMetadata
```

- [ ] **Step 2: Fix V0.1 class body — Part construction and Role enum**

In `CallableAgentAdapter.from_framework()` (around line 304-309), replace:
```python
        return Message(
            role=Role.agent,
            message_id=str(uuid.uuid4()),
            context_id=context_id,
            parts=[Part(root=TextPart(text=response_text))],
        )
```
with:
```python
        return Message(
            role=Role.ROLE_AGENT,
            message_id=str(uuid.uuid4()),
            context_id=context_id,
            parts=[Part(text=response_text)],
        )
```

In `CallableAgentAdapter.handle_stream()` (around line 383-388), replace:
```python
        final_message = Message(
            role=Role.agent,
            message_id=message_id,
            context_id=context_id,
            parts=[Part(root=TextPart(text=accumulated_text))],
        )
```
with:
```python
        final_message = Message(
            role=Role.ROLE_AGENT,
            message_id=message_id,
            context_id=context_id,
            parts=[Part(text=accumulated_text)],
        )
```

Also fix `to_framework()` (around line 217-219) — replace the `.root` accessor:
```python
                    if hasattr(part, "root") and hasattr(part.root, "text"):
                        text_parts.append(part.root.text)
                    # Handle direct TextPart
                    elif hasattr(part, "text"):
                        text_parts.append(part.text)
```
with:
```python
                    text_value = getattr(part, "text", None)
                    if text_value is not None:
                        text_parts.append(text_value)
```

- [ ] **Step 3: Verify import**

Run: `uv run python -c "from a2a_adapter.integrations.callable import CallableAdapter, CallableAgentAdapter; print('callable OK')"`
Expected: `callable OK`

- [ ] **Step 4: Commit**

```bash
git add a2a_adapter/integrations/callable.py
git commit -m "fix: V1.0 compat for callable adapter — import restructure + V0.1 body fixes"
```

---

## Task 4: Import restructure — crewai.py

**Files:**
- Modify: `a2a_adapter/integrations/crewai.py`

- [ ] **Step 1: Add `from __future__ import annotations` and restructure imports**

Replace lines 1-33 of `a2a_adapter/integrations/crewai.py` imports section (keep the module docstring, add future annotations after it):

After the module docstring (line 15), add `from __future__ import annotations` as the first import.

Then replace the `a2a.types` import block (lines 24-33):
```python
from a2a.types import (
    Message,
    MessageSendParams,
    Part,
    Role,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)
```
with:
```python
from a2a.types import (
    Message,
    Part,
    Role,
    Task,
    TaskState,
    TaskStatus,
)

try:
    from a2a.types import TextPart
except ImportError:
    TextPart = None  # type: ignore

try:
    from a2a.types import SendMessageRequest as MessageSendParams
except ImportError:
    MessageSendParams = None  # type: ignore
```

- [ ] **Step 2: Fix ALL `Part(root=TextPart(...))` in V0.1 class body**

Search for every `Part(root=TextPart(text=` in the file and replace with `Part(text=`. There are instances in:
- `_handle_sync` (~line 347): `Part(root=TextPart(text="Crew completed"))` → `Part(text="Crew completed")`
- `_execute_crew_with_timeout` (~line 427): `Part(root=TextPart(text=f"Crew timed out..."))` → `Part(text=f"Crew timed out...")`
- `_execute_crew_background` (~line 465): `Part(root=TextPart(text=response_text))` → `Part(text=response_text)`
- `_execute_crew_background` (~line 505): `Part(root=TextPart(text=f"Crew failed: {str(e)}"))` → `Part(text=f"Crew failed: {str(e)}")`
- `from_framework` (~line 637): `Part(root=TextPart(text=response_text))` → `Part(text=response_text)`

- [ ] **Step 3: Fix ALL Role and TaskState enum values in V0.1 class body**

Replace all occurrences:
- `Role.agent` → `Role.ROLE_AGENT` (in `_handle_sync`, `_execute_crew_with_timeout`, `_execute_crew_background`, `from_framework`)
- `TaskState.working` → `TaskState.TASK_STATE_WORKING` (in `_handle_async`)
- `TaskState.completed` → `TaskState.TASK_STATE_COMPLETED` (in `_execute_crew_background`)
- `TaskState.failed` → `TaskState.TASK_STATE_FAILED` (in `_execute_crew_with_timeout`, `_execute_crew_background`)
- `TaskState.canceled` → `TaskState.TASK_STATE_CANCELED` (in `cancel_task`)

- [ ] **Step 4: Verify import**

Run: `uv run python -c "from a2a_adapter.integrations.crewai import CrewAIAdapter, CrewAIAgentAdapter; print('crewai OK')"`
Expected: `crewai OK`

- [ ] **Step 5: Commit**

```bash
git add a2a_adapter/integrations/crewai.py
git commit -m "fix: V1.0 compat for crewai adapter — import restructure + V0.1 body fixes"
```

---

## Task 5: Import restructure — langchain.py

**Files:**
- Modify: `a2a_adapter/integrations/langchain.py`

- [ ] **Step 1: Add `from __future__ import annotations` and restructure imports**

After the module docstring (line 15), add `from __future__ import annotations`.

Replace the `a2a.types` import block (lines 23-30):
```python
from a2a.types import (
    Message,
    MessageSendParams,
    Part,
    Role,
    Task,
    TextPart,
)
```
with:
```python
from a2a.types import (
    Message,
    Part,
    Role,
    Task,
)

try:
    from a2a.types import TextPart
except ImportError:
    TextPart = None  # type: ignore

try:
    from a2a.types import SendMessageRequest as MessageSendParams
except ImportError:
    MessageSendParams = None  # type: ignore
```

- [ ] **Step 2: Fix V0.1 class body — Part construction and Role enum**

Replace all `Part(root=TextPart(text=...))` with `Part(text=...)`:
- `from_framework` (~line 417): `Part(root=TextPart(text=response_text))` → `Part(text=response_text)`
- `handle_stream` (~line 507): `Part(root=TextPart(text=accumulated_text))` → `Part(text=accumulated_text)`

Replace all `Role.agent` with `Role.ROLE_AGENT`:
- `from_framework` (~line 413)
- `handle_stream` (~line 503)

- [ ] **Step 3: Verify and commit**

Run: `uv run python -c "from a2a_adapter.integrations.langchain import LangChainAdapter, LangChainAgentAdapter; print('langchain OK')"`

```bash
git add a2a_adapter/integrations/langchain.py
git commit -m "fix: V1.0 compat for langchain adapter — import restructure + V0.1 body fixes"
```

---

## Task 6: Import restructure — langgraph.py

**Files:**
- Modify: `a2a_adapter/integrations/langgraph.py`

- [ ] **Step 1: Add `from __future__ import annotations` and restructure imports**

After the module docstring (line 15), add `from __future__ import annotations`.

Replace the `a2a.types` import block (lines 24-33):
```python
from a2a.types import (
    Message,
    MessageSendParams,
    Part,
    Role,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)
```
with:
```python
from a2a.types import (
    Message,
    Part,
    Role,
    Task,
    TaskState,
    TaskStatus,
)

try:
    from a2a.types import TextPart
except ImportError:
    TextPart = None  # type: ignore

try:
    from a2a.types import SendMessageRequest as MessageSendParams
except ImportError:
    MessageSendParams = None  # type: ignore
```

- [ ] **Step 2: Fix ALL Part construction, Role, and TaskState in V0.1 class body**

Replace all `Part(root=TextPart(text=...))` with `Part(text=...)` — instances in:
- `_handle_sync` (~line 447)
- `_execute_workflow_with_timeout` (~line 525)
- `_execute_workflow_background` (~line 564)
- `_execute_workflow_background` error handler (~line 604)
- `from_framework` (~line 735)
- `handle_stream` (~line 872)

Replace all `Role.agent` with `Role.ROLE_AGENT`.
Replace all `TaskState.working` with `TaskState.TASK_STATE_WORKING`.
Replace all `TaskState.completed` with `TaskState.TASK_STATE_COMPLETED`.
Replace all `TaskState.failed` with `TaskState.TASK_STATE_FAILED`.
Replace all `TaskState.canceled` with `TaskState.TASK_STATE_CANCELED`.

- [ ] **Step 3: Verify and commit**

Run: `uv run python -c "from a2a_adapter.integrations.langgraph import LangGraphAdapter, LangGraphAgentAdapter; print('langgraph OK')"`

```bash
git add a2a_adapter/integrations/langgraph.py
git commit -m "fix: V1.0 compat for langgraph adapter — import restructure + V0.1 body fixes"
```

---

## Task 7: N8n adapter — V0.2 code rewrite + import restructure + V0.1 fixes

This is the most complex task. N8n is the only adapter where the V0.2 class body directly uses removed types.

**Files:**
- Modify: `a2a_adapter/integrations/n8n.py`

- [ ] **Step 1: Add `from __future__ import annotations` and restructure imports**

After the module docstring (line 15), add `from __future__ import annotations`.

Replace the `a2a.types` import block (lines 30-43):
```python
from a2a.types import (
    Artifact,
    FilePart,
    FileWithBytes,
    FileWithUri,
    Message,
    MessageSendParams,
    Part,
    Role,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)
```
with:
```python
from a2a.types import (
    Artifact,
    Message,
    Part,
    Role,
    Task,
    TaskState,
    TaskStatus,
)

try:
    from a2a.types import TextPart
except ImportError:
    TextPart = None  # type: ignore

try:
    from a2a.types import FilePart, FileWithBytes, FileWithUri
except ImportError:
    FilePart = None  # type: ignore
    FileWithBytes = None  # type: ignore
    FileWithUri = None  # type: ignore

try:
    from a2a.types import SendMessageRequest as MessageSendParams
except ImportError:
    MessageSendParams = None  # type: ignore
```

- [ ] **Step 2: Rewrite V0.2 `_build_multimodal_payload()` (~lines 277-335)**

Replace the entire method body with proto Part field detection:

```python
    async def _build_multimodal_payload(
        self,
        user_input: str,
        context_id: str | None,
        context: RequestContext,
    ) -> Dict[str, Any]:
        """Build webhook payload with files/images from context.message.parts."""
        payload = self._build_payload(user_input, context_id)

        files = []
        images = []

        for part in context.message.parts:
            if part.HasField("url") or part.HasField("raw"):
                try:
                    file_data = await self._fetch_file_content(part)
                    file_entry: Dict[str, Any] = {
                        "name": part.filename or "file",
                        "mime_type": part.media_type,
                    }

                    if self.encode_files_base64:
                        file_entry["data"] = base64.b64encode(file_data).decode("utf-8")
                    else:
                        file_entry["data"] = file_data

                    if part.HasField("url"):
                        file_entry["uri"] = part.url

                    if part.media_type and part.media_type.startswith("image/"):
                        images.append(file_entry)
                    else:
                        files.append(file_entry)

                except Exception as e:
                    logger.warning(
                        "Failed to fetch file %s: %s", part.filename or "unknown", e
                    )

        if files:
            payload[self.file_field] = files
        if images:
            payload[self.image_field] = images

        return payload
```

- [ ] **Step 3: Rewrite V0.2 `_fetch_file_content()` (~lines 337-357)**

Replace the entire method:

```python
    async def _fetch_file_content(self, part: Part) -> bytes:
        """Fetch file content from URL or return embedded raw data.

        Args:
            part: A2A Part with url or raw field set.

        Returns:
            File content as bytes.
        """
        if part.HasField("url"):
            client = await self._get_client()
            resp = await client.get(part.url)
            resp.raise_for_status()
            return resp.content
        elif part.HasField("raw"):
            return part.raw
        else:
            raise ValueError(f"Part has no url or raw data: {part}")
```

- [ ] **Step 4: Rewrite V0.2 `_extract_response()` (~lines 455-514)**

Replace the entire method:

```python
    def _extract_response(self, output: Any) -> list[Part]:
        """Extract response with multimodal content detection."""
        if not isinstance(output, dict):
            text = self._extract_response_text(output)
            return [Part(text=text)]

        parts: list[Part] = []

        text = self._extract_text_from_item(output)
        if text:
            parts.append(Part(text=text))

        if self.file_field in output:
            for file_ref in output[self.file_field]:
                if isinstance(file_ref, dict):
                    parts.append(
                        Part(
                            url=file_ref.get("url") or file_ref.get("uri"),
                            filename=file_ref.get("name"),
                            media_type=file_ref.get("mime_type")
                            or file_ref.get("mimeType"),
                        )
                    )

        if self.image_field in output:
            for img_ref in output[self.image_field]:
                if isinstance(img_ref, dict):
                    parts.append(
                        Part(
                            url=img_ref.get("url") or img_ref.get("uri"),
                            filename=img_ref.get("name"),
                            media_type=img_ref.get("mime_type")
                            or img_ref.get("mimeType", "image/png"),
                        )
                    )

        if not parts:
            return [Part(text="[Empty response]")]
        return parts
```

- [ ] **Step 5: Fix V0.1 N8nAgentAdapter class body**

Replace all `Part(root=TextPart(text=...))` with `Part(text=...)` throughout the V0.1 class. Instances in:
- `_handle_sync` (~line 665): `Part(root=TextPart(text="Task completed"))` → `Part(text="Task completed")`
- `_execute_workflow_with_timeout` (~line 758): timeout message
- `_execute_workflow_background` (~line 801): response message
- `_execute_workflow_background` (~line 825): artifact parts
- `_execute_workflow_background` error (~line 851): error message
- `from_framework` (~line 1069): `Part(root=TextPart(text=response_text))` → `Part(text=response_text)`

Replace all `Role.agent` → `Role.ROLE_AGENT`.
Replace all `TaskState.working` → `TaskState.TASK_STATE_WORKING`.
Replace all `TaskState.completed` → `TaskState.TASK_STATE_COMPLETED`.
Replace all `TaskState.failed` → `TaskState.TASK_STATE_FAILED`.
Replace all `TaskState.canceled` → `TaskState.TASK_STATE_CANCELED`.

Also fix the `delete_task` terminal_states set (~line 1165):
```python
        terminal_states = {TaskState.completed, TaskState.failed, TaskState.canceled}
```
→
```python
        terminal_states = {TaskState.TASK_STATE_COMPLETED, TaskState.TASK_STATE_FAILED, TaskState.TASK_STATE_CANCELED}
```

And fix the `.value` reference in the error message on that same block:
```python
                f"Only tasks in terminal states ({', '.join(s.value for s in terminal_states)}) can be deleted."
```
→
```python
                f"Only tasks in terminal states (completed, failed, canceled) can be deleted."
```

- [ ] **Step 6: Verify import**

Run: `uv run python -c "from a2a_adapter.integrations.n8n import N8nAdapter, N8nAgentAdapter; print('n8n OK')"`

- [ ] **Step 7: Commit**

```bash
git add a2a_adapter/integrations/n8n.py
git commit -m "fix: V1.0 compat for n8n adapter — V0.2 multimodal rewrite + import restructure + V0.1 fixes"
```

---

## Task 7b: Targeted tests for N8n V0.2 multimodal rewrite

This is the riskiest behavior change in the sweep — the only V0.2 runtime code rewrite. The existing repo has no `test_v02_n8n.py`. We add targeted tests for the three rewritten methods.

**Files:**
- Create: `tests/unit/test_v02_n8n_multimodal.py`

- [ ] **Step 1: Write tests for `_build_multimodal_payload()`**

```python
"""Tests for N8nAdapter V0.2 multimodal methods after V1.0 proto migration."""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from a2a.types import Part

from a2a_adapter.integrations.n8n import N8nAdapter


class TestBuildMultimodalPayload:
    """Test _build_multimodal_payload with V1.0 proto Part fields."""

    @pytest.fixture
    def adapter(self):
        return N8nAdapter(
            webhook_url="http://localhost:5678/webhook/test",
            multimodal_mode=True,
        )

    async def test_text_only_parts_ignored(self, adapter):
        """Parts with only text set should not appear in files/images."""
        ctx = MagicMock()
        text_part = Part(text="hello")
        ctx.message.parts = [text_part]

        payload = await adapter._build_multimodal_payload("hello", "ctx-1", ctx)
        assert "files" not in payload
        assert "images" not in payload

    async def test_url_part_detected_as_file(self, adapter):
        """A Part with url set should be fetched and added to files."""
        ctx = MagicMock()
        file_part = Part(url="http://example.com/doc.pdf", filename="doc.pdf", media_type="application/pdf")
        ctx.message.parts = [file_part]

        mock_resp = MagicMock()
        mock_resp.content = b"pdf-bytes"
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.is_closed = False
        adapter._client = mock_client

        payload = await adapter._build_multimodal_payload("check this", "ctx-1", ctx)
        assert "files" in payload
        assert len(payload["files"]) == 1
        assert payload["files"][0]["name"] == "doc.pdf"
        assert payload["files"][0]["mime_type"] == "application/pdf"
        assert payload["files"][0]["data"] == base64.b64encode(b"pdf-bytes").decode("utf-8")
        assert payload["files"][0]["uri"] == "http://example.com/doc.pdf"

    async def test_image_categorized_separately(self, adapter):
        """A Part with image/* media_type should go to images, not files."""
        ctx = MagicMock()
        img_part = Part(url="http://example.com/photo.png", filename="photo.png", media_type="image/png")
        ctx.message.parts = [img_part]

        mock_resp = MagicMock()
        mock_resp.content = b"png-bytes"
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.is_closed = False
        adapter._client = mock_client

        payload = await adapter._build_multimodal_payload("look at this", "ctx-1", ctx)
        assert "images" in payload
        assert "files" not in payload
        assert payload["images"][0]["name"] == "photo.png"

    async def test_raw_part_no_fetch(self, adapter):
        """A Part with raw bytes should use data directly, not HTTP fetch."""
        ctx = MagicMock()
        raw_part = Part(raw=b"raw-content", filename="data.bin", media_type="application/octet-stream")
        ctx.message.parts = [raw_part]

        payload = await adapter._build_multimodal_payload("process", "ctx-1", ctx)
        assert "files" in payload
        assert payload["files"][0]["data"] == base64.b64encode(b"raw-content").decode("utf-8")


class TestFetchFileContent:
    """Test _fetch_file_content with V1.0 proto Part fields."""

    @pytest.fixture
    def adapter(self):
        return N8nAdapter(webhook_url="http://localhost:5678/webhook/test")

    async def test_url_part_fetches(self, adapter):
        part = Part(url="http://example.com/file.bin")
        mock_resp = MagicMock()
        mock_resp.content = b"fetched"
        mock_resp.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.is_closed = False
        adapter._client = mock_client

        result = await adapter._fetch_file_content(part)
        assert result == b"fetched"

    async def test_raw_part_returns_directly(self, adapter):
        part = Part(raw=b"inline-data")
        result = await adapter._fetch_file_content(part)
        assert result == b"inline-data"

    async def test_empty_part_raises(self, adapter):
        part = Part(text="just text")
        with pytest.raises(ValueError, match="no url or raw"):
            await adapter._fetch_file_content(part)


class TestExtractResponse:
    """Test _extract_response with V1.0 Part construction."""

    @pytest.fixture
    def adapter(self):
        return N8nAdapter(
            webhook_url="http://localhost:5678/webhook/test",
            multimodal_mode=True,
        )

    def test_text_only_response(self, adapter):
        output = {"output": "hello world"}
        parts = adapter._extract_response(output)
        assert len(parts) == 1
        assert parts[0].text == "hello world"

    def test_with_files(self, adapter):
        output = {
            "output": "here are results",
            "files": [{"url": "http://example.com/report.pdf", "name": "report.pdf", "mime_type": "application/pdf"}],
        }
        parts = adapter._extract_response(output)
        assert len(parts) == 2
        assert parts[0].text == "here are results"
        assert parts[1].url == "http://example.com/report.pdf"
        assert parts[1].filename == "report.pdf"
        assert parts[1].media_type == "application/pdf"

    def test_with_images(self, adapter):
        output = {
            "output": "chart generated",
            "images": [{"url": "http://example.com/chart.png", "name": "chart.png", "mimeType": "image/png"}],
        }
        parts = adapter._extract_response(output)
        assert len(parts) == 2
        assert parts[1].url == "http://example.com/chart.png"
        assert parts[1].media_type == "image/png"

    def test_empty_response(self, adapter):
        parts = adapter._extract_response({})
        assert len(parts) == 1
        assert parts[0].text == "[Empty response]"

    def test_non_dict_response(self, adapter):
        parts = adapter._extract_response([{"output": "from list"}])
        assert len(parts) == 1
        assert "from list" in parts[0].text
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/unit/test_v02_n8n_multimodal.py -v`
Expected: All tests pass (depends on Task 7 being done first)

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_v02_n8n_multimodal.py
git commit -m "test: add targeted tests for N8n V0.2 multimodal rewrite (V1.0 proto Part fields)"
```

---

## Task 8: Import restructure — openclaw.py

**Files:**
- Modify: `a2a_adapter/integrations/openclaw.py`

- [ ] **Step 1: Add `from __future__ import annotations` and restructure imports**

After the module docstring (line 10), add `from __future__ import annotations`.

Replace the `a2a.types` import block (lines 24-37):
```python
from a2a.types import (
    Artifact,
    FilePart,
    FileWithUri,
    Message,
    MessageSendParams,
    Part,
    PushNotificationConfig,
    Role,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)
```
with:
```python
from a2a.types import (
    Artifact,
    Message,
    Part,
    Role,
    Task,
    TaskState,
    TaskStatus,
)

try:
    from a2a.types import TextPart
except ImportError:
    TextPart = None  # type: ignore

try:
    from a2a.types import FilePart, FileWithUri
except ImportError:
    FilePart = None  # type: ignore
    FileWithUri = None  # type: ignore

try:
    from a2a.types import TaskPushNotificationConfig as PushNotificationConfig
except ImportError:
    try:
        from a2a.types import PushNotificationConfig
    except ImportError:
        PushNotificationConfig = None  # type: ignore

try:
    from a2a.types import SendMessageRequest as MessageSendParams
except ImportError:
    MessageSendParams = None  # type: ignore
```

- [ ] **Step 2: Fix V0.1 OpenClawAgentAdapter class body**

Apply the same pattern as other adapters:
- All `Part(root=TextPart(text=...))` → `Part(text=...)`
- All `Part(root=FilePart(file=FileWithUri(...)))` → `Part(url=..., filename=..., media_type=...)`
- All `Role.agent` → `Role.ROLE_AGENT`
- All `TaskState.working` → `TaskState.TASK_STATE_WORKING`
- All `TaskState.completed` → `TaskState.TASK_STATE_COMPLETED`
- All `TaskState.failed` → `TaskState.TASK_STATE_FAILED`
- All `TaskState.canceled` → `TaskState.TASK_STATE_CANCELED`
- `isinstance(result.parts[0].root, TextPart)` → check `result.parts[0].text is not None`
- Any `.root.text` accessors → `.text`

**Important: also fix `extract_raw_input()` (~line 943-950).** This block has both code and comments referencing removed types. Replace:
```python
                for part in msg.parts:
                    # Handle Part(root=TextPart(...)) structure
                    if hasattr(part, "root") and hasattr(part.root, "text"):
                        text_parts.append(part.root.text)
                    # Handle direct TextPart
                    elif hasattr(part, "text"):
                        text_parts.append(part.text)
```
with:
```python
                for part in msg.parts:
                    text_value = getattr(part, "text", None)
                    if text_value is not None:
                        text_parts.append(text_value)
```

- [ ] **Step 3: Verify and commit**

Run: `uv run python -c "from a2a_adapter.integrations.openclaw import OpenClawAdapter, OpenClawAgentAdapter; print('openclaw OK')"`

```bash
git add a2a_adapter/integrations/openclaw.py
git commit -m "fix: V1.0 compat for openclaw adapter — import restructure + V0.1 body fixes"
```

---

## Task 9: Smoke test — all adapters import cleanly

- [ ] **Step 1: Verify all V0.2 adapters load**

Run:
```bash
uv run python -c "
from a2a_adapter import (
    N8nAdapter, CrewAIAdapter, LangChainAdapter, LangGraphAdapter,
    CallableAdapter, OpenClawAdapter, OllamaAdapter, HermesAdapter,
    ClaudeCodeAdapter, CodexAdapter,
)
print('All V0.2 adapters OK')
"
```

- [ ] **Step 2: Verify V0.1 legacy loads**

Run:
```bash
uv run python -c "
from a2a_adapter import BaseAgentAdapter, build_agent_app, load_a2a_agent
print('V0.1 exports OK')
"
```

- [ ] **Step 3: Verify build_agent_app raises RuntimeError**

Run:
```bash
uv run python -c "
from a2a_adapter import build_agent_app
try:
    build_agent_app()
except (RuntimeError, TypeError) as e:
    print(f'Expected error: {e}')
"
```

---

## Task 10: Fix test fixtures — conftest.py

**Files:**
- Modify: `tests/conftest.py:37`

- [ ] **Step 1: Fix `.root` accessor in StreamingStubAdapter**

In `tests/conftest.py`, line 37, replace:
```python
            c if isinstance(c, str) else getattr(c.root, "text", "")
```
with:
```python
            c if isinstance(c, str) else (c.text or "")
```

- [ ] **Step 2: Run conftest import check**

Run: `uv run python -c "from tests.conftest import StreamingStubAdapter; print('conftest OK')"`

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "fix: update conftest.py Part accessor from .root.text to .text for V1.0"
```

---

## Task 11: Fix V0.2 core tests — test_base_adapter.py, test_callable.py, test_executor.py

**Files:**
- Modify: `tests/test_base_adapter.py`
- Modify: `tests/test_callable.py`
- Modify: `tests/test_executor.py`

- [ ] **Step 1: Fix test_base_adapter.py**

Line 6: Replace `from a2a.types import Part, TextPart` with `from a2a.types import Part`
Line 110: Replace `Part(root=TextPart(text="multi"))` with `Part(text="multi")`
Line 119: Replace `result[0].root.text` with `result[0].text`

- [ ] **Step 2: Fix test_callable.py**

Line 4: Replace `from a2a.types import Part, TextPart` with `from a2a.types import Part`

- [ ] **Step 3: Fix test_executor.py**

Line 11: Replace `from a2a.types import Part, TextPart` with `from a2a.types import Part`

Then globally in the file:
- Replace all `Part(root=TextPart(text="..."))` with `Part(text="...")`
- Replace all `.root.text` accessors with `.text`
- Replace all `part.root = MagicMock()` / `del part.root.text` mock patterns with V1.0 equivalents — for tests that check "non-text part" behavior, use `Part()` (empty Part with no text field set) instead of mocking `.root`
- Fix TaskState enum comparison: if the test checks `"failed" in states` or similar string comparisons, update to compare against `TaskState.TASK_STATE_FAILED` enum values

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_base_adapter.py tests/test_callable.py tests/test_executor.py -v`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add tests/test_base_adapter.py tests/test_callable.py tests/test_executor.py
git commit -m "fix: update V0.2 core tests for A2A SDK V1.0 Part/TextPart migration"
```

---

## Task 12: Fix V0.1 legacy test — test_adapter.py

**Files:**
- Modify: `tests/unit/test_adapter.py`

- [ ] **Step 1: Fix imports**

Line 7: Replace:
```python
from a2a.types import Message, MessageSendParams, TextPart, Role, Part
```
with:
```python
from a2a.types import Message, SendMessageRequest, Role, Part
```

- [ ] **Step 2: Fix helper and all type usage**

Update `make_message_send_params()` function:
```python
def make_message_send_params(text: str) -> SendMessageRequest:
    return SendMessageRequest(
        message=Message(
            role=Role.ROLE_USER,
            parts=[Part(text=text)],
        )
    )
```

Fix all `Part(root=TextPart(text=...))` → `Part(text=...)`.
Fix all `Role.agent` → `Role.ROLE_AGENT`, `Role.user` → `Role.ROLE_USER`.
Fix all `.root.text` → `.text`.
Fix all `isinstance(x.root, TextPart)` → check `x.text is not None`.

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/unit/test_adapter.py -v`

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_adapter.py
git commit -m "fix: migrate test_adapter.py to A2A SDK V1.0 types"
```

---

## Task 13: Fix V0.1 legacy tests — test_callable_adapter.py

**Files:**
- Modify: `tests/unit/test_callable_adapter.py`

- [ ] **Step 1: Fix imports and helper**

Line 7: Replace:
```python
from a2a.types import Message, MessageSendParams, TextPart, Role, Part
```
with:
```python
from a2a.types import Message, SendMessageRequest, Role, Part
```

Update `make_message_send_params()`:
```python
def make_message_send_params(text: str, context_id: str | None = None) -> SendMessageRequest:
    return SendMessageRequest(
        message=Message(
            role=Role.ROLE_USER,
            parts=[Part(text=text)],
            context_id=context_id,
        )
    )
```

- [ ] **Step 2: Fix all `.root.text` accessors**

Replace all `result.parts[0].root.text` with `result.parts[0].text`.

- [ ] **Step 3: Run, commit**

Run: `uv run pytest tests/unit/test_callable_adapter.py -v`

```bash
git add tests/unit/test_callable_adapter.py
git commit -m "fix: migrate test_callable_adapter.py to A2A SDK V1.0 types"
```

---

## Task 14: Fix V0.1 legacy tests — test_crewai_adapter.py

**Files:**
- Modify: `tests/unit/test_crewai_adapter.py`

- [ ] **Step 1: Fix imports**

Replace `MessageSendParams` with `SendMessageRequest`, remove `TextPart`.

- [ ] **Step 2: Fix helper, Part construction, Role/TaskState enums, .root accessors**

Same patterns as Task 12-13. Additionally fix any `MagicMock(spec=MessageSendParams)` → `MagicMock(spec=SendMessageRequest)`.

Fix the `make_message_send_params_with_dict_text` helper — replace `mock_part.root.text = data` with setting `mock_part.text = data`.

- [ ] **Step 3: Run, commit**

Run: `uv run pytest tests/unit/test_crewai_adapter.py -v`

```bash
git add tests/unit/test_crewai_adapter.py
git commit -m "fix: migrate test_crewai_adapter.py to A2A SDK V1.0 types"
```

---

## Task 15: Fix V0.1 legacy tests — test_langchain_adapter.py

**Files:**
- Modify: `tests/unit/test_langchain_adapter.py`

- [ ] **Step 1: Fix imports, helper, Part construction, Role, .root accessors**

Same patterns. Replace `MessageSendParams` import with `SendMessageRequest`. Fix `make_message_send_params()`. Replace all `Part(root=TextPart(...))` → `Part(text=...)`. Fix `Role.agent` → `Role.ROLE_AGENT`. Fix `.root.text` → `.text`. Fix any `MagicMock(spec=MessageSendParams)` → `MagicMock(spec=SendMessageRequest)`.

- [ ] **Step 2: Run, commit**

Run: `uv run pytest tests/unit/test_langchain_adapter.py -v`

```bash
git add tests/unit/test_langchain_adapter.py
git commit -m "fix: migrate test_langchain_adapter.py to A2A SDK V1.0 types"
```

---

## Task 16: Fix V0.1 legacy tests — test_langgraph_adapter.py

**Files:**
- Modify: `tests/unit/test_langgraph_adapter.py`

- [ ] **Step 1: Fix imports, helper, Part construction, Role, TaskState, .root accessors**

Same patterns as above. This file has TaskState usage in async mode tests — fix `TaskState.working` → `TaskState.TASK_STATE_WORKING`, etc.

- [ ] **Step 2: Run, commit**

Run: `uv run pytest tests/unit/test_langgraph_adapter.py -v`

```bash
git add tests/unit/test_langgraph_adapter.py
git commit -m "fix: migrate test_langgraph_adapter.py to A2A SDK V1.0 types"
```

---

## Task 17: Fix V0.1 legacy tests — test_n8n_adapter.py

**Files:**
- Modify: `tests/unit/test_n8n_adapter.py`

- [ ] **Step 1: Fix imports, helper, Part construction, Role, TaskState, .root accessors**

Same patterns. This file also has Task/TaskState assertions for async mode.

- [ ] **Step 2: Run, commit**

Run: `uv run pytest tests/unit/test_n8n_adapter.py -v`

```bash
git add tests/unit/test_n8n_adapter.py
git commit -m "fix: migrate test_n8n_adapter.py to A2A SDK V1.0 types"
```

---

## Task 18: Fix V0.1 legacy tests — test_openclaw_adapter.py

**Files:**
- Modify: `tests/unit/test_openclaw_adapter.py`

- [ ] **Step 1: Fix imports, helper, Part construction, Role, TaskState, .root accessors, FilePart assertions**

Same patterns. Additionally fix any `isinstance(x, TextPart)` assertions and `FilePart` references.

- [ ] **Step 2: Run, commit**

Run: `uv run pytest tests/unit/test_openclaw_adapter.py -v`

```bash
git add tests/unit/test_openclaw_adapter.py
git commit -m "fix: migrate test_openclaw_adapter.py to A2A SDK V1.0 types"
```

---

## Task 19: Fix integration tests — test_a2a_compatibility.py

**Files:**
- Modify: `tests/integration/test_a2a_compatibility.py`

- [ ] **Step 1: Fix imports**

Replace:
```python
from a2a.types import MessageSendParams, Message, Role, Part, TextPart
```
with:
```python
from a2a.types import SendMessageRequest, Message, Role, Part
```

- [ ] **Step 2: Fix Part construction, Role values, TextPart assertions**

- `Part(root=TextPart(text='What is AI?'))` → `Part(text='What is AI?')`
- `MessageSendParams(message=user_message)` → `SendMessageRequest(message=user_message)`
- `Role.user` → `Role.ROLE_USER`
- `isinstance(result.parts[0].root, TextPart)` → `result.parts[0].text is not None`

- [ ] **Step 3: Run, commit**

Run: `uv run pytest tests/integration/test_a2a_compatibility.py -v`

```bash
git add tests/integration/test_a2a_compatibility.py
git commit -m "fix: migrate test_a2a_compatibility.py to A2A SDK V1.0 types"
```

---

## Task 20: Fix any remaining V0.2 unit tests

**Files:**
- Check: `tests/unit/test_v02_executor.py`, `tests/unit/test_v02_core.py`, `tests/unit/test_v02_crewai.py`, `tests/unit/test_v02_langchain.py`, `tests/unit/test_v02_langgraph.py`, `tests/unit/test_v02_openclaw.py`, `tests/unit/test_v02_hermes.py`
- Check: `tests/integration/test_v02_integration.py`

- [ ] **Step 1: Grep for remaining broken patterns**

Run:
```bash
uv run python -c "
import subprocess, sys
result = subprocess.run(
    ['grep', '-rn', 'TextPart\|root\.text\|Part(root=\|Role\.agent\|Role\.user\b\|TaskState\.working\|TaskState\.completed\|TaskState\.failed\|TaskState\.canceled\b\|MessageSendParams', 'tests/'],
    capture_output=True, text=True
)
print(result.stdout or 'No remaining patterns found')
"
```

- [ ] **Step 2: Fix any files found**

Apply the same migration patterns to any remaining files.

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests pass

- [ ] **Step 4: Commit if any changes**

```bash
git add tests/
git commit -m "fix: clean up remaining V0.2 test files for A2A SDK V1.0"
```

---

## Task 21: Final validation

- [ ] **Step 1: Verify all validation criteria**

Run each check from the spec:

```bash
# 1. import a2a_adapter succeeds
uv run python -c "import a2a_adapter; print('1. OK:', a2a_adapter.__version__)"

# 2. All V0.2 adapters load
uv run python -c "
from a2a_adapter import N8nAdapter, CrewAIAdapter, LangChainAdapter, LangGraphAdapter, CallableAdapter, OpenClawAdapter
print('2. All V0.2 adapters OK')
"

# 3. BaseAgentAdapter loads
uv run python -c "from a2a_adapter import BaseAgentAdapter; print('3. BaseAgentAdapter OK')"

# 4. build_agent_app loads but raises on call
uv run python -c "
from a2a_adapter import build_agent_app
try:
    build_agent_app()
except (RuntimeError, TypeError):
    print('4. build_agent_app raises as expected')
"

# 7. Full test suite
uv run pytest -v
```

- [ ] **Step 2: Grep for any remaining removed type usage in non-test source**

Run:
```bash
grep -rn 'TextPart\|FilePart\|FileWithUri\|FileWithBytes' a2a_adapter/ --include='*.py' | grep -v 'try:\|except\|import\|# type: ignore\|__pycache__'
```

Expected: No matches outside of try/except import blocks.

- [ ] **Step 3: Final commit if any cleanup needed**

```bash
git add -A
git commit -m "chore: final validation pass for v0.2.9 A2A SDK V1.0 compatibility"
```
