"""
OpenClaw adapter for A2A Protocol.

This adapter enables OpenClaw agents to be exposed as A2A-compliant agents
by wrapping the OpenClaw CLI as a subprocess.

Contains:
    - OpenClawAdapter (v0.2): New simplified interface based on BaseA2AAdapter
    - OpenClawAgentAdapter (v0.1): Legacy interface, deprecated
"""

import asyncio
from asyncio.subprocess import PIPE, Process as AsyncProcess
import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

import httpx

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

from ..adapter import BaseAgentAdapter
from ..base_adapter import AdapterMetadata, BaseA2AAdapter

# Lazy import for TaskStore to avoid hard dependency
try:
    from a2a.server.tasks import InMemoryTaskStore, TaskStore

    _HAS_TASK_STORE = True
except ImportError:
    _HAS_TASK_STORE = False
    TaskStore = None  # type: ignore
    InMemoryTaskStore = None  # type: ignore

logger = logging.getLogger(__name__)

# Valid thinking levels for OpenClaw
VALID_THINKING_LEVELS = {"off", "minimal", "low", "medium", "high", "xhigh"}

# Regex for sanitizing session IDs (matches OpenClaw's VALID_ID_RE pattern)
# OpenClaw session IDs must be alphanumeric with underscores/hyphens, max 64 chars
_INVALID_SESSION_CHARS_RE = re.compile(r"[^a-z0-9_-]+")
_LEADING_TRAILING_DASH_RE = re.compile(r"^-+|-+$")
_SESSION_ID_MAX_LEN = 64


# ═══════════════════════════════════════════════════════════════
# v0.2 Adapter (new, recommended)
# ═══════════════════════════════════════════════════════════════


class OpenClawAdapter(BaseA2AAdapter):
    """Adapter for OpenClaw agents via the CLI.

    Wraps the ``openclaw agent --local --json`` command as a subprocess,
    parses the JSON output, and returns the response text. Supports
    ``cancel()`` to kill a running subprocess.

    Example::

        from a2a_adapter import OpenClawAdapter, serve_agent

        adapter = OpenClawAdapter(
            session_id="my-session",
            agent_id="main",
            thinking="low",
        )
        serve_agent(adapter, port=9005)
    """

    def __init__(
        self,
        session_id: str | None = None,
        agent_id: str | None = None,
        thinking: str = "low",
        timeout: int = 600,
        openclaw_path: str = "openclaw",
        working_directory: str | None = None,
        env_vars: Dict[str, str] | None = None,
        name: str = "",
        description: str = "",
    ) -> None:
        """Initialize the OpenClaw adapter.

        Args:
            session_id: Session ID for conversation continuity.
                Auto-generates a unique ID if not provided.
            agent_id: OpenClaw agent ID. Uses default agent if not provided.
            thinking: Thinking level (off/minimal/low/medium/high/xhigh).
            timeout: Command timeout in seconds (default: 600).
            openclaw_path: Path to the openclaw binary (default: "openclaw").
            working_directory: Working directory for the subprocess.
            env_vars: Additional environment variables for the subprocess.
            name: Optional agent name for AgentCard generation.
            description: Optional agent description for AgentCard generation.
        """
        if thinking not in VALID_THINKING_LEVELS:
            raise ValueError(
                f"Invalid thinking level: {thinking}. "
                f"Valid values: {', '.join(sorted(VALID_THINKING_LEVELS))}"
            )

        self.session_id = session_id or f"a2a-{uuid.uuid4().hex[:12]}"
        self.agent_id = agent_id
        self.thinking = thinking
        self.timeout = timeout
        self.openclaw_path = openclaw_path
        self.working_directory = working_directory
        self.env_vars = dict(env_vars) if env_vars else {}
        self._name = name
        self._description = description

        # Track current subprocess for cancel support
        self._current_process: AsyncProcess | None = None

    async def invoke(self, user_input: str, context_id: str | None = None, **kwargs) -> str:
        """Execute the OpenClaw CLI and return the response text."""
        import os

        session_id = self._context_id_to_session_id(context_id)
        cmd = self._build_command(user_input, session_id)
        logger.debug("Executing OpenClaw command: %s", " ".join(cmd))

        env = os.environ.copy()
        env.update(self.env_vars)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=PIPE,
                stderr=PIPE,
                cwd=self.working_directory,
                env=env,
            )
            self._current_process = proc

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout,
                )
            finally:
                self._current_process = None

        except FileNotFoundError:
            raise FileNotFoundError(
                f"OpenClaw binary not found at '{self.openclaw_path}'. "
                "Ensure OpenClaw is installed and in PATH."
            )
        except asyncio.TimeoutError:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
            raise RuntimeError(
                f"OpenClaw command timed out after {self.timeout} seconds"
            )

        if proc.returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"OpenClaw command failed with exit code {proc.returncode}: {stderr_text}"
            )

        stdout_text = stdout.decode("utf-8", errors="replace").strip()
        if not stdout_text:
            raise RuntimeError("OpenClaw command returned empty output")

        try:
            parsed = json.loads(stdout_text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse OpenClaw JSON output: {e}") from e

        return self._extract_response_text(parsed)

    async def cancel(self, **kwargs) -> None:
        """Kill the running OpenClaw subprocess."""
        proc = self._current_process
        if proc and proc.returncode is None:
            logger.debug("Killing OpenClaw subprocess")
            proc.kill()

    async def close(self) -> None:
        """Ensure subprocess is cleaned up."""
        await self.cancel()

    def get_metadata(self) -> AdapterMetadata:
        """Return adapter metadata for AgentCard generation."""
        return AdapterMetadata(
            name=self._name or "OpenClawAdapter",
            description=self._description or "OpenClaw AI agent",
            streaming=False,
        )

    # ──── Internal: Command Building ────

    def _build_command(self, message: str, session_id: str) -> list[str]:
        """Build the OpenClaw CLI command."""
        cmd = [
            self.openclaw_path,
            "agent",
            "--local",
            "--message", message,
            "--json",
            "--session-id", session_id,
            "--thinking", self.thinking,
        ]
        if self.agent_id:
            cmd.extend(["--agent", self.agent_id])
        return cmd

    def _context_id_to_session_id(self, context_id: str | None) -> str:
        """Convert A2A context_id to a valid OpenClaw session_id.

        OpenClaw session IDs must match: ``^[a-z0-9][a-z0-9_-]{0,63}$``
        """
        if not context_id:
            return self.session_id

        sanitized = _INVALID_SESSION_CHARS_RE.sub("-", context_id.lower())
        sanitized = _LEADING_TRAILING_DASH_RE.sub("", sanitized)
        if not sanitized:
            return self.session_id

        max_suffix_len = _SESSION_ID_MAX_LEN - 4
        sanitized = sanitized[:max_suffix_len]
        return f"a2a-{sanitized}"

    # ──── Internal: Output Extraction ────

    @staticmethod
    def _extract_response_text(output: Dict[str, Any]) -> str:
        """Extract text from OpenClaw JSON output (payloads array)."""
        parts: list[str] = []
        for payload in output.get("payloads", []):
            text = payload.get("text", "")
            if text:
                parts.append(text)
        return "\n\n".join(parts) if parts else ""

    @staticmethod
    def _detect_mime_type(url: str) -> str:
        """Detect MIME type from URL extension."""
        url_lower = url.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".svg": "image/svg+xml",
            ".pdf": "application/pdf",
            ".mp4": "video/mp4",
            ".webm": "video/webm",
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
        }
        for ext, mime in mime_map.items():
            if url_lower.endswith(ext):
                return mime
        return "application/octet-stream"


# ═══════════════════════════════════════════════════════════════
# v0.1 Adapter (legacy, deprecated — will be removed in v0.3)
# ═══════════════════════════════════════════════════════════════


class OpenClawAgentAdapter(BaseAgentAdapter):
    """
    Adapter for integrating OpenClaw agents as A2A agents.

    This adapter wraps the OpenClaw CLI (`openclaw agent --local --json`) as a
    subprocess and translates between A2A protocol messages and OpenClaw's
    JSON output format.

    Supports two execution modes:

    1. **Async Task Mode** (default, async_mode=True):
       - Returns a Task with state="working" immediately
       - Processes the command in the background
       - Clients can poll get_task() for status updates
       - Best for typical OpenClaw operations (can take minutes)
       - Tasks time out after timeout seconds (default: 300)

    2. **Synchronous Mode** (async_mode=False):
       - Blocks until the OpenClaw command completes
       - Returns a Message with the response
       - Best for quick operations or testing

    **Requirements**:
    - OpenClaw CLI must be installed and in PATH (or provide custom path)
    - OpenClaw must be configured with API keys (ANTHROPIC_API_KEY, etc.)
    - Valid OpenClaw configuration at ~/.openclaw/config.yaml

    **Memory Considerations (Async Mode)**:

    When using InMemoryTaskStore (the default), completed tasks are automatically
    cleaned up after `task_ttl_seconds` (default: 1 hour). You can also:

    1. Call delete_task() after retrieving completed tasks to free memory immediately
    2. Use DatabaseTaskStore for persistent storage with external cleanup
    3. Set task_ttl_seconds=None to disable auto-cleanup (manual cleanup only)

    Example:
        >>> adapter = OpenClawAgentAdapter(
        ...     session_id="my-session",
        ...     agent_id="main",
        ...     thinking="low",
        ... )
        >>> task = await adapter.handle(params)  # Returns Task immediately
        >>> # Poll for completion
        >>> completed = await adapter.get_task(task.id)
    """

    def __init__(
        self,
        session_id: str | None = None,
        agent_id: str | None = None,
        thinking: str = "low",
        timeout: int = 600,
        openclaw_path: str = "openclaw",
        working_directory: str | None = None,
        env_vars: Dict[str, str] | None = None,
        async_mode: bool = True,
        task_store: "TaskStore | None" = None,
        task_ttl_seconds: int | None = 3600,
        cleanup_interval_seconds: int = 300,
    ):
        """
        Initialize the OpenClaw adapter.

        Args:
            session_id: Session ID for conversation continuity. If not provided,
                        auto-generates a unique session ID.
            agent_id: OpenClaw agent ID (from `openclaw agents list`). If not
                      provided, uses the default agent.
            thinking: Thinking level for the agent. Valid values:
                      off, minimal, low, medium, high, xhigh. Default: "low".
            timeout: Command timeout in seconds (default: 600).
            openclaw_path: Path to the openclaw binary (default: "openclaw").
            working_directory: Working directory for the subprocess. If not
                               provided, uses the current directory.
            env_vars: Additional environment variables to pass to the subprocess.
            async_mode: If True (default), return Task immediately and process
                        in background. If False, block until command completes.
            task_store: Optional TaskStore for persisting task state. If not
                        provided and async_mode is True, uses InMemoryTaskStore.
            task_ttl_seconds: Time-to-live for completed tasks in seconds. After
                              this duration, completed/failed/canceled tasks are
                              automatically deleted. Set to None to disable
                              auto-cleanup. Default: 3600 (1 hour).
            cleanup_interval_seconds: How often to run the cleanup routine in
                                      seconds. Default: 300 (5 minutes).
        """
        # Validate thinking level
        if thinking not in VALID_THINKING_LEVELS:
            raise ValueError(
                f"Invalid thinking level: {thinking}. "
                f"Valid values: {', '.join(sorted(VALID_THINKING_LEVELS))}"
            )

        # Generate session ID if not provided
        self.session_id = session_id or f"a2a-{uuid.uuid4().hex[:12]}"
        self.agent_id = agent_id
        self.thinking = thinking
        self.timeout = timeout
        self.openclaw_path = openclaw_path
        self.working_directory = working_directory
        self.env_vars = dict(env_vars) if env_vars else {}

        # Async task mode configuration
        self.async_mode = async_mode
        self._background_tasks: Dict[str, asyncio.Task[None]] = {}
        self._background_processes: Dict[str, AsyncProcess] = {}
        self._cancelled_tasks: set[str] = set()
        
        # Push notification configuration per task
        self._push_configs: Dict[str, PushNotificationConfig] = {}
        self._http_client: httpx.AsyncClient | None = None

        # TTL-based cleanup configuration
        self._task_ttl = task_ttl_seconds
        self._cleanup_interval = cleanup_interval_seconds
        self._task_completion_times: Dict[str, float] = {}  # task_id -> completion timestamp
        self._cleanup_task: asyncio.Task[None] | None = None

        # Initialize task store for async mode
        if async_mode:
            if not _HAS_TASK_STORE:
                raise ImportError(
                    "Async task mode requires the A2A SDK with task support. "
                    "Install with: pip install a2a-sdk"
                )
            self.task_store: "TaskStore" = task_store or InMemoryTaskStore()
            # Note: cleanup task is started lazily on first handle() call
            # to avoid requiring a running event loop at init time
        else:
            self.task_store = task_store  # type: ignore

    async def handle(self, params: MessageSendParams) -> Message | Task:
        """
        Handle a non-streaming A2A message request.

        In async mode (default): Returns Task immediately, processes in background.
        In sync mode: Blocks until command completes, returns Message.
        """
        if self.async_mode:
            return await self._handle_async(params)
        else:
            return await self._handle_sync(params)

    async def _handle_sync(self, params: MessageSendParams) -> Message:
        """Handle request synchronously - blocks until command completes."""
        framework_input = await self.to_framework(params)
        framework_output = await self.call_framework(framework_input, params)
        result = await self.from_framework(framework_output, params)
        # In sync mode, always return Message
        if isinstance(result, Task):
            # Extract message from completed task if needed
            if result.status and result.status.message:
                return result.status.message
            # Fallback: create a message from task
            return Message(
                role=Role.agent,
                message_id=str(uuid.uuid4()),
                context_id=result.context_id,
                parts=[Part(root=TextPart(text="Task completed"))],
            )
        return result

    async def _handle_async(self, params: MessageSendParams) -> Task:
        """
        Handle request asynchronously - returns Task immediately, processes in background.

        1. Creates a Task with state="working"
        2. Saves the task to the TaskStore
        3. Stores push notification config if provided
        4. Starts a background coroutine to execute the command
        5. Returns the Task immediately
        """
        # Start cleanup loop lazily (requires running event loop)
        self._ensure_cleanup_task_started()

        # Generate IDs
        task_id = str(uuid.uuid4())
        context_id = self._extract_context_id(params) or str(uuid.uuid4())

        # Extract the initial message for history
        initial_message = None
        if hasattr(params, "message") and params.message:
            initial_message = params.message

        # Extract and store push notification config if provided
        push_config = getattr(params, "configuration", None)
        if push_config and hasattr(push_config, "push_notification_config"):
            pn_config = push_config.push_notification_config
            if pn_config and pn_config.url:
                self._push_configs[task_id] = pn_config
                logger.debug("Stored push notification config for task %s: %s", task_id, pn_config.url)

        # Create initial task with "working" state
        now = datetime.now(timezone.utc).isoformat()
        task = Task(
            id=task_id,
            context_id=context_id,
            status=TaskStatus(
                state=TaskState.working,
                timestamp=now,
            ),
            history=[initial_message] if initial_message else None,
        )

        # Save initial task state
        await self.task_store.save(task)
        logger.debug("Created async task %s with state=working", task_id)

        # Start background processing with timeout
        bg_task = asyncio.create_task(
            self._execute_command_with_timeout(task_id, context_id, params)
        )
        self._background_tasks[task_id] = bg_task

        # Clean up background task reference when done and handle exceptions
        def _on_task_done(t: asyncio.Task[None]) -> None:
            self._background_tasks.pop(task_id, None)
            self._background_processes.pop(task_id, None)
            self._cancelled_tasks.discard(task_id)
            # Clean up push config after task completes
            self._push_configs.pop(task_id, None)
            # Check for unhandled exceptions (shouldn't happen, but log if they do)
            if not t.cancelled():
                exc = t.exception()
                if exc:
                    logger.error(
                        "Unhandled exception in background task %s: %s",
                        task_id,
                        exc,
                    )

        bg_task.add_done_callback(_on_task_done)

        return task

    # ---------- TTL-based Cleanup ----------

    def _ensure_cleanup_task_started(self) -> None:
        """Start the cleanup task if TTL is enabled and not already running."""
        if (
            self.async_mode
            and self._task_ttl is not None
            and self._task_ttl > 0
            and self._cleanup_task is None
        ):
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self) -> None:
        """
        Background loop that periodically cleans up expired tasks.
        
        Runs every cleanup_interval_seconds and removes tasks that have been
        in a terminal state for longer than task_ttl_seconds.
        """
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_expired_tasks()
            except asyncio.CancelledError:
                logger.debug("Cleanup loop cancelled")
                break
            except Exception as e:
                # Log but don't crash the cleanup loop
                logger.error("Error in cleanup loop: %s", e)

    async def _cleanup_expired_tasks(self) -> None:
        """Remove tasks that have exceeded their TTL."""
        if self._task_ttl is None:
            return

        now = time.time()
        expired_task_ids = [
            task_id
            for task_id, completion_time in list(self._task_completion_times.items())
            if now - completion_time > self._task_ttl
        ]

        if not expired_task_ids:
            return

        deleted_count = 0
        for task_id in expired_task_ids:
            try:
                await self.task_store.delete(task_id)
                self._task_completion_times.pop(task_id, None)
                deleted_count += 1
                logger.debug("Auto-deleted expired task %s", task_id)
            except Exception as e:
                # Task may already be deleted or store may have issues
                logger.debug("Failed to delete expired task %s: %s", task_id, e)
                # Still remove from tracking to avoid repeated attempts
                self._task_completion_times.pop(task_id, None)

        if deleted_count > 0:
            logger.info(
                "Task cleanup: removed %d expired tasks, %d remaining",
                deleted_count,
                len(self._task_completion_times),
            )

    def _record_task_completion(self, task_id: str) -> None:
        """Record the completion time of a task for TTL tracking."""
        if self._task_ttl is not None:
            self._task_completion_times[task_id] = time.time()

    async def _execute_command_with_timeout(
        self,
        task_id: str,
        context_id: str,
        params: MessageSendParams,
    ) -> None:
        """
        Execute the command with a timeout wrapper.

        This ensures that long-running commands don't hang indefinitely.
        """
        try:
            await asyncio.wait_for(
                self._execute_command_background(task_id, context_id, params),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            # Kill the subprocess if still running to prevent zombie processes
            proc = self._background_processes.get(task_id)
            if proc and proc.returncode is None:
                logger.debug("Killing subprocess for task %s due to timeout", task_id)
                proc.kill()
                try:
                    await proc.wait()  # Reap the zombie process
                except Exception:
                    pass  # Best effort cleanup

            # Check if task was cancelled (don't overwrite canceled state)
            if task_id in self._cancelled_tasks:
                logger.debug("Task %s was cancelled, not marking as failed", task_id)
                return

            logger.error("Task %s timed out after %s seconds", task_id, self.timeout)
            now = datetime.now(timezone.utc).isoformat()
            error_message = Message(
                role=Role.agent,
                message_id=str(uuid.uuid4()),
                context_id=context_id,
                parts=[
                    Part(
                        root=TextPart(
                            text=f"OpenClaw command timed out after {self.timeout} seconds"
                        )
                    )
                ],
            )

            timeout_task = Task(
                id=task_id,
                context_id=context_id,
                status=TaskStatus(
                    state=TaskState.failed,
                    message=error_message,
                    timestamp=now,
                ),
            )
            await self.task_store.save(timeout_task)

            # Record completion time for TTL cleanup
            self._record_task_completion(task_id)

            # Send push notification for timeout failure
            await self._send_push_notification(task_id, timeout_task)

    async def _execute_command_background(
        self,
        task_id: str,
        context_id: str,
        params: MessageSendParams,
    ) -> None:
        """
        Execute the OpenClaw command in the background and update task state.

        This runs as a background coroutine after the initial Task is returned.
        Sends push notifications on completion/failure if configured.
        """
        try:
            logger.debug("Starting background execution for task %s", task_id)

            # Execute the command (this may take a while)
            framework_input = await self.to_framework(params)
            framework_output = await self._call_framework_with_tracking(
                framework_input, params, task_id
            )

            # Check if task was cancelled during execution
            if task_id in self._cancelled_tasks:
                logger.debug(
                    "Task %s was cancelled during execution, not updating state", task_id
                )
                return

            # Convert to message for history
            response_message = self._create_response_message(framework_output, context_id)

            # Build history (Messages for conversation tracking)
            history = []
            if hasattr(params, "message") and params.message:
                history.append(params.message)
            history.append(response_message)

            # Create artifact for the response (A2A spec: task outputs go in artifacts)
            response_artifact = self._create_response_artifact(framework_output)

            # Update task to completed state
            now = datetime.now(timezone.utc).isoformat()
            completed_task = Task(
                id=task_id,
                context_id=context_id,
                status=TaskStatus(
                    state=TaskState.completed,
                    timestamp=now,
                ),
                artifacts=[response_artifact],  # A2A-compliant: outputs in artifacts
                history=history,
            )

            await self.task_store.save(completed_task)
            logger.debug("Task %s completed successfully", task_id)

            # Record completion time for TTL cleanup
            self._record_task_completion(task_id)

            # Send push notification for completion
            await self._send_push_notification(task_id, completed_task)

        except asyncio.CancelledError:
            # Task was cancelled - don't update state, cancel_task() handles it
            logger.debug("Task %s was cancelled", task_id)
            raise  # Re-raise to properly cancel the task

        except Exception as e:
            # Check if task was cancelled (don't overwrite canceled state)
            if task_id in self._cancelled_tasks:
                logger.debug("Task %s was cancelled, not marking as failed", task_id)
                return

            # Update task to failed state
            logger.error("Task %s failed: %s", task_id, e)
            now = datetime.now(timezone.utc).isoformat()
            error_message = Message(
                role=Role.agent,
                message_id=str(uuid.uuid4()),
                context_id=context_id,
                parts=[Part(root=TextPart(text=f"OpenClaw command failed: {str(e)}"))],
            )

            failed_task = Task(
                id=task_id,
                context_id=context_id,
                status=TaskStatus(
                    state=TaskState.failed,
                    message=error_message,
                    timestamp=now,
                ),
            )

            await self.task_store.save(failed_task)

            # Record completion time for TTL cleanup
            self._record_task_completion(task_id)

            # Send push notification for failure
            await self._send_push_notification(task_id, failed_task)

    async def _call_framework_with_tracking(
        self,
        framework_input: Dict[str, Any],
        params: MessageSendParams,
        task_id: str,
    ) -> Dict[str, Any]:
        """
        Execute the OpenClaw command with process tracking for cancellation.

        This is similar to call_framework but tracks the subprocess for
        cancellation support.
        """
        cmd = self._build_command(framework_input)
        logger.debug("Executing OpenClaw command: %s", " ".join(cmd))

        # Prepare environment
        import os

        env = os.environ.copy()
        env.update(self.env_vars)

        # Create subprocess
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=PIPE,
            stderr=PIPE,
            cwd=self.working_directory,
            env=env,
        )

        # Track the process for cancellation
        self._background_processes[task_id] = proc

        try:
            stdout, stderr = await proc.communicate()
        finally:
            # Remove from tracking
            self._background_processes.pop(task_id, None)

        # Check return code
        if proc.returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"OpenClaw command failed with exit code {proc.returncode}: {stderr_text}"
            )

        # Parse JSON output
        stdout_text = stdout.decode("utf-8", errors="replace").strip()
        logger.debug("OpenClaw raw stdout length: %d chars", len(stdout_text))
        logger.debug("OpenClaw raw stdout (first 500 chars): %s", stdout_text[:500])
        if not stdout_text:
            raise RuntimeError("OpenClaw command returned empty output")

        try:
            parsed = json.loads(stdout_text)
            logger.debug("OpenClaw parsed JSON keys: %s", list(parsed.keys()) if isinstance(parsed, dict) else type(parsed))
            if isinstance(parsed, dict) and "payloads" in parsed:
                logger.debug("OpenClaw payloads count: %d", len(parsed.get("payloads", [])))
                for i, p in enumerate(parsed.get("payloads", [])):
                    text_preview = (p.get("text", "") or "")[:100]
                    logger.debug("OpenClaw payload[%d] text preview: %s", i, text_preview)
            return parsed
        except json.JSONDecodeError as e:
            logger.error("OpenClaw JSON parse error. Raw output: %s", stdout_text[:1000])
            raise RuntimeError(f"Failed to parse OpenClaw JSON output: {e}") from e

    def _extract_context_id(self, params: MessageSendParams) -> str | None:
        """Extract context_id from MessageSendParams."""
        if hasattr(params, "message") and params.message:
            return getattr(params.message, "context_id", None)
        return None

    def _context_id_to_session_id(self, context_id: str | None) -> str:
        """
        Convert A2A context_id to a valid OpenClaw session_id.

        OpenClaw session IDs must match the pattern: ^[a-z0-9][a-z0-9_-]{0,63}$
        This method sanitizes the A2A context_id to conform to that format.

        If context_id is provided, it's sanitized and prefixed with 'a2a-' to
        namespace it. If context_id is None or empty, falls back to the
        adapter's default session_id.

        Args:
            context_id: The A2A context_id to convert

        Returns:
            A valid OpenClaw session_id
        """
        if not context_id:
            return self.session_id

        # Lowercase and replace invalid characters with hyphen
        sanitized = _INVALID_SESSION_CHARS_RE.sub("-", context_id.lower())
        # Remove leading/trailing hyphens
        sanitized = _LEADING_TRAILING_DASH_RE.sub("", sanitized)

        if not sanitized:
            return self.session_id

        # Prefix with 'a2a-' to namespace and truncate to max length
        # Account for 'a2a-' prefix (4 chars) in the max length
        max_suffix_len = _SESSION_ID_MAX_LEN - 4
        sanitized = sanitized[:max_suffix_len]

        return f"a2a-{sanitized}"

    # ---------- Input mapping ----------

    async def to_framework(self, params: MessageSendParams) -> Dict[str, Any]:
        """
        Convert A2A message parameters to OpenClaw command input.

        Extracts the user's message text for passing to the OpenClaw CLI.
        Maps A2A context_id to OpenClaw session_id for conversation continuity.

        Args:
            params: A2A message parameters

        Returns:
            Dictionary with command input data
        """
        user_message = ""

        # Extract message from A2A params (new format with message.parts)
        if hasattr(params, "message") and params.message:
            msg = params.message
            if hasattr(msg, "parts") and msg.parts:
                text_parts = []
                for part in msg.parts:
                    # Handle Part(root=TextPart(...)) structure
                    if hasattr(part, "root") and hasattr(part.root, "text"):
                        text_parts.append(part.root.text)
                    # Handle direct TextPart
                    elif hasattr(part, "text"):
                        text_parts.append(part.text)
                user_message = self._join_text_parts(text_parts)

        # Legacy support for messages array (deprecated)
        elif getattr(params, "messages", None):
            last = params.messages[-1]
            content = getattr(last, "content", "")
            if isinstance(content, str):
                user_message = content.strip()
            elif isinstance(content, list):
                text_parts = []
                for item in content:
                    txt = getattr(item, "text", None)
                    if txt and isinstance(txt, str) and txt.strip():
                        text_parts.append(txt.strip())
                user_message = self._join_text_parts(text_parts)

        # Map A2A context_id to OpenClaw session_id
        # This enables conversation continuity: each A2A context gets its own
        # OpenClaw session, so the agent remembers previous messages in that context
        context_id = self._extract_context_id(params)
        effective_session_id = self._context_id_to_session_id(context_id)

        return {
            "message": user_message,
            "session_id": effective_session_id,
            "agent_id": self.agent_id,
            "thinking": self.thinking,
        }

    @staticmethod
    def _join_text_parts(parts: list[str]) -> str:
        """Join text parts into a single string."""
        if not parts:
            return ""
        text = " ".join(p.strip() for p in parts if p)
        return text.strip()

    # ---------- Framework call ----------

    def _build_command(self, framework_input: Dict[str, Any]) -> list[str]:
        """Build the OpenClaw CLI command."""
        cmd = [
            self.openclaw_path,
            "agent",
            "--local",  # CRITICAL: Run embedded, not via gateway
            "--message",
            framework_input["message"],
            "--json",
            "--session-id",
            framework_input["session_id"],
            "--thinking",
            framework_input["thinking"],
        ]

        # Add agent ID if specified
        if framework_input.get("agent_id"):
            cmd.extend(["--agent", framework_input["agent_id"]])

        return cmd

    async def call_framework(
        self, framework_input: Dict[str, Any], params: MessageSendParams
    ) -> Dict[str, Any]:
        """
        Execute the OpenClaw CLI command.

        Args:
            framework_input: Input dictionary from to_framework()
            params: Original A2A parameters (for context)

        Returns:
            Parsed JSON output from OpenClaw

        Raises:
            RuntimeError: If command execution fails
            FileNotFoundError: If openclaw binary is not found
        """
        import os

        cmd = self._build_command(framework_input)
        logger.debug("Executing OpenClaw command: %s", " ".join(cmd))

        # Prepare environment
        env = os.environ.copy()
        env.update(self.env_vars)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=PIPE,
                stderr=PIPE,
                cwd=self.working_directory,
                env=env,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )

        except FileNotFoundError:
            raise FileNotFoundError(
                f"OpenClaw binary not found at '{self.openclaw_path}'. "
                "Ensure OpenClaw is installed and in PATH."
            )
        except asyncio.TimeoutError:
            # Kill the process if it times out
            proc.kill()
            await proc.wait()
            raise RuntimeError(
                f"OpenClaw command timed out after {self.timeout} seconds"
            )

        # Check return code
        if proc.returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"OpenClaw command failed with exit code {proc.returncode}: {stderr_text}"
            )

        # Parse JSON output
        stdout_text = stdout.decode("utf-8", errors="replace").strip()
        logger.debug("OpenClaw sync raw stdout length: %d chars", len(stdout_text))
        logger.debug("OpenClaw sync raw stdout (first 500 chars): %s", stdout_text[:500])
        if not stdout_text:
            raise RuntimeError("OpenClaw command returned empty output")

        try:
            parsed = json.loads(stdout_text)
            logger.debug("OpenClaw sync parsed JSON keys: %s", list(parsed.keys()) if isinstance(parsed, dict) else type(parsed))
            if isinstance(parsed, dict) and "payloads" in parsed:
                logger.debug("OpenClaw sync payloads count: %d", len(parsed.get("payloads", [])))
                for i, p in enumerate(parsed.get("payloads", [])):
                    text_preview = (p.get("text", "") or "")[:100]
                    logger.debug("OpenClaw sync payload[%d] text preview: %s", i, text_preview)
            return parsed
        except json.JSONDecodeError as e:
            logger.error("OpenClaw sync JSON parse error. Raw output: %s", stdout_text[:1000])
            raise RuntimeError(f"Failed to parse OpenClaw JSON output: {e}") from e

    # ---------- Output mapping ----------

    async def from_framework(
        self, framework_output: Dict[str, Any], params: MessageSendParams
    ) -> Message | Task:
        """
        Convert OpenClaw JSON output to A2A Message.

        Handles the OpenClaw JSON output format:
        {
            "payloads": [
                {
                    "text": "Response text",
                    "mediaUrl": null,
                    "mediaUrls": ["https://..."]
                }
            ],
            "meta": {...}
        }

        Args:
            framework_output: JSON output from OpenClaw
            params: Original A2A parameters

        Returns:
            A2A Message with the response
        """
        context_id = self._extract_context_id(params)
        return self._create_response_message(framework_output, context_id)

    def _create_response_message(
        self, framework_output: Dict[str, Any], context_id: str | None
    ) -> Message:
        """Create a response Message from OpenClaw output."""
        logger.debug("_create_response_message called with framework_output keys: %s", 
                     list(framework_output.keys()) if isinstance(framework_output, dict) else type(framework_output))
        parts: list[Part] = []

        # Extract payloads
        payloads = framework_output.get("payloads", [])
        logger.debug("Extracting from %d payloads", len(payloads))
        for i, payload in enumerate(payloads):
            # Extract text
            text = payload.get("text", "")
            logger.debug("Payload[%d] text length: %d, preview: %s", i, len(text) if text else 0, (text or "")[:100])
            if text:
                parts.append(Part(root=TextPart(text=text)))

            # Extract media URLs
            media_urls = payload.get("mediaUrls") or []
            if payload.get("mediaUrl"):
                media_urls.append(payload["mediaUrl"])

            for url in media_urls:
                # Detect MIME type from URL extension
                mime_type = self._detect_mime_type(url)
                parts.append(
                    Part(
                        root=FilePart(
                            file=FileWithUri(uri=url, mimeType=mime_type),
                        )
                    )
                )

        # Fallback if no parts extracted
        if not parts:
            logger.warning("No parts extracted from OpenClaw output, using empty fallback")
            parts.append(Part(root=TextPart(text="")))

        logger.debug("Created Message with %d parts", len(parts))
        for i, part in enumerate(parts):
            if hasattr(part, 'root') and hasattr(part.root, 'text'):
                logger.debug("Part[%d] text length: %d", i, len(part.root.text) if part.root.text else 0)

        return Message(
            role=Role.agent,
            message_id=str(uuid.uuid4()),
            context_id=context_id,
            parts=parts,
        )

    def _create_response_artifact(self, framework_output: Dict[str, Any]) -> Artifact:
        """
        Create an Artifact from OpenClaw output.
        
        Per A2A spec, task outputs should be stored in artifacts, not messages.
        Messages are for conversation history; artifacts are for task results.
        """
        parts: list[Part] = []

        # Extract payloads
        payloads = framework_output.get("payloads", [])
        for payload in payloads:
            # Extract text
            text = payload.get("text", "")
            if text:
                parts.append(Part(root=TextPart(text=text)))

            # Extract media URLs
            media_urls = payload.get("mediaUrls") or []
            if payload.get("mediaUrl"):
                media_urls.append(payload["mediaUrl"])

            for url in media_urls:
                mime_type = self._detect_mime_type(url)
                parts.append(
                    Part(
                        root=FilePart(
                            file=FileWithUri(uri=url, mimeType=mime_type),
                        )
                    )
                )

        # Fallback if no parts extracted
        if not parts:
            parts.append(Part(root=TextPart(text="")))

        return Artifact(
            artifact_id=str(uuid.uuid4()),
            name="response",
            description="OpenClaw agent response",
            parts=parts,
        )

    @staticmethod
    def _detect_mime_type(url: str) -> str:
        """Detect MIME type from URL extension."""
        url_lower = url.lower()
        if url_lower.endswith(".png"):
            return "image/png"
        elif url_lower.endswith((".jpg", ".jpeg")):
            return "image/jpeg"
        elif url_lower.endswith(".gif"):
            return "image/gif"
        elif url_lower.endswith(".webp"):
            return "image/webp"
        elif url_lower.endswith(".svg"):
            return "image/svg+xml"
        elif url_lower.endswith(".pdf"):
            return "application/pdf"
        elif url_lower.endswith(".mp4"):
            return "video/mp4"
        elif url_lower.endswith(".webm"):
            return "video/webm"
        elif url_lower.endswith(".mp3"):
            return "audio/mpeg"
        elif url_lower.endswith(".wav"):
            return "audio/wav"
        else:
            return "application/octet-stream"

    # ---------- Push Notification Support ----------

    def supports_push_notifications(self) -> bool:
        """Check if this adapter supports push notifications."""
        return self.async_mode

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client for push notifications."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def _send_push_notification(self, task_id: str, task: Task) -> bool:
        """
        Send a push notification for a task status update.

        Per A2A spec section 4.3.3, push notifications use StreamResponse format.
        We send the full Task object (including artifacts) wrapped in StreamResponse.

        Args:
            task_id: The task ID
            task: The updated Task object

        Returns:
            True if notification was sent successfully, False otherwise
        """
        push_config = self._push_configs.get(task_id)
        if not push_config or not push_config.url:
            return False

        try:
            client = await self._get_http_client()

            # A2A-compliant: Send full Task wrapped in StreamResponse format
            # This ensures artifacts (with response content) are included
            payload = {
                "task": task.model_dump(mode="json")
            }

            # Build headers
            headers = {"Content-Type": "application/json"}

            # Add Bearer token if provided
            if push_config.token:
                headers["Authorization"] = f"Bearer {push_config.token}"

            # Send the notification
            response = await client.post(
                push_config.url,
                json=payload,
                headers=headers,
            )

            if response.status_code in (200, 201, 202, 204):
                logger.info(
                    "Push notification sent for task %s to %s (status=%s)",
                    task_id,
                    push_config.url,
                    task.status.state,
                )
                return True
            else:
                logger.warning(
                    "Push notification failed for task %s: HTTP %s - %s",
                    task_id,
                    response.status_code,
                    response.text[:200],
                )
                return False

        except Exception as e:
            logger.error(
                "Failed to send push notification for task %s: %s",
                task_id,
                e,
            )
            return False

    async def set_push_notification_config(
        self, task_id: str, config: PushNotificationConfig
    ) -> bool:
        """
        Set or update push notification config for a task.

        Args:
            task_id: The task ID
            config: The push notification configuration

        Returns:
            True if config was set successfully
        """
        if not self.async_mode:
            raise RuntimeError(
                "Push notifications are only available in async mode. "
                "Initialize adapter with async_mode=True"
            )

        task = await self.task_store.get(task_id)
        if not task:
            return False

        self._push_configs[task_id] = config
        logger.debug("Set push notification config for task %s: %s", task_id, config.url)
        return True

    async def get_push_notification_config(
        self, task_id: str
    ) -> PushNotificationConfig | None:
        """
        Get push notification config for a task.

        Args:
            task_id: The task ID

        Returns:
            The push notification config, or None if not set
        """
        return self._push_configs.get(task_id)

    async def delete_push_notification_config(self, task_id: str) -> bool:
        """
        Delete push notification config for a task.

        Args:
            task_id: The task ID

        Returns:
            True if config was deleted, False if not found
        """
        if task_id in self._push_configs:
            del self._push_configs[task_id]
            logger.debug("Deleted push notification config for task %s", task_id)
            return True
        return False

    # ---------- Async Task Support ----------

    def supports_async_tasks(self) -> bool:
        """Check if this adapter supports async task execution."""
        return self.async_mode

    async def get_task(self, task_id: str) -> Task | None:
        """
        Get the current status of a task by ID.

        This method is used for polling task status in async task execution mode.

        Args:
            task_id: The ID of the task to retrieve

        Returns:
            The Task object with current status, or None if not found

        Raises:
            RuntimeError: If async mode is not enabled
        """
        if not self.async_mode:
            raise RuntimeError(
                "get_task() is only available in async mode. "
                "Initialize adapter with async_mode=True"
            )

        task = await self.task_store.get(task_id)
        if task:
            logger.debug("Retrieved task %s with state=%s", task_id, task.status.state)
        else:
            logger.debug("Task %s not found", task_id)
        return task

    async def delete_task(self, task_id: str) -> bool:
        """
        Delete a task from the task store.

        This can be used to clean up completed/failed tasks to prevent memory leaks
        when using InMemoryTaskStore. Only tasks in terminal states (completed,
        failed, canceled) should be deleted.

        Args:
            task_id: The ID of the task to delete

        Returns:
            True if the task was deleted, False if not found or still running

        Raises:
            RuntimeError: If async mode is not enabled
            ValueError: If the task is still running (not in a terminal state)
        """
        if not self.async_mode:
            raise RuntimeError(
                "delete_task() is only available in async mode. "
                "Initialize adapter with async_mode=True"
            )

        task = await self.task_store.get(task_id)
        if not task:
            return False

        # Only allow deletion of tasks in terminal states
        terminal_states = {TaskState.completed, TaskState.failed, TaskState.canceled}
        if task.status.state not in terminal_states:
            raise ValueError(
                f"Cannot delete task {task_id} with state={task.status.state}. "
                f"Only tasks in terminal states ({', '.join(s.value for s in terminal_states)}) can be deleted."
            )

        await self.task_store.delete(task_id)
        logger.debug("Deleted task %s", task_id)
        return True

    async def cancel_task(self, task_id: str) -> Task | None:
        """
        Attempt to cancel a running task.

        This cancels the background asyncio task and kills the subprocess if
        it's still running.

        Args:
            task_id: The ID of the task to cancel

        Returns:
            The updated Task object with state="canceled", or None if not found
        """
        if not self.async_mode:
            raise RuntimeError(
                "cancel_task() is only available in async mode. "
                "Initialize adapter with async_mode=True"
            )

        # Mark task as cancelled to prevent race conditions
        self._cancelled_tasks.add(task_id)

        # Kill the subprocess if still running
        proc = self._background_processes.get(task_id)
        if proc and proc.returncode is None:
            logger.debug("Killing subprocess for task %s", task_id)
            proc.kill()

        # Cancel the background task if still running and wait for it
        bg_task = self._background_tasks.get(task_id)
        if bg_task and not bg_task.done():
            bg_task.cancel()
            logger.debug("Cancelling background task for %s", task_id)
            # Wait for the task to actually finish
            try:
                await bg_task
            except asyncio.CancelledError:
                pass  # Expected when task is cancelled
            except Exception:
                pass  # Task may have failed, we're cancelling anyway

        # Update task state to canceled
        task = await self.task_store.get(task_id)
        if task:
            now = datetime.now(timezone.utc).isoformat()
            canceled_task = Task(
                id=task_id,
                context_id=task.context_id,
                status=TaskStatus(
                    state=TaskState.canceled,
                    timestamp=now,
                ),
                history=task.history,
            )
            await self.task_store.save(canceled_task)
            logger.debug("Task %s marked as canceled", task_id)

            # Record completion time for TTL cleanup
            self._record_task_completion(task_id)

            # Send push notification for cancellation
            await self._send_push_notification(task_id, canceled_task)

            return canceled_task

        return None

    # ---------- Lifecycle ----------

    async def close(self) -> None:
        """Close the adapter and cancel pending background tasks."""
        # Cancel cleanup task first
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass  # Expected

        # Mark all tasks as cancelled to prevent state updates
        for task_id in self._background_tasks:
            self._cancelled_tasks.add(task_id)

        # Kill all running subprocesses
        for task_id, proc in list(self._background_processes.items()):
            if proc.returncode is None:
                logger.debug("Killing subprocess %s during close", task_id)
                proc.kill()

        # Cancel all pending background tasks
        tasks_to_cancel = []
        for task_id, bg_task in list(self._background_tasks.items()):
            if not bg_task.done():
                bg_task.cancel()
                tasks_to_cancel.append(bg_task)
                logger.debug("Cancelling background task %s during close", task_id)

        # Wait for all cancelled tasks to complete
        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

        self._background_tasks.clear()
        self._background_processes.clear()
        self._cancelled_tasks.clear()
        self._push_configs.clear()
        self._task_completion_times.clear()

        # Close HTTP client
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    def supports_streaming(self) -> bool:
        """This adapter does not support streaming responses."""
        return False
