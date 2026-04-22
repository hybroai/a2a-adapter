"""
New v0.2 adapter interface for A2A Protocol integration.

This module defines the simplified BaseA2AAdapter abstract class and
AdapterMetadata dataclass. Framework-specific adapters implement only
invoke() to become A2A-compatible.

Design philosophy:
    The adapter answers ONE question: "Given text, return text or multimodal content."
    Everything else (task management, SSE streaming, push notifications,
    resubscription, state persistence) is handled by the A2A SDK via
    the AdapterAgentExecutor bridge layer.
"""

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from asyncio import wait_for
from asyncio.subprocess import DEVNULL, PIPE, Process, create_subprocess_exec
from dataclasses import dataclass, field
from typing import AsyncIterator

from a2a.types import Part

from .exceptions import CancelledByAdapterError

logger = logging.getLogger(__name__)


@dataclass
class AdapterMetadata:
    """Self-describing metadata for automatic AgentCard generation.

    Instead of forcing users to manually construct AgentCard objects,
    adapters declare their capabilities here. The server layer reads
    this to auto-generate a well-known agent card.

    Attributes:
        name: Human-readable adapter name (defaults to class name).
        description: What this agent does.
        version: Semantic version string.
        skills: List of skill dicts (each with 'id', 'name', 'description').
        input_modes: Supported input MIME types (default: ["text"]).
        output_modes: Supported output MIME types (default: ["text"]).
        streaming: Whether the adapter supports streaming responses.
        provider: Optional dict with 'organization' and 'url' keys.
        documentation_url: Optional URL to the agent's documentation.
        icon_url: Optional URL to an icon for the agent.
    """

    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    skills: list[dict] = field(default_factory=list)
    input_modes: list[str] = field(default_factory=lambda: ["text"])
    output_modes: list[str] = field(default_factory=lambda: ["text"])
    streaming: bool = False
    provider: dict | None = None
    documentation_url: str | None = None
    icon_url: str | None = None


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Output of _build_command() — carries resume metadata for stale retry."""
    args: list[str]
    used_resume: bool


@dataclass(frozen=True, slots=True)
class ParseResult:
    """Output of _parse_invoke_output() — response text + optional session ID."""
    text: str
    session_id: str | None = None


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """Output of _handle_stream_line() — one parsed stdout line."""
    text: str | None = None
    session_id: str | None = None
    error: str | None = None


class BaseA2AAdapter(ABC):
    """The only interface framework developers need to implement.

    Design philosophy:
        - invoke() is the single required method — answers "given text, return text or multimodal"
        - stream() is optional — for frameworks that support token-by-token output
        - cancel() is optional — for frameworks where execution can be interrupted
        - close() is optional — for resource cleanup (HTTP clients, subprocesses)
        - get_metadata() is optional — for automatic AgentCard generation

    Everything else (task management, SSE streaming, push notifications,
    resubscription, state persistence) is handled by the A2A SDK via
    the AdapterAgentExecutor bridge layer.

    Three levels of control:
        Level 1: invoke(input) -> str | list[Part]     (90% of use cases)
        Level 2: stream(input) -> AsyncIterator        (streaming frameworks)
        Level 3: Implement AgentExecutor directly      (full SDK access)
    """

    def __init__(
        self,
        *,
        working_dir: str = "",
        timeout: int = 600,
        env_vars: dict[str, str] | None = None,
        session_store_path: str = "",
    ) -> None:
        """Initialize subprocess infrastructure for CLI-based adapters.

        Non-CLI adapters do NOT need to call this. Only adapters that use
        _run_subprocess() or _run_subprocess_streaming() should call
        super().__init__().
        """
        self._cli_runtime_enabled = True

        self.working_dir = working_dir
        self.timeout = timeout
        self.env_vars = dict(env_vars) if env_vars else {}

        self._context_locks: dict[str, asyncio.Lock] = {}
        self._active_processes: dict[str, Process] = {}
        self._cancelled_tasks: set[str] = set()
        self._killed_tasks: set[str] = set()

        self._sessions: dict[str, str] = {}
        self._session_store_path = session_store_path
        if self._session_store_path:
            self._load_sessions()

    @abstractmethod
    async def invoke(
        self,
        user_input: str,
        context_id: str | None = None,
        **kwargs,
    ) -> str | list[Part]:
        """Execute the agent and return a response.

        This is the ONLY method you must implement.

        Args:
            user_input: The user's message as plain text.
                Extracted from A2A SendMessageRequest by the bridge layer
                using SDK's RequestContext.get_user_input().
            context_id: Conversation context ID for multi-turn support.
                Same context_id = same conversation. None for single-turn.
            **kwargs: Additional keyword arguments from the bridge layer.

        Keyword Args:
            context: The A2A SDK ``RequestContext`` object, providing access
                to the full message including non-text parts (parts with
                ``url``, ``raw``, or ``data`` fields). Access via
                ``kwargs.get('context')``.
                Use ``context.message.parts`` to iterate over all parts.

        Returns:
            str: Text-only response (backward compatible).
            list[Part]: Multimodal response with text, files, images, etc.

        Example:
            # Text-only response (backward compatible)
            async def invoke(self, user_input, context_id=None, **kwargs):
                return "Hello, world!"

            # Multimodal response with text and file
            from a2a.types import Part
            async def invoke(self, user_input, context_id=None, **kwargs):
                return [
                    Part(text="Generated report"),
                    Part(url="http://example.com/report.pdf",
                         filename="report.pdf",
                         media_type="application/pdf")
                ]

        Raises:
            Any exception will be caught by the bridge layer and converted
            to a Task with state=failed.
        """
        ...

    async def stream(
        self,
        user_input: str,
        context_id: str | None = None,
        **kwargs,
    ) -> AsyncIterator[str | Part]:
        """Stream the agent response, yielding chunks.

        Optional. If not implemented, the bridge layer falls back to invoke()
        and delivers the full result as a single event.

        When implemented, each yielded chunk becomes a TaskArtifactUpdateEvent
        in the A2A SSE stream, giving clients real-time token-by-token output.

        Args:
            user_input: The user's message as plain text.
            context_id: Conversation context ID for multi-turn support.
            **kwargs: Additional keyword arguments from the bridge layer.

        Keyword Args:
            context: The A2A SDK ``RequestContext`` object. See
                :meth:`invoke` for details.

        Yields:
            str: Text chunk (backward compatible).
            Part: Multimodal content chunk (images, files, etc.).

        Example:
            # Text-only streaming (backward compatible)
            async def stream(self, user_input, context_id=None, **kwargs):
                for token in ["Hello", ", ", "world", "!"]:
                    yield token

            # Multimodal streaming
            async def stream(self, user_input, context_id=None, **kwargs):
                from a2a.types import Part
                yield "Generating chart..."
                yield Part(url="http://example.com/chart.png",
                           filename="chart.png",
                           media_type="image/png")
        """
        raise NotImplementedError
        # Make this an async generator so type checkers are happy
        yield  # pragma: no cover

    def supports_streaming(self) -> bool:
        """Whether this adapter supports streaming responses.

        Auto-detects by checking if stream() is overridden.
        Override this method for explicit control.
        """
        return type(self).stream is not BaseA2AAdapter.stream

    async def cancel(self, context_id: str | None = None, **kwargs) -> None:
        """Cancel execution. Two-set mechanism if super().__init__() was called."""
        if not getattr(self, '_cli_runtime_enabled', False):
            return

        context = kwargs.get("context")
        if not context:
            return
        task_id = context.task_id

        proc = self._active_processes.get(task_id)
        if proc and proc.returncode is None:
            self._killed_tasks.add(task_id)
            logger.debug("Killing subprocess for task %s", task_id)
            proc.kill()
        else:
            self._cancelled_tasks.add(task_id)
            logger.debug("Queued cancellation for task %s", task_id)

    async def close(self) -> None:
        """Kill all active subprocesses. No-op if super().__init__() wasn't called."""
        if not getattr(self, '_cli_runtime_enabled', False):
            return

        for task_id, proc in list(self._active_processes.items()):
            if proc.returncode is None:
                logger.debug("Killing subprocess %s during close", task_id)
                proc.kill()
                try:
                    await proc.wait()
                except Exception:
                    pass
        self._active_processes.clear()
        self._cancelled_tasks.clear()
        self._killed_tasks.clear()

    async def _run_subprocess(
        self,
        context_key: str,
        task_id: str,
        *,
        user_input: str,
        allow_retry: bool = True,
    ) -> str:
        """Template: spawn CLI subprocess with full lifecycle management."""
        # 1. Cancel check
        if task_id in self._cancelled_tasks:
            self._cancelled_tasks.discard(task_id)
            raise CancelledByAdapterError(
                f"Task {task_id} was cancelled before execution started"
            )

        # 2. Build command (subclass hook)
        cmd_result = self._build_command(user_input, context_key)
        logger.debug("Executing command: %s", " ".join(cmd_result.args))

        # 3. Spawn
        try:
            proc = await create_subprocess_exec(
                *cmd_result.args,
                stdin=DEVNULL,
                stdout=PIPE,
                stderr=PIPE,
                cwd=self.working_dir,
                env=self._build_env(),
            )
        except FileNotFoundError:
            if not os.path.isdir(self.working_dir):
                raise FileNotFoundError(
                    f"Working directory does not exist: '{self.working_dir}'"
                )
            raise FileNotFoundError(self._binary_not_found_message())

        # 4. Track
        self._active_processes[task_id] = proc
        try:
            try:
                stdout_bytes, stderr_bytes = await wait_for(
                    proc.communicate(), timeout=self.timeout
                )
            except asyncio.TimeoutError:
                if proc.returncode is None:
                    proc.kill()
                    await proc.wait()
                raise RuntimeError(
                    f"Command timed out after {self.timeout} seconds"
                )
        finally:
            self._active_processes.pop(task_id, None)

        # 5. Kill detection
        if proc.returncode is not None and proc.returncode < 0:
            if task_id in self._killed_tasks:
                self._killed_tasks.discard(task_id)
                raise CancelledByAdapterError(
                    f"Task {task_id} was killed by cancel request"
                )

        stdout_text = stdout_bytes.decode("utf-8", errors="replace")
        stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()

        # 6. Stale retry — only when BOTH stdout and stderr are empty.
        # If stderr has content, it signals a real error (auth, env, CLI bug)
        # and the session may still be valid. Don't mask the real failure.
        if (
            allow_retry
            and proc.returncode is not None
            and proc.returncode > 0
            and not stdout_text.strip()
            and not stderr_text
            and cmd_result.used_resume
        ):
            logger.info(
                "Stale session for context %s, retrying without resume",
                context_key,
            )
            self._sessions.pop(context_key, None)
            self._save_sessions()
            return await self._run_subprocess(
                context_key, task_id,
                user_input=user_input, allow_retry=False,
            )

        # 7. Non-zero exit
        if proc.returncode != 0:
            raise RuntimeError(
                f"Command failed with exit code {proc.returncode}: "
                f"{stderr_text or '(no stderr)'}"
            )

        if not stdout_text.strip():
            raise RuntimeError("Command returned empty output")

        # 8. Parse (subclass hook)
        result = self._parse_invoke_output(stdout_text, context_key)

        # 9. Persist session
        if result.session_id:
            self._sessions[context_key] = result.session_id
            self._save_sessions()

        return result.text

    async def _run_subprocess_streaming(
        self,
        context_key: str,
        task_id: str,
        *,
        user_input: str,
        allow_retry: bool = True,
    ) -> AsyncIterator[str]:
        """Template: stream CLI subprocess output line by line."""
        # 1. Cancel check
        if task_id in self._cancelled_tasks:
            self._cancelled_tasks.discard(task_id)
            raise CancelledByAdapterError(
                f"Task {task_id} was cancelled before execution started"
            )

        # 2. Build command
        cmd_result = self._build_command(user_input, context_key)
        logger.debug("Streaming command: %s", " ".join(cmd_result.args))

        # 3. Spawn
        try:
            proc = await create_subprocess_exec(
                *cmd_result.args,
                stdin=DEVNULL,
                stdout=PIPE,
                stderr=PIPE,
                cwd=self.working_dir,
                env=self._build_env(),
            )
        except FileNotFoundError:
            if not os.path.isdir(self.working_dir):
                raise FileNotFoundError(
                    f"Working directory does not exist: '{self.working_dir}'"
                )
            raise FileNotFoundError(self._binary_not_found_message())

        # 4. Track
        self._active_processes[task_id] = proc
        yielded_any = False

        stderr_chunks: list[bytes] = []

        async def _drain_stderr():
            assert proc.stderr is not None
            while True:
                chunk = await proc.stderr.read(8192)
                if not chunk:
                    break
                stderr_chunks.append(chunk)

        stderr_task = asyncio.create_task(_drain_stderr())
        aborted = False

        try:
            assert proc.stdout is not None
            while True:
                line = await asyncio.wait_for(
                    proc.stdout.readline(), timeout=self.timeout
                )
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue

                event = self._handle_stream_line(text)
                if event is None:
                    continue

                if event.error:
                    raise RuntimeError(event.error)
                if event.session_id:
                    self._sessions[context_key] = event.session_id
                    self._save_sessions()
                if event.text:
                    yielded_any = True
                    yield event.text

        except asyncio.TimeoutError:
            aborted = True
            if proc.returncode is None:
                proc.kill()
            raise RuntimeError(
                f"Command timed out after {self.timeout} seconds"
            )
        except (GeneratorExit, asyncio.CancelledError):
            aborted = True
            if proc.returncode is None:
                proc.kill()
            raise
        except Exception:
            aborted = True
            if proc.returncode is None:
                proc.kill()
            raise
        finally:
            self._active_processes.pop(task_id, None)

            if aborted:
                stderr_task.cancel()
                try:
                    await stderr_task
                except asyncio.CancelledError:
                    pass
                await proc.wait()
            else:
                await proc.wait()
                await stderr_task

        stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="replace").strip()

        # Kill detection
        if proc.returncode is not None and proc.returncode < 0:
            if task_id in self._killed_tasks:
                self._killed_tasks.discard(task_id)
                raise CancelledByAdapterError(
                    f"Task {task_id} was killed by cancel request"
                )

        # Stale retry — only when BOTH no output yielded and stderr empty.
        # If stderr has content, it signals a real error; don't mask it.
        if (
            allow_retry
            and proc.returncode is not None
            and proc.returncode > 0
            and not yielded_any
            and not stderr_text
            and cmd_result.used_resume
        ):
            logger.info(
                "Stale session for context %s (stream), retrying",
                context_key,
            )
            self._sessions.pop(context_key, None)
            self._save_sessions()
            async for chunk in self._run_subprocess_streaming(
                context_key, task_id,
                user_input=user_input, allow_retry=False,
            ):
                yield chunk
            return

        # Terminal state
        if proc.returncode != 0:
            raise RuntimeError(
                f"Command failed with exit code {proc.returncode}: "
                f"{stderr_text or '(no stderr)'}"
            )

        if not yielded_any:
            raise RuntimeError("Command returned no output")

    # ──── Utility methods ────

    def _get_context_lock(self, context_key: str) -> asyncio.Lock:
        """Get or create a per-context asyncio.Lock."""
        if context_key not in self._context_locks:
            self._context_locks[context_key] = asyncio.Lock()
        return self._context_locks[context_key]

    def _build_env(self) -> dict[str, str]:
        """Build subprocess environment: os.environ + self.env_vars."""
        env = os.environ.copy()
        env.update(self.env_vars)
        return env

    def _load_sessions(self) -> None:
        """Load session mapping from disk. Missing/corrupt -> empty dict."""
        try:
            with open(self._session_store_path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._sessions = {str(k): str(v) for k, v in data.items()}
                logger.debug(
                    "Loaded %d session(s) from %s",
                    len(self._sessions), self._session_store_path,
                )
            else:
                self._sessions = {}
        except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
            logger.debug("Could not load sessions from %s: %s", self._session_store_path, e)
            self._sessions = {}

    def _save_sessions(self) -> None:
        """Persist session mapping to disk. Creates parent dirs."""
        try:
            parent_dir = os.path.dirname(self._session_store_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            with open(self._session_store_path, "w") as f:
                json.dump(self._sessions, f, indent=2)
            logger.debug(
                "Saved %d session(s) to %s",
                len(self._sessions), self._session_store_path,
            )
        except OSError as e:
            logger.warning("Could not save sessions to %s: %s", self._session_store_path, e)

    # ──── Hooks (subclasses override) ────

    def _build_command(self, message: str, context_key: str) -> CommandResult:
        """Build CLI command. CLI adapters MUST override."""
        raise NotImplementedError("CLI adapters must implement _build_command")

    def _parse_invoke_output(self, stdout_text: str, context_key: str) -> ParseResult:
        """Parse subprocess stdout. CLI adapters MUST override."""
        raise NotImplementedError("CLI adapters must implement _parse_invoke_output")

    def _handle_stream_line(self, line: str) -> StreamEvent | None:
        """Parse one stdout line for streaming. Override if supports_streaming()."""
        raise NotImplementedError("Streaming CLI adapters must implement _handle_stream_line")

    def _binary_not_found_message(self) -> str:
        """Error message when CLI binary not found. Override for custom message."""
        return "CLI binary not found. Ensure it is installed and in PATH."

    def get_metadata(self) -> AdapterMetadata:
        """Return adapter metadata for automatic AgentCard generation. Optional.

        Override to provide agent name, description, skills, and capabilities.
        The server layer uses this to auto-generate the AgentCard served at
        /.well-known/agent.json.
        """
        return AdapterMetadata()

    async def __aenter__(self):
        """Support async context manager for resource management."""
        return self

    async def __aexit__(self, *args):
        """Cleanup resources when exiting async context."""
        await self.close()
