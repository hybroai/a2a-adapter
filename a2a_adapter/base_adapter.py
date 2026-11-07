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
import atexit
import contextlib
import hashlib
import json
import logging
import os
import signal
import tempfile
import weakref
from abc import ABC, abstractmethod
from asyncio import wait_for
from asyncio.subprocess import DEVNULL, PIPE, Process, create_subprocess_exec
from dataclasses import dataclass, field
from typing import AsyncIterator

from a2a.types import Part

from .exceptions import CancelledByAdapterError

logger = logging.getLogger(__name__)

# ──── Process-tree management ────
#
# CLI agents (claude, codex, pi, ...) start their own children: shells, MCP
# servers, tool processes. Signalling only the direct child leaves those
# running and holding its stdout/stderr pipes. On POSIX each CLI therefore
# starts in a new session, so it leads its own process group and the whole
# tree can be signalled at once with ``os.killpg``. A CLI helper that moves
# itself into yet another session escapes the group; Windows keeps the
# single-process kill.

_SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)
_POST_KILL_WAIT = 5.0  # seconds to wait for a killed process to be reaped
_MIN_TAIL_WAIT = 0.1  # seconds a streaming run gets to exit after its output ends

# Processes this module spawned as process-group leaders. Only these are ever
# passed to ``os.killpg``; anything else (mocks with fake PIDs, processes
# spawned elsewhere) gets the single-process path.
_OWN_GROUPS: "weakref.WeakSet[Process]" = weakref.WeakSet()
_atexit_registered = False


def _register_own_group(proc: Process) -> None:
    """Record ``proc`` as a process-group leader we may ``killpg``.

    Only a real ``asyncio.subprocess.Process`` that leads its own group
    (``pgid == pid``) is recorded.
    """
    global _atexit_registered
    if os.name != "posix" or not isinstance(proc, Process):
        return
    try:
        if os.getpgid(proc.pid) != proc.pid:
            return
    except (ProcessLookupError, PermissionError):
        return
    _OWN_GROUPS.add(proc)
    if not _atexit_registered:
        atexit.register(_kill_own_groups_at_exit)
        _atexit_registered = True


def _forget_own_group(proc: Process) -> None:
    """Stop treating ``proc``'s group as ours (its run is over and reaped)."""
    _OWN_GROUPS.discard(proc)


def _owns_group(proc: Process) -> bool:
    return proc in _OWN_GROUPS


async def _spawn_cli(factory, *args, **kwargs) -> Process:
    """Spawn a CLI with ``factory`` (a ``create_subprocess_exec``) in its own group.

    Callers pass their module's own ``create_subprocess_exec`` so tests that
    patch that name keep working.
    """
    if os.name == "posix":
        kwargs["start_new_session"] = True
    proc = await factory(*args, **kwargs)
    _register_own_group(proc)
    return proc


def _kill_process_tree(proc: Process, sig: int = _SIGKILL) -> None:
    """Signal ``proc`` and, on POSIX, every process in its group.

    The group is signalled even when the leader already exited, because its
    children can outlive it. Call this only on abnormal paths (cancel,
    timeout, error, close, shutdown), never after a clean exit, and forget the
    process once it is reaped: a group ID can be reused once all its members
    are gone. Uses only calls that are safe inside a signal handler.
    """
    if _owns_group(proc):
        try:
            os.killpg(proc.pid, sig)
            return
        except ProcessLookupError:
            return  # the whole group is already gone
        except PermissionError as error:
            logger.debug("killpg(%s) failed (%s); signalling the process only", proc.pid, error)
    if proc.returncode is not None:
        return
    try:
        if sig == _SIGKILL:
            proc.kill()
        elif sig == signal.SIGTERM:
            proc.terminate()
        else:
            proc.send_signal(sig)
    except ProcessLookupError:
        pass


async def _wait_reaped(proc: Process, timeout: float = _POST_KILL_WAIT) -> None:
    """Wait for a killed process, bounded.

    ``Process.wait()`` returns only once the process exited and all its pipes
    closed, so a process outside the group that still holds a pipe would
    block it forever. The leader is dead either way.
    """
    try:
        await asyncio.wait_for(proc.wait(), timeout)
    except asyncio.TimeoutError:
        logger.warning(
            "Process %s was killed but its pipes are still open after %ss; moving on",
            getattr(proc, "pid", "?"),
            timeout,
        )


def _kill_own_groups_at_exit() -> None:
    """Last resort at interpreter exit: SIGKILL every group whose leader still runs."""
    for proc in list(_OWN_GROUPS):
        if proc.returncode is None:
            with contextlib.suppress(OSError):
                os.killpg(proc.pid, _SIGKILL)


def default_state_dir(agent: str, working_dir: str) -> str:
    """Per-user state directory for one agent working in one project.

    Returns ``$XDG_STATE_HOME/a2a-adapter/<agent>/<hash>`` (falling back to
    ``~/.local/state`` when ``XDG_STATE_HOME`` is unset or not absolute),
    where ``<hash>`` is a short digest of the resolved working directory.
    Keeps adapter state out of the agent's project while giving each
    project its own store.
    """
    state_home = os.environ.get("XDG_STATE_HOME", "")
    if not os.path.isabs(state_home):
        state_home = os.path.join(os.path.expanduser("~"), ".local", "state")
    resolved = os.path.realpath(working_dir or ".")
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]
    return os.path.join(state_home, "a2a-adapter", agent, digest)


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
    """Output of _build_command() — carries resume metadata for stale retry.

    ``stdin`` is written to the subprocess's standard input (then closed) when
    set. Use it for the user message instead of an argv element: Linux limits a
    single argument to 128 KiB (``MAX_ARG_STRLEN``), so large prompts passed on
    the command line fail with ``OSError: [Errno 7] Argument list too long``.
    """

    args: list[str]
    used_resume: bool
    stdin: str | None = None


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
        if not getattr(self, "_cli_runtime_enabled", False):
            return

        context = kwargs.get("context")
        if not context:
            return
        task_id = context.task_id

        proc = self._active_processes.get(task_id)
        # A group we spawned is killed even if the CLI itself already exited:
        # its children may still be running and holding the pipes.
        if proc and (proc.returncode is None or _owns_group(proc)):
            self._killed_tasks.add(task_id)
            logger.debug("Killing subprocess tree for task %s", task_id)
            _kill_process_tree(proc)
        else:
            self._cancelled_tasks.add(task_id)
            logger.debug("Queued cancellation for task %s", task_id)

    def kill_children_now(self) -> None:
        """SIGKILL every running CLI process tree, synchronously.

        Called from the server's signal handler on Ctrl+C, so it does no
        awaiting and clears no state; in-flight requests then fail promptly
        and ``close()`` does the rest. No-op for non-CLI adapters.
        """
        if not getattr(self, "_cli_runtime_enabled", False):
            return
        for proc in list(self._active_processes.values()):
            try:
                _kill_process_tree(proc)
            except Exception:  # never let one failure stop the others
                logger.debug("Could not kill subprocess %s", getattr(proc, "pid", "?"))

    async def close(self) -> None:
        """Kill all active subprocess trees. No-op if super().__init__() wasn't called."""
        if not getattr(self, "_cli_runtime_enabled", False):
            return

        for task_id, proc in list(self._active_processes.items()):
            logger.debug("Killing subprocess tree %s during close", task_id)
            _kill_process_tree(proc)
            try:
                await _wait_reaped(proc)
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
            raise CancelledByAdapterError(f"Task {task_id} was cancelled before execution started")

        # 2. Build command (subclass hook)
        cmd_result = self._build_command(user_input, context_key)
        logger.debug("Executing command: %s", " ".join(cmd_result.args))
        stdin_bytes = cmd_result.stdin.encode("utf-8") if cmd_result.stdin is not None else None

        # 3. Spawn
        try:
            proc = await _spawn_cli(
                create_subprocess_exec,
                *cmd_result.args,
                stdin=PIPE if stdin_bytes is not None else DEVNULL,
                stdout=PIPE,
                stderr=PIPE,
                cwd=self.working_dir,
                env=self._build_env(),
                limit=10 * 1024 * 1024,  # 10 MB — prevents LimitOverrunError on large output lines
            )
        except FileNotFoundError:
            if not os.path.isdir(self.working_dir):
                raise FileNotFoundError(f"Working directory does not exist: '{self.working_dir}'")
            raise FileNotFoundError(self._binary_not_found_message())

        # 4. Track
        self._active_processes[task_id] = proc
        aborted = False
        try:
            try:
                communicate = (
                    proc.communicate(input=stdin_bytes)
                    if stdin_bytes is not None
                    else proc.communicate()
                )
                stdout_bytes, stderr_bytes = await wait_for(communicate, timeout=self.timeout)
            except asyncio.TimeoutError:
                aborted = True
                _kill_process_tree(proc)
                await _wait_reaped(proc)
                raise RuntimeError(f"Command timed out after {self.timeout} seconds")
            except asyncio.CancelledError:
                # Task cancelled (tasks/cancel or shutdown): don't orphan the tree
                aborted = True
                _kill_process_tree(proc)
                await _wait_reaped(proc)
                raise
        finally:
            self._active_processes.pop(task_id, None)
            _forget_own_group(proc)
            if aborted:
                # After the pop, so a cancel() racing the reap can't re-add a
                # marker that would mark a later run with this ID cancelled.
                self._killed_tasks.discard(task_id)

        # 5. Kill detection (the CLI may have exited on its own before cancel()
        # killed the rest of its group, so don't require a negative exit code)
        if task_id in self._killed_tasks:
            self._killed_tasks.discard(task_id)
            raise CancelledByAdapterError(f"Task {task_id} was killed by cancel request")

        stdout_text = stdout_bytes.decode("utf-8", errors="replace")
        stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()

        # 6. Stale retry. A resumed run that fails with no output at all is
        # treated as a stale session; so is one whose output the adapter's
        # _is_stale_session_error() hook recognises. Any other stderr is a
        # real error (auth, env, CLI bug) and the session may still be valid.
        retryable = allow_retry and cmd_result.used_resume
        if (
            retryable
            and proc.returncode is not None
            and proc.returncode > 0
            and (
                (not stdout_text.strip() and not stderr_text)
                or self._is_stale_session_error(stdout_text, stderr_text)
            )
        ):
            return await self._retry_without_resume(context_key, task_id, user_input)

        # 7. Non-zero exit — prefer the CLI's structured error over stderr noise
        if proc.returncode != 0:
            detail = self._extract_error(stdout_text) or stderr_text or "(no stderr)"
            raise RuntimeError(f"Command failed with exit code {proc.returncode}: {detail}")

        output_text = self._select_invoke_output(stdout_text, stderr_text)
        if not output_text.strip():
            raise RuntimeError("Command returned empty output")

        # 8. Parse (subclass hook). A CLI may exit 0 and report the missing
        # session only as a failed turn, so the hook is checked here too.
        try:
            result = self._parse_invoke_output(output_text, context_key)
        except Exception:
            if retryable and self._is_stale_session_error(stdout_text, stderr_text):
                return await self._retry_without_resume(context_key, task_id, user_input)
            raise

        # 9. Persist session
        if result.session_id:
            self._sessions[context_key] = result.session_id
            self._save_sessions()

        return result.text

    async def _retry_without_resume(self, context_key: str, task_id: str, user_input: str) -> str:
        self._forget_stale_session(context_key)
        return await self._run_subprocess(
            context_key, task_id, user_input=user_input, allow_retry=False
        )

    def _forget_stale_session(self, context_key: str) -> None:
        """Drop the stored session for ``context_key`` (memory and disk)."""
        logger.warning(
            "Stale session for context %s, retrying once without resume", context_key
        )
        self._sessions.pop(context_key, None)
        self._save_sessions()

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
            raise CancelledByAdapterError(f"Task {task_id} was cancelled before execution started")

        # 2. Build command
        cmd_result = self._build_command(user_input, context_key)
        logger.debug("Streaming command: %s", " ".join(cmd_result.args))

        # 3. Spawn
        try:
            proc = await _spawn_cli(
                create_subprocess_exec,
                *cmd_result.args,
                stdin=PIPE if cmd_result.stdin is not None else DEVNULL,
                stdout=PIPE,
                stderr=PIPE,
                cwd=self.working_dir,
                env=self._build_env(),
                limit=10 * 1024 * 1024,  # 10 MB — prevents LimitOverrunError on large output lines
            )
        except FileNotFoundError:
            if not os.path.isdir(self.working_dir):
                raise FileNotFoundError(f"Working directory does not exist: '{self.working_dir}'")
            raise FileNotFoundError(self._binary_not_found_message())

        # 4. Track. ``self.timeout`` bounds the whole run, not each line: a
        # CLI that keeps printing must still stop at the deadline. Each await
        # gets the time that is left instead of one asyncio.timeout() around
        # the loop, which would fire while the generator is suspended at
        # ``yield`` and cancel the consumer at an unrelated await. Time the
        # consumer spends between chunks counts toward the deadline.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout
        self._active_processes[task_id] = proc
        yielded_any = False

        def _remaining() -> float:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError
            return remaining

        stderr_chunks: list[bytes] = []

        async def _drain_stderr():
            assert proc.stderr is not None
            while True:
                chunk = await proc.stderr.read(8192)
                if not chunk:
                    break
                stderr_chunks.append(chunk)

        stderr_task = asyncio.create_task(_drain_stderr())
        # Feed stdin concurrently with reading stdout so a large prompt cannot
        # deadlock against a child that starts writing before it finishes reading.
        stdin_task = (
            asyncio.create_task(self._feed_stdin(proc, cmd_result.stdin))
            if cmd_result.stdin is not None
            else None
        )
        helpers = [task for task in (stderr_task, stdin_task) if task is not None]

        async def _stop_after_kill() -> None:
            for task in helpers:
                task.cancel()
            # return_exceptions: a helper that failed (or was cancelled) must
            # not skip the others or the reap.
            await asyncio.gather(*helpers, return_exceptions=True)
            await _wait_reaped(proc)

        aborted = False
        tail_timed_out = False
        stale = False

        try:
            assert proc.stdout is not None
            while True:
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=_remaining())
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue

                event = self._handle_stream_line(text)
                if event is None:
                    continue

                if event.error:
                    if (
                        allow_retry
                        and cmd_result.used_resume
                        and not yielded_any
                        and self._is_stale_session_error(text, "")
                    ):
                        # Retry below. Kill the CLI first so a process (or MCP
                        # child) that keeps running can't hold the tail wait
                        # for the whole deadline.
                        stale = True
                        _kill_process_tree(proc)
                        break
                    raise RuntimeError(event.error)
                if event.session_id:
                    self._sessions[context_key] = event.session_id
                    self._save_sessions()
                if event.text:
                    yielded_any = True
                    yield event.text

        except asyncio.TimeoutError:
            aborted = True
            _kill_process_tree(proc)
            raise RuntimeError(f"Command timed out after {self.timeout} seconds")
        except (GeneratorExit, asyncio.CancelledError):
            aborted = True
            _kill_process_tree(proc)
            raise
        except Exception:
            aborted = True
            _kill_process_tree(proc)
            raise
        finally:
            try:
                if aborted:
                    await _stop_after_kill()
                else:
                    # stdout closed; the CLI (or a child holding stderr) may
                    # still run, so the tail is bounded by the same deadline.
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(proc.wait(), *helpers),
                            # A floor, so output that ended right at the
                            # deadline isn't reported as a timeout.
                            timeout=max(deadline - loop.time(), _MIN_TAIL_WAIT),
                        )
                    except asyncio.TimeoutError:
                        tail_timed_out = True
                        _kill_process_tree(proc)
                        await _stop_after_kill()
                    except BaseException:  # consumer cancelled, or a helper failed
                        aborted = True
                        _kill_process_tree(proc)
                        await _stop_after_kill()
                        raise
            finally:
                # Popped only now, so cancel() and kill_children_now() still
                # reach a CLI that closed stdout but keeps running.
                self._active_processes.pop(task_id, None)
                _forget_own_group(proc)
                if aborted or tail_timed_out:
                    # After the pop, so a cancel() racing the reap can't re-add
                    # a marker that would mark a later run with this ID cancelled.
                    self._killed_tasks.discard(task_id)

        if tail_timed_out:
            raise RuntimeError(f"Command timed out after {self.timeout} seconds")

        stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="replace").strip()

        # Kill detection (see _run_subprocess)
        if task_id in self._killed_tasks:
            self._killed_tasks.discard(task_id)
            raise CancelledByAdapterError(f"Task {task_id} was killed by cancel request")

        # Stale retry — never after output reached the client. A resumed run
        # that failed with no output and no stderr, or whose error line or
        # stderr the hook recognises, is treated as a stale session.
        if (
            allow_retry
            and cmd_result.used_resume
            and not yielded_any
            and (
                stale
                or (proc.returncode is not None and proc.returncode > 0 and not stderr_text)
                or (
                    proc.returncode is not None
                    and proc.returncode >= 0
                    and self._is_stale_session_error("", stderr_text)
                )
            )
        ):
            self._forget_stale_session(context_key)
            # The retry gets a fresh timeout budget, as in _run_subprocess.
            async for chunk in self._run_subprocess_streaming(
                context_key,
                task_id,
                user_input=user_input,
                allow_retry=False,
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

    @staticmethod
    async def _feed_stdin(proc: Process, data: str) -> None:
        """Write ``data`` to the subprocess's stdin, then close it to signal EOF."""
        assert proc.stdin is not None
        try:
            proc.stdin.write(data.encode("utf-8"))
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            # The child exited before reading everything; its exit code and
            # stderr describe the real failure, so don't mask it here.
            pass
        finally:
            proc.stdin.close()

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
        except FileNotFoundError:
            self._sessions = {}
            return
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
            logger.warning(
                "Could not load sessions from %s, starting with none: %s",
                self._session_store_path,
                e,
            )
            self._sessions = {}
            return
        if isinstance(data, dict):
            self._sessions = {str(k): str(v) for k, v in data.items()}
            logger.debug(
                "Loaded %d session(s) from %s",
                len(self._sessions),
                self._session_store_path,
            )
        else:
            logger.warning(
                "Session store %s is not a JSON object, starting with none",
                self._session_store_path,
            )
            self._sessions = {}

    def _save_sessions(self) -> None:
        """Persist session mapping to disk atomically. Creates parent dirs.

        Writes to a temp file in the same directory and replaces the store
        with it, so a crash mid-write never leaves a truncated store.
        """
        tmp_path = None
        try:
            parent_dir = os.path.dirname(self._session_store_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                dir=parent_dir or ".",
                prefix=f".{os.path.basename(self._session_store_path)}.",
                suffix=".tmp",
            )
            with os.fdopen(fd, "w") as f:
                json.dump(self._sessions, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self._session_store_path)
            tmp_path = None
            logger.debug(
                "Saved %d session(s) to %s",
                len(self._sessions),
                self._session_store_path,
            )
        except OSError as e:
            logger.warning("Could not save sessions to %s: %s", self._session_store_path, e)
        finally:
            if tmp_path is not None:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)

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

    def _select_invoke_output(self, stdout_text: str, stderr_text: str) -> str:
        """Pick the text to parse from a successful run. Optional.

        Defaults to stdout. Override for CLIs that sometimes write their
        result to stderr instead.
        """
        return stdout_text

    def _is_stale_session_error(self, stdout_text: str, stderr_text: str) -> bool:
        """Whether a failed resumed run says its session no longer exists. Optional.

        When True, the stored session is dropped and the run is retried once
        without resume (never after streamed output reached the client).
        Defaults to False: only a resumed run with no output at all counts.
        Match the CLI's own error messages, not the whole transcript: model
        text or tool output may quote the same words. ``stdout_text`` is the
        run's stdout (or one error line when streaming).
        """
        return False

    def _extract_error(self, stdout_text: str) -> str | None:
        """Return the CLI's own error message from stdout of a failed run. Optional.

        Used for non-zero exits, where stderr often holds only warnings.
        """
        return None

    def _binary_not_found_message(self) -> str:
        """Error message when CLI binary not found. Override for custom message."""
        return "CLI binary not found. Ensure it is installed and in PATH."

    def get_metadata(self) -> AdapterMetadata:
        """Return adapter metadata for automatic AgentCard generation. Optional.

        Override to provide agent name, description, skills, and capabilities.
        The server layer uses this to auto-generate the AgentCard served at
        /.well-known/agent-card.json.
        """
        return AdapterMetadata()

    async def __aenter__(self):
        """Support async context manager for resource management."""
        return self

    async def __aexit__(self, *args):
        """Cleanup resources when exiting async context."""
        await self.close()
