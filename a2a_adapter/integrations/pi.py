"""Pi coding agent adapter for the A2A Protocol."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
from asyncio.subprocess import PIPE, Process, create_subprocess_exec
from collections import deque
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import suppress
from typing import Any

from ..base_adapter import (
    AdapterMetadata,
    BaseA2AAdapter,
    _child_env,
    _forget_own_group,
    _forget_own_group_if_empty,
    _kill_process_tree,
    _owns_group,
    _spawn_cli,
    _wait_reaped,
    default_state_dir,
)
from ..exceptions import CancelledByAdapterError


logger = logging.getLogger(__name__)

_BLOCKING_EXTENSION_UI_METHODS = {"select", "confirm", "input", "editor"}


_GROUP_POLL_INTERVAL = 0.5  # how often to check whether Pi's exited group is empty


class PiRpcError(RuntimeError):
    """Base error for Pi RPC failures."""


class PiCommandError(PiRpcError):
    """Pi rejected an RPC command."""


class PiCommandTimeoutError(PiRpcError):
    """Pi did not acknowledge an RPC command before its deadline."""


class PiProtocolError(PiRpcError):
    """Pi emitted malformed or inconsistent protocol output."""


class PiTurnAbortedError(PiRpcError):
    """The active Pi turn was aborted."""


class PiTurnTimeoutError(PiRpcError):
    """The active Pi turn exceeded its configured timeout."""


class PiProcessLostError(PiRpcError):
    """The Pi process exited during a turn.

    ``accepted`` distinguishes safe pre-acceptance failures from an uncertain
    outcome after Pi accepted a prompt and may have executed side effects.
    """

    def __init__(self, message: str, *, accepted: bool) -> None:
        super().__init__(message)
        self.accepted = accepted


class _ProcessExitedError(PiRpcError):
    """Internal process-exit signal used by pending requests and event readers."""


class PiRpcProcess:
    """Own one long-lived Pi RPC process and one durable Pi session."""

    def __init__(
        self,
        *,
        command: Sequence[str],
        cli_args: Sequence[str] | None = None,
        working_dir: str,
        session_id: str,
        session_dir: str,
        env_vars: dict[str, str] | None = None,
        provider: str | None = None,
        model: str | None = None,
        timeout: float = 600,
        startup_timeout: float = 30,
        cancel_grace_timeout: float = 10,
    ) -> None:
        if not command:
            raise ValueError("Pi command must not be empty")
        if isinstance(cli_args, str):
            raise TypeError("cli_args must be a sequence of arguments, not a string")

        self.command = list(command)
        self.cli_args = list(cli_args or [])
        self.working_dir = working_dir
        self.session_id = session_id
        self.session_dir = session_dir
        self.env_vars = dict(env_vars) if env_vars else {}
        self.provider = provider
        self.model = model
        self.timeout = timeout
        self.startup_timeout = startup_timeout
        self.cancel_grace_timeout = cancel_grace_timeout

        self._process: Process | None = None
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._events: asyncio.Queue[dict[str, Any] | BaseException] = asyncio.Queue()
        self._write_lock = asyncio.Lock()
        self._request_number = 0
        self._stdout_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._watch_task: asyncio.Task[None] | None = None
        self._group_task: asyncio.Task[None] | None = None
        self._failure: BaseException | None = None
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None
        self._turn_active = False
        self._turn_settled = asyncio.Event()
        self._turn_settled.set()
        self._stderr_tail: deque[bytes] = deque(maxlen=64)

    @property
    def is_running(self) -> bool:
        return (
            self._process is not None
            and self._process.returncode is None
            and self._failure is None
            and not self._closing
        )

    async def start(self) -> None:
        """Start Pi and verify protocol readiness with ``get_state``."""
        if self.is_running:
            return
        if self._process is not None:
            raise PiRpcError("A PiRpcProcess instance cannot be restarted")
        if not os.path.isdir(self.working_dir):
            raise FileNotFoundError(f"Working directory does not exist: '{self.working_dir}'")

        os.makedirs(self.session_dir, exist_ok=True)
        args = [
            *self.command,
            *self.cli_args,
            "--mode",
            "rpc",
            "--session-id",
            self.session_id,
            "--session-dir",
            self.session_dir,
        ]
        if self.provider:
            args.extend(["--provider", self.provider])
        if self.model:
            args.extend(["--model", self.model])

        env = _child_env(self.env_vars)

        try:
            self._process = await _spawn_cli(
                create_subprocess_exec,
                *args,
                stdin=PIPE,
                stdout=PIPE,
                stderr=PIPE,
                cwd=self.working_dir,
                env=env,
                limit=10 * 1024 * 1024,
            )
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Pi executable not found at '{self.command[0]}'. "
                "Install Pi or provide a source command with pi_command."
            ) from error

        self._stdout_task = asyncio.create_task(self._read_stdout())
        self._stderr_task = asyncio.create_task(self._read_stderr())
        self._watch_task = asyncio.create_task(self._watch_process())
        self._group_task = asyncio.create_task(self._watch_group())

        try:
            response = await self.request({"type": "get_state"}, timeout=self.startup_timeout)
            actual_session_id = response.get("data", {}).get("sessionId")
            if actual_session_id != self.session_id:
                raise PiProtocolError(
                    "Pi readiness check returned an unexpected session ID: "
                    f"{actual_session_id!r}"
                )
        except Exception:
            await self.close()
            raise

    async def request(
        self, command: dict[str, Any], *, timeout: float | None = None
    ) -> dict[str, Any]:
        """Send one correlated RPC command and await its response."""
        if not self.is_running:
            if self._failure:
                raise self._failure
            raise PiRpcError("Pi RPC process is not running")

        self._request_number += 1
        request_id = f"a2a-{self._request_number}"
        payload = dict(command)
        payload["id"] = request_id

        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[request_id] = future

        try:
            await self._write_payload(payload)
            response = await asyncio.wait_for(
                asyncio.shield(future), timeout=timeout or self.timeout
            )
        except asyncio.TimeoutError:
            self._pending.pop(request_id, None)
            future.cancel()
            raise PiCommandTimeoutError(
                f"Pi RPC command {command.get('type')!r} timed out"
            ) from None
        except asyncio.CancelledError:
            self._pending.pop(request_id, None)
            future.cancel()
            raise
        except Exception:
            self._pending.pop(request_id, None)
            raise

        if not response.get("success", False):
            error = response.get("error") or "unknown Pi RPC error"
            raise PiCommandError(f"Pi rejected {command.get('type')!r}: {error}")
        return response

    async def prompt(self, message: str) -> AsyncIterator[str]:
        """Run one prompt, yielding text deltas until ``agent_settled``."""
        if self._turn_active:
            raise PiRpcError("Pi already has an active turn")

        self._turn_active = True
        self._turn_settled.clear()
        accepted = False
        turn_error: PiRpcError | None = None

        try:
            try:
                await self.request({"type": "prompt", "message": message})
                accepted = True
            except _ProcessExitedError as error:
                raise self._process_lost(error, accepted=False) from error
            except PiCommandTimeoutError:
                # The prompt may have been accepted even though its response
                # was lost or delayed. Retire the process so session-scoped
                # events cannot leak into the next turn.
                await self._terminate_process()
                raise

            try:
                async with asyncio.timeout(self.timeout):
                    while True:
                        event = await self._events.get()
                        if isinstance(event, BaseException):
                            raise self._process_lost(event, accepted=accepted) from event

                        event_type = event.get("type")
                        if event_type == "message_update":
                            delta = event.get("assistantMessageEvent", {})
                            delta_type = delta.get("type")
                            if delta_type == "text_delta":
                                text = delta.get("delta", "")
                                if text:
                                    yield str(text)
                            elif delta_type == "error":
                                reason = str(delta.get("reason", "error"))
                                if reason == "aborted":
                                    turn_error = PiTurnAbortedError("Pi turn was aborted")
                                else:
                                    turn_error = PiRpcError(f"Pi assistant stream failed: {reason}")
                        elif event_type == "agent_settled":
                            if turn_error:
                                raise turn_error
                            return
            except TimeoutError:
                await self._terminate_process()
                raise PiTurnTimeoutError(
                    f"Pi turn timed out after {self.timeout} seconds"
                ) from None
        except (asyncio.CancelledError, GeneratorExit):
            await self._abort_cancelled_coroutine()
            raise
        finally:
            self._turn_active = False
            self._turn_settled.set()

    async def abort(self) -> None:
        """Abort the active turn, killing Pi only if it does not settle."""
        if not self._turn_active or not self.is_running:
            return

        try:
            await self.request({"type": "abort"}, timeout=self.cancel_grace_timeout)
            await asyncio.wait_for(self._turn_settled.wait(), timeout=self.cancel_grace_timeout)
        except (PiRpcError, asyncio.TimeoutError):
            await self._terminate_process()

    async def close(self) -> None:
        """Close the process and all background protocol tasks.

        The close runs once, as its own task: cancelling a caller (or a
        repeat call) does not cut it short, and every caller waits for it.
        """
        if self._close_task is None:
            self._closing = True
            self._close_task = asyncio.create_task(self._close(asyncio.current_task()))
        await asyncio.shield(self._close_task)

    async def _close(self, caller: asyncio.Task | None) -> None:
        # A registered process group is signalled even after Pi exited on its
        # own: Pi's tool children can outlive it.
        if self._process is not None and (
            self._process.returncode is None or _owns_group(self._process)
        ):
            self._signal_failure(PiRpcError("Pi RPC process is closing"))
            try:
                await self._terminate_process()
            except BaseException:
                # Cancelled during the SIGTERM grace (e.g. loop shutdown):
                # don't leave the group half-stopped.
                self.kill_now()
                raise

        tasks = [self._stdout_task, self._stderr_task, self._watch_task, self._group_task]
        for task in tasks:
            if task is not None and task is not caller and not task.done():
                task.cancel()
        for task in tasks:
            if task is not None and task is not caller:
                with suppress(asyncio.CancelledError):
                    await task

    async def _read_stdout(self) -> None:
        process = self._process
        assert process is not None and process.stdout is not None

        try:
            while True:
                line = await process.stdout.readline()
                if not line:
                    return
                try:
                    value = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    protocol_error = PiProtocolError(f"Pi emitted malformed JSONL: {error}")
                    self._signal_failure(protocol_error)
                    await self._terminate_process()
                    return

                if not isinstance(value, dict):
                    protocol_error = PiProtocolError(
                        "Pi emitted a JSONL record that was not an object"
                    )
                    self._signal_failure(protocol_error)
                    await self._terminate_process()
                    return

                if value.get("type") == "response" and value.get("id"):
                    request_id = str(value["id"])
                    future = self._pending.pop(request_id, None)
                    if future is not None and not future.done():
                        future.set_result(value)
                    continue

                if value.get("type") == "extension_ui_request":
                    method = value.get("method")
                    request_id = value.get("id")
                    if method in _BLOCKING_EXTENSION_UI_METHODS and request_id:
                        await self._write_payload(
                            {
                                "type": "extension_ui_response",
                                "id": request_id,
                                "cancelled": True,
                            }
                        )
                    continue

                await self._events.put(value)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._signal_failure(PiProtocolError(f"Pi stdout reader failed: {error}"))

    async def _read_stderr(self) -> None:
        process = self._process
        assert process is not None and process.stderr is not None
        try:
            while True:
                chunk = await process.stderr.read(8192)
                if not chunk:
                    return
                self._stderr_tail.append(chunk)
        except asyncio.CancelledError:
            raise

    async def _watch_process(self) -> None:
        process = self._process
        assert process is not None
        # Known gap: wait() returns only once all of Pi's pipes close, so while
        # a grandchild still holds them Pi's own death goes unnoticed here.
        return_code = await process.wait()
        if not self._closing:
            stderr = b"".join(self._stderr_tail).decode("utf-8", errors="replace")[-2000:].strip()
            message = f"Pi RPC process exited with code {return_code}"
            if stderr:
                logger.error("%s; stderr tail: %s", message, stderr)
            self._signal_failure(_ProcessExitedError(message))

    async def _watch_group(self) -> None:
        """Forget Pi's process group once Pi exited and the group is empty.

        Pi's tool children may outlive it: until then the group stays ours,
        so close(), kill_now() and the exit hook still kill them. Runs apart
        from _watch_process, whose wait() also waits for every pipe to close
        (a child in its own session can hold them long after the group empties).
        """
        process = self._process
        assert process is not None
        while _owns_group(process):
            await asyncio.sleep(_GROUP_POLL_INTERVAL)
            _forget_own_group_if_empty(process)

    async def _write_payload(self, payload: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None or not self.is_running:
            raise PiRpcError("Pi RPC process is not running")
        encoded = (json.dumps(payload, separators=(",", ":")) + "\n").encode()
        async with self._write_lock:
            process.stdin.write(encoded)
            await process.stdin.drain()

    async def _abort_cancelled_coroutine(self) -> None:
        """Prevent a disconnected caller from leaving a turn running in Pi."""
        if self.is_running:
            with suppress(PiRpcError, asyncio.TimeoutError):
                await self.request({"type": "abort"}, timeout=self.cancel_grace_timeout)
        await self._terminate_process()

    def _signal_failure(self, error: BaseException) -> None:
        if self._failure is not None:
            return
        self._failure = error
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)
        self._pending.clear()
        self._events.put_nowait(error)
        self._turn_settled.set()

    def kill_now(self) -> None:
        """SIGKILL Pi and its process group immediately (signal-handler safe)."""
        if self._process is not None:
            _kill_process_tree(self._process)
            # SIGKILL can't be ignored: the group is done, and its ID must not
            # be signalled again once it is reused.
            _forget_own_group(self._process)

    async def _terminate_process(self) -> None:
        """SIGTERM Pi's process group, wait the grace period, then SIGKILL it.

        For a group we spawned, the group is signalled even when Pi already
        exited, since its tool children may not have; only the waits are
        skipped then.
        """
        process = self._process
        if process is None:
            return
        owned = _owns_group(process)
        if process.returncode is not None and not owned:
            return

        leader_running = process.returncode is None
        _kill_process_tree(process, signal.SIGTERM)
        if leader_running:
            try:
                await asyncio.wait_for(process.wait(), timeout=self.cancel_grace_timeout)
                if not owned:
                    return
            except asyncio.TimeoutError:
                pass
        _kill_process_tree(process)
        # SIGKILL can't be ignored: the group is done, and its ID may be
        # reused once its members are reaped.
        _forget_own_group(process)
        if leader_running:
            await _wait_reaped(process)

    @staticmethod
    def _process_lost(error: BaseException, *, accepted: bool) -> PiProcessLostError:
        qualifier = "after" if accepted else "before"
        suffix = (
            " The execution outcome is unknown and the prompt was not replayed." if accepted else ""
        )
        return PiProcessLostError(
            f"Pi process was lost {qualifier} accepting the prompt: {error}.{suffix}",
            accepted=accepted,
        )


class PiAdapter(BaseA2AAdapter):
    """Expose one persistent Pi RPC process and session as an A2A agent.

    The first request starts Pi lazily. Every later request reuses the same
    process and durable session. Turns are globally serialized because Pi's
    asynchronous agent events are session-scoped rather than prompt-correlated.
    """

    def __init__(
        self,
        working_dir: str,
        *,
        pi_command: Sequence[str] | None = None,
        cli_args: Sequence[str] | None = None,
        session_id: str = "a2a-pi-agent",
        session_dir: str | None = None,
        timeout: float = 600,
        startup_timeout: float = 30,
        cancel_grace_timeout: float = 10,
        env_vars: dict[str, str] | None = None,
        model: str | None = None,
        model_provider: str | None = None,
        enforce_single_context: bool = True,
        name: str = "",
        description: str = "",
        skills: list[dict] | None = None,
        provider: dict | None = None,
        documentation_url: str | None = None,
        icon_url: str | None = None,
        _rpc_factory: Callable[[], PiRpcProcess] | None = None,
    ) -> None:
        if isinstance(pi_command, str):
            raise TypeError("pi_command must be a sequence of arguments, not a string")
        if isinstance(cli_args, str):
            raise TypeError("cli_args must be a sequence of arguments, not a string")
        if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", session_id):
            raise ValueError(
                "session_id must contain only alphanumeric characters, '-', '_', "
                "and '.', and must start and end with an alphanumeric character"
            )

        self.working_dir = working_dir
        self.pi_command = list(pi_command or ["pi"])
        self.cli_args = list(cli_args or [])
        self.session_id = session_id
        self.session_dir = session_dir or os.path.join(
            default_state_dir("pi", working_dir), "sessions"
        )
        self.timeout = timeout
        self.startup_timeout = startup_timeout
        self.cancel_grace_timeout = cancel_grace_timeout
        self.env_vars = dict(env_vars) if env_vars else {}
        self.model = model
        self.model_provider = model_provider
        self.enforce_single_context = enforce_single_context

        self._name = name
        self._description = description
        self._skills = skills or []
        self._provider = provider
        self._documentation_url = documentation_url
        self._icon_url = icon_url

        self._rpc: PiRpcProcess | None = None
        self._rpc_factory = _rpc_factory
        self._startup_lock = asyncio.Lock()
        self._turn_lock = asyncio.Lock()
        self._active_task_id: str | None = None
        self._cancelled_tasks: set[str] = set()
        self._bound_context_id: str | None = None
        self._closed = False

    async def invoke(self, user_input: str, context_id: str | None = None, **kwargs: Any) -> str:
        chunks = [chunk async for chunk in self.stream(user_input, context_id=context_id, **kwargs)]
        return "".join(chunks)

    async def stream(
        self, user_input: str, context_id: str | None = None, **kwargs: Any
    ) -> AsyncIterator[str]:
        context = kwargs.get("context")
        task_id = context.task_id if context else (context_id or "_direct")

        self._raise_if_cancelled(task_id)
        async with self._turn_lock:
            self._raise_if_cancelled(task_id)
            self._bind_context(context_id)
            rpc = await self._ensure_started()
            self._active_task_id = task_id
            lost = False
            try:
                async for chunk in rpc.prompt(user_input):
                    yield chunk
            except PiTurnAbortedError as error:
                if task_id in self._cancelled_tasks:
                    raise CancelledByAdapterError(f"Task {task_id} was cancelled") from error
                raise
            except PiProcessLostError as error:
                lost = True
                if task_id in self._cancelled_tasks:
                    raise CancelledByAdapterError(f"Task {task_id} was cancelled") from error
                raise
            finally:
                self._active_task_id = None
                self._cancelled_tasks.discard(task_id)
                if lost or not rpc.is_running:
                    # Kills what Pi left in its process group. _rpc is cleared
                    # only afterwards so kill_children_now() can still reach it.
                    try:
                        # Shielded: cancelling this request must not abandon
                        # the close between SIGTERM and SIGKILL.
                        await asyncio.shield(rpc.close())
                    finally:
                        if self._rpc is rpc:
                            self._rpc = None

    async def cancel(self, context_id: str | None = None, **kwargs: Any) -> None:
        context = kwargs.get("context")
        if context is None:
            return
        task_id = context.task_id
        self._cancelled_tasks.add(task_id)

        if task_id == self._active_task_id and self._rpc is not None:
            await self._rpc.abort()

    def _release_task(self, task_id: str) -> None:
        self._cancelled_tasks.discard(task_id)

    def kill_children_now(self) -> None:
        """SIGKILL the Pi RPC process group, synchronously (no SIGTERM grace)."""
        rpc = self._rpc
        if rpc is not None:
            rpc.kill_now()

    async def close(self) -> None:
        async with self._startup_lock:
            if self._closed:
                return
            self._closed = True
            rpc = self._rpc
            self._rpc = None
        if rpc is not None:
            await rpc.close()

    def get_metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name=self._name or "Pi",
            description=self._description or "Pi coding agent",
            streaming=True,
            skills=self._skills,
            provider=self._provider,
            documentation_url=self._documentation_url,
            icon_url=self._icon_url,
        )

    async def _ensure_started(self) -> PiRpcProcess:
        if self._closed:
            raise RuntimeError("PiAdapter is closed")
        if self._rpc is not None and self._rpc.is_running:
            return self._rpc

        async with self._startup_lock:
            if self._closed:
                raise RuntimeError("PiAdapter is closed")
            if self._rpc is not None and self._rpc.is_running:
                return self._rpc

            stale_rpc = self._rpc
            self._rpc = None
            if stale_rpc is not None:
                await stale_rpc.close()

            rpc = self._create_rpc()
            try:
                await rpc.start()
            except Exception:
                await rpc.close()
                raise
            self._rpc = rpc
            return rpc

    def _create_rpc(self) -> PiRpcProcess:
        if self._rpc_factory is not None:
            return self._rpc_factory()
        return PiRpcProcess(
            command=self.pi_command,
            cli_args=self.cli_args,
            working_dir=self.working_dir,
            session_id=self.session_id,
            session_dir=self.session_dir,
            env_vars=self.env_vars,
            provider=self.model_provider,
            model=self.model,
            timeout=self.timeout,
            startup_timeout=self.startup_timeout,
            cancel_grace_timeout=self.cancel_grace_timeout,
        )

    def _bind_context(self, context_id: str | None) -> None:
        if not self.enforce_single_context or context_id is None:
            return
        if self._bound_context_id is None:
            self._bound_context_id = context_id
            return
        if self._bound_context_id != context_id:
            raise RuntimeError(
                "This PiAdapter exposes one session and is already bound to "
                f"A2A context {self._bound_context_id!r}; received {context_id!r}"
            )

    def _raise_if_cancelled(self, task_id: str) -> None:
        if task_id in self._cancelled_tasks:
            self._cancelled_tasks.discard(task_id)
            raise CancelledByAdapterError(f"Task {task_id} was cancelled before execution started")
