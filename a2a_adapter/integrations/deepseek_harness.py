"""DeepSeek Harness (dsh) SDK adapter for the A2A Protocol.

Wraps ``deepseek-harness-sdk``: one ``DeepSeekHarness`` runtime (a local
``dsh`` subprocess speaking JSON-RPC over stdio) is launched lazily and reused
for every turn. The SDK is synchronous, so every SDK call runs on a single
worker thread and turns are serialized through one lock.

The SDK is imported lazily, so the adapter class always loads for
registry/loader purposes. Install it with
``pip install "a2a-adapter[deepseek-harness]"``.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
import re
import shutil
import uuid
import weakref
from collections.abc import AsyncIterator, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..base_adapter import AdapterMetadata, BaseA2AAdapter, default_state_dir
from ..exceptions import CancelledByAdapterError

logger = logging.getLogger(__name__)

_SAFE_SESSION_ID = re.compile(r"[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?")
_MAX_VERBATIM_CONTEXT_ID = 100

# Upgrading never adds an extra, so name the reinstall for each installer.
_INSTALL_COMMANDS = (
    "Add the deepseek-harness extra with the tool that installed a2a-adapter:\n"
    '  uv tool install "a2a-adapter[deepseek-harness]"\n'
    '  pipx install --force "a2a-adapter[deepseek-harness]"\n'
    '  pip install "a2a-adapter[deepseek-harness]"\n'
    "Supported platforms: Linux x64/arm64, macOS 14+, Windows x64."
)
_INSTALL_HINT = f"DeepSeekHarnessAdapter requires deepseek-harness-sdk.\n{_INSTALL_COMMANDS}"


class DeepSeekHarnessAdapter(BaseA2AAdapter):
    """Expose a DeepSeek Harness runtime as an A2A agent.

    Each A2A ``context_id`` maps to a durable dsh session, so conversations
    continue across turns and across server restarts:

    - an id matching ``[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?`` (up to 100
      characters) becomes ``a2a-<context_id>``; ids with uppercase letters are
      hashed, because dsh names session directories after the id and macOS
      and Windows filesystems ignore case;
    - any other id becomes ``a2ah-<first 16 hex chars of its sha256>``;
    - a request without a ``context_id`` gets a one-off ``a2a-<12 hex>`` session,
      which stays on disk under ``dsh_home`` like any other.

    The default ``sdk`` profile starts new sessions in dsh's
    ``workspace-write`` preset: shell commands and file writes are confined to
    the workspace and temp directories, but reads and network access are not
    (``web_fetch`` can reach loopback and private hosts). ``sdk-minimal`` runs
    with ``danger-full-access``, and ``DSH_PERMISSION_MODE``, ``patches`` or a
    custom profile can change the preset too; the warning logged for
    ``sdk-minimal`` is a best-effort notice, not a guarantee. Either way the
    agent executes tools for whoever can reach the A2A server, so keep the
    server on a trusted interface.

    ``dsh_bin`` defaults to the runtime bundled with the SDK. On Windows, point
    it at the runtime executable rather than the ``dsh`` console script, which
    waits on the runtime instead of replacing itself, so closing it would
    leave the runtime running.

    Sessions live under ``dsh_home``: the explicit argument, else a non-empty
    ``DSH_HOME`` environment variable, else a per-user, per-project state
    directory. Prefer the ``DEEPSEEK_API_KEY`` environment variable over the
    ``api_key`` argument; either way the key is only passed to the SDK.

    A turn that ends without ``completed`` (for example ``max-tokens``) but
    produced text returns that text, with a warning in the log; one that
    produced no text raises.

    The SDK has no interrupt call, so cancelling the running task closes the
    runtime; the next turn launches a new one and resumes its session. The SDK
    has no call to close one session either, so every conversation seen since
    the runtime launched stays open in it until the runtime is closed.

    Example::

        from a2a_adapter import DeepSeekHarnessAdapter, serve_agent

        adapter = DeepSeekHarnessAdapter(working_dir="/path/to/project")
        serve_agent(adapter, host="127.0.0.1", port=9010)
    """

    def __init__(
        self,
        working_dir: str,
        *,
        dsh_home: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
        profile: str = "sdk",
        patches: Sequence[str] = (),
        base_url: str | None = None,
        api_key: str | None = None,
        dsh_bin: str | None = None,
        timeout: float = 600,
        startup_timeout: float = 30,
        name: str = "",
        description: str = "",
        skills: list[dict] | None = None,
        agent_provider: dict | None = None,
        documentation_url: str | None = None,
        icon_url: str | None = None,
    ) -> None:
        if isinstance(patches, str):
            raise TypeError("patches must be a sequence of paths, not a string")

        self.working_dir = os.path.abspath(working_dir)
        self.dsh_home = self._resolve_dsh_home(dsh_home, self.working_dir)
        self.provider = provider
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.profile = profile
        if profile == "sdk-minimal":
            logger.warning(
                "DeepSeekHarnessAdapter: profile 'sdk-minimal' runs tools with "
                "danger-full-access (no sandbox or approvals). Only use it in "
                "trusted, sandboxed environments."
            )
        self.patches = tuple(patches)
        self.base_url = base_url
        self._api_key = api_key
        self.dsh_bin = dsh_bin
        self.timeout = timeout
        self.startup_timeout = startup_timeout

        self._name = name
        self._description = description
        self._skills = skills or []
        self._agent_provider = agent_provider
        self._documentation_url = documentation_url
        self._icon_url = icon_url

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dsh")
        self._harness: Any = None
        self._rpc_error_type: type[BaseException] | None = None
        # Live Session handles for the current runtime. Replaced (never
        # cleared in place) when the runtime goes away, so a worker that
        # finishes late can only write into the dict of its own runtime.
        self._session_handles: dict[str, Any] = {}
        self._startup_lock = asyncio.Lock()
        # Harnesses that were closed. The SDK launches its runtime lazily on
        # the first session call and close() before that launch is a no-op,
        # so a close can race a launch on the worker. _retire_harness marks
        # before closing and the worker checks after opening a session:
        # either the close sees the live runtime or the worker sees the mark
        # and closes it itself, so no runtime is left running.
        self._retired: weakref.WeakSet[Any] = weakref.WeakSet()
        self._turn_lock = asyncio.Lock()
        self._active_task_id: str | None = None
        self._cancelled_tasks: set[str] = set()
        self._closed = False

    # ──── Configuration ────

    @staticmethod
    def _resolve_dsh_home(dsh_home: str | None, working_dir: str) -> str:
        if dsh_home:
            return os.path.abspath(os.path.expanduser(dsh_home))
        env_home = os.environ.get("DSH_HOME", "").strip()
        if env_home:
            return os.path.abspath(os.path.expanduser(env_home))
        return default_state_dir("deepseek-harness", working_dir)

    @staticmethod
    def _session_key(context_id: str | None) -> str:
        """Map an A2A context_id to a deterministic dsh session id."""
        if not context_id:
            return f"a2a-{uuid.uuid4().hex[:12]}"
        if len(context_id) <= _MAX_VERBATIM_CONTEXT_ID and _SAFE_SESSION_ID.fullmatch(context_id):
            return f"a2a-{context_id}"
        # A separate prefix keeps a hashed id from matching a verbatim one.
        digest = hashlib.sha256(context_id.encode("utf-8")).hexdigest()[:16]
        return f"a2ah-{digest}"

    # ──── Required: invoke ────

    async def invoke(self, user_input: str, context_id: str | None = None, **kwargs: Any) -> str:
        context = kwargs.get("context")
        task_id = context.task_id if context else uuid.uuid4().hex
        session_id = self._session_key(context_id)
        result = await self._turn(task_id, session_id, user_input, bool(context_id))
        return self._check_result(result, session_id)

    # ──── Optional: stream ────

    async def stream(
        self, user_input: str, context_id: str | None = None, **kwargs: Any
    ) -> AsyncIterator[str]:
        """Yield the text of each assistant message as the runtime finishes it.

        The runtime reports whole messages, not tokens, so a turn that calls
        tools streams one chunk per model step, joined by blank lines;
        ``invoke`` returns only the last message.
        """
        context = kwargs.get("context")
        task_id = context.task_id if context else uuid.uuid4().hex
        session_id = self._session_key(context_id)
        loop = asyncio.get_running_loop()
        chunks: asyncio.Queue[str] = asyncio.Queue()

        def on_event(event: dict) -> None:
            # Runs on the worker thread.
            text = _message_text(event)
            if text:
                # An abandoned turn may still report after the loop closed.
                with contextlib.suppress(RuntimeError):
                    loop.call_soon_threadsafe(chunks.put_nowait, text)

        turn = asyncio.create_task(
            self._turn(task_id, session_id, user_input, bool(context_id), on_event=on_event)
        )
        # Read the outcome even when the consumer leaves before the turn ends.
        turn.add_done_callback(_consume_result)
        streamed = False
        getter: asyncio.Future[str] | None = None
        try:
            while not turn.done():
                getter = asyncio.ensure_future(chunks.get())
                done, _ = await asyncio.wait({getter, turn}, return_when=asyncio.FIRST_COMPLETED)
                if getter not in done:
                    getter.cancel()
                    continue
                if task_id in self._cancelled_tasks or _cancelled(turn):
                    # cancel() already reported the task; send nothing more.
                    continue
                yield f"\n\n{getter.result()}" if streamed else getter.result()
                streamed = True
            if _cancelled(turn):
                raise turn.exception()
            # The worker queues every chunk before the turn completes.
            while not chunks.empty():
                self._raise_if_cancelled_late(task_id)
                text = chunks.get_nowait()
                yield f"\n\n{text}" if streamed else text
                streamed = True
            text = self._check_result(turn.result(), session_id)
            if not streamed and text:
                self._raise_if_cancelled_late(task_id)
                yield text
            self._raise_if_cancelled_late(task_id)
        finally:
            if getter is not None:
                getter.cancel()
            if not turn.done():
                # The consumer went away: cancelling the turn closes the runtime.
                # wait() lets a cancellation of this task propagate.
                turn.cancel()
                await asyncio.wait({turn})

    async def _turn(
        self,
        task_id: str,
        session_id: str,
        prompt: str,
        resumable: bool,
        *,
        on_event: Callable[[dict], None] | None = None,
    ) -> Any:
        """Run one turn under the turn lock, honouring cancel() for ``task_id``."""
        self._raise_if_cancelled(task_id)
        async with self._turn_lock:
            self._raise_if_cancelled(task_id)
            self._active_task_id = task_id
            try:
                result = await self._run_turn_relaunching(
                    task_id, session_id, prompt, resumable=resumable, on_event=on_event
                )
                if task_id in self._cancelled_tasks:
                    # cancel() already reported the task as cancelled.
                    raise CancelledByAdapterError(f"Task {task_id} was cancelled")
            except CancelledByAdapterError:
                raise
            except Exception as error:
                if task_id in self._cancelled_tasks:
                    raise CancelledByAdapterError(f"Task {task_id} was cancelled") from error
                raise
            finally:
                self._active_task_id = None
                self._cancelled_tasks.discard(task_id)
        return result

    # ──── Optional: cancel ────

    async def cancel(self, context_id: str | None = None, **kwargs: Any) -> None:
        """Cancel a task: drop the runtime if it is running, else skip it when it starts."""
        context = kwargs.get("context")
        if context is None:
            return
        task_id = context.task_id
        self._cancelled_tasks.add(task_id)
        harness = self._harness
        if task_id == self._active_task_id and harness is not None:
            await self._drop_harness(harness)

    # ──── Optional: metadata ────

    def get_metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name=self._name or "DeepSeek Harness",
            description=self._description or "DeepSeek Harness coding agent",
            streaming=True,
            skills=self._skills,
            provider=self._agent_provider,
            documentation_url=self._documentation_url,
            icon_url=self._icon_url,
        )

    # ──── Optional: cleanup ────

    async def close(self) -> None:
        """Close the runtime, then the worker thread."""
        async with self._startup_lock:
            if self._closed:
                return
            self._closed = True
            harness = self._harness
            self._harness = None
            self._session_handles = {}
        if harness is not None:
            await self._close_harness(harness)
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._cancelled_tasks.clear()

    # ──── Internal: runtime lifecycle ────

    async def _ensure_harness(self) -> Any:
        async with self._startup_lock:
            if self._closed:
                raise RuntimeError("DeepSeekHarnessAdapter is closed")
            if self._harness is None:
                self._harness = self._create_harness()
                self._session_handles = {}
            return self._harness

    def _create_harness(self) -> Any:
        """Build the SDK object. The runtime itself launches on first use."""
        try:
            from deepseek_harness import DeepSeekHarness
            from deepseek_harness.errors import JsonRpcError
        except ImportError as error:
            raise ImportError(_INSTALL_HINT) from error

        os.makedirs(self.dsh_home, exist_ok=True)
        options: dict[str, Any] = {
            "cwd": self.working_dir,
            "dsh_home": self.dsh_home,
            "profile": self.profile,
            "patches": self.patches,
            "initialize_timeout_seconds": self.startup_timeout,
            "request_timeout_seconds": self.timeout,
        }
        # Unset options are left out so the SDK keeps its own defaults.
        optional = {
            "provider": self.provider,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "max_tokens": self.max_tokens,
            "base_url": self.base_url,
            "api_key": self._api_key,
            "dsh_bin": resolve_dsh_bin(self.dsh_bin),
        }
        options.update({key: value for key, value in optional.items() if value is not None})
        self._rpc_error_type = JsonRpcError
        return DeepSeekHarness(**options)

    async def _drop_harness(self, harness: Any) -> None:
        """Retire ``harness`` if it is still current; a stale one is left alone."""
        async with self._startup_lock:
            if harness is not self._harness:
                return
            self._harness = None
            self._session_handles = {}
        await self._close_harness(harness)

    async def _close_harness(self, harness: Any) -> None:
        try:
            await asyncio.to_thread(self._retire_harness, harness)
        except Exception:
            # The SDK fails pending calls only at the end of close(), so the
            # worker may still be blocked on this runtime. Give later turns a
            # fresh worker instead of queueing them behind it.
            logger.error("Closing the DeepSeek Harness runtime failed", exc_info=True)
            stuck, self._executor = self._executor, ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="dsh"
            )
            stuck.shutdown(wait=False, cancel_futures=True)

    def _retire_harness(self, harness: Any) -> None:
        self._retired.add(harness)
        harness.close()

    def _raise_if_retired(self, harness: Any) -> None:
        if harness in self._retired:
            raise RuntimeError("DeepSeek Harness runtime was closed before the turn started")

    # ──── Internal: turns ────

    async def _run_turn_relaunching(
        self,
        task_id: str,
        session_id: str,
        prompt: str,
        *,
        resumable: bool,
        on_event: Callable[[dict], None] | None = None,
    ) -> Any:
        """Run one turn, relaunching once if the runtime died while idle."""
        for attempt in range(2):
            harness = await self._ensure_harness()
            if task_id in self._cancelled_tasks:
                raise CancelledByAdapterError(f"Task {task_id} was cancelled")
            try:
                return await self._run_turn(
                    harness, session_id, prompt, resumable=resumable, on_event=on_event
                )
            except _RuntimeExitedError:
                # Nothing was sent, and _run_turn already dropped the runtime.
                if attempt:
                    raise
        raise AssertionError("unreachable")  # pragma: no cover

    async def _run_turn(
        self,
        harness: Any,
        session_id: str,
        prompt: str,
        *,
        resumable: bool,
        on_event: Callable[[dict], None] | None = None,
    ) -> Any:
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            self._executor,
            self._turn_job,
            harness,
            self._session_handles,
            session_id,
            prompt,
            resumable,
            on_event,
        )
        try:
            done, _ = await asyncio.wait({future}, timeout=self.timeout)
        except asyncio.CancelledError:
            # The caller went away. Closing the runtime releases the worker.
            future.add_done_callback(_consume_result)
            await self._drop_harness(harness)
            raise
        if not done:
            future.add_done_callback(_consume_result)
            await self._drop_harness(harness)
            raise TimeoutError(f"DeepSeek Harness turn timed out after {self.timeout} seconds")

        try:
            return future.result()
        except Exception as error:
            # A JSON-RPC error is an answer from a healthy runtime. Anything
            # else (exited child, protocol error, request timeout) may leave
            # it unusable, so the next turn starts a fresh one.
            if self._rpc_error_type is None or not isinstance(error, self._rpc_error_type):
                await self._drop_harness(harness)
            raise

    def _turn_job(
        self,
        harness: Any,
        handles: dict[str, Any],
        session_id: str,
        prompt: str,
        resumable: bool,
        on_event: Callable[[dict], None] | None = None,
    ) -> Any:
        """Worker-thread job: open the session if needed, then run one turn.

        Opening a session is itself a JSON-RPC call that may launch the
        runtime, so it must not run on the event loop.
        """
        self._raise_if_retired(harness)
        if _runtime_exited(harness):
            raise _RuntimeExitedError("DeepSeek Harness runtime exited before the turn started")
        session = handles.get(session_id)
        try:
            if session is None:
                if resumable:
                    session = self._resume_or_start(harness, session_id)
                    handles[session_id] = session
                else:
                    session = harness.start_session(session_id)
        finally:
            if harness in self._retired:
                # Retired while this job was launching it: that close may
                # have run before the launch and done nothing.
                harness.close()
        self._raise_if_retired(harness)
        # A close after this point makes run() fail: Session.run only writes
        # to a running runtime and never relaunches it.
        if on_event is None:
            return session.run(prompt)

        def forward(notification: Any) -> None:
            payload = notification.payload
            if notification.method == "session.event" and payload.get("sessionId") == session_id:
                event = payload.get("event")
                if isinstance(event, dict):
                    on_event(event)

        return session.run(prompt, on_notification=forward)

    def _resume_or_start(self, harness: Any, session_id: str) -> Any:
        try:
            return harness.resume_session(session_id)
        except Exception as error:
            # The runtime reports a missing session with a generic JSON-RPC
            # code, so the exact message is the only signal.
            if (
                self._rpc_error_type is not None
                and isinstance(error, self._rpc_error_type)
                and str(error) == f'session "{session_id}" not found'
            ):
                return harness.start_session(session_id)
            raise

    @staticmethod
    def _check_result(result: Any, session_id: str) -> str:
        """Return the reply; raise only when the turn failed with nothing to show."""
        text = result.final_response or ""
        reason = result.finish_reason
        detail = _turn_error(result.events) if reason != "completed" else ""
        suffix = f": {detail}" if detail else ""
        if reason != "completed":
            logger.warning(
                "DeepSeek Harness turn for session %s finished with %r%s",
                session_id,
                reason,
                suffix,
            )
        if text or reason == "completed":
            return text
        raise RuntimeError(
            f"DeepSeek Harness turn produced no response (finish reason: {reason!r}){suffix}"
        )

    def _raise_if_cancelled_late(self, task_id: str) -> None:
        """Stop a stream whose task was cancelled after its turn finished."""
        if task_id in self._cancelled_tasks:
            self._cancelled_tasks.discard(task_id)
            raise CancelledByAdapterError(f"Task {task_id} was cancelled")

    def _raise_if_cancelled(self, task_id: str) -> None:
        if task_id in self._cancelled_tasks:
            self._cancelled_tasks.discard(task_id)
            raise CancelledByAdapterError(f"Task {task_id} was cancelled before execution started")


def resolve_dsh_bin(dsh_bin: str | None) -> str | None:
    """Look a bare executable name up on PATH; the SDK only accepts paths."""
    if dsh_bin is None or os.sep in dsh_bin or (os.altsep and os.altsep in dsh_bin):
        return dsh_bin
    found = shutil.which(dsh_bin)
    if found is None:
        raise FileNotFoundError(f"dsh executable {dsh_bin!r} not found on PATH")
    return found


class _RuntimeExitedError(RuntimeError):
    """The runtime's child process exited between turns; nothing was sent."""


def _runtime_exited(harness: Any) -> bool:
    """Whether the SDK's child exited without close().

    The SDK keeps a dead child until close() and its start() does not
    relaunch it, so the next call would fail. This reads the SDK's private
    ``client._proc``; if that changes, the check is skipped.
    """
    proc = getattr(getattr(harness, "client", None), "_proc", None)
    poll = getattr(proc, "poll", None)
    return callable(poll) and poll() is not None


def _cancelled(turn: asyncio.Task) -> bool:
    """Whether a finished turn ended because cancel() was called for its task."""
    return turn.done() and isinstance(turn.exception(), CancelledByAdapterError)


def _message_text(event: dict) -> str:
    """Return the text blocks of an ``assistant/message`` event, as the SDK reads them."""
    if event.get("type") != "assistant/message":
        return ""
    data = event.get("data")
    if not isinstance(data, dict):
        return ""
    message = data.get("message")
    content = (message if isinstance(message, dict) else data).get("content")
    if not isinstance(content, list):
        return ""
    return "".join(
        str(block.get("text") or "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _turn_error(events: list[Any]) -> str:
    """Return ``code: message`` from the last ``turn/end`` error, or ``""``."""
    for event in reversed(events or []):
        if not isinstance(event, dict) or event.get("type") != "turn/end":
            continue
        data = event.get("data")
        reason = data.get("reason") if isinstance(data, dict) else None
        error = reason.get("error") if isinstance(reason, dict) else None
        if not isinstance(error, dict):
            return ""
        message = str(error.get("message") or "")
        code = str(error.get("code") or "")
        return f"{code}: {message}" if code and message else code or message
    return ""


def _consume_result(future: asyncio.Future) -> None:
    """Retrieve an abandoned turn's outcome so asyncio does not log it as unhandled."""
    if not future.cancelled():
        future.exception()
