"""
Hermes Agent adapter for A2A Protocol (Gateway Pattern).

This module provides HermesAdapter — a BaseA2AAdapter subclass that wraps
Hermes Agent's AIAgent via the same gateway pattern used by Hermes's own
messaging platforms (Telegram, Discord, Slack, etc.):

1. Load conversation history from SessionDB (SQLite) using context_id.
2. Create a fresh AIAgent instance per turn.
3. Call agent.run_conversation() with the loaded history.
4. Let AIAgent persist the updated history back to SessionDB internally.

AIAgent.run_conversation() is synchronous, so all calls are dispatched to
a ThreadPoolExecutor via loop.run_in_executor().

Hermes-agent must be importable at runtime (e.g. via PYTHONPATH). All Hermes
imports are lazy — inside _make_agent() and _ensure_session_db() — so the
adapter class always loads for registry/loader purposes.
"""

import asyncio
import logging
import uuid
import weakref
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from typing import Any, AsyncIterator

from ..base_adapter import AdapterMetadata, BaseA2AAdapter

logger = logging.getLogger(__name__)

_SENTINEL = object()


class _Turn:
    """One agent run, whose session lock is held until its thread ends.

    The coroutine that started the run can go away first (tasks/cancel, a
    consumer closing the stream). ``run_conversation`` keeps going in its
    thread, so the lock must outlive the coroutine: otherwise the next
    message on the conversation races the run on SessionDB.

    ``_started``, ``_abandoned`` and ``_finished`` are shared with the worker
    thread. ``run`` sets started and then checks abandoned; ``abandon`` sets
    abandoned and then checks started, so one always sees the other. The lock
    is released only on the event loop.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, lock: asyncio.Lock | None) -> None:
        self._loop = loop
        self._lock = lock  # also the strong reference the weak lock dict relies on
        self._agent: Any = None
        self._submitted = False
        self._started = False
        self._abandoned = False
        self._finished = False

    def submit(
        self, executor: ThreadPoolExecutor, agent: Any, call: Callable[[], Any]
    ) -> asyncio.Future:
        """Run ``call`` on the pool. Wait with ``asyncio.wait``, then use :meth:`result`."""
        future = self._loop.run_in_executor(executor, self._run, call)
        self._agent = agent
        self._submitted = True
        future.add_done_callback(self._on_done)
        return future

    @staticmethod
    def result(future: asyncio.Future) -> Any:
        if future.cancelled():
            raise RuntimeError("HermesAdapter closed before the turn started")
        return future.result()

    def _run(self, call: Callable[[], Any]) -> Any:
        self._started = True
        try:
            if self._abandoned:
                return None
            return call()
        finally:
            self._finished = True
            with suppress(RuntimeError):  # the loop closed while this ran
                self._loop.call_soon_threadsafe(self._release)

    def _on_done(self, future: asyncio.Future) -> None:
        if future.cancelled():  # close() dropped it from the pool's queue
            self._finished = True
            self._release()
        else:
            future.exception()  # retrieved, for a run nobody waits for any more

    def abandon(self) -> None:
        """The coroutine is leaving: release now, or interrupt the running thread."""
        if not self._submitted:
            self._release()
            return
        self._abandoned = True
        if self._started and not self._finished:
            try:
                self._agent.interrupt()
            except Exception:
                logger.warning("Interrupting an abandoned Hermes run failed", exc_info=True)

    def _release(self) -> None:
        if self._lock is not None:
            lock, self._lock = self._lock, None
            lock.release()


class HermesAdapter(BaseA2AAdapter):
    """A2A adapter for Hermes Agent using the Gateway pattern.

    Each invoke()/stream() call creates a fresh AIAgent, loads conversation
    history from a shared SessionDB, runs the synchronous
    run_conversation() in a thread pool, and lets the agent persist the
    updated history back to SQLite.

    Example::

        from a2a_adapter import HermesAdapter, serve_agent

        adapter = HermesAdapter(
            model="anthropic/claude-sonnet-4",
            enabled_toolsets=["hermes-cli"],
        )
        serve_agent(adapter, port=9010)
    """

    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        enabled_toolsets: list[str] | None = None,
        max_workers: int = 4,
        *,
        name: str = "",
        description: str = "",
        skills: list[dict] | None = None,
        agent_provider: dict | None = None,
        documentation_url: str | None = None,
        icon_url: str | None = None,
    ) -> None:
        self._model = model
        self._provider_name = provider
        self._enabled_toolsets = enabled_toolsets
        self._name = name
        self._description = description
        self._skills = skills or []
        self._agent_provider = agent_provider
        self._documentation_url = documentation_url
        self._icon_url = icon_url

        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="hermes")
        self._session_db = None  # lazy — see _ensure_session_db()
        self._running_agents: dict[str, object] = {}  # task_id → AIAgent
        # context_id → lock. Weak: an entry lives only while a holder or waiter
        # references the lock.
        self._session_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )

    # ──── Internal helpers ────

    @staticmethod
    def _session_key(context_id: str | None) -> str:
        """Map an A2A context_id to a Hermes SessionDB session id.

        Deterministic, so a restarted server finds the same history. The
        ``a2a-`` prefix keeps these apart from Hermes's other platforms.
        Without a context_id each call gets a fresh one-off session.
        """
        if context_id:
            return f"a2a-{context_id}"
        return f"a2a-{uuid.uuid4().hex[:12]}"

    def _session_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create a per-session lock.

        Serializes concurrent requests within the same conversation so
        they don't race on SessionDB snapshots or Hermes task-scoped
        resources (VMs, background processes).  Requests across different
        conversations run fully in parallel.

        Called only from the asyncio event loop thread, so the
        get-or-create sequence is safe without its own lock.
        """
        lock = self._session_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[session_id] = lock
        return lock

    def _ensure_session_db(self):
        """Lazily initialize the shared SessionDB instance.

        Deferred so the adapter class always loads for registry/loader
        purposes even when hermes-agent is not on sys.path.
        """
        if self._session_db is None:
            try:
                from hermes_state import SessionDB
            except ImportError as e:
                raise ImportError(
                    "HermesAdapter requires hermes-agent to be importable. "
                    "Add the hermes-agent directory to PYTHONPATH: "
                    "export PYTHONPATH=/path/to/hermes-agent:$PYTHONPATH"
                ) from e
            self._session_db = SessionDB()
        return self._session_db

    def _make_agent(self, session_id: str):
        """Create a fresh AIAgent for a single turn.

        Mirrors the gateway pattern at gateway/run.py — new agent per
        message with a shared SessionDB for persistence.
        """
        from run_agent import AIAgent
        from hermes_cli.config import load_config
        from hermes_cli.runtime_provider import resolve_runtime_provider

        config = load_config()
        model_cfg = config.get("model")
        default_model = ""
        config_provider = None
        if isinstance(model_cfg, dict):
            default_model = str(model_cfg.get("default") or "")
            config_provider = model_cfg.get("provider")
        elif isinstance(model_cfg, str) and model_cfg.strip():
            default_model = model_cfg.strip()

        kwargs = {
            "platform": "a2a",
            "enabled_toolsets": self._enabled_toolsets,
            "quiet_mode": True,
            "session_id": session_id,
            "session_db": self._ensure_session_db(),
            "model": self._model or default_model,
        }

        try:
            runtime = resolve_runtime_provider(
                requested=self._provider_name or config_provider,
            )
            kwargs.update(
                {
                    "provider": runtime.get("provider"),
                    "api_mode": runtime.get("api_mode"),
                    "base_url": runtime.get("base_url"),
                    "api_key": runtime.get("api_key"),
                    "command": runtime.get("command"),
                    "args": list(runtime.get("args") or []),
                }
            )
        except Exception:
            logger.debug(
                "Could not resolve runtime provider; using AIAgent defaults",
                exc_info=True,
            )

        return AIAgent(**kwargs)

    @staticmethod
    def _extract_task_id(kwargs) -> str:
        """Pull the task identifier from the bridge-provided RequestContext."""
        context = kwargs.get("context")
        if context and hasattr(context, "task_id"):
            return context.task_id
        return uuid.uuid4().hex

    async def _begin_turn(self, session_id: str, context_id: str | None) -> _Turn:
        # A one-off session (no context_id) is never shared, so it needs no
        # lock, and skipping it keeps _session_locks from growing per call.
        lock = self._session_lock(session_id) if context_id else None
        if lock is not None:
            await lock.acquire()
        return _Turn(asyncio.get_running_loop(), lock)

    # ──── Internal: result inspection ────

    @staticmethod
    def _check_hermes_result(result: dict, session_id: str) -> str:
        """Inspect a run_conversation() result dict and return the response.

        Hermes returns structured error metadata on many non-exception exit
        paths (truncation, iteration exhaustion, compression failure, etc.).
        If a usable ``final_response`` exists we return it — the user got
        useful output even if the run was not fully successful.  When no
        response is available **and** the result signals failure, we raise
        so the executor can emit A2A ``failed`` state.

        Raises:
            RuntimeError: When the run failed and produced no usable output.
        """
        final = result.get("final_response") or ""
        error = result.get("error")
        failed = result.get("failed", False)
        completed = result.get("completed", True)

        if error:
            logger.warning(
                "Hermes run_conversation error for session %s: %s",
                session_id,
                error,
            )

        if final:
            return final

        if failed or not completed:
            raise RuntimeError(error or "Hermes run did not complete successfully")

        return final

    # ──── Required: invoke ────

    async def invoke(self, user_input: str, context_id: str | None = None, **kwargs) -> str:
        """Execute Hermes Agent and return the full response.

        Loads conversation history from SessionDB, runs run_conversation()
        in a thread, and lets the agent persist updates internally.
        """
        session_id = self._session_key(context_id)
        turn = await self._begin_turn(session_id, context_id)
        task_id = self._extract_task_id(kwargs)
        try:
            db = self._ensure_session_db()
            history = db.get_messages_as_conversation(session_id) or []

            agent = self._make_agent(session_id)
            self._running_agents[task_id] = agent

            future = turn.submit(
                self._executor,
                agent,
                lambda: agent.run_conversation(
                    user_message=user_input,
                    conversation_history=history,
                    task_id=session_id,
                ),
            )
            await asyncio.wait({future})
            result = turn.result(future)
        except BaseException:
            turn.abandon()
            raise
        finally:
            self._running_agents.pop(task_id, None)

        return self._check_hermes_result(result, session_id)

    # ──── Optional: streaming ────

    async def stream(
        self, user_input: str, context_id: str | None = None, **kwargs
    ) -> AsyncIterator[str]:
        """Stream Hermes Agent response via callback-to-queue bridge.

        Uses AIAgent's stream_callback parameter to push chunks from the
        synchronous thread into an asyncio.Queue, yielding them as an
        async iterator for the bridge layer.
        """
        session_id = self._session_key(context_id)
        turn = await self._begin_turn(session_id, context_id)
        task_id = self._extract_task_id(kwargs)
        try:
            db = self._ensure_session_db()
            history = db.get_messages_as_conversation(session_id) or []

            agent = self._make_agent(session_id)
            self._running_agents[task_id] = agent

            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[str | None] = asyncio.Queue()

            def post(item) -> None:
                with suppress(RuntimeError):  # an abandoned run can outlive the loop
                    loop.call_soon_threadsafe(queue.put_nowait, item)

            def _run():
                try:
                    return agent.run_conversation(
                        user_message=user_input,
                        conversation_history=history,
                        task_id=session_id,
                        stream_callback=post,
                    )
                finally:
                    post(_SENTINEL)

            future = turn.submit(self._executor, agent, _run)

            def wake_if_dropped(done: asyncio.Future) -> None:
                if done.cancelled():  # _run never ran, so it sent no sentinel
                    queue.put_nowait(_SENTINEL)

            future.add_done_callback(wake_if_dropped)

            while True:
                chunk = await queue.get()
                if chunk is _SENTINEL:
                    break
                if not chunk:
                    continue
                yield chunk

            await asyncio.wait({future})
            result = turn.result(future)
            if result:
                self._check_hermes_result(result, session_id)
        except BaseException:
            turn.abandon()
            raise
        finally:
            self._running_agents.pop(task_id, None)

    # ──── Optional: cancel ────

    async def cancel(self, context_id: str | None = None, **kwargs) -> None:
        """Cancel an in-flight Hermes Agent execution.

        Calls AIAgent.interrupt() which is thread-safe — it sets an atomic
        flag checked between tool-use iterations.

        The conversation's lock stays held until the run's thread ends, so
        the next message on it waits, with no time limit: a tool call that
        never returns blocks that conversation until the server restarts.
        """
        task_id = self._extract_task_id(kwargs)
        agent = self._running_agents.get(task_id)
        if agent:
            agent.interrupt()

    # ──── Optional: metadata ────

    def get_metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name=self._name or "Hermes",
            description=self._description
            or (
                "Hermes AI Agent \u2014 multi-purpose assistant with tool use, "
                "persistent memory, and subagent delegation."
            ),
            streaming=True,
            skills=self._skills,
            provider=self._agent_provider,
            documentation_url=self._documentation_url,
            icon_url=self._icon_url,
        )

    # ──── Optional: cleanup ────

    async def close(self) -> None:
        """Shut down the thread pool executor and release internal state."""
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._session_locks.clear()
