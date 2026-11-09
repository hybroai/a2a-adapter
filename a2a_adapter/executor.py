"""
Bridge layer: AdapterAgentExecutor.

This module translates the simplified BaseA2AAdapter interface into the
A2A SDK's event-driven AgentExecutor model. Users never interact with
this class directly — it's an internal implementation detail.

Design rationale:
    Adapters should never import or use EventQueue, TaskUpdater, etc.
    This bridge handles all the "protocol plumbing" in one place:
    - If the adapter raises an exception, we emit a failed state.
    - If the adapter supports streaming, we emit artifact chunks incrementally.
    - If not, we call invoke() and emit a single artifact + completion.
"""

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, Task, TaskState, TaskStatus

from .base_adapter import BaseA2AAdapter
from .exceptions import CancelledByAdapterError, UserFacingError

logger = logging.getLogger(__name__)

# How long a cancel whose adapter.cancel() raised waits for the interrupted
# run to unwind before it gives up and reports the task failed. Also bounds
# the wait for a terminal-state enqueue that a cancellation interrupted.
# Longer than base_adapter._POST_KILL_WAIT, so a killed CLI can be reaped.
_CANCEL_SETTLE_TIMEOUT = 10.0


@dataclass
class _Run:
    """State of one execute() call, shared with a concurrent cancel()."""

    finished: asyncio.Event  # set when _execute() has unwound
    terminal: bool = False  # a terminal state was (or is being) enqueued


class TaskObserver(Protocol):
    """Receives task lifecycle events, e.g. to print them in a terminal.

    Observer errors are logged and swallowed: they never change what the
    executor sends over A2A.
    """

    def task_submitted(
        self, task_id: str, context_id: str, prompt: str, message_id: str | None = None
    ) -> None: ...

    def task_working(self, task_id: str, context_id: str) -> None: ...

    def task_completed(
        self, task_id: str, context_id: str, text: str, chunks: int | None
    ) -> None: ...

    def task_failed(self, task_id: str, context_id: str, error: str) -> None: ...

    def task_canceled(self, task_id: str, context_id: str) -> None: ...


class AdapterAgentExecutor(AgentExecutor):
    """Bridge: translates BaseA2AAdapter into SDK's AgentExecutor.

    This is the key innovation of v0.2 — a thin bridge (~80 lines) that
    connects the simple invoke()/stream() contract with the SDK's full
    event-driven execution model.

    Responsibilities:
        1. Extract user input from RequestContext
        2. Create TaskUpdater for emitting lifecycle events
        3. Route to invoke() or stream() based on adapter capabilities
        4. Convert text responses into proper A2A Part/Message/Artifact objects
        5. Handle errors and emit failed state
        6. Handle cancellation by delegating to adapter.cancel()
    """

    def __init__(
        self,
        adapter: BaseA2AAdapter,
        observer: TaskObserver | None = None,
        *,
        expose_error_details: bool = False,
    ) -> None:
        self._adapter = adapter
        self._observer = observer
        self._expose_error_details = expose_error_details
        # tasks/cancel calls in progress, by task id. The SDK cancels the
        # running execute() before it calls cancel(), and closes the event
        # queue once execute() returns, so execute() waits for these: the
        # terminal state that cancel() sends would otherwise be dropped.
        self._cancelling: dict[str, asyncio.Future[None]] = {}
        # The execute() call in flight, by task id. The SDK runs at most one
        # execute() per task at a time.
        self._runs: dict[str, _Run] = {}

    def _notify(self, event: str, *args) -> None:
        if self._observer is None:
            return
        try:
            getattr(self._observer, event)(*args)
        except Exception:
            logger.debug("Task observer %s failed", event, exc_info=True)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute the adapter and emit A2A events to the queue.

        Called by DefaultRequestHandler for each incoming message/send
        or message/stream request. Each call runs in its own asyncio task.

        Args:
            context: Request context with user input, task_id, context_id.
            event_queue: Queue for publishing A2A lifecycle events.
        """
        run = _Run(asyncio.Event())
        self._runs[context.task_id] = run
        try:
            await self._execute(context, event_queue, run)
        finally:
            run.finished.set()
            try:
                cancelling = self._cancelling.get(context.task_id)
                if cancelling is not None:
                    await asyncio.shield(cancelling)
            finally:
                # Also on a second cancellation (e.g. shutdown) during the
                # wait; cancel() then releases again, which is harmless.
                if self._runs.get(context.task_id) is run:
                    del self._runs[context.task_id]
                self._release(context.task_id)

    def _client_error(self, prefix: str, task_id: str, e: Exception) -> str:
        """The failure text sent to the client.

        Exception text can hold URLs, paths, tokens or tool output, so it
        stays in the server log unless the adapter marked it safe or the
        operator opted in. The task id lets a person find the log line.
        """
        if self._expose_error_details or isinstance(e, UserFacingError):
            return f"{prefix}: {e}"
        return f"{prefix} (task {task_id}). Details are in the server log."

    def _release(self, task_id: str) -> None:
        """Let the adapter forget the task's cancel markers."""
        try:
            self._adapter._release_task(task_id)
        except Exception:
            logger.debug("Adapter _release_task failed for task %s", task_id, exc_info=True)

    def _cancel_owns(self, task_id: str, run: _Run) -> bool:
        """Whether execute() must leave the terminal state alone.

        True while a cancel is in progress (cancel() sends the terminal
        state) or once a terminal state was already enqueued.
        """
        return task_id in self._cancelling or run.terminal

    @staticmethod
    async def _send_terminal(
        run: _Run, send: Awaitable[None], on_sent: Callable[[], None] | None = None
    ) -> None:
        """Enqueue a terminal state, even if the caller is cancelled.

        ``run.terminal`` is set before the first await, so a concurrent
        cancel() defers to this state. If the caller is cancelled while the
        enqueue is blocked (e.g. on a full queue), the enqueue keeps going
        and is awaited, bounded, before the cancellation propagates: the
        SDK closes the queue once execute() returns. ``on_sent`` runs once
        the state was enqueued, also on that cancelled path.
        """
        run.terminal = True
        enqueue = asyncio.ensure_future(send)
        try:
            await asyncio.shield(enqueue)
        except asyncio.CancelledError:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(enqueue), _CANCEL_SETTLE_TIMEOUT)
            if on_sent and enqueue.done() and not enqueue.cancelled() and not enqueue.exception():
                on_sent()
            raise
        except Exception:
            # Nothing was enqueued: let the caller report the failure.
            run.terminal = False
            raise
        if on_sent:
            on_sent()

    async def _execute(self, context: RequestContext, event_queue: EventQueue, run: _Run) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)

        try:
            user_input = context.get_user_input()
            if not user_input:
                logger.warning("Empty user input for task %s", context.task_id)

            # A follow-up message to an existing task must not replace it;
            # the SDK rejects that with an error log.
            if context.current_task is None:
                await event_queue.enqueue_event(
                    Task(
                        id=context.task_id,
                        context_id=context.context_id,
                        status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                    )
                )
            self._notify(
                "task_submitted",
                context.task_id,
                context.context_id,
                user_input,
                getattr(context.message, "message_id", None) or None,
            )
            await updater.start_work()
            self._notify("task_working", context.task_id, context.context_id)

            if self._adapter.supports_streaming():
                await self._execute_streaming(updater, user_input, context, run)
            else:
                await self._execute_invoke(updater, user_input, context, run)

        except CancelledByAdapterError:
            # cancel() already emitted terminal canceled state.
            # Do NOT emit failed — just exit silently.
            logger.info("Task %s canceled by adapter", context.task_id)
        except Exception as e:
            if self._cancel_owns(context.task_id, run):
                # cancel() sends (or sent) the terminal state.
                logger.info(
                    "Task %s raised during cancel; ignored, cancel in progress: %s",
                    context.task_id,
                    e,
                )
                return
            logger.error(
                "Adapter execution failed for task %s: %s",
                context.task_id,
                e,
                exc_info=True,
                extra={"a2a_activity_reported": self._observer is not None},
            )
            error = f"{type(e).__name__}: {e}"
            error_msg = updater.new_agent_message(
                [Part(text=self._client_error("Agent execution failed", context.task_id, e))]
            )
            await self._send_final_state(
                context,
                run,
                updater.failed(message=error_msg),
                lambda: self._notify("task_failed", context.task_id, context.context_id, error),
                failure=error,
            )

    def _drop_late_result(self, context: RequestContext, run: _Run) -> bool:
        if not self._cancel_owns(context.task_id, run):
            return False
        logger.info("Task %s: result dropped, task is being canceled", context.task_id)
        return True

    @staticmethod
    def _is_empty_chunk(chunk: str | Part) -> bool:
        """Check if a chunk carries no meaningful content.

        Empty text chunks (``text=""``) carry no meaningful content and should
        be filtered out before reaching the wire. Also catches Parts where no
        ``oneof content`` variant was set at all (e.g. ``Part()``), which
        serialize to ``{}`` on the wire.
        """
        if isinstance(chunk, str):
            return chunk == ""
        if isinstance(chunk, Part):
            try:
                if not chunk.HasField("content"):
                    return True
            except ValueError:
                pass
            variant = chunk.WhichOneof("content")
            if variant == "text":
                return chunk.text == ""
            return variant is None
        return False

    @staticmethod
    def _to_parts(chunk: str | Part) -> list[Part]:
        """Convert a str or Part to a list[Part]."""
        if isinstance(chunk, str):
            return [Part(text=chunk)]
        elif isinstance(chunk, Part):
            return [chunk]
        else:
            raise TypeError(f"Unexpected type {type(chunk).__name__}. Expected str or Part.")

    async def _execute_invoke(
        self,
        updater: TaskUpdater,
        user_input: str,
        context: RequestContext,
        run: _Run,
    ) -> None:
        """Non-streaming execution: invoke() -> single artifact -> complete.

        Handles both text-only (str) and multimodal (list[Part]) responses.
        A result that arrives while the task is being canceled is dropped.
        """
        result = await self._adapter.invoke(user_input, context.context_id, context=context)
        if self._drop_late_result(context, run):
            return

        # Type detection: str (backward compatible) or list[Part] (multimodal)
        if isinstance(result, str):
            parts = self._to_parts(result)
            response_text = result
        elif isinstance(result, list):
            parts = result
            response_text = self._extract_text_from_parts(parts)
        else:
            raise TypeError(
                f"Adapter invoke() returned unexpected type {type(result).__name__}. "
                f"Expected str or list[Part]."
            )

        # Match the streaming path: Parts with no content serialize to {}.
        parts = [p for p in parts if not self._is_empty_chunk(p)]
        if parts:
            await updater.add_artifact(parts, name="response")
        if self._drop_late_result(context, run):
            return

        # Part(text="") would also serialize to {} in the completion message.
        message = updater.new_agent_message([Part(text=response_text or "[Empty response]")])
        await self._send_terminal(
            run,
            updater.complete(message=message),
            lambda: self._notify(
                "task_completed", context.task_id, context.context_id, response_text, None
            ),
        )

    def _extract_text_from_parts(self, parts: list[Part]) -> str:
        """Extract text content from multimodal parts for completion message.

        Args:
            parts: List of Part objects (may contain text, file, or data parts).

        Returns:
            Concatenated text from all text parts, or a placeholder
            if no text parts are found.
        """
        texts = []
        for part in parts:
            if part.text:
                texts.append(part.text)
        return " ".join(texts) if texts else "[Non-text response]"

    async def _execute_streaming(
        self,
        updater: TaskUpdater,
        user_input: str,
        context: RequestContext,
        run: _Run,
    ) -> None:
        """Streaming execution: stream() -> incremental artifacts -> complete.

        Handles both text chunks (str) and multimodal chunks (Part).
        Empty text chunks are silently dropped to avoid producing malformed
        wire-format Parts (protobuf omits default-value fields).

        If the stream raises mid-flight, the open artifact is closed with
        ``last_chunk=True`` before the exception propagates so downstream
        consumers never see an unclosed artifact. While the task is being
        canceled nothing more is sent, so the artifact is left open
        (``last_chunk=False``): no event may follow the canceled state.
        """
        chunks: list[str | Part] = []
        prev_chunk: str | Part | None = None
        artifact_id = uuid.uuid4().hex

        stream = self._adapter.stream(user_input, context.context_id, context=context)
        # Close the generator before returning, also when cancelled, so its
        # cleanup (e.g. killing and reaping a CLI) runs before cancel()
        # decides whether the run stopped, not later in a finalizer.
        closing = (
            contextlib.aclosing(stream) if hasattr(stream, "aclose") else contextlib.nullcontext()
        )
        try:
            async with closing:
                async for chunk in stream:
                    if self._cancel_owns(context.task_id, run):
                        # A stream that ignored the cancel: stop reading it
                        # (it is closed below) and send nothing more.
                        break
                    if self._is_empty_chunk(chunk):
                        continue

                    if prev_chunk is not None:
                        chunks.append(prev_chunk)

                        await updater.add_artifact(
                            self._to_parts(prev_chunk),
                            artifact_id=artifact_id,
                            append=len(chunks) > 1,
                            last_chunk=False,
                        )
                    prev_chunk = chunk
        except CancelledByAdapterError:
            # cancel() already emitted the canceled state; sending the
            # held-back chunk now would follow the task's terminal state.
            raise
        except Exception:
            # Close the artifact with the held-back chunk, which has not
            # been sent yet; re-sending chunks[-1] would duplicate content.
            if prev_chunk is not None and not self._cancel_owns(context.task_id, run):
                chunks.append(prev_chunk)
                await updater.add_artifact(
                    self._to_parts(prev_chunk),
                    artifact_id=artifact_id,
                    append=len(chunks) > 1,
                    last_chunk=True,
                )
            raise

        if self._drop_late_result(context, run):
            return
        if prev_chunk is not None:
            chunks.append(prev_chunk)

            await updater.add_artifact(
                self._to_parts(prev_chunk),
                artifact_id=artifact_id,
                append=len(chunks) > 1,
                last_chunk=True,
            )

        if self._drop_late_result(context, run):
            return
        full_text = self._concatenate_chunks(chunks)
        message = updater.new_agent_message([Part(text=full_text)])
        await self._send_terminal(
            run,
            updater.complete(message=message),
            lambda: self._notify(
                "task_completed", context.task_id, context.context_id, full_text, len(chunks)
            ),
        )

    def _concatenate_chunks(self, chunks: list[str | Part]) -> str:
        """Concatenate text from streaming chunks for completion message.

        Args:
            chunks: List of chunks (str or Part objects)

        Returns:
            Concatenated text from all text chunks, or a placeholder
            if no text content is found.
        """
        texts = []
        for chunk in chunks:
            if isinstance(chunk, str):
                texts.append(chunk)
            elif isinstance(chunk, Part) and chunk.text:
                texts.append(chunk.text)
        return "".join(texts) if texts else "[Streamed non-text content]"

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel an ongoing execution.

        Called by DefaultRequestHandler when a tasks/cancel request arrives.
        Delegates to the adapter's cancel() method, then emits canceled state.
        If adapter.cancel() raises, this waits (up to
        ``_CANCEL_SETTLE_TIMEOUT``) for the interrupted run to unwind. If it
        stopped (invoke() returned or raised, or the stream() generator was
        closed), the task is canceled and the error is only logged.
        Otherwise the work may still be running, so the task is not
        reported canceled: the executor sends a single failed state
        ("Cancel failed ...") and does not re-raise, so tasks/cancel
        returns that failed task instead of a JSON-RPC error. The decision
        and its enqueue finish even if this call is cancelled.

        The interrupted execute() waits until this returns, so the event
        queue is still open for the terminal state even when
        adapter.cancel() awaits. While a cancel is in progress, execute()
        sends no terminal state and no late artifact of its own. If the
        in-flight execute() had already sent one, this sends nothing; a
        cancel that arrives after execute() returned is not covered.

        Args:
            context: Request context for the task being canceled.
            event_queue: Queue for publishing the cancellation event.
        """
        # Registered before the first await, so execute() sees it.
        cancelling = asyncio.get_running_loop().create_future()
        self._cancelling[context.task_id] = cancelling
        # execute() drops its own result while this runs, so the decision
        # and its enqueue must finish even if this call is cancelled (e.g.
        # on a client disconnect): run them as a shielded task.
        decision = asyncio.ensure_future(self._cancel(context, event_queue))
        try:
            await asyncio.shield(decision)
        finally:
            if not decision.done():
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await asyncio.wait_for(asyncio.shield(decision), _CANCEL_SETTLE_TIMEOUT)
            if not decision.done():
                self._abandon(context.task_id, decision)
            if self._cancelling.get(context.task_id) is cancelling:
                del self._cancelling[context.task_id]
            cancelling.set_result(None)
            # With an execute() in flight, it releases once it has returned.
            if context.task_id not in self._runs:
                self._release(context.task_id)

    def _abandon(self, task_id: str, decision: asyncio.Future[None]) -> None:
        """Stop waiting for a cancel decision that outlasted the bounded wait.

        The decision may still send its terminal state, so the run in
        flight must send none of its own from here on; the markers that
        adapter.cancel() may still queue are released when it finishes.
        """
        logger.warning("Cancel of task %s did not finish in time; left running", task_id)
        run = self._runs.get(task_id)
        if run is not None:
            run.terminal = True

        def done(future: asyncio.Future[None]) -> None:
            if not future.cancelled() and future.exception() is not None:
                logger.warning(
                    "Abandoned cancel of task %s failed: %s", task_id, future.exception()
                )
            self._release(task_id)

        decision.add_done_callback(done)

    async def _cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        run = self._runs.get(context.task_id)
        if run is not None and run.terminal:
            # execute() sent (or is sending) the terminal state; the run is
            # over. If that enqueue then fails or times out, no terminal
            # state is sent (a known, rare gap).
            logger.info("Task %s already reached a terminal state; not canceling", context.task_id)
            return

        try:
            await self._adapter.cancel(context=context)
        except Exception as e:
            # Not re-raised: the SDK would add a second, bare failed state
            # and hand the error to every stream subscriber.
            if await self._run_stopped(run):
                logger.warning(
                    "Adapter cancel() raised for task %s, but the run had already "
                    "stopped; reporting canceled: %s",
                    context.task_id,
                    e,
                )
            else:
                logger.warning(
                    "Adapter cancel() raised for task %s and the run is still going; "
                    "marking it failed: %s",
                    context.task_id,
                    e,
                    exc_info=True,
                )
                error = f"cancel failed: {type(e).__name__}: {e}"
                message = updater.new_agent_message(
                    [Part(text=self._client_error("Cancel failed", context.task_id, e))]
                )
                await self._send_final_state(
                    context,
                    run,
                    updater.failed(message=message),
                    lambda: self._notify("task_failed", context.task_id, context.context_id, error),
                    failure=error,
                )
                return

        await self._send_final_state(
            context,
            run,
            updater.cancel(),
            lambda: self._notify("task_canceled", context.task_id, context.context_id),
        )

    async def _send_final_state(
        self,
        context: RequestContext,
        run: _Run | None,
        send: Awaitable[None],
        on_sent: Callable[[], None],
        failure: str | None = None,
    ) -> None:
        """Send a state that nothing can recover from if its enqueue raises.

        The error then leaves execute() or cancel() and the SDK adds its own
        failed state. That one is the task's terminal state, so a run still
        going must send no other, and the observer hears failed (``failure``,
        or else the enqueue error) instead of what could not be sent. If the
        cancel() call was itself cancelled the SDK never sees the error, so
        no failed state follows that report (a known, rare gap).
        """
        # With no execute() in flight, nothing reads the flag.
        owner = run if run is not None else _Run(asyncio.Event())
        try:
            await self._send_terminal(owner, send, on_sent)
        except Exception as e:
            owner.terminal = True
            self._notify(
                "task_failed",
                context.task_id,
                context.context_id,
                failure or f"{type(e).__name__}: {e}",
            )
            raise

    @staticmethod
    async def _run_stopped(run: _Run | None) -> bool:
        """Wait, bounded, for the interrupted run to unwind; True if it did.

        No run means no execute() is in flight, so nothing of ours runs.
        """
        if run is None:
            return True
        if not run.finished.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(run.finished.wait(), _CANCEL_SETTLE_TIMEOUT)
        return run.finished.is_set()
