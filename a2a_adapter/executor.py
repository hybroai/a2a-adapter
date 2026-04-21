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

import logging
import uuid

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, Task, TaskState, TaskStatus

from .base_adapter import BaseA2AAdapter
from .exceptions import CancelledByAdapterError

logger = logging.getLogger(__name__)


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

    def __init__(self, adapter: BaseA2AAdapter) -> None:
        self._adapter = adapter

    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Execute the adapter and emit A2A events to the queue.

        Called by DefaultRequestHandler for each incoming message/send
        or message/stream request. Each call runs in its own asyncio task.

        Args:
            context: Request context with user input, task_id, context_id.
            event_queue: Queue for publishing A2A lifecycle events.
        """
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)

        try:
            user_input = context.get_user_input()
            if not user_input:
                logger.warning("Empty user input for task %s", context.task_id)

            await event_queue.enqueue_event(
                Task(
                    id=context.task_id,
                    context_id=context.context_id,
                    status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                )
            )
            await updater.start_work()

            if self._adapter.supports_streaming():
                await self._execute_streaming(updater, user_input, context)
            else:
                await self._execute_invoke(updater, user_input, context)

        except CancelledByAdapterError:
            # cancel() already emitted terminal canceled state.
            # Do NOT emit failed — just exit silently.
            logger.info("Task %s canceled by adapter", context.task_id)
        except Exception as e:
            logger.error(
                "Adapter execution failed for task %s: %s",
                context.task_id,
                e,
                exc_info=True,
            )
            error_msg = updater.new_agent_message(
                [Part(text=f"Agent execution failed: {e}")]
            )
            await updater.failed(message=error_msg)

    @staticmethod
    def _is_empty_chunk(chunk: str | Part) -> bool:
        """Check if a chunk carries no meaningful content.

        Empty text chunks produce protobuf Parts where ``text=""`` is the
        default value, so ``MessageToDict`` omits the field entirely.  The
        resulting JSON ``{"kind": "text"}`` (no ``text`` key) fails Pydantic
        validation on the receiving end.  Filter these out before they reach
        the wire.
        """
        if isinstance(chunk, str):
            return chunk == ""
        if isinstance(chunk, Part):
            return chunk.text == ""
        return False

    @staticmethod
    def _to_parts(chunk: str | Part) -> list[Part]:
        """Convert a str or Part to a list[Part]."""
        if isinstance(chunk, str):
            return [Part(text=chunk)]
        elif isinstance(chunk, Part):
            return [chunk]
        else:
            raise TypeError(
                f"Unexpected type {type(chunk).__name__}. Expected str or Part."
            )

    async def _execute_invoke(
        self,
        updater: TaskUpdater,
        user_input: str,
        context: RequestContext,
    ) -> None:
        """Non-streaming execution: invoke() -> single artifact -> complete.

        Handles both text-only (str) and multimodal (list[Part]) responses.
        """
        result = await self._adapter.invoke(
            user_input, context.context_id, context=context
        )

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

        await updater.add_artifact(parts, name="response")

        message = updater.new_agent_message([Part(text=response_text)])
        await updater.complete(message=message)

    def _extract_text_from_parts(self, parts: list[Part]) -> str:
        """Extract text content from multimodal parts for completion message.

        Args:
            parts: List of Part objects (may contain TextPart, FilePart, etc.)

        Returns:
            Concatenated text from all TextPart objects, or a placeholder
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
    ) -> None:
        """Streaming execution: stream() -> incremental artifacts -> complete.

        Handles both text chunks (str) and multimodal chunks (Part).
        Empty text chunks are silently dropped to avoid producing malformed
        wire-format Parts (protobuf omits default-value fields).

        If the stream raises mid-flight, the open artifact is closed with
        ``last_chunk=True`` before the exception propagates so downstream
        consumers never see an unclosed artifact.
        """
        chunks: list[str | Part] = []
        prev_chunk: str | Part | None = None
        artifact_id = uuid.uuid4().hex

        try:
            async for chunk in self._adapter.stream(
                user_input, context.context_id, context=context
            ):
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
        except Exception:
            if chunks:
                await updater.add_artifact(
                    self._to_parts(chunks[-1]),
                    artifact_id=artifact_id,
                    append=True,
                    last_chunk=True,
                )
            raise

        if prev_chunk is not None:
            chunks.append(prev_chunk)

            await updater.add_artifact(
                self._to_parts(prev_chunk),
                artifact_id=artifact_id,
                append=len(chunks) > 1,
                last_chunk=True,
            )

        full_text = self._concatenate_chunks(chunks)
        message = updater.new_agent_message([Part(text=full_text)])
        await updater.complete(message=message)

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

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Cancel an ongoing execution.

        Called by DefaultRequestHandler when a tasks/cancel request arrives.
        Delegates to the adapter's cancel() method, then emits canceled state.

        Args:
            context: Request context for the task being canceled.
            event_queue: Queue for publishing the cancellation event.
        """
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)

        try:
            await self._adapter.cancel(context=context)
        except Exception as e:
            logger.warning(
                "Adapter cancel() raised for task %s: %s",
                context.task_id,
                e,
            )

        await updater.cancel()
