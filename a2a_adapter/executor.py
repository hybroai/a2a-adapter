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

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, TextPart
from a2a.utils.message import new_agent_text_message

from .base_adapter import BaseA2AAdapter

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

            await updater.start_work()

            if self._adapter.supports_streaming():
                await self._execute_streaming(updater, user_input, context)
            else:
                await self._execute_invoke(updater, user_input, context)

        except Exception as e:
            logger.error(
                "Adapter execution failed for task %s: %s",
                context.task_id,
                e,
                exc_info=True,
            )
            error_msg = new_agent_text_message(
                f"Agent execution failed: {e}",
                context.context_id,
                context.task_id,
            )
            await updater.failed(message=error_msg)

    async def _execute_invoke(
        self,
        updater: TaskUpdater,
        user_input: str,
        context: RequestContext,
    ) -> None:
        """Non-streaming execution: invoke() -> single artifact -> complete."""
        result_text = await self._adapter.invoke(
            user_input, context.context_id, context=context
        )

        parts = [Part(root=TextPart(text=result_text))]
        await updater.add_artifact(parts, name="response")

        message = new_agent_text_message(
            result_text, context.context_id, context.task_id
        )
        await updater.complete(message=message)

    async def _execute_streaming(
        self,
        updater: TaskUpdater,
        user_input: str,
        context: RequestContext,
    ) -> None:
        """Streaming execution: stream() -> incremental artifacts -> complete."""
        chunks: list[str] = []
        prev_chunk: str | None = None

        async for chunk in self._adapter.stream(
            user_input, context.context_id, context=context
        ):
            if prev_chunk is not None:
                chunks.append(prev_chunk)
                parts = [Part(root=TextPart(text=prev_chunk))]
                await updater.add_artifact(
                    parts,
                    append=len(chunks) > 1,
                    last_chunk=False,
                )
            prev_chunk = chunk

        if prev_chunk is not None:
            chunks.append(prev_chunk)
            parts = [Part(root=TextPart(text=prev_chunk))]
            await updater.add_artifact(
                parts,
                append=len(chunks) > 1,
                last_chunk=True,
            )

        full_text = "".join(chunks)
        message = new_agent_text_message(
            full_text, context.context_id, context.task_id
        )
        await updater.complete(message=message)

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
