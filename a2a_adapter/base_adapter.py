"""
New v0.2 adapter interface for A2A Protocol integration.

This module defines the simplified BaseA2AAdapter abstract class and
AdapterMetadata dataclass. Framework-specific adapters implement only
invoke() to become A2A-compatible.

Design philosophy:
    The adapter answers ONE question: "Given text, return text."
    Everything else (task management, SSE streaming, push notifications,
    resubscription, state persistence) is handled by the A2A SDK via
    the AdapterAgentExecutor bridge layer.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator

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
    """

    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    skills: list[dict] = field(default_factory=list)
    input_modes: list[str] = field(default_factory=lambda: ["text"])
    output_modes: list[str] = field(default_factory=lambda: ["text"])
    streaming: bool = False


class BaseA2AAdapter(ABC):
    """The only interface framework developers need to implement.

    Design philosophy:
        - invoke() is the single required method — answers "given text, return text"
        - stream() is optional — for frameworks that support token-by-token output
        - cancel() is optional — for frameworks where execution can be interrupted
        - close() is optional — for resource cleanup (HTTP clients, subprocesses)
        - get_metadata() is optional — for automatic AgentCard generation

    Everything else (task management, SSE streaming, push notifications,
    resubscription, state persistence) is handled by the A2A SDK via
    the AdapterAgentExecutor bridge layer.

    Three levels of control:
        Level 1: invoke(input) -> str             (90% of use cases)
        Level 2: stream(input) -> AsyncIterator   (streaming frameworks)
        Level 3: Implement AgentExecutor directly  (full SDK access)
    """

    @abstractmethod
    async def invoke(
        self,
        user_input: str,
        context_id: str | None = None,
        **kwargs,
    ) -> str:
        """Execute the agent and return a text response.

        This is the ONLY method you must implement.

        Args:
            user_input: The user's message as plain text.
                Extracted from A2A MessageSendParams by the bridge layer
                using SDK's RequestContext.get_user_input().
            context_id: Conversation context ID for multi-turn support.
                Same context_id = same conversation. None for single-turn.
            **kwargs: Additional keyword arguments from the bridge layer.

        Keyword Args:
            context: The A2A SDK ``RequestContext`` object, providing access
                to the full message including non-text parts (``FilePart``,
                ``DataPart``, etc.). Access via ``kwargs.get('context')``.
                Use ``context.message.parts`` to iterate over all parts.

        Returns:
            The agent's text response.

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
    ) -> AsyncIterator[str]:
        """Stream the agent response, yielding text chunks.

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
            Text chunks of the agent's response.
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
        """Cancel the current execution. Optional.

        Override for frameworks where execution can be interrupted
        (e.g., OpenClaw can kill a subprocess).

        Args:
            context_id: The context ID of the task to cancel.

        Keyword Args:
            context: The A2A SDK ``RequestContext`` for the task being
                canceled. Use ``kwargs.get('context')`` to identify which
                task to cancel (e.g., ``context.task_id``). This is
                important for concurrent-safe cancellation when multiple
                requests are in-flight.
        """
        pass

    async def close(self) -> None:
        """Release resources held by this adapter. Optional.

        Called when the adapter is no longer needed (e.g., closing
        HTTP clients, terminating subprocesses).
        """
        pass

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
