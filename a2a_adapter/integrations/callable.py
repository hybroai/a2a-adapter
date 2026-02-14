"""
Generic callable adapter for A2A Protocol.

This adapter allows any async Python function to be exposed as an A2A-compliant
agent, providing maximum flexibility for custom implementations.

Contains:
    - CallableAdapter (v0.2): New simplified interface based on BaseA2AAdapter
    - CallableAgentAdapter (v0.1): Legacy interface, deprecated
"""

import json
import logging
import uuid
from typing import Any, AsyncIterator, Callable, Dict

from a2a.types import (
    Message,
    MessageSendParams,
    Task,
    TextPart,
    Role,
    Part,
)
from ..adapter import BaseAgentAdapter
from ..base_adapter import BaseA2AAdapter, AdapterMetadata

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# v0.2 Adapter (new, recommended)
# ═══════════════════════════════════════════════════════════════


class CallableAdapter(BaseA2AAdapter):
    """Adapter for any async Python function.

    The simplest adapter — wraps any async callable as an A2A agent.

    For non-streaming functions:
        The function should accept a dict and return a string (or anything
        that can be converted to a string).

    For streaming functions:
        The function should be an async generator that yields string chunks.

    Example (non-streaming)::

        from a2a_adapter import CallableAdapter, serve_agent

        async def my_agent(inputs: dict) -> str:
            return f"You said: {inputs['message']}"

        adapter = CallableAdapter(func=my_agent)
        serve_agent(adapter, port=9000)

    Example (streaming)::

        async def my_streaming_agent(inputs: dict):
            for word in inputs['message'].split():
                yield word + " "

        adapter = CallableAdapter(func=my_streaming_agent, streaming=True)
        serve_agent(adapter, port=9000)
    """

    def __init__(
        self,
        func: Callable,
        streaming: bool = False,
        name: str = "",
        description: str = "",
    ) -> None:
        """Initialize the callable adapter.

        Args:
            func: An async callable that processes agent logic.
                For non-streaming: accepts dict, returns str.
                For streaming: async generator yielding str chunks.
            streaming: Whether the function supports streaming.
            name: Optional agent name for AgentCard generation.
            description: Optional agent description for AgentCard generation.
        """
        self.func = func
        self._streaming = streaming
        self._name = name
        self._description = description

    async def invoke(self, user_input: str, context_id: str | None = None, **kwargs) -> str:
        """Call the function and return a text response."""
        inputs = {"message": user_input, "context_id": context_id}
        result = await self.func(inputs)
        return self._extract_text(result)

    async def stream(self, user_input: str, context_id: str | None = None, **kwargs) -> AsyncIterator[str]:
        """Stream chunks from an async generator function."""
        inputs = {"message": user_input, "context_id": context_id}
        async for chunk in self.func(inputs):
            yield str(chunk) if not isinstance(chunk, str) else chunk

    def supports_streaming(self) -> bool:
        """Whether streaming is enabled for this adapter."""
        return self._streaming

    def get_metadata(self) -> AdapterMetadata:
        """Return adapter metadata for AgentCard generation."""
        return AdapterMetadata(
            name=self._name or (self.func.__name__ if hasattr(self.func, "__name__") else "CallableAdapter"),
            description=self._description,
            streaming=self._streaming,
        )

    @staticmethod
    def _extract_text(output: Any) -> str:
        """Extract text from function output."""
        if isinstance(output, str):
            return output
        if isinstance(output, dict):
            for key in ("response", "output", "result", "answer", "text", "message"):
                if key in output:
                    return str(output[key])
            return json.dumps(output, indent=2)
        return str(output)


# ═══════════════════════════════════════════════════════════════
# v0.1 Adapter (legacy, deprecated — will be removed in v0.3)
# ═══════════════════════════════════════════════════════════════


class CallableAgentAdapter(BaseAgentAdapter):
    """
    Adapter for integrating custom async functions as A2A agents.

    This adapter wraps any async callable (function, coroutine) and handles
    the A2A protocol translation. The callable should accept a dictionary
    input and return either a string or dictionary output.

    For streaming support, the callable should be an async generator that
    yields string chunks.

    Example (non-streaming):
        >>> async def my_agent(inputs: dict) -> str:
        ...     message = inputs["message"]
        ...     return f"Processed: {message}"
        >>>
        >>> adapter = CallableAgentAdapter(func=my_agent)

    Example (streaming):
        >>> async def my_streaming_agent(inputs: dict):
        ...     message = inputs["message"]
        ...     for word in message.split():
        ...         yield word + " "
        >>>
        >>> adapter = CallableAgentAdapter(
        ...     func=my_streaming_agent,
        ...     supports_streaming=True
        ... )
    """

    def __init__(
        self,
        func: Callable,
        supports_streaming: bool = False,
    ):
        """
        Initialize the callable adapter.

        Args:
            func: An async callable that processes the agent logic.
                  For non-streaming: Should accept Dict[str, Any] and return str or Dict.
                  For streaming: Should be an async generator yielding str chunks.
            supports_streaming: Whether the function supports streaming (default: False)
        """
        self.func = func
        self._supports_streaming = supports_streaming

    # ---------- Input mapping ----------

    async def to_framework(self, params: MessageSendParams) -> Dict[str, Any]:
        """
        Convert A2A message parameters to a dictionary for the callable.

        Extracts the user's message and relevant metadata.

        Args:
            params: A2A message parameters

        Returns:
            Dictionary with input data for the callable
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

        # Extract metadata
        context_id = self._extract_context_id(params)

        # Build input dictionary with useful fields
        return {
            "message": user_message,
            "context_id": context_id,
            "params": params,  # Full params for advanced use cases
        }

    @staticmethod
    def _join_text_parts(parts: list[str]) -> str:
        """Join text parts into a single string."""
        if not parts:
            return ""
        text = " ".join(p.strip() for p in parts if p)
        return text.strip()

    def _extract_context_id(self, params: MessageSendParams) -> str | None:
        """Extract context_id from MessageSendParams."""
        if hasattr(params, "message") and params.message:
            return getattr(params.message, "context_id", None)
        return None

    # ---------- Framework call ----------

    async def call_framework(
        self, framework_input: Dict[str, Any], params: MessageSendParams
    ) -> Any:
        """
        Execute the callable function with the provided input.

        Args:
            framework_input: Input dictionary for the function
            params: Original A2A parameters (for context)

        Returns:
            Function execution output

        Raises:
            Exception: If function execution fails
        """
        logger.debug("Invoking callable with input keys: %s", list(framework_input.keys()))
        result = await self.func(framework_input)
        logger.debug("Callable returned: %s", type(result).__name__)
        return result

    # ---------- Output mapping ----------

    async def from_framework(
        self, framework_output: Any, params: MessageSendParams
    ) -> Message | Task:
        """
        Convert callable output to A2A Message.

        Args:
            framework_output: Output from the callable
            params: Original A2A parameters

        Returns:
            A2A Message with the function's response
        """
        response_text = self._extract_output_text(framework_output)
        context_id = self._extract_context_id(params)

        return Message(
            role=Role.agent,
            message_id=str(uuid.uuid4()),
            context_id=context_id,
            parts=[Part(root=TextPart(text=response_text))],
        )

    def _extract_output_text(self, framework_output: Any) -> str:
        """
        Extract text content from callable output.

        Args:
            framework_output: Output from the callable

        Returns:
            Extracted text string
        """
        # Dictionary output
        if isinstance(framework_output, dict):
            # Try common output keys
            for key in ["response", "output", "result", "answer", "text", "message"]:
                if key in framework_output:
                    return str(framework_output[key])
            # Fallback: serialize as JSON
            return json.dumps(framework_output, indent=2)

        # String or other type - convert to string
        return str(framework_output)

    # ---------- Streaming support ----------

    async def handle_stream(
        self, params: MessageSendParams
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Handle a streaming A2A message request.

        The wrapped function must be an async generator for streaming to work.

        Args:
            params: A2A message parameters

        Yields:
            Server-Sent Events compatible dictionaries with streaming chunks

        Raises:
            NotImplementedError: If streaming is not enabled for this adapter
        """
        if not self._supports_streaming:
            raise NotImplementedError(
                "CallableAgentAdapter: streaming not enabled for this function. "
                "Initialize with supports_streaming=True and provide an async generator."
            )

        framework_input = await self.to_framework(params)
        context_id = self._extract_context_id(params)
        message_id = str(uuid.uuid4())

        logger.debug("Starting streaming call")

        accumulated_text = ""

        # Call the async generator function
        async for chunk in self.func(framework_input):
            # Convert chunk to string if needed
            text = str(chunk) if not isinstance(chunk, str) else chunk

            if text:
                accumulated_text += text
                # Yield SSE-compatible event
                yield {
                    "event": "message",
                    "data": json.dumps({
                        "type": "content",
                        "content": text,
                    }),
                }

        # Send final message with complete response
        final_message = Message(
            role=Role.agent,
            message_id=message_id,
            context_id=context_id,
            parts=[Part(root=TextPart(text=accumulated_text))],
        )

        # Send completion event
        yield {
            "event": "done",
            "data": json.dumps({
                "status": "completed",
                "message": final_message.model_dump() if hasattr(final_message, "model_dump") else str(final_message),
            }),
        }

        logger.debug("Streaming call completed")

    def supports_streaming(self) -> bool:
        """
        Check if this adapter supports streaming.

        Returns:
            True if streaming is enabled, False otherwise
        """
        return self._supports_streaming
