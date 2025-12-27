"""
Generic callable adapter for A2A Protocol.

This adapter allows any async Python function to be exposed as an A2A-compliant
agent, providing maximum flexibility for custom implementations.
"""

import json
from typing import Any, AsyncIterator, Callable, Dict

from a2a.types import Message, MessageSendParams, Task, TextPart


class CallableAgentAdapter:
    """
    Adapter for integrating custom async functions as A2A agents.
    
    This adapter wraps any async callable (function, coroutine) and handles
    the A2A protocol translation. The callable should accept a dictionary
    input and return either a string or dictionary output.
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

    async def handle(self, params: MessageSendParams) -> Message | Task:
        """Handle a non-streaming A2A message request."""
        framework_input = await self.to_framework(params)
        framework_output = await self.call_framework(framework_input, params)
        return await self.from_framework(framework_output, params)

    async def handle_stream(
        self, params: MessageSendParams
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Handle a streaming A2A message request.
        
        The wrapped function must be an async generator for streaming to work.
        """
        if not self._supports_streaming:
            raise NotImplementedError(
                "CallableAgentAdapter: streaming not enabled for this function"
            )

        framework_input = await self.to_framework(params)

        # Call the async generator function
        async for chunk in self.func(framework_input):
            # Convert chunk to string if needed
            text = str(chunk) if not isinstance(chunk, str) else chunk

            # Yield SSE-compatible event
            if text:
                yield {
                    "event": "message",
                    "data": json.dumps({
                        "type": "content",
                        "content": text,
                    }),
                }

        # Send completion event
        yield {
            "event": "done",
            "data": json.dumps({"status": "completed"}),
        }

    def supports_streaming(self) -> bool:
        """Check if this adapter supports streaming."""
        return self._supports_streaming

    async def to_framework(self, params: MessageSendParams) -> Dict[str, Any]:
        """
        Convert A2A message parameters to a dictionary for the callable.
        
        Args:
            params: A2A message parameters
            
        Returns:
            Dictionary with input data for the callable
        """
        # Extract text from the last user message
        user_message = ""
        if params.messages:
            last_message = params.messages[-1]
            if hasattr(last_message, "content"):
                if isinstance(last_message.content, list):
                    # Extract text from content blocks
                    text_parts = [
                        item.text
                        for item in last_message.content
                        if hasattr(item, "text")
                    ]
                    user_message = " ".join(text_parts)
                elif isinstance(last_message.content, str):
                    user_message = last_message.content

        # Build input dictionary
        return {
            "message": user_message,
            "messages": params.messages,
            "session_id": getattr(params, "session_id", None),
            "context": getattr(params, "context", None),
        }

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
        result = await self.func(framework_input)
        return result

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
        # Convert output to string
        if isinstance(framework_output, dict):
            # If output has a 'response' or 'output' key, use that
            if "response" in framework_output:
                response_text = str(framework_output["response"])
            elif "output" in framework_output:
                response_text = str(framework_output["output"])
            else:
                response_text = json.dumps(framework_output, indent=2)
        else:
            response_text = str(framework_output)

        return Message(
            role="assistant",
            content=[TextPart(type="text", text=response_text)],
        )

    def supports_streaming(self) -> bool:
        """Check if this adapter supports streaming responses."""
        return False

