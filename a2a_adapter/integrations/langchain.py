"""
LangChain adapter for A2A Protocol.

This adapter enables LangChain runnables (chains, agents) to be exposed
as A2A-compliant agents with support for both streaming and non-streaming modes.
"""

import json
from typing import Any, AsyncIterator, Dict

from a2a.types import Message, MessageSendParams, Task, TextPart


class LangChainAgentAdapter:
    """
    Adapter for integrating LangChain runnables as A2A agents.
    
    This adapter works with any LangChain Runnable (chains, agents, etc.)
    and supports both streaming and non-streaming execution modes.
    """

    def __init__(
        self,
        runnable: Any,  # Type: Runnable (avoiding hard dependency)
        input_key: str = "input",
        output_key: str | None = None,
    ):
        """
        Initialize the LangChain adapter.
        
        Args:
            runnable: A LangChain Runnable instance (chain, agent, etc.)
            input_key: The key name for passing input to the runnable (default: "input")
            output_key: Optional key to extract from runnable output. If None, uses the entire output.
        """
        self.runnable = runnable
        self.input_key = input_key
        self.output_key = output_key

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
        
        Yields Server-Sent Events compatible dictionaries with streaming chunks.
        """
        framework_input = await self.to_framework(params)

        # Stream from LangChain runnable
        async for chunk in self.runnable.astream(framework_input):
            # Extract text from chunk
            if hasattr(chunk, "content"):
                text = chunk.content
            elif isinstance(chunk, dict):
                text = chunk.get(self.output_key or "output", str(chunk))
            else:
                text = str(chunk)

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
        """Check if the runnable supports streaming."""
        return hasattr(self.runnable, "astream")

    async def to_framework(self, params: MessageSendParams) -> Dict[str, Any]:
        """
        Convert A2A message parameters to LangChain runnable input.
        
        Args:
            params: A2A message parameters
            
        Returns:
            Dictionary with runnable input data
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

        # Build runnable input
        return {
            self.input_key: user_message,
        }

    async def call_framework(
        self, framework_input: Dict[str, Any], params: MessageSendParams
    ) -> Any:
        """
        Execute the LangChain runnable with the provided input.
        
        Args:
            framework_input: Input dictionary for the runnable
            params: Original A2A parameters (for context)
            
        Returns:
            Runnable execution output
            
        Raises:
            Exception: If runnable execution fails
        """
        result = await self.runnable.ainvoke(framework_input)
        return result

    async def from_framework(
        self, framework_output: Any, params: MessageSendParams
    ) -> Message | Task:
        """
        Convert LangChain runnable output to A2A Message.
        
        Args:
            framework_output: Output from runnable execution
            params: Original A2A parameters
            
        Returns:
            A2A Message with the runnable's response
        """
        # Extract output based on type
        if hasattr(framework_output, "content"):
            # AIMessage or similar
            response_text = framework_output.content
        elif isinstance(framework_output, dict):
            # Dictionary output - extract using output_key or serialize
            if self.output_key and self.output_key in framework_output:
                response_text = str(framework_output[self.output_key])
            else:
                response_text = json.dumps(framework_output, indent=2)
        else:
            # String or other type - convert to string
            response_text = str(framework_output)

        return Message(
            role="assistant",
            content=[TextPart(type="text", text=response_text)],
        )

    def supports_streaming(self) -> bool:
        """Check if this adapter supports streaming responses."""
        return False

