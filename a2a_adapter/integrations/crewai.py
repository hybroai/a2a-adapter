"""
CrewAI adapter for A2A Protocol.

This adapter enables CrewAI crews to be exposed as A2A-compliant agents
by translating A2A messages to crew inputs and crew outputs back to A2A.
"""

import json
from typing import Any, Dict

from a2a.types import Message, MessageSendParams, Task, TextPart


class CrewAIAgentAdapter:
    """
    Adapter for integrating CrewAI crews as A2A agents.
    
    This adapter handles the translation between A2A protocol messages
    and CrewAI's crew execution model.
    """

    def __init__(
        self,
        crew: Any,  # Type: crewai.Crew (avoiding hard dependency)
        inputs_key: str = "inputs",
    ):
        """
        Initialize the CrewAI adapter.
        
        Args:
            crew: A CrewAI Crew instance to execute
            inputs_key: The key name for passing inputs to the crew (default: "inputs")
        """
        self.crew = crew
        self.inputs_key = inputs_key

    async def handle(self, params: MessageSendParams) -> Message | Task:
        """Handle a non-streaming A2A message request."""
        framework_input = await self.to_framework(params)
        framework_output = await self.call_framework(framework_input, params)
        return await self.from_framework(framework_output, params)

    async def to_framework(self, params: MessageSendParams) -> Dict[str, Any]:
        """
        Convert A2A message parameters to CrewAI crew inputs.
        
        Extracts the user's message and prepares it as input for the crew.
        
        Args:
            params: A2A message parameters
            
        Returns:
            Dictionary with crew input data
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

        # Build crew inputs
        # CrewAI typically expects a dict with task-specific keys
        return {
            self.inputs_key: user_message,
            "message": user_message,
            "session_id": getattr(params, "session_id", None),
        }

    async def call_framework(
        self, framework_input: Dict[str, Any], params: MessageSendParams
    ) -> Any:
        """
        Execute the CrewAI crew with the provided inputs.
        
        Args:
            framework_input: Input dictionary for the crew
            params: Original A2A parameters (for context)
            
        Returns:
            CrewAI crew execution output
            
        Raises:
            Exception: If crew execution fails
        """
        # CrewAI supports async execution via kickoff_async
        try:
            result = await self.crew.kickoff_async(inputs=framework_input)
            return result
        except AttributeError:
            # Fallback for older CrewAI versions without async support
            # Note: This will block the event loop
            import asyncio
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self.crew.kickoff, framework_input
            )
            return result

    async def from_framework(
        self, framework_output: Any, params: MessageSendParams
    ) -> Message | Task:
        """
        Convert CrewAI crew output to A2A Message.
        
        Args:
            framework_output: Output from crew execution
            params: Original A2A parameters
            
        Returns:
            A2A Message with the crew's response
        """
        # CrewAI output can be various types (string, dict, CrewOutput object)
        if hasattr(framework_output, "raw"):
            # CrewOutput object
            response_text = str(framework_output.raw)
        elif isinstance(framework_output, dict):
            # Dictionary output - serialize as JSON
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

