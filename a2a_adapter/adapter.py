"""
Core adapter abstraction for A2A Protocol integration.

This module defines the BaseAgentAdapter abstract class that all framework-specific
adapters must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict

from a2a.types import Message, MessageSendParams, Task


class BaseAgentAdapter(ABC):
    """
    Abstract base class for adapting agent frameworks to the A2A Protocol.
    
    This class provides the core interface for translating between A2A Protocol
    messages and framework-specific inputs/outputs. Concrete implementations
    handle the specifics of n8n, CrewAI, LangChain, etc.
    
    The adapter follows a three-step process:
    1. to_framework: Convert A2A MessageSendParams to framework input
    2. call_framework: Execute the framework-specific logic
    3. from_framework: Convert framework output back to A2A Message/Task
    """

    async def handle(self, params: MessageSendParams) -> Message | Task:
        """
        Handle a non-streaming A2A message request.
        
        Args:
            params: A2A protocol message parameters
            
        Returns:
            A2A Message or Task response
            
        Raises:
            Exception: If the underlying framework call fails
        """
        framework_input = await self.to_framework(params)
        framework_output = await self.call_framework(framework_input, params)
        return await self.from_framework(framework_output, params)

    async def handle_stream(
        self, params: MessageSendParams
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Handle a streaming A2A message request.
        
        Default implementation raises NotImplementedError. Override this method
        in subclasses that support streaming responses.
        
        Args:
            params: A2A protocol message parameters
            
        Yields:
            Server-Sent Events compatible dictionaries with 'event' and 'data' keys
            
        Raises:
            NotImplementedError: If streaming is not supported by this adapter
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support streaming"
        )

    @abstractmethod
    async def to_framework(self, params: MessageSendParams) -> Any:
        """
        Convert A2A message parameters to framework-specific input format.
        
        Args:
            params: A2A protocol message parameters
            
        Returns:
            Framework-specific input (format varies by implementation)
        """
        ...

    @abstractmethod
    async def call_framework(
        self, framework_input: Any, params: MessageSendParams
    ) -> Any:
        """
        Execute the underlying agent framework with the prepared input.
        
        Args:
            framework_input: Framework-specific input from to_framework()
            params: Original A2A message parameters (for context)
            
        Returns:
            Framework-specific output
        """
        ...

    @abstractmethod
    async def from_framework(
        self, framework_output: Any, params: MessageSendParams
    ) -> Message | Task:
        """
        Convert framework output to A2A Message or Task.
        
        Args:
            framework_output: Output from call_framework()
            params: Original A2A message parameters (for context)
            
        Returns:
            A2A Message or Task response
        """
        ...

    def supports_streaming(self) -> bool:
        """
        Check if this adapter supports streaming responses.
        
        Returns:
            True if streaming is supported, False otherwise
        """
        try:
            # Check if handle_stream is overridden
            return (
                self.__class__.handle_stream
                != BaseAgentAdapter.handle_stream
            )
        except AttributeError:
            return False

