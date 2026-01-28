"""
Core adapter abstraction for A2A Protocol integration.

This module defines the BaseAgentAdapter abstract class that all framework-specific
adapters must implement.

Common capabilities provided to all adapters:
- Input mapping: input_mapper, parse_json_input, default_inputs
- Timeout configuration: timeout (sync mode)
- Raw input extraction utilities
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Callable, Dict

from a2a.types import Message, MessageSendParams, Task

logger = logging.getLogger(__name__)


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
    
    Common Input Handling (available to all adapters):
    
    All adapters can use these input mapping features (in priority order):
    
    1. **input_mapper** (highest priority):
       Custom function for full control over input transformation.
       Signature: (raw_input: str, context_id: str | None) -> dict
    
    2. **parse_json_input** (default: True):
       Automatically parse JSON input and pass directly to framework.
       Perfect for structured inputs.
    
    3. **Framework-specific fallback**:
       Each adapter has its own fallback behavior (e.g., inputs_key for CrewAI).
    
    For adapters that support async task execution, the adapter can:
    - Return a Task with state="submitted" or "working" immediately
    - Run the actual work in the background
    - Allow clients to poll for task status via get_task()
    """

    # Common configuration attributes (can be set by subclasses)
    timeout: int = 30  # Default sync timeout in seconds
    parse_json_input: bool = True
    input_mapper: Callable[[str, str | None], Dict[str, Any]] | None = None
    default_inputs: Dict[str, Any] | None = None

    # ---------- Common Input Utilities ----------

    def extract_raw_input(self, params: MessageSendParams) -> str:
        """
        Extract raw text content from A2A message parameters.
        
        This utility method handles:
        - New format: message.parts with Part(root=TextPart(...)) structure
        - Legacy format: messages array (deprecated)
        - Edge case: part.root.text returning dict instead of str
        
        All adapters can use this method to extract user input consistently.
        
        Args:
            params: A2A message parameters
            
        Returns:
            Extracted text as string
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
                        text_value = part.root.text
                        # Handle dict type - convert to JSON string
                        if isinstance(text_value, dict):
                            text_parts.append(json.dumps(text_value, ensure_ascii=False))
                        elif isinstance(text_value, str):
                            text_parts.append(text_value)
                        else:
                            text_parts.append(str(text_value))
                    # Handle direct TextPart
                    elif hasattr(part, "text"):
                        text_value = part.text
                        if isinstance(text_value, dict):
                            text_parts.append(json.dumps(text_value, ensure_ascii=False))
                        elif isinstance(text_value, str):
                            text_parts.append(text_value)
                        else:
                            text_parts.append(str(text_value))
                user_message = self._join_text_parts(text_parts)

        # Legacy support for messages array (deprecated)
        elif getattr(params, "messages", None):
            last = params.messages[-1]
            content = getattr(last, "content", "")
            if isinstance(content, str):
                user_message = content.strip()
            elif isinstance(content, dict):
                user_message = json.dumps(content, ensure_ascii=False)
            elif isinstance(content, list):
                text_parts = []
                for item in content:
                    txt = getattr(item, "text", None)
                    if txt is not None:
                        if isinstance(txt, dict):
                            text_parts.append(json.dumps(txt, ensure_ascii=False))
                        elif isinstance(txt, str) and txt.strip():
                            text_parts.append(txt.strip())
                user_message = self._join_text_parts(text_parts)

        return user_message

    def extract_context_id(self, params: MessageSendParams) -> str | None:
        """
        Extract context_id from MessageSendParams.
        
        Args:
            params: A2A message parameters
            
        Returns:
            The context_id string or None if not present
        """
        if hasattr(params, "message") and params.message:
            return getattr(params.message, "context_id", None)
        return None

    def try_parse_json(self, raw_input: str) -> Dict[str, Any] | None:
        """
        Attempt to parse raw input as JSON object.
        
        Args:
            raw_input: Raw text input
            
        Returns:
            Parsed dict if successful, None otherwise
        """
        if not raw_input or not raw_input.strip():
            return None

        raw_input = raw_input.strip()

        # Quick check: must start with { to be a JSON object
        if not raw_input.startswith("{"):
            return None

        try:
            parsed = json.loads(raw_input)
            if isinstance(parsed, dict):
                return parsed
            return None
        except json.JSONDecodeError:
            return None

    def apply_input_mapping(
        self,
        params: MessageSendParams,
        fallback_key: str = "input",
    ) -> Dict[str, Any]:
        """
        Apply the standard input mapping pipeline.
        
        Processing priority:
        1. input_mapper (custom function) - highest priority
        2. parse_json_input (auto JSON parsing)
        3. fallback_key (simple text mapping)
        
        Args:
            params: A2A message parameters
            fallback_key: Key to use for plain text when JSON parsing fails
            
        Returns:
            Dictionary with mapped input data
        """
        raw_input = self.extract_raw_input(params)
        context_id = self.extract_context_id(params)
        defaults = self.default_inputs or {}

        # Priority 1: Custom input_mapper function
        if self.input_mapper is not None:
            try:
                mapped_inputs = self.input_mapper(raw_input, context_id)
                logger.debug("Used input_mapper to transform input")
                return {**defaults, **mapped_inputs, "context_id": context_id}
            except Exception as e:
                logger.warning("input_mapper failed: %s, falling back", e)

        # Priority 2: Auto JSON parsing
        if self.parse_json_input:
            parsed = self.try_parse_json(raw_input)
            if parsed is not None:
                logger.debug("Parsed JSON input")
                return {**defaults, **parsed, "context_id": context_id}

        # Priority 3: Fallback to simple text mapping
        logger.debug("Using fallback_key '%s' for plain text input", fallback_key)
        return {
            **defaults,
            fallback_key: raw_input,
            "message": raw_input,
            "context_id": context_id,
        }

    @staticmethod
    def _join_text_parts(parts: list[str]) -> str:
        """Join text parts into a single string."""
        if not parts:
            return ""
        text = " ".join(p.strip() for p in parts if p)
        return text.strip()

    # ---------- Core Handler ----------

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

    def supports_async_tasks(self) -> bool:
        """
        Check if this adapter supports async task execution.
        
        Async task execution allows the adapter to return a Task immediately
        with state="working" and process the request in the background.
        Clients can then poll for task status via get_task().
        
        Returns:
            True if async tasks are supported, False otherwise
        """
        return False

    async def get_task(self, task_id: str) -> Task | None:
        """
        Get the current status of a task by ID.
        
        This method is used for polling task status in async task execution mode.
        Override this method in subclasses that support async tasks.
        
        Args:
            task_id: The ID of the task to retrieve
            
        Returns:
            The Task object with current status, or None if not found
            
        Raises:
            NotImplementedError: If async tasks are not supported by this adapter
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support async task execution"
        )

    async def cancel_task(self, task_id: str) -> Task | None:
        """
        Attempt to cancel a running task.
        
        This method is used to cancel async tasks that are still in progress.
        Override this method in subclasses that support async tasks.
        
        Args:
            task_id: The ID of the task to cancel
            
        Returns:
            The updated Task object with state="canceled", or None if not found
            
        Raises:
            NotImplementedError: If async tasks are not supported by this adapter
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support async task execution"
        )

