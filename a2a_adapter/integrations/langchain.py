"""
LangChain adapter for A2A Protocol.

This adapter enables LangChain runnables (chains, agents, RAG pipelines) to be
exposed as A2A-compliant agents with support for both streaming and non-streaming modes.

Contains:
    - LangChainAdapter (v0.2): New simplified interface based on BaseA2AAdapter
    - LangChainAgentAdapter (v0.1): Legacy interface, deprecated

Supports flexible input handling:
- input_mapper: Custom function for full control over input transformation
- parse_json_input: Automatic JSON parsing for structured inputs
- input_key: Simple text mapping to a single key (default fallback)
"""

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncIterator, Callable, Dict

from a2a.types import (
    Message,
    MessageSendParams,
    Part,
    Role,
    Task,
    TextPart,
)

from ..adapter import BaseAgentAdapter
from ..base_adapter import AdapterMetadata, BaseA2AAdapter

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# v0.2 Adapter (new, recommended)
# ═══════════════════════════════════════════════════════════════


class LangChainAdapter(BaseA2AAdapter):
    """Adapter for LangChain runnables (chains, agents, RAG pipelines).

    Supports both streaming (via ``astream()``) and non-streaming
    (via ``ainvoke()``) execution. Streaming is auto-detected based
    on whether the runnable has an ``astream`` method.

    Example::

        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from a2a_adapter import LangChainAdapter, serve_agent

        llm = ChatOpenAI(model="gpt-4o-mini")
        prompt = ChatPromptTemplate.from_template("Answer: {input}")
        chain = prompt | llm

        adapter = LangChainAdapter(runnable=chain, input_key="input")
        serve_agent(adapter, port=9002)
        # Automatically supports both message/send AND message/stream!
    """

    def __init__(
        self,
        runnable: Any,
        input_key: str = "input",
        output_key: str | None = None,
        timeout: int = 60,
        parse_json_input: bool = True,
        input_mapper: Callable[[str, str | None], Dict[str, Any]] | None = None,
        default_inputs: Dict[str, Any] | None = None,
        name: str = "",
        description: str = "",
    ) -> None:
        """Initialize the LangChain adapter.

        Args:
            runnable: A LangChain Runnable instance (chain, agent, etc.).
            input_key: Key name for passing input to the runnable (default: "input").
            output_key: Optional key to extract from runnable output.
            timeout: Execution timeout in seconds (default: 60).
            parse_json_input: Auto-parse JSON input (default: True).
            input_mapper: Custom function for input transformation.
            default_inputs: Default values to merge with parsed inputs.
            name: Optional agent name for AgentCard generation.
            description: Optional agent description for AgentCard generation.
        """
        self.runnable = runnable
        self.input_key = input_key
        self.output_key = output_key
        self.timeout = timeout
        self.parse_json_input = parse_json_input
        self.input_mapper = input_mapper
        self.default_inputs = default_inputs or {}
        self._name = name
        self._description = description

    async def invoke(self, user_input: str, context_id: str | None = None, **kwargs) -> str:
        """Invoke the runnable and return a text response."""
        inputs = self._build_inputs(user_input, context_id)
        result = await asyncio.wait_for(
            self.runnable.ainvoke(inputs), timeout=self.timeout,
        )
        return self._extract_output(result)

    async def stream(self, user_input: str, context_id: str | None = None, **kwargs) -> AsyncIterator[str]:
        """Stream tokens from the runnable via astream()."""
        inputs = self._build_inputs(user_input, context_id)
        async for chunk in self.runnable.astream(inputs):
            text = self._extract_chunk_text(chunk)
            if text:
                yield text

    def supports_streaming(self) -> bool:
        """Auto-detect streaming support from the runnable."""
        return hasattr(self.runnable, "astream")

    def get_metadata(self) -> AdapterMetadata:
        """Return adapter metadata for AgentCard generation."""
        return AdapterMetadata(
            name=self._name or "LangChainAdapter",
            description=self._description,
            streaming=self.supports_streaming(),
        )

    # ──── Internal: Input Building ────

    def _build_inputs(self, user_input: str, context_id: str | None) -> Dict[str, Any]:
        """Build runnable inputs with 3-priority pipeline."""
        # Priority 1: Custom input_mapper
        if self.input_mapper is not None:
            try:
                return {**self.default_inputs, **self.input_mapper(user_input, context_id)}
            except Exception as e:
                logger.warning("input_mapper failed: %s, falling back", e)

        # Priority 2: Auto JSON parsing
        if self.parse_json_input and user_input.strip().startswith("{"):
            try:
                parsed = json.loads(user_input)
                if isinstance(parsed, dict):
                    clean = {k: v for k, v in parsed.items() if k != "context_id"}
                    return {**self.default_inputs, **clean}
            except json.JSONDecodeError:
                pass

        # Priority 3: Fallback to input_key
        return {**self.default_inputs, self.input_key: user_input}

    # ──── Internal: Output Extraction ────

    def _extract_output(self, output: Any) -> str:
        """Extract text from runnable output (AIMessage, dict, str)."""
        # AIMessage or similar with content attribute
        if hasattr(output, "content"):
            content = output.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif hasattr(item, "text"):
                        parts.append(item.text)
                    elif isinstance(item, dict) and "text" in item:
                        parts.append(item["text"])
                return " ".join(parts)
            return str(content)

        # Dictionary output
        if isinstance(output, dict):
            if self.output_key and self.output_key in output:
                return str(output[self.output_key])
            for key in ("output", "result", "answer", "response", "text"):
                if key in output:
                    return str(output[key])
            return json.dumps(output, indent=2)

        return str(output)

    def _extract_chunk_text(self, chunk: Any) -> str:
        """Extract text from a streaming chunk."""
        # AIMessageChunk or similar
        if hasattr(chunk, "content"):
            content = chunk.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    item if isinstance(item, str)
                    else getattr(item, "text", "")
                    for item in content
                )
            return str(content) if content else ""

        # Dictionary chunk
        if isinstance(chunk, dict):
            if self.output_key and self.output_key in chunk:
                return str(chunk[self.output_key])
            for key in ("output", "result", "content", "text"):
                if key in chunk:
                    return str(chunk[key])
            return ""

        if isinstance(chunk, str):
            return chunk

        return ""


# ═══════════════════════════════════════════════════════════════
# v0.1 Adapter (legacy, deprecated — will be removed in v0.3)
# ═══════════════════════════════════════════════════════════════


class LangChainAgentAdapter(BaseAgentAdapter):
    """
    Adapter for integrating LangChain runnables as A2A agents.

    This adapter works with any LangChain Runnable (chains, agents, RAG pipelines)
    and supports both streaming and non-streaming execution modes.

    Input Handling (in priority order):

    1. **input_mapper** (highest priority):
       Custom function for full control over input transformation.
       Signature: (raw_input: str, context_id: str | None) -> dict

    2. **parse_json_input** (default: True):
       Automatically parse JSON input and pass directly to runnable.
       Perfect for chains that expect multiple input keys.

    3. **input_key** (fallback):
       Map plain text to a single key when JSON parsing fails.

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> from langchain_core.prompts import ChatPromptTemplate
        >>>
        >>> llm = ChatOpenAI(model="gpt-4o-mini")
        >>> prompt = ChatPromptTemplate.from_template("Answer: {input}")
        >>> chain = prompt | llm
        >>>
        >>> # Basic usage with auto JSON parsing
        >>> adapter = LangChainAgentAdapter(runnable=chain, input_key="input")
        >>>
        >>> # With custom input mapper
        >>> def my_mapper(raw_input: str, context_id: str | None) -> dict:
        ...     return {"input": raw_input, "context": context_id or "default"}
        >>> adapter = LangChainAgentAdapter(runnable=chain, input_mapper=my_mapper)
    """

    def __init__(
        self,
        runnable: Any,  # Type: Runnable (avoiding hard dependency)
        input_key: str = "input",
        output_key: str | None = None,
        timeout: int = 60,  # Default timeout in seconds
        # Flexible input handling parameters
        parse_json_input: bool = True,
        input_mapper: Callable[[str, str | None], Dict[str, Any]] | None = None,
        default_inputs: Dict[str, Any] | None = None,
    ):
        """
        Initialize the LangChain adapter.

        Args:
            runnable: A LangChain Runnable instance (chain, agent, etc.)
            input_key: The key name for passing input to the runnable (default: "input").
                       Used as fallback when JSON parsing fails or is disabled.
            output_key: Optional key to extract from runnable output. If None,
                        the adapter will attempt to extract text intelligently.
            timeout: Timeout for runnable execution in seconds (default: 60).
            parse_json_input: If True (default), attempt to parse input as JSON and use
                              the parsed dict directly as runnable inputs.
            input_mapper: Optional custom function to transform raw input to runnable inputs.
                          Signature: (raw_input: str, context_id: str | None) -> dict.
                          When provided, this takes highest priority over other methods.
            default_inputs: Optional dict of default values to merge with parsed inputs.
        """
        self.runnable = runnable
        self.input_key = input_key
        self.output_key = output_key
        self.timeout = timeout

        # Flexible input handling configuration
        self.parse_json_input = parse_json_input
        self.input_mapper = input_mapper
        self.default_inputs = default_inputs or {}

    # ---------- Input mapping ----------

    async def to_framework(self, params: MessageSendParams) -> Dict[str, Any]:
        """
        Convert A2A message parameters to LangChain runnable input.

        Processing priority:
        1. input_mapper (custom function) - highest priority
        2. parse_json_input (auto JSON parsing)
        3. input_key (fallback for plain text)

        Args:
            params: A2A message parameters

        Returns:
            Dictionary with runnable input data
        """
        # Use base class utility for raw input extraction
        raw_input = self.extract_raw_input(params)
        context_id = self.extract_context_id(params)

        # Priority 1: Custom input_mapper function (highest priority)
        if self.input_mapper is not None:
            try:
                mapped_inputs = self.input_mapper(raw_input, context_id)
                logger.debug("Used input_mapper to transform input")
                return {**self.default_inputs, **mapped_inputs}
            except Exception as e:
                logger.warning("input_mapper failed: %s, falling back", e)

        # Priority 2: Auto JSON parsing
        if self.parse_json_input:
            parsed = self.try_parse_json(raw_input)
            if parsed is not None:
                logger.debug("Parsed JSON input")
                # Remove context_id from parsed input as LangChain doesn't need it
                parsed_clean = {k: v for k, v in parsed.items() if k != "context_id"}
                return {**self.default_inputs, **parsed_clean}

        # Priority 3: Fallback to simple text mapping with input_key
        logger.debug("Using input_key '%s' fallback for plain text input", self.input_key)
        return {
            **self.default_inputs,
            self.input_key: raw_input,
        }

    # ---------- Framework call ----------

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
            asyncio.TimeoutError: If execution exceeds timeout
        """
        logger.debug("Invoking LangChain runnable with input: %s", framework_input)

        try:
            result = await asyncio.wait_for(
                self.runnable.ainvoke(framework_input),
                timeout=self.timeout,
            )
            logger.debug("LangChain runnable returned: %s", type(result).__name__)
            return result
        except asyncio.TimeoutError as e:
            logger.error("LangChain runnable timed out after %s seconds", self.timeout)
            raise RuntimeError(f"Runnable timed out after {self.timeout} seconds") from e

    # ---------- Output mapping ----------

    async def from_framework(
        self, framework_output: Any, params: MessageSendParams
    ) -> Message | Task:
        """
        Convert LangChain runnable output to A2A Message.

        Handles various LangChain output types:
        - AIMessage: Extract content attribute
        - Dict: Extract using output_key or serialize
        - String: Use directly

        Args:
            framework_output: Output from runnable execution
            params: Original A2A parameters

        Returns:
            A2A Message with the runnable's response
        """
        response_text = self._extract_output_text(framework_output)

        # Use base class utility for context_id extraction
        context_id = self.extract_context_id(params)

        return Message(
            role=Role.agent,
            message_id=str(uuid.uuid4()),
            context_id=context_id,
            parts=[Part(root=TextPart(text=response_text))],
        )

    def _extract_output_text(self, framework_output: Any) -> str:
        """
        Extract text content from LangChain runnable output.

        Args:
            framework_output: Output from the runnable

        Returns:
            Extracted text string
        """
        # AIMessage or similar with content attribute
        if hasattr(framework_output, "content"):
            content = framework_output.content
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                # Handle list of content blocks (multimodal)
                text_parts = []
                for item in content:
                    if isinstance(item, str):
                        text_parts.append(item)
                    elif hasattr(item, "text"):
                        text_parts.append(item.text)
                    elif isinstance(item, dict) and "text" in item:
                        text_parts.append(item["text"])
                return " ".join(text_parts)
            return str(content)

        # Dictionary output - extract using output_key or serialize
        if isinstance(framework_output, dict):
            if self.output_key and self.output_key in framework_output:
                return str(framework_output[self.output_key])
            # Try common output keys
            for key in ["output", "result", "answer", "response", "text"]:
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

        Uses LangChain's astream() method to yield tokens as they are generated.

        Args:
            params: A2A message parameters

        Yields:
            Server-Sent Events compatible dictionaries with streaming chunks
        """
        framework_input = await self.to_framework(params)
        context_id = self.extract_context_id(params)
        message_id = str(uuid.uuid4())

        logger.debug("Starting LangChain stream with input: %s", framework_input)

        accumulated_text = ""

        # Stream from LangChain runnable
        async for chunk in self.runnable.astream(framework_input):
            # Extract text from chunk
            text = self._extract_chunk_text(chunk)

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

        logger.debug("LangChain stream completed")

    def _extract_chunk_text(self, chunk: Any) -> str:
        """
        Extract text from a streaming chunk.

        Args:
            chunk: A streaming chunk from LangChain

        Returns:
            Extracted text string
        """
        # AIMessageChunk or similar
        if hasattr(chunk, "content"):
            content = chunk.content
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, str):
                        text_parts.append(item)
                    elif hasattr(item, "text"):
                        text_parts.append(item.text)
                return "".join(text_parts)
            return str(content) if content else ""

        # Dictionary chunk
        if isinstance(chunk, dict):
            if self.output_key and self.output_key in chunk:
                return str(chunk[self.output_key])
            for key in ["output", "result", "content", "text"]:
                if key in chunk:
                    return str(chunk[key])
            return ""

        # String chunk
        if isinstance(chunk, str):
            return chunk

        return ""

    def supports_streaming(self) -> bool:
        """
        Check if the runnable supports streaming.

        Returns:
            True if the runnable has an astream method
        """
        return hasattr(self.runnable, "astream")
