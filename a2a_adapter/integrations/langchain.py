"""
LangChain adapter for A2A Protocol.

This adapter enables LangChain runnables (chains, agents, RAG pipelines) to be
exposed as A2A-compliant agents with support for both streaming and non-streaming modes.
"""

import json
import logging
import uuid
from typing import Any, AsyncIterator, Dict

from a2a.types import (
    Message,
    MessageSendParams,
    Task,
    TextPart,
    Role,
    Part,
)
from ..adapter import BaseAgentAdapter

logger = logging.getLogger(__name__)


class LangChainAgentAdapter(BaseAgentAdapter):
    """
    Adapter for integrating LangChain runnables as A2A agents.

    This adapter works with any LangChain Runnable (chains, agents, RAG pipelines)
    and supports both streaming and non-streaming execution modes.

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> from langchain_core.prompts import ChatPromptTemplate
        >>>
        >>> llm = ChatOpenAI(model="gpt-4o-mini")
        >>> prompt = ChatPromptTemplate.from_template("Answer: {input}")
        >>> chain = prompt | llm
        >>>
        >>> adapter = LangChainAgentAdapter(runnable=chain, input_key="input")
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
            output_key: Optional key to extract from runnable output. If None,
                        the adapter will attempt to extract text intelligently.
        """
        self.runnable = runnable
        self.input_key = input_key
        self.output_key = output_key

    # ---------- Input mapping ----------

    async def to_framework(self, params: MessageSendParams) -> Dict[str, Any]:
        """
        Convert A2A message parameters to LangChain runnable input.

        Extracts the user's message text and formats it for the runnable.

        Args:
            params: A2A message parameters

        Returns:
            Dictionary with runnable input data
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

        # Build runnable input
        return {
            self.input_key: user_message,
        }

    @staticmethod
    def _join_text_parts(parts: list[str]) -> str:
        """Join text parts into a single string."""
        if not parts:
            return ""
        text = " ".join(p.strip() for p in parts if p)
        return text.strip()

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
        """
        logger.debug("Invoking LangChain runnable with input: %s", framework_input)
        result = await self.runnable.ainvoke(framework_input)
        logger.debug("LangChain runnable returned: %s", type(result).__name__)
        return result

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

        # Preserve context_id from the request for multi-turn conversation tracking
        context_id = self._extract_context_id(params)

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

    def _extract_context_id(self, params: MessageSendParams) -> str | None:
        """Extract context_id from MessageSendParams."""
        if hasattr(params, "message") and params.message:
            return getattr(params.message, "context_id", None)
        return None

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
        context_id = self._extract_context_id(params)
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
