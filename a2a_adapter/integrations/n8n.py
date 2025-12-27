"""
n8n adapter for A2A Protocol.

This adapter enables n8n workflows to be exposed as A2A-compliant agents
by forwarding A2A messages to n8n webhooks.
"""

import json
import asyncio
import time
import uuid
from typing import Any, Dict

import httpx
from httpx import HTTPStatusError, ConnectError, ReadTimeout

from a2a.types import Message, MessageSendParams, Task, TextPart, Role, Part
from ..adapter import BaseAgentAdapter


class N8nAgentAdapter(BaseAgentAdapter):
    """
    Adapter for integrating n8n workflows as A2A agents.

    This adapter forwards A2A message requests to an n8n webhook URL and
    translates the response back to A2A format.
    """

    def __init__(
        self,
        webhook_url: str,
        timeout: int = 30,
        headers: Dict[str, str] | None = None,
        max_retries: int = 2,
        backoff: float = 0.25,
        payload_template: Dict[str, Any] | None = None,
        message_field: str = "message",
    ):
        """
        Initialize the n8n adapter.

        Args:
            webhook_url: The n8n webhook URL to send requests to.
            timeout: HTTP request timeout in seconds (default: 30).
            headers: Optional additional HTTP headers to include in requests.
            max_retries: Number of retry attempts for transient failures (default: 2).
            backoff: Base backoff seconds; multiplied by 2**attempt between retries.
            payload_template: Optional base payload dict to merge with message.
                              Use this to add static fields your n8n workflow expects.
            message_field: Field name for the user message (default: "message").
                           Change this if your n8n workflow expects a different field name.
        """
        self.webhook_url = webhook_url
        self.timeout = timeout
        self.headers = dict(headers) if headers else {}
        self.max_retries = max(0, int(max_retries))
        self.backoff = float(backoff)
        self.payload_template = dict(payload_template) if payload_template else {}
        self.message_field = message_field
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def handle(self, params: MessageSendParams) -> Message | Task:
        """Handle a non-streaming A2A message request."""
        framework_input = await self.to_framework(params)
        framework_output = await self.call_framework(framework_input, params)
        return await self.from_framework(framework_output, params)

    # ---------- Input mapping ----------

    async def to_framework(self, params: MessageSendParams) -> Dict[str, Any]:
        """
        Build the n8n webhook payload from A2A params.

        Extracts the latest user message text and constructs a JSON-serializable
        payload for posting to an n8n webhook. Supports custom payload templates
        and message field names for flexibility with different n8n workflows.

        Args:
            params: A2A message parameters.

        Returns:
            dict with the user message and any configured template fields.
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

        # Legacy support for messages array
        elif getattr(params, "messages", None):
            last = params.messages[-1]
            content = getattr(last, "content", "")
            if isinstance(content, str):
                user_message = content.strip()
            elif isinstance(content, list):
                text_parts: list[str] = []
                for item in content:
                    txt = getattr(item, "text", None)
                    if txt and isinstance(txt, str) and txt.strip():
                        text_parts.append(txt.strip())
                user_message = self._join_text_parts(text_parts)

        # Build payload with custom template support
        payload: Dict[str, Any] = {
            **self.payload_template,  # Start with template (e.g., {"name": "A2A Agent"})
            self.message_field: user_message,  # Add message with custom field name
        }

        # Add metadata only if not using custom template
        if not self.payload_template:
            payload["metadata"] = {
                "session_id": getattr(params, "session_id", None),
                "context": getattr(params, "context", None),
            }

        return payload

    @staticmethod
    def _join_text_parts(parts: list[str]) -> str:
        """
        Join text parts into a single string.
        """
        if not parts:
            return ""
        text = " ".join(p.strip() for p in parts if p)
        return text.strip()

    # ---------- Framework call ----------

    async def call_framework(
        self, framework_input: Dict[str, Any], params: MessageSendParams
    ) -> Dict[str, Any] | list:
        """
        Execute the n8n workflow by POSTing to the webhook URL with retries/backoff.

        Error policy:
          - 4xx: no retry, raise ValueError with a concise message (likely bad request/user/config).
          - 5xx / network timeouts / connect errors: retry with exponential backoff, then raise RuntimeError.
        """
        client = await self._get_client()
        req_id = str(uuid.uuid4())
        headers = {
            "Content-Type": "application/json",
            "X-Request-Id": req_id,
            **self.headers,
        }

        for attempt in range(self.max_retries + 1):
            start = time.monotonic()
            try:
                resp = await client.post(
                    self.webhook_url,
                    json=framework_input,
                    headers=headers,
                )
                dur_ms = int((time.monotonic() - start) * 1000)

                # Explicitly surface 4xx without retry.
                if 400 <= resp.status_code < 500:
                    text = (await resp.aread()).decode(errors="ignore")
                    raise ValueError(
                        f"n8n webhook returned {resp.status_code} "
                        f"(req_id={req_id}, {dur_ms}ms): {text[:512]}"
                    )

                # For 5xx, httpx will raise in raise_for_status().
                resp.raise_for_status()
                return resp.json()

            except HTTPStatusError as e:
                # Only 5xx should reach here (4xx is handled above).
                if attempt < self.max_retries:
                    await asyncio.sleep(self.backoff * (2**attempt))
                    continue
                raise RuntimeError(
                    f"n8n upstream 5xx after retries (req_id={req_id}): {e}"
                ) from e

            except (ConnectError, ReadTimeout) as e:
                if attempt < self.max_retries:
                    await asyncio.sleep(self.backoff * (2**attempt))
                    continue
                raise RuntimeError(
                    f"n8n upstream unavailable/timeout after retries (req_id={req_id}): {e}"
                ) from e

        # Should never reach here, but keeps type-checkers happy.
        raise RuntimeError("Unexpected error in call_framework retry loop.")

    # ---------- Output mapping ----------

    async def from_framework(
        self, framework_output: Dict[str, Any] | list, params: MessageSendParams
    ) -> Message | Task:
        """
        Convert n8n webhook response to A2A Message.

        Handles both n8n response formats:
        - Single object: {"output": "..."} (first entry only)
        - Array of objects: [{"output": "..."}, ...] (all entries)

        Args:
            framework_output: JSON response from n8n (dict or list).
            params: Original A2A parameters.

        Returns:
            A2A Message with the n8n response text.
        """
        # Handle array format (all entries from last node)
        if isinstance(framework_output, list):
            if len(framework_output) == 0:
                response_text = ""
            elif len(framework_output) == 1:
                # Single item in array - extract it
                response_text = self._extract_text_from_item(framework_output[0])
            else:
                # Multiple items - combine all outputs
                texts = []
                for item in framework_output:
                    if isinstance(item, dict):
                        text = self._extract_text_from_item(item)
                        if text:
                            texts.append(text)
                response_text = "\n".join(texts) if texts else json.dumps(framework_output, indent=2)
        elif isinstance(framework_output, dict):
            # Handle single object format (first entry only)
            response_text = self._extract_text_from_item(framework_output)
        else:
            # Fallback for unexpected types
            response_text = str(framework_output)

        return Message(
            role=Role.agent,
            message_id=str(uuid.uuid4()),
            parts=[Part(root=TextPart(text=response_text))],
        )

    def _extract_text_from_item(self, item: Dict[str, Any]) -> str:
        """
        Extract text content from a single n8n output item.

        Checks common field names in order of priority.

        Args:
            item: A dictionary from n8n workflow output.

        Returns:
            Extracted text string.
        """
        if not isinstance(item, dict):
            return str(item)
        if "output" in item:
            return str(item["output"])
        elif "result" in item:
            return str(item["result"])
        elif "message" in item:
            return str(item["message"])
        elif "text" in item:
            return str(item["text"])
        elif "response" in item:
            return str(item["response"])
        elif "content" in item:
            return str(item["content"])
        else:
            # Fallback: serialize entire item as JSON
            return json.dumps(item, indent=2)

    # ---------- Lifecycle ----------

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    def supports_streaming(self) -> bool:
        """This adapter does not support streaming responses."""
        return False


