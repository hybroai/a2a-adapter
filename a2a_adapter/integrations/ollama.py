"""
Ollama adapter for A2A Protocol.

This adapter enables local Ollama models to be exposed as A2A-compliant agents
by forwarding A2A messages to the Ollama HTTP API (/api/chat).

Contains:
    - OllamaAdapter (v0.2): Simplified interface based on BaseA2AAdapter

Supports both streaming and non-streaming execution via Ollama's
NDJSON streaming protocol.
"""

import json
import logging
from typing import Any, AsyncIterator, Dict

import httpx

from ..base_adapter import AdapterMetadata, BaseA2AAdapter

logger = logging.getLogger(__name__)


class OllamaAdapter(BaseA2AAdapter):
    """Adapter for local Ollama language models.

    Wraps the Ollama HTTP API (``/api/chat``) to expose any Ollama model
    as an A2A-compliant agent with streaming support.

    Example::

        from a2a_adapter import OllamaAdapter, serve_agent

        adapter = OllamaAdapter(model="llama3.2:8b", name="My Local LLM")
        serve_agent(adapter, port=10010)

    Example with system prompt::

        adapter = OllamaAdapter(
            model="llama3.2:8b",
            system_prompt="You are a helpful coding assistant.",
            temperature=0.3,
        )
        serve_agent(adapter, port=10010)
    """

    def __init__(
        self,
        model: str = "llama3.2:8b",
        base_url: str = "http://localhost:11434",
        system_prompt: str | None = None,
        temperature: float | None = None,
        timeout: int = 120,
        keep_alive: str | None = None,
        name: str = "",
        description: str = "",
        skills: list[dict] | None = None,
        provider: dict | None = None,
        documentation_url: str | None = None,
        icon_url: str | None = None,
    ) -> None:
        """Initialize the Ollama adapter.

        Args:
            model: Ollama model name (e.g. "llama3.2:8b", "mistral", "codellama").
            base_url: Ollama server base URL (default: http://localhost:11434).
            system_prompt: Optional system prompt prepended to every conversation.
            temperature: Sampling temperature. None uses Ollama's default.
            timeout: HTTP request timeout in seconds (default: 120).
            keep_alive: How long Ollama keeps the model loaded (e.g. "5m", "0" to unload).
            name: Optional agent name for AgentCard generation.
            description: Optional agent description for AgentCard generation.
            skills: Optional list of skill dicts for AgentCard generation.
            provider: Optional dict with 'organization' and 'url' keys.
            documentation_url: Optional URL to the agent's documentation.
            icon_url: Optional URL to an icon for the agent.
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.timeout = timeout
        self.keep_alive = keep_alive
        self._name = name
        self._description = description
        self._skills = skills or []
        self._provider = provider
        self._documentation_url = documentation_url
        self._icon_url = icon_url
        self._client: httpx.AsyncClient | None = None

    # ──── Required: invoke ────

    async def invoke(self, user_input: str, context_id: str | None = None, **kwargs) -> str:
        """Send a message to Ollama and return the full response."""
        payload = self._build_payload(user_input, stream=False)
        client = await self._get_client()

        try:
            resp = await client.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Is Ollama running? (ollama serve): {e}"
            ) from e
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Ollama returned HTTP {e.response.status_code}: "
                f"{e.response.text[:512]}"
            ) from e

    # ──── Optional: streaming ────

    async def stream(
        self, user_input: str, context_id: str | None = None, **kwargs
    ) -> AsyncIterator[str]:
        """Stream response tokens from Ollama.

        Ollama's /api/chat with stream=true returns NDJSON lines,
        each containing {"message": {"content": "token"}, "done": false}.
        """
        payload = self._build_payload(user_input, stream=True)
        client = await self._get_client()

        try:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if chunk.get("done", False):
                        return
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Is Ollama running? (ollama serve): {e}"
            ) from e
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Ollama returned HTTP {e.response.status_code}"
            ) from e

    # ──── Optional: metadata ────

    def get_metadata(self) -> AdapterMetadata:
        """Return adapter metadata for AgentCard generation."""
        return AdapterMetadata(
            name=self._name or f"Ollama ({self.model})",
            description=self._description or f"Local LLM via Ollama ({self.model})",
            skills=self._skills,
            streaming=True,
            provider=self._provider,
            documentation_url=self._documentation_url,
            icon_url=self._icon_url,
        )

    # ──── Optional: cleanup ────

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ──── Internal helpers ────

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy-init httpx client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    def _build_payload(self, user_input: str, stream: bool) -> Dict[str, Any]:
        """Build the Ollama /api/chat request payload."""
        messages: list[Dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_input})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }
        if self.temperature is not None:
            payload["options"] = {"temperature": self.temperature}
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        return payload
