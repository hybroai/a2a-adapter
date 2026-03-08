"""
Ollama adapter for A2A Protocol.

This module provides two classes:
    - OllamaClient: HTTP client for the Ollama API (/api/chat)
    - OllamaAdapter: A2A adapter that wraps an OllamaClient

Separating the client from the adapter follows the same pattern as
LangChainAdapter (accepts a runnable) and LangGraphAdapter (accepts a graph),
keeping infrastructure concerns out of the adapter layer.

Supports both streaming and non-streaming execution via Ollama's
NDJSON streaming protocol.
"""

import json
import logging
from typing import Any, AsyncIterator, Dict

import httpx

from ..base_adapter import AdapterMetadata, BaseA2AAdapter

logger = logging.getLogger(__name__)


class OllamaClient:
    """HTTP client for the Ollama API.

    Encapsulates all Ollama-specific configuration (model, base_url,
    system_prompt, sampling parameters) and HTTP communication, keeping
    these concerns separate from the A2A adapter layer.

    Example::

        client = OllamaClient(model="llama3.2:8b")
        text = await client.chat("Why is the sky blue?")

    Example with streaming::

        async for token in client.chat_stream("Tell me a story"):
            print(token, end="")
    """

    def __init__(
        self,
        model: str = "llama3.2:8b",
        base_url: str = "http://localhost:11434",
        system_prompt: str | None = None,
        temperature: float | None = None,
        timeout: int = 120,
        keep_alive: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.timeout = timeout
        self.keep_alive = keep_alive
        self._client: httpx.AsyncClient | None = None

    async def chat(self, user_input: str) -> str:
        """Send a message and return the full response."""
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

    async def chat_stream(self, user_input: str) -> AsyncIterator[str]:
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

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _get_client(self) -> httpx.AsyncClient:
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

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


class OllamaAdapter(BaseA2AAdapter):
    """A2A adapter for Ollama, wrapping an OllamaClient.

    Follows the same pattern as LangChainAdapter (accepts a runnable) and
    LangGraphAdapter (accepts a graph): the user constructs the client with
    model/infrastructure config, then passes it to the adapter.

    Example::

        from a2a_adapter import OllamaAdapter, serve_agent
        from a2a_adapter.integrations.ollama import OllamaClient

        client = OllamaClient(model="llama3.2:8b")
        adapter = OllamaAdapter(client=client, name="My Local LLM")
        serve_agent(adapter, port=10010)

    Convenience shorthand (creates OllamaClient internally)::

        adapter = OllamaAdapter(
            model="llama3.2:8b",
            name="My Local LLM",
        )
    """

    def __init__(
        self,
        client: OllamaClient | None = None,
        *,
        # Convenience params — used to create OllamaClient if client is None
        model: str = "llama3.2:8b",
        base_url: str = "http://localhost:11434",
        system_prompt: str | None = None,
        temperature: float | None = None,
        timeout: int = 120,
        keep_alive: str | None = None,
        # AgentCard metadata
        name: str = "",
        description: str = "",
        skills: list[dict] | None = None,
        provider: dict | None = None,
        documentation_url: str | None = None,
        icon_url: str | None = None,
    ) -> None:
        if client is not None:
            self.client = client
        else:
            self.client = OllamaClient(
                model=model,
                base_url=base_url,
                system_prompt=system_prompt,
                temperature=temperature,
                timeout=timeout,
                keep_alive=keep_alive,
            )
        self._name = name
        self._description = description
        self._skills = skills or []
        self._provider = provider
        self._documentation_url = documentation_url
        self._icon_url = icon_url

    # ──── Required: invoke ────

    async def invoke(self, user_input: str, context_id: str | None = None, **kwargs) -> str:
        """Send a message to Ollama and return the full response."""
        return await self.client.chat(user_input)

    # ──── Optional: streaming ────

    async def stream(
        self, user_input: str, context_id: str | None = None, **kwargs
    ) -> AsyncIterator[str]:
        """Stream response tokens from Ollama."""
        async for token in self.client.chat_stream(user_input):
            yield token

    # ──── Optional: metadata ────

    def get_metadata(self) -> AdapterMetadata:
        """Return adapter metadata for AgentCard generation."""
        return AdapterMetadata(
            name=self._name or "OllamaAdapter",
            description=self._description,
            skills=self._skills,
            streaming=True,
            provider=self._provider,
            documentation_url=self._documentation_url,
            icon_url=self._icon_url,
        )

    # ──── Optional: cleanup ────

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self.client.close()
