"""Tests for OllamaAdapter (v0.2)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from a2a_adapter.integrations.ollama import OllamaAdapter


# ──── Fixtures ────


@pytest.fixture
def adapter():
    return OllamaAdapter(
        model="llama3.2:8b",
        name="Test Ollama",
        description="Test adapter",
        system_prompt="You are helpful.",
        temperature=0.5,
    )


@pytest.fixture
def adapter_no_system():
    return OllamaAdapter(model="mistral")


# ──── Payload building ────


class TestBuildPayload:
    def test_basic_payload(self, adapter_no_system):
        payload = adapter_no_system._build_payload("Hello", stream=False)
        assert payload["model"] == "mistral"
        assert payload["stream"] is False
        assert len(payload["messages"]) == 1
        assert payload["messages"][0] == {"role": "user", "content": "Hello"}
        assert "options" not in payload

    def test_payload_with_system_prompt(self, adapter):
        payload = adapter._build_payload("Hi", stream=True)
        assert payload["stream"] is True
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][0]["content"] == "You are helpful."
        assert payload["messages"][1] == {"role": "user", "content": "Hi"}

    def test_payload_with_temperature(self, adapter):
        payload = adapter._build_payload("test", stream=False)
        assert payload["options"]["temperature"] == 0.5

    def test_payload_with_keep_alive(self):
        a = OllamaAdapter(model="x", keep_alive="10m")
        payload = a._build_payload("test", stream=False)
        assert payload["keep_alive"] == "10m"


# ──── Metadata ────


class TestMetadata:
    def test_default_metadata(self):
        a = OllamaAdapter(model="codellama")
        meta = a.get_metadata()
        assert "codellama" in meta.name
        assert meta.streaming is True

    def test_custom_metadata(self, adapter):
        meta = adapter.get_metadata()
        assert meta.name == "Test Ollama"
        assert meta.description == "Test adapter"
        assert meta.streaming is True

    def test_supports_streaming(self, adapter):
        assert adapter.supports_streaming() is True


# ──── invoke() ────


class TestInvoke:
    @pytest.mark.asyncio
    async def test_invoke_success(self, adapter):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "Hello there!"},
            "done": True,
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        adapter._client = mock_client

        result = await adapter.invoke("Hi")
        assert result == "Hello there!"
        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        assert call_kwargs[1]["json"]["stream"] is False

    @pytest.mark.asyncio
    async def test_invoke_connection_error(self, adapter):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        adapter._client = mock_client

        with pytest.raises(RuntimeError, match="Cannot connect to Ollama"):
            await adapter.invoke("Hi")

    @pytest.mark.asyncio
    async def test_invoke_http_error(self, adapter):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "model not found"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=mock_response
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        adapter._client = mock_client

        with pytest.raises(RuntimeError, match="Ollama returned HTTP 404"):
            await adapter.invoke("Hi")


# ──── stream() ────


class TestStream:
    @pytest.mark.asyncio
    async def test_stream_success(self, adapter):
        lines = [
            json.dumps({"message": {"content": "Hello"}, "done": False}),
            json.dumps({"message": {"content": " world"}, "done": False}),
            json.dumps({"message": {"content": ""}, "done": True}),
        ]

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = MagicMock(return_value=_async_iter(lines))
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_response)
        adapter._client = mock_client

        chunks = []
        async for chunk in adapter.stream("Hi"):
            chunks.append(chunk)
        assert chunks == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_stream_connection_error(self, adapter):
        mock_response = AsyncMock()
        mock_response.__aenter__ = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_response)
        adapter._client = mock_client

        with pytest.raises(RuntimeError, match="Cannot connect to Ollama"):
            async for _ in adapter.stream("Hi"):
                pass


# ──── close() ────


class TestClose:
    @pytest.mark.asyncio
    async def test_close(self, adapter):
        mock_client = AsyncMock()
        adapter._client = mock_client
        await adapter.close()
        mock_client.aclose.assert_called_once()
        assert adapter._client is None

    @pytest.mark.asyncio
    async def test_close_no_client(self, adapter):
        await adapter.close()  # Should not raise


# ──── Config-driven loading ────


class TestLoading:
    def test_load_adapter(self):
        from a2a_adapter import load_adapter

        adapter = load_adapter({"adapter": "ollama", "model": "phi3"})
        assert isinstance(adapter, OllamaAdapter)
        assert adapter.model == "phi3"


# ──── Helpers ────


async def _async_iter(items):
    for item in items:
        yield item
