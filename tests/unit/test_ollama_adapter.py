"""Tests for OllamaClient and OllamaAdapter (v0.2)."""

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from a2a_adapter.integrations.ollama import OllamaAdapter, OllamaClient


# ══════════════════════════════════════════════════════════════
# OllamaClient tests
# ══════════════════════════════════════════════════════════════


@pytest.fixture
def client():
    return OllamaClient(
        model="llama3.2:8b",
        system_prompt="You are helpful.",
        temperature=0.5,
    )


@pytest.fixture
def client_no_system():
    return OllamaClient(model="mistral")


class TestBuildPayload:
    def test_basic_payload(self, client_no_system):
        payload = client_no_system._build_payload("Hello", stream=False)
        assert payload["model"] == "mistral"
        assert payload["stream"] is False
        assert len(payload["messages"]) == 1
        assert payload["messages"][0] == {"role": "user", "content": "Hello"}
        assert "options" not in payload

    def test_payload_with_system_prompt(self, client):
        payload = client._build_payload("Hi", stream=True)
        assert payload["stream"] is True
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][0]["content"] == "You are helpful."
        assert payload["messages"][1] == {"role": "user", "content": "Hi"}

    def test_payload_with_temperature(self, client):
        payload = client._build_payload("test", stream=False)
        assert payload["options"]["temperature"] == 0.5

    def test_payload_with_keep_alive(self):
        c = OllamaClient(model="x", keep_alive="10m")
        payload = c._build_payload("test", stream=False)
        assert payload["keep_alive"] == "10m"


class TestClientChat:
    @pytest.mark.asyncio
    async def test_chat_success(self, client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "Hello there!"},
            "done": True,
        }

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        client._client = mock_http

        result = await client.chat("Hi")
        assert result == "Hello there!"
        mock_http.post.assert_called_once()
        call_kwargs = mock_http.post.call_args
        assert call_kwargs[1]["json"]["stream"] is False

    @pytest.mark.asyncio
    async def test_chat_connection_error(self, client):
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        client._client = mock_http

        with pytest.raises(RuntimeError, match="Cannot connect to Ollama"):
            await client.chat("Hi")

    @pytest.mark.asyncio
    async def test_chat_http_error(self, client):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "model not found"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=mock_response
        )

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        client._client = mock_http

        with pytest.raises(RuntimeError, match="Ollama returned HTTP 404"):
            await client.chat("Hi")


class TestClientStream:
    @pytest.mark.asyncio
    async def test_chat_stream_success(self, client):
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

        mock_http = AsyncMock()
        mock_http.stream = MagicMock(return_value=mock_response)
        client._client = mock_http

        chunks = []
        async for chunk in client.chat_stream("Hi"):
            chunks.append(chunk)
        assert chunks == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_chat_stream_connection_error(self, client):
        mock_response = AsyncMock()
        mock_response.__aenter__ = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_http = AsyncMock()
        mock_http.stream = MagicMock(return_value=mock_response)
        client._client = mock_http

        with pytest.raises(RuntimeError, match="Cannot connect to Ollama"):
            async for _ in client.chat_stream("Hi"):
                pass


class TestClientClose:
    @pytest.mark.asyncio
    async def test_close(self, client):
        mock_http = AsyncMock()
        client._client = mock_http
        await client.close()
        mock_http.aclose.assert_called_once()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_close_no_client(self, client):
        await client.close()  # Should not raise


# ══════════════════════════════════════════════════════════════
# OllamaAdapter tests
# ══════════════════════════════════════════════════════════════


@pytest.fixture
def mock_client():
    """OllamaClient with mocked methods."""
    c = MagicMock(spec=OllamaClient)
    c.chat = AsyncMock(return_value="Hello there!")
    c.close = AsyncMock()
    return c


@pytest.fixture
def adapter(mock_client):
    return OllamaAdapter(
        client=mock_client,
        name="Test Ollama",
        description="Test adapter",
    )


class TestAdapterWithClient:
    """Test that OllamaAdapter correctly delegates to OllamaClient."""

    @pytest.mark.asyncio
    async def test_invoke_delegates_to_client(self, adapter, mock_client):
        result = await adapter.invoke("Hi")
        assert result == "Hello there!"
        mock_client.chat.assert_called_once_with("Hi")

    @pytest.mark.asyncio
    async def test_stream_delegates_to_client(self, adapter, mock_client):
        mock_client.chat_stream = MagicMock(
            return_value=_async_iter(["Hello", " world"])
        )
        chunks = []
        async for chunk in adapter.stream("Hi"):
            chunks.append(chunk)
        assert chunks == ["Hello", " world"]
        mock_client.chat_stream.assert_called_once_with("Hi")

    @pytest.mark.asyncio
    async def test_close_delegates_to_client(self, adapter, mock_client):
        await adapter.close()
        mock_client.close.assert_called_once()


class TestAdapterConvenienceConstructor:
    """Test that OllamaAdapter can be created with convenience params."""

    def test_creates_client_from_params(self):
        adapter = OllamaAdapter(
            model="codellama",
            base_url="http://custom:11434",
            system_prompt="Be concise.",
            temperature=0.3,
        )
        assert isinstance(adapter.client, OllamaClient)
        assert adapter.client.model == "codellama"
        assert adapter.client.base_url == "http://custom:11434"
        assert adapter.client.system_prompt == "Be concise."
        assert adapter.client.temperature == 0.3

    def test_explicit_client_takes_precedence(self):
        explicit = OllamaClient(model="phi3")
        adapter = OllamaAdapter(client=explicit, model="ignored")
        assert adapter.client is explicit
        assert adapter.client.model == "phi3"


class TestAdapterMetadata:
    def test_default_metadata(self):
        a = OllamaAdapter(model="codellama")
        meta = a.get_metadata()
        assert meta.name == "OllamaAdapter"
        assert meta.streaming is True

    def test_custom_metadata(self):
        a = OllamaAdapter(
            model="x",
            name="Test Ollama",
            description="Test adapter",
        )
        meta = a.get_metadata()
        assert meta.name == "Test Ollama"
        assert meta.description == "Test adapter"
        assert meta.streaming is True

    def test_supports_streaming(self, adapter):
        assert adapter.supports_streaming() is True

    def test_model_not_leaked_in_metadata(self):
        a = OllamaAdapter(model="llama3.2:8b")
        meta = a.get_metadata()
        assert "llama3.2" not in meta.name
        assert "llama3.2" not in meta.description


class TestLoading:
    def test_load_adapter(self):
        from a2a_adapter import load_adapter

        adapter = load_adapter({"adapter": "ollama", "model": "phi3"})
        assert isinstance(adapter, OllamaAdapter)
        assert adapter.client.model == "phi3"


# ──── Helpers ────


async def _async_iter(items):
    for item in items:
        yield item
