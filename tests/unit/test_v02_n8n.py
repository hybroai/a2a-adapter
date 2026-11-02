"""Tests for the v0.2 N8nAdapter webhook retry policy and payload building."""

import httpx
import pytest

from a2a_adapter.integrations.n8n import N8nAdapter

WEBHOOK = "http://localhost:5678/webhook/test"


def _adapter_with_transport(handler, **kwargs) -> N8nAdapter:
    """Build an adapter whose HTTP client is backed by ``httpx.MockTransport``."""
    adapter = N8nAdapter(webhook_url=WEBHOOK, backoff=0, **kwargs)
    adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return adapter


def _counting_handler(responder):
    """Wrap a responder so the test can count webhook calls."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return responder(request, len(calls))

    return handler, calls


class TestRetryPolicy:
    async def test_5xx_not_retried_by_default(self):
        handler, calls = _counting_handler(lambda req, n: httpx.Response(500))
        adapter = _adapter_with_transport(handler, max_retries=2)

        with pytest.raises(RuntimeError, match="500"):
            await adapter._post_with_retry({"message": "hi"})
        assert len(calls) == 1

    async def test_5xx_retried_when_opted_in(self):
        def responder(req, n):
            if n < 3:
                return httpx.Response(502)
            return httpx.Response(200, json={"output": "ok"})

        handler, calls = _counting_handler(responder)
        adapter = _adapter_with_transport(
            handler, max_retries=2, retry_on_server_error=True
        )

        assert await adapter._post_with_retry({"message": "hi"}) == {"output": "ok"}
        assert len(calls) == 3

    async def test_5xx_retries_exhausted_when_opted_in(self):
        handler, calls = _counting_handler(lambda req, n: httpx.Response(503))
        adapter = _adapter_with_transport(
            handler, max_retries=2, retry_on_server_error=True
        )

        with pytest.raises(RuntimeError, match="503"):
            await adapter._post_with_retry({"message": "hi"})
        assert len(calls) == 3

    async def test_read_timeout_not_retried_by_default(self):
        def responder(req, n):
            raise httpx.ReadTimeout("still running", request=req)

        handler, calls = _counting_handler(responder)
        adapter = _adapter_with_transport(handler, max_retries=2)

        with pytest.raises(RuntimeError, match="timeout"):
            await adapter._post_with_retry({"message": "hi"})
        assert len(calls) == 1

    async def test_read_timeout_retried_when_opted_in(self):
        def responder(req, n):
            if n == 1:
                raise httpx.ReadTimeout("still running", request=req)
            return httpx.Response(200, json={"output": "ok"})

        handler, calls = _counting_handler(responder)
        adapter = _adapter_with_transport(
            handler, max_retries=2, retry_on_server_error=True
        )

        assert await adapter._post_with_retry({"message": "hi"}) == {"output": "ok"}
        assert len(calls) == 2

    async def test_remote_protocol_error_not_retried_by_default(self):
        def responder(req, n):
            raise httpx.RemoteProtocolError("server disconnected", request=req)

        handler, calls = _counting_handler(responder)
        adapter = _adapter_with_transport(handler, max_retries=2)

        with pytest.raises(RuntimeError):
            await adapter._post_with_retry({"message": "hi"})
        assert len(calls) == 1

    @pytest.mark.parametrize(
        "exc_type", [httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout]
    )
    async def test_pre_send_failures_retried(self, exc_type):
        def responder(req, n):
            if n <= 2:
                raise exc_type("not connected", request=req)
            return httpx.Response(200, json={"output": "ok"})

        handler, calls = _counting_handler(responder)
        adapter = _adapter_with_transport(handler, max_retries=2)

        assert await adapter._post_with_retry({"message": "hi"}) == {"output": "ok"}
        assert len(calls) == 3

    async def test_pre_send_failures_exhaust_retries(self):
        def responder(req, n):
            raise httpx.ConnectTimeout("not connected", request=req)

        handler, calls = _counting_handler(responder)
        adapter = _adapter_with_transport(handler, max_retries=2)

        with pytest.raises(RuntimeError, match="unavailable"):
            await adapter._post_with_retry({"message": "hi"})
        assert len(calls) == 3

    @pytest.mark.parametrize("retry_on_server_error", [False, True])
    async def test_4xx_never_retried(self, retry_on_server_error):
        handler, calls = _counting_handler(lambda req, n: httpx.Response(422))
        adapter = _adapter_with_transport(
            handler, max_retries=2, retry_on_server_error=retry_on_server_error
        )

        with pytest.raises(ValueError, match="422"):
            await adapter._post_with_retry({"message": "hi"})
        assert len(calls) == 1
