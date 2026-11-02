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
        "exc_type",
        [httpx.ReadError, httpx.WriteError, httpx.WriteTimeout, httpx.RemoteProtocolError],
    )
    async def test_mid_request_transport_errors_wrapped_not_retried(self, exc_type):
        def responder(req, n):
            raise exc_type("connection reset", request=req)

        handler, calls = _counting_handler(responder)
        adapter = _adapter_with_transport(handler, max_retries=2)

        with pytest.raises(RuntimeError, match="req_id="):
            await adapter._post_with_retry({"message": "hi"})
        assert len(calls) == 1

    async def test_read_error_retried_when_opted_in(self):
        def responder(req, n):
            if n == 1:
                raise httpx.ReadError("connection reset", request=req)
            return httpx.Response(200, json={"output": "ok"})

        handler, calls = _counting_handler(responder)
        adapter = _adapter_with_transport(
            handler, max_retries=2, retry_on_server_error=True
        )

        assert await adapter._post_with_retry({"message": "hi"}) == {"output": "ok"}
        assert len(calls) == 2

    async def test_client_keepalive_shorter_than_n8n_idle_timeout(self, monkeypatch):
        """Idle sockets expire before n8n's 5s keep-alive closes them, so a
        stale-connection disconnect (no longer retried) is unlikely."""
        created = {}

        class FakeClient:
            is_closed = False

            def __init__(self, **kwargs):
                created.update(kwargs)

        monkeypatch.setattr(
            "a2a_adapter.integrations.n8n.httpx.AsyncClient", FakeClient
        )
        await N8nAdapter(webhook_url=WEBHOOK)._get_client()
        assert created["limits"].keepalive_expiry < 5

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


class TestPayloadPrecedence:
    """default_inputs < client JSON < payload_template < adapter context fields."""

    def test_client_json_cannot_override_template_key(self):
        adapter = N8nAdapter(
            webhook_url=WEBHOOK, payload_template={"role": "agent", "tenant": "acme"}
        )
        payload = adapter._build_payload(
            '{"role": "admin", "tenant": "evil", "query": "hi"}', "ctx-1"
        )
        assert payload["role"] == "agent"
        assert payload["tenant"] == "acme"
        assert payload["query"] == "hi"

    def test_client_json_fills_default_inputs_key(self):
        adapter = N8nAdapter(webhook_url=WEBHOOK, default_inputs={"lang": "en", "n": 1})
        payload = adapter._build_payload('{"lang": "fr"}', "ctx-1")
        assert payload["lang"] == "fr"
        assert payload["n"] == 1

    def test_client_json_cannot_forge_context_metadata(self):
        adapter = N8nAdapter(webhook_url=WEBHOOK)
        payload = adapter._build_payload(
            '{"query": "hi", "metadata": {"context_id": "forged", "tag": "x"}}',
            "ctx-1",
        )
        assert payload["metadata"]["context_id"] == "ctx-1"
        assert payload["metadata"]["tag"] == "x"

    def test_client_json_cannot_forge_context_id_with_template(self):
        adapter = N8nAdapter(webhook_url=WEBHOOK, payload_template={"name": "bot"})
        payload = adapter._build_payload('{"context_id": "forged"}', "ctx-1")
        assert payload["context_id"] == "ctx-1"

    def test_client_metadata_context_id_overwritten_with_template(self):
        adapter = N8nAdapter(webhook_url=WEBHOOK, payload_template={"name": "bot"})
        payload = adapter._build_payload(
            '{"metadata": {"context_id": "forged", "tag": "x"}}', "ctx-1"
        )
        assert payload["context_id"] == "ctx-1"
        assert payload["metadata"] == {"context_id": "ctx-1", "tag": "x"}

    def test_client_metadata_context_id_overwritten_with_template_context_id(self):
        adapter = N8nAdapter(
            webhook_url=WEBHOOK, payload_template={"context_id": "fixed"}
        )
        payload = adapter._build_payload(
            '{"metadata": {"context_id": "forged"}}', "ctx-1"
        )
        assert payload["metadata"] == {"context_id": "ctx-1"}

    def test_template_metadata_wins_over_client_metadata(self):
        adapter = N8nAdapter(
            webhook_url=WEBHOOK, payload_template={"metadata": {"source": "op"}}
        )
        payload = adapter._build_payload(
            '{"metadata": {"context_id": "forged"}}', "ctx-1"
        )
        assert payload["metadata"] == {"source": "op"}
        assert payload["context_id"] == "ctx-1"

    def test_template_wins_over_default_inputs_for_plain_text(self):
        adapter = N8nAdapter(
            webhook_url=WEBHOOK,
            payload_template={"lang": "en"},
            default_inputs={"lang": "fr", "n": 1},
        )
        payload = adapter._build_payload("hello", "ctx-1")
        assert payload["lang"] == "en"
        assert payload["n"] == 1

    def test_template_context_id_still_wins(self):
        adapter = N8nAdapter(
            webhook_url=WEBHOOK, payload_template={"context_id": "fixed"}
        )
        payload = adapter._build_payload('{"context_id": "forged"}', "ctx-1")
        assert payload["context_id"] == "fixed"

    def test_input_mapper_cannot_override_template_key(self):
        adapter = N8nAdapter(
            webhook_url=WEBHOOK,
            payload_template={"role": "agent"},
            default_inputs={"lang": "en"},
            input_mapper=lambda raw, ctx: {"role": raw, "lang": "fr"},
        )
        payload = adapter._build_payload("admin", "ctx-1")
        assert payload["role"] == "agent"
        assert payload["lang"] == "fr"

    def test_plain_text_payload_unchanged(self):
        adapter = N8nAdapter(webhook_url=WEBHOOK, default_inputs={"lang": "en"})
        assert adapter._build_payload("hello", "ctx-1") == {
            "lang": "en",
            "message": "hello",
            "metadata": {"context_id": "ctx-1"},
        }
