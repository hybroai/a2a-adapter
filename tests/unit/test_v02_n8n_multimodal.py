"""Tests for N8nAdapter V0.2 multimodal methods after V1.0 proto migration."""

import base64
from unittest.mock import MagicMock

import httpx
import pytest
from a2a.types import Part

from a2a_adapter.integrations.n8n import N8nAdapter


def _client_for(handler, **kwargs) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)


def _fail_if_called(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"unexpected HTTP request to {request.url}")


class TestBuildMultimodalPayload:
    """Test _build_multimodal_payload with V1.0 proto Part fields."""

    @pytest.fixture
    def adapter(self):
        return N8nAdapter(
            webhook_url="http://localhost:5678/webhook/test",
            multimodal_mode=True,
        )

    async def test_text_only_parts_ignored(self, adapter):
        """Parts with only text set should not appear in files/images."""
        ctx = MagicMock()
        text_part = Part(text="hello")
        ctx.message.parts = [text_part]

        payload = await adapter._build_multimodal_payload("hello", "ctx-1", ctx)
        assert "files" not in payload
        assert "images" not in payload

    async def test_url_part_passed_through_without_fetch(self, adapter):
        """By default a url Part is forwarded as a uri and never fetched."""
        ctx = MagicMock()
        file_part = Part(url="http://169.254.169.254/latest/meta-data", filename="doc.pdf", media_type="application/pdf")
        ctx.message.parts = [file_part]
        adapter._client = _client_for(_fail_if_called)

        payload = await adapter._build_multimodal_payload("check this", "ctx-1", ctx)
        assert payload["files"] == [
            {
                "name": "doc.pdf",
                "mime_type": "application/pdf",
                "uri": "http://169.254.169.254/latest/meta-data",
            }
        ]

    async def test_image_categorized_separately(self, adapter):
        """A Part with image/* media_type should go to images, not files."""
        ctx = MagicMock()
        img_part = Part(url="http://example.com/photo.png", filename="photo.png", media_type="image/png")
        ctx.message.parts = [img_part]
        adapter._client = _client_for(_fail_if_called)

        payload = await adapter._build_multimodal_payload("look at this", "ctx-1", ctx)
        assert "images" in payload
        assert "files" not in payload
        assert payload["images"][0]["name"] == "photo.png"
        assert payload["images"][0]["uri"] == "http://example.com/photo.png"

    async def test_raw_part_no_fetch(self, adapter):
        """A Part with raw bytes should use data directly, not HTTP fetch."""
        ctx = MagicMock()
        raw_part = Part(raw=b"raw-content", filename="data.bin", media_type="application/octet-stream")
        ctx.message.parts = [raw_part]

        payload = await adapter._build_multimodal_payload("process", "ctx-1", ctx)
        assert "files" in payload
        assert payload["files"][0]["data"] == base64.b64encode(b"raw-content").decode("utf-8")


class TestFetchFileUrls:
    """Opt-in fetching of client-supplied file URLs (fetch_file_urls=True)."""

    @pytest.fixture
    def adapter(self):
        return N8nAdapter(
            webhook_url="http://localhost:5678/webhook/test",
            multimodal_mode=True,
            fetch_file_urls=True,
            max_file_bytes=16,
        )

    @staticmethod
    def _ctx(url, media_type="application/pdf"):
        ctx = MagicMock()
        ctx.message.parts = [Part(url=url, filename="doc.pdf", media_type=media_type)]
        return ctx

    async def test_fetched_and_base64_encoded(self, adapter):
        adapter._client = _client_for(lambda req: httpx.Response(200, content=b"pdf-bytes"))

        payload = await adapter._build_multimodal_payload(
            "check this", "ctx-1", self._ctx("http://example.com/doc.pdf")
        )
        assert payload["files"][0]["data"] == base64.b64encode(b"pdf-bytes").decode("utf-8")
        assert payload["files"][0]["uri"] == "http://example.com/doc.pdf"

    async def test_content_length_over_limit_raises(self, adapter):
        async def body():
            raise AssertionError("body read despite oversized Content-Length")
            yield b""

        adapter._client = _client_for(
            lambda req: httpx.Response(200, headers={"Content-Length": "17"}, content=body())
        )

        with pytest.raises(ValueError, match=r"\(17 > 16\)"):
            await adapter._build_multimodal_payload(
                "check this", "ctx-1", self._ctx("http://example.com/doc.pdf")
            )

    async def test_streamed_body_over_limit_raises(self, adapter):
        """A body without Content-Length is capped while streaming."""

        async def body():
            for _ in range(10):
                yield b"xxxx"

        def handler(req):
            resp = httpx.Response(200, content=body())
            assert "content-length" not in resp.headers
            return resp

        adapter._client = _client_for(handler)

        with pytest.raises(ValueError, match="exceeds"):
            await adapter._build_multimodal_payload(
                "check this", "ctx-1", self._ctx("http://example.com/doc.pdf")
            )

    async def test_http_error_raises(self, adapter):
        adapter._client = _client_for(lambda req: httpx.Response(404))

        with pytest.raises(ValueError, match="Failed to fetch"):
            await adapter._build_multimodal_payload(
                "check this", "ctx-1", self._ctx("http://example.com/doc.pdf")
            )

    async def test_redirect_not_followed(self, adapter):
        seen = []

        def handler(req):
            seen.append(str(req.url))
            if req.url.host == "example.com":
                return httpx.Response(302, headers={"Location": "http://10.0.0.1/"})
            return httpx.Response(200, content=b"internal")

        # The client follows redirects; only the per-request override stops it.
        adapter._client = _client_for(handler, follow_redirects=True)

        with pytest.raises(ValueError, match="Failed to fetch"):
            await adapter._build_multimodal_payload(
                "check this", "ctx-1", self._ctx("http://example.com/doc.pdf")
            )
        assert seen == ["http://example.com/doc.pdf"]

    async def test_slow_drip_body_hits_total_deadline(self, adapter):
        """timeout bounds the whole fetch, not just each read."""
        import asyncio

        async def body():
            for _ in range(100):
                await asyncio.sleep(0.01)
                yield b""

        adapter.timeout = 0.1
        adapter._client = _client_for(lambda req: httpx.Response(200, content=body()))

        with pytest.raises(ValueError, match="timed out"):
            await adapter._build_multimodal_payload(
                "check this", "ctx-1", self._ctx("http://example.com/doc.pdf")
            )

    async def test_invalid_url_raises_value_error(self, adapter):
        adapter._client = _client_for(_fail_if_called)

        with pytest.raises(ValueError):
            await adapter._build_multimodal_payload("x", "ctx-1", self._ctx("http://a:port/"))

    async def test_connect_error_raises(self, adapter):
        def handler(req):
            raise httpx.ConnectError("refused", request=req)

        adapter._client = _client_for(handler)

        with pytest.raises(ValueError, match="Failed to fetch"):
            await adapter._build_multimodal_payload(
                "check this", "ctx-1", self._ctx("http://example.com/doc.pdf")
            )

    @pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x", "gopher://x"])
    async def test_non_http_scheme_rejected(self, adapter, url):
        adapter._client = _client_for(_fail_if_called)

        with pytest.raises(ValueError, match="scheme"):
            await adapter._build_multimodal_payload("check this", "ctx-1", self._ctx(url))

    def test_default_max_file_bytes_is_10_mib(self):
        adapter = N8nAdapter(webhook_url="http://localhost:5678/webhook/test")
        assert adapter.fetch_file_urls is False
        assert adapter.max_file_bytes == 10 * 1024 * 1024


class TestFetchFileContent:
    """Test _fetch_file_content with V1.0 proto Part fields."""

    @pytest.fixture
    def adapter(self):
        return N8nAdapter(
            webhook_url="http://localhost:5678/webhook/test", fetch_file_urls=True
        )

    async def test_url_part_fetches(self, adapter):
        part = Part(url="http://example.com/file.bin")
        adapter._client = _client_for(lambda req: httpx.Response(200, content=b"fetched"))

        result = await adapter._fetch_file_content(part)
        assert result == b"fetched"

    async def test_raw_part_returns_directly(self, adapter):
        part = Part(raw=b"inline-data")
        result = await adapter._fetch_file_content(part)
        assert result == b"inline-data"

    async def test_empty_part_raises(self, adapter):
        part = Part(text="just text")
        with pytest.raises(ValueError, match="no url or raw"):
            await adapter._fetch_file_content(part)


class TestExtractResponse:
    """Test _extract_response with V1.0 Part construction."""

    @pytest.fixture
    def adapter(self):
        return N8nAdapter(
            webhook_url="http://localhost:5678/webhook/test",
            multimodal_mode=True,
        )

    def test_text_only_response(self, adapter):
        output = {"output": "hello world"}
        parts = adapter._extract_response(output)
        assert len(parts) == 1
        assert parts[0].text == "hello world"

    def test_with_files(self, adapter):
        output = {
            "output": "here are results",
            "files": [{"url": "http://example.com/report.pdf", "name": "report.pdf", "mime_type": "application/pdf"}],
        }
        parts = adapter._extract_response(output)
        assert len(parts) == 2
        assert parts[0].text == "here are results"
        assert parts[1].url == "http://example.com/report.pdf"
        assert parts[1].filename == "report.pdf"
        assert parts[1].media_type == "application/pdf"

    def test_with_images(self, adapter):
        output = {
            "output": "chart generated",
            "images": [{"url": "http://example.com/chart.png", "name": "chart.png", "mimeType": "image/png"}],
        }
        parts = adapter._extract_response(output)
        assert len(parts) == 2
        assert parts[1].url == "http://example.com/chart.png"
        assert parts[1].media_type == "image/png"

    def test_empty_response(self, adapter):
        parts = adapter._extract_response({})
        assert len(parts) == 1
        assert parts[0].text == "[Empty response]"

    def test_non_dict_response(self, adapter):
        parts = adapter._extract_response([{"output": "from list"}])
        assert len(parts) == 1
        assert "from list" in parts[0].text


class TestExtractResponseMalformedRefs:
    """File/image refs without a usable location must not become empty Parts."""

    @pytest.fixture
    def adapter(self):
        return N8nAdapter(
            webhook_url="http://localhost:5678/webhook/test",
            multimodal_mode=True,
        )

    @staticmethod
    def _assert_no_empty_parts(parts):
        assert all(p.WhichOneof("content") is not None for p in parts)

    def test_file_ref_without_location_skipped(self, adapter):
        output = {"output": "done", "files": [{"name": "report.pdf"}], "images": [{}]}
        parts = adapter._extract_response(output)
        self._assert_no_empty_parts(parts)
        assert [p.text for p in parts] == ["done"]

    def test_file_ref_with_inline_data_becomes_raw_part(self, adapter):
        output = {
            "files": [
                {
                    "name": "a.txt",
                    "mime_type": "text/plain",
                    "data": base64.b64encode(b"inline").decode(),
                }
            ]
        }
        parts = adapter._extract_response(output)
        assert len(parts) == 1
        assert parts[0].raw == b"inline"
        assert parts[0].filename == "a.txt"
        assert parts[0].media_type == "text/plain"

    def test_file_ref_with_invalid_base64_skipped(self, adapter):
        output = {"output": "done", "files": [{"name": "a.txt", "data": "not base64!"}]}
        parts = adapter._extract_response(output)
        self._assert_no_empty_parts(parts)
        assert [p.text for p in parts] == ["done"]

    @pytest.mark.parametrize("value", [None, "files.pdf", {"url": "http://x/a"}, 3])
    def test_non_list_file_field_treated_as_empty(self, adapter, value):
        parts = adapter._extract_response({"output": "done", "files": value, "images": value})
        assert [p.text for p in parts] == ["done"]

    def test_only_malformed_refs_gives_placeholder(self, adapter):
        parts = adapter._extract_response({"files": [{"name": "x"}], "images": None})
        assert [p.text for p in parts] == ["[Empty response]"]

    def test_non_string_name_and_mime_ignored(self, adapter):
        output = {"files": [{"url": "http://x/a.pdf", "name": 123, "mime_type": {}}]}
        parts = adapter._extract_response(output)
        assert len(parts) == 1
        assert parts[0].url == "http://x/a.pdf"
        assert parts[0].filename == ""

    def test_non_string_url_falls_back_to_uri(self, adapter):
        output = {"files": [{"url": {"bad": 1}, "uri": "http://x/a.pdf"}]}
        parts = adapter._extract_response(output)
        assert [p.url for p in parts] == ["http://x/a.pdf"]
