"""Tests for server.py — build_agent_card() and to_a2a()."""

import httpx
import pytest
from a2a.types import AgentCard

from a2a_adapter.base_adapter import AdapterMetadata, BaseA2AAdapter
from a2a_adapter.server import build_agent_card, to_a2a

from .conftest import StubAdapter, StreamingStubAdapter


# ── build_agent_card ─────────────────────────────────────────


class TestBuildAgentCard:
    def test_default_card(self):
        adapter = StubAdapter()
        card = build_agent_card(adapter)
        assert isinstance(card, AgentCard)
        assert card.name == "StubAdapter"
        assert len(card.supported_interfaces) == 1
        assert card.supported_interfaces[0].url == "http://localhost:9000/"
        assert card.capabilities.streaming is False

    def test_streaming_auto_detected(self):
        adapter = StreamingStubAdapter(["a", "b"])
        card = build_agent_card(adapter)
        assert card.capabilities.streaming is True

    def test_overrides(self):
        adapter = StubAdapter()
        card = build_agent_card(
            adapter,
            name="Custom",
            description="desc",
            url="http://myhost:8080",
            version="2.0.0",
        )
        assert card.name == "Custom"
        assert card.description == "desc"
        assert len(card.supported_interfaces) == 1
        assert card.supported_interfaces[0].url == "http://myhost:8080"
        assert card.version == "2.0.0"

    def test_metadata_skills(self):
        class SkillAdapter(BaseA2AAdapter):
            async def invoke(self, user_input, context_id=None, **kwargs):
                return "ok"

            def get_metadata(self):
                return AdapterMetadata(
                    name="SkillAgent",
                    skills=[
                        {"id": "s1", "name": "Search", "description": "Web search"},
                    ],
                )

        card = build_agent_card(SkillAdapter())
        assert len(card.skills) == 1
        assert card.skills[0].name == "Search"

    def test_provider_from_dict(self):
        class ProviderAdapter(BaseA2AAdapter):
            async def invoke(self, user_input, context_id=None, **kwargs):
                return "ok"

            def get_metadata(self):
                return AdapterMetadata(
                    provider={"organization": "Acme", "url": "https://acme.com"}
                )

        card = build_agent_card(ProviderAdapter())
        assert card.provider is not None
        assert card.provider.organization == "Acme"

    def test_documentation_and_icon_urls(self):
        class UrlAdapter(BaseA2AAdapter):
            async def invoke(self, user_input, context_id=None, **kwargs):
                return "ok"

            def get_metadata(self):
                return AdapterMetadata(
                    documentation_url="https://docs.example.com",
                    icon_url="https://icon.example.com/logo.png",
                )

        card = build_agent_card(UrlAdapter())
        assert card.documentation_url == "https://docs.example.com"
        assert card.icon_url == "https://icon.example.com/logo.png"


# ── to_a2a ───────────────────────────────────────────────────


class TestToA2a:
    def test_returns_asgi_app(self):
        adapter = StubAdapter()
        app = to_a2a(adapter)
        assert callable(app)

    def test_with_custom_card(self):
        adapter = StubAdapter()
        card = build_agent_card(adapter, name="Custom")
        app = to_a2a(adapter, agent_card=card)
        assert callable(app)

    def test_with_card_overrides(self):
        adapter = StubAdapter()
        app = to_a2a(adapter, name="Override")
        assert callable(app)

    @pytest.mark.asyncio
    async def test_legacy_agent_card_includes_empty_skills(self):
        adapter = StubAdapter()
        app = to_a2a(adapter)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            resp = await client.get("/.well-known/agent-card.json")

        assert resp.status_code == 200
        data = resp.json()
        assert data["skills"] == []
        assert data["url"] == data["supportedInterfaces"][0]["url"]
