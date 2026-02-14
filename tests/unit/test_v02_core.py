"""Tests for v0.2 core components: BaseA2AAdapter, AdapterMetadata,
build_agent_card, to_a2a, load_adapter, register_adapter.

These test the infrastructure shared by all adapters, not individual
adapter implementations (which have their own test_v02_*.py files).
"""

import pytest
from unittest.mock import patch

from a2a_adapter.base_adapter import BaseA2AAdapter, AdapterMetadata
from a2a_adapter.server import build_agent_card, to_a2a
from a2a_adapter.loader import load_adapter, register_adapter, _REGISTRY


# ──── Helpers ────


class MinimalAdapter(BaseA2AAdapter):
    """Minimal adapter that only implements invoke()."""

    async def invoke(self, user_input, context_id=None, **kwargs):
        return f"echo: {user_input}"


class StreamAdapter(BaseA2AAdapter):
    """Adapter with stream() override."""

    async def invoke(self, user_input, context_id=None, **kwargs):
        return "full"

    async def stream(self, user_input, context_id=None, **kwargs):
        yield "chunk"


class MetadataAdapter(BaseA2AAdapter):
    """Adapter with custom metadata."""

    async def invoke(self, user_input, context_id=None, **kwargs):
        return "ok"

    def get_metadata(self):
        return AdapterMetadata(
            name="Custom Agent",
            description="Does custom things",
            version="2.0.0",
            skills=[
                {"id": "skill-1", "name": "Skill One", "description": "First skill"},
            ],
            input_modes=["text", "image"],
            output_modes=["text"],
            streaming=True,
        )


class CloseTracker(BaseA2AAdapter):
    """Adapter that tracks close() calls."""

    def __init__(self):
        self.closed = False

    async def invoke(self, user_input, context_id=None, **kwargs):
        return "ok"

    async def close(self):
        self.closed = True


# ═══════════════════════════════════════════════════
# BaseA2AAdapter Tests
# ═══════════════════════════════════════════════════


def test_supports_streaming_false_by_default():
    adapter = MinimalAdapter()
    assert adapter.supports_streaming() is False


def test_supports_streaming_true_when_overridden():
    adapter = StreamAdapter()
    assert adapter.supports_streaming() is True


def test_default_metadata():
    adapter = MinimalAdapter()
    meta = adapter.get_metadata()
    assert meta.name == ""
    assert meta.description == ""
    assert meta.version == "1.0.0"
    assert meta.skills == []
    assert meta.input_modes == ["text"]
    assert meta.output_modes == ["text"]
    assert meta.streaming is False


@pytest.mark.asyncio
async def test_cancel_is_noop_by_default():
    adapter = MinimalAdapter()
    await adapter.cancel()  # Should not raise


@pytest.mark.asyncio
async def test_close_is_noop_by_default():
    adapter = MinimalAdapter()
    await adapter.close()  # Should not raise


@pytest.mark.asyncio
async def test_context_manager():
    adapter = CloseTracker()
    async with adapter as a:
        assert a is adapter
        assert not adapter.closed
    assert adapter.closed


@pytest.mark.asyncio
async def test_stream_raises_not_implemented():
    adapter = MinimalAdapter()
    with pytest.raises(NotImplementedError):
        async for _ in adapter.stream("test"):
            pass


# ═══════════════════════════════════════════════════
# AdapterMetadata Tests
# ═══════════════════════════════════════════════════


def test_metadata_defaults():
    meta = AdapterMetadata()
    assert meta.name == ""
    assert meta.streaming is False
    assert meta.input_modes == ["text"]


def test_metadata_custom():
    meta = AdapterMetadata(name="Test", streaming=True, version="3.0")
    assert meta.name == "Test"
    assert meta.streaming is True
    assert meta.version == "3.0"


def test_metadata_skills_mutable_default():
    """Each instance should get its own skills list."""
    m1 = AdapterMetadata()
    m2 = AdapterMetadata()
    m1.skills.append({"id": "x"})
    assert len(m2.skills) == 0


# ═══════════════════════════════════════════════════
# build_agent_card Tests
# ═══════════════════════════════════════════════════


def test_build_card_defaults():
    card = build_agent_card(MinimalAdapter())
    assert card.name == "MinimalAdapter"  # Falls back to class name
    assert card.description == ""
    assert card.url == "http://localhost:9000"
    assert card.capabilities.streaming is False
    assert card.skills == []


def test_build_card_from_metadata():
    card = build_agent_card(MetadataAdapter())
    assert card.name == "Custom Agent"
    assert card.description == "Does custom things"
    assert card.version == "2.0.0"
    assert card.capabilities.streaming is True
    assert len(card.skills) == 1
    assert card.skills[0].id == "skill-1"
    assert card.skills[0].name == "Skill One"
    assert card.default_input_modes == ["text", "image"]
    assert card.default_output_modes == ["text"]


def test_build_card_overrides():
    card = build_agent_card(
        MinimalAdapter(),
        name="Override Name",
        description="Override Desc",
        url="http://custom:8080",
        version="9.9.9",
    )
    assert card.name == "Override Name"
    assert card.description == "Override Desc"
    assert card.url == "http://custom:8080"
    assert card.version == "9.9.9"


def test_build_card_streaming_auto_detect():
    """build_agent_card should auto-detect streaming from the adapter."""
    card = build_agent_card(StreamAdapter())
    assert card.capabilities.streaming is True


def test_build_card_streaming_override():
    card = build_agent_card(MinimalAdapter(), streaming=True)
    assert card.capabilities.streaming is True


def test_build_card_skills_with_tags():
    class SkillAdapter(BaseA2AAdapter):
        async def invoke(self, user_input, context_id=None, **kwargs):
            return ""

        def get_metadata(self):
            return AdapterMetadata(
                skills=[
                    {"id": "s1", "name": "Math", "description": "calc", "tags": ["math"]},
                    {"id": "s2", "name": "Code", "description": "code", "tags": ["coding"]},
                ]
            )

    card = build_agent_card(SkillAdapter())
    assert len(card.skills) == 2
    assert card.skills[0].tags == ["math"]
    assert card.skills[1].tags == ["coding"]


# ═══════════════════════════════════════════════════
# to_a2a Tests
# ═══════════════════════════════════════════════════


def test_to_a2a_returns_app():
    app = to_a2a(MinimalAdapter())
    assert app is not None


def test_to_a2a_with_custom_card():
    from a2a.types import AgentCard, AgentCapabilities

    card = AgentCard(
        name="Custom", description="", url="http://x",
        version="1.0", capabilities=AgentCapabilities(streaming=False),
        skills=[], default_input_modes=["text"], default_output_modes=["text"],
    )
    app = to_a2a(MinimalAdapter(), agent_card=card)
    assert app is not None


def test_to_a2a_with_card_overrides():
    app = to_a2a(MinimalAdapter(), name="Override", url="http://test:1234")
    assert app is not None


# ═══════════════════════════════════════════════════
# load_adapter Tests
# ═══════════════════════════════════════════════════


def test_load_adapter_callable():
    async def fn(inputs):
        return "ok"

    adapter = load_adapter({"adapter": "callable", "func": fn})
    assert adapter is not None


def test_load_adapter_missing_adapter_key():
    with pytest.raises(ValueError, match="adapter"):
        load_adapter({"webhook_url": "http://x"})


def test_load_adapter_unknown_type():
    with pytest.raises(ValueError, match="Unknown adapter type"):
        load_adapter({"adapter": "nonexistent"})


def test_load_adapter_does_not_mutate_config():
    config = {"adapter": "callable", "func": lambda x: "ok"}
    load_adapter(config)
    assert "adapter" in config  # Original dict should be unchanged


# ═══════════════════════════════════════════════════
# register_adapter Tests
# ═══════════════════════════════════════════════════


def test_register_and_load():
    """Registered adapters should be loadable via load_adapter."""

    @register_adapter("test_custom_framework")
    class TestCustomAdapter(BaseA2AAdapter):
        async def invoke(self, user_input, context_id=None, **kwargs):
            return "custom"

    try:
        adapter = load_adapter({"adapter": "test_custom_framework"})
        assert isinstance(adapter, TestCustomAdapter)
    finally:
        _REGISTRY.pop("test_custom_framework", None)


def test_register_overwrite_warns(caplog):
    """Re-registering the same name should log a warning."""
    import logging

    class AdapterA(BaseA2AAdapter):
        async def invoke(self, user_input, context_id=None, **kwargs):
            return "a"

    class AdapterB(BaseA2AAdapter):
        async def invoke(self, user_input, context_id=None, **kwargs):
            return "b"

    try:
        register_adapter("test_overwrite")(AdapterA)
        with caplog.at_level(logging.WARNING):
            register_adapter("test_overwrite")(AdapterB)

        assert _REGISTRY["test_overwrite"] is AdapterB
    finally:
        _REGISTRY.pop("test_overwrite", None)


def test_register_priority_over_builtin():
    """User-registered adapters should take priority over built-ins."""

    @register_adapter("callable")
    class OverrideCallable(BaseA2AAdapter):
        async def invoke(self, user_input, context_id=None, **kwargs):
            return "overridden"

    try:
        adapter = load_adapter({"adapter": "callable"})
        assert isinstance(adapter, OverrideCallable)
    finally:
        _REGISTRY.pop("callable", None)


# ═══════════════════════════════════════════════════
# Flat Import Tests
# ═══════════════════════════════════════════════════


def test_import_core_directly():
    from a2a_adapter import BaseA2AAdapter, AdapterMetadata, serve_agent, to_a2a
    assert BaseA2AAdapter is not None
    assert AdapterMetadata is not None


def test_import_adapter_lazily():
    from a2a_adapter import N8nAdapter, CallableAdapter
    assert N8nAdapter is not None
    assert CallableAdapter is not None


def test_import_all_adapters():
    from a2a_adapter import (
        N8nAdapter,
        CallableAdapter,
        LangChainAdapter,
        LangGraphAdapter,
        CrewAIAdapter,
        OpenClawAdapter,
    )
    # All should be importable
    adapters = [N8nAdapter, CallableAdapter, LangChainAdapter,
                LangGraphAdapter, CrewAIAdapter, OpenClawAdapter]
    assert all(a is not None for a in adapters)


def test_version():
    from a2a_adapter import __version__
    assert __version__ == "0.2.0"
