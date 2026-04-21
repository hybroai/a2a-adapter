"""Shared fixtures for a2a-adapter tests."""

import asyncio
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.types import Part

from a2a_adapter.base_adapter import AdapterMetadata, BaseA2AAdapter


class StubAdapter(BaseA2AAdapter):
    """Minimal adapter returning a fixed response."""

    def __init__(self, response: str = "hello"):
        self._response = response

    async def invoke(
        self, user_input: str, context_id: str | None = None, **kwargs
    ) -> str:
        return self._response


class StreamingStubAdapter(BaseA2AAdapter):
    """Adapter that streams predetermined chunks."""

    def __init__(self, chunks: list[str | Part]):
        self._chunks = chunks

    async def invoke(
        self, user_input: str, context_id: str | None = None, **kwargs
    ) -> str:
        return "".join(
            c if isinstance(c, str) else getattr(c.root, "text", "")
            for c in self._chunks
        )

    async def stream(
        self, user_input: str, context_id: str | None = None, **kwargs
    ) -> AsyncIterator[str | Part]:
        for chunk in self._chunks:
            yield chunk


class FailingAdapter(BaseA2AAdapter):
    """Adapter that raises on invoke."""

    async def invoke(
        self, user_input: str, context_id: str | None = None, **kwargs
    ) -> str:
        raise RuntimeError("boom")


class FailingStreamAdapter(BaseA2AAdapter):
    """Adapter that yields some chunks then raises mid-stream."""

    def __init__(self, ok_chunks: list[str], error: Exception | None = None):
        self._ok_chunks = ok_chunks
        self._error = error or RuntimeError("stream exploded")

    async def invoke(
        self, user_input: str, context_id: str | None = None, **kwargs
    ) -> str:
        raise RuntimeError("invoke not supported")

    async def stream(
        self, user_input: str, context_id: str | None = None, **kwargs
    ) -> AsyncIterator[str]:
        for chunk in self._ok_chunks:
            yield chunk
        raise self._error


@pytest.fixture
def stub_adapter():
    return StubAdapter()


@pytest.fixture
def streaming_stub_adapter():
    return StreamingStubAdapter(["Hello", ", ", "world!"])


@pytest.fixture
def failing_adapter():
    return FailingAdapter()


@pytest.fixture
def event_queue():
    return EventQueue()


@pytest.fixture
def request_context():
    ctx = MagicMock(spec=RequestContext)
    ctx.task_id = "test-task-001"
    ctx.context_id = "test-ctx-001"
    ctx.get_user_input.return_value = "hi"
    return ctx
