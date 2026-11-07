"""Shared fixtures for a2a-adapter tests."""

from typing import AsyncIterator
from unittest.mock import MagicMock

import pytest
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.types import Part

from a2a_adapter.base_adapter import BaseA2AAdapter


class StubAdapter(BaseA2AAdapter):
    """Minimal adapter returning a fixed response."""

    def __init__(self, response: str = "hello"):
        self._response = response

    async def invoke(self, user_input: str, context_id: str | None = None, **kwargs) -> str:
        return self._response


class StreamingStubAdapter(BaseA2AAdapter):
    """Adapter that streams predetermined chunks."""

    def __init__(self, chunks: list[str | Part]):
        self._chunks = chunks

    async def invoke(self, user_input: str, context_id: str | None = None, **kwargs) -> str:
        return "".join(c if isinstance(c, str) else (c.text or "") for c in self._chunks)

    async def stream(
        self, user_input: str, context_id: str | None = None, **kwargs
    ) -> AsyncIterator[str | Part]:
        for chunk in self._chunks:
            yield chunk


class FailingAdapter(BaseA2AAdapter):
    """Adapter that raises on invoke."""

    async def invoke(self, user_input: str, context_id: str | None = None, **kwargs) -> str:
        raise RuntimeError("boom")


class FailingStreamAdapter(BaseA2AAdapter):
    """Adapter that yields some chunks then raises mid-stream."""

    def __init__(self, ok_chunks: list[str], error: Exception | None = None):
        self._ok_chunks = ok_chunks
        self._error = error or RuntimeError("stream exploded")

    async def invoke(self, user_input: str, context_id: str | None = None, **kwargs) -> str:
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


@pytest.fixture(autouse=True)
def _isolated_state_home(tmp_path_factory, monkeypatch):
    """Keep CLI adapter session stores out of the real ~/.local/state."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path_factory.mktemp("state")))


@pytest.fixture(autouse=True)
def _killpg_only_own_children(monkeypatch):
    """Fail any test that would signal a process group it did not spawn.

    Many unit tests hand the adapters mock processes with made-up PIDs; a
    stray ``os.killpg`` on one of those would signal a real process group
    (for example the user's shell). This fixture records, on its own, the
    PID of every real child spawned during the test (asyncio's Unix
    subprocess transport goes through ``subprocess.Popen``) and lets
    ``os.killpg`` reach only those. It also empties the adapter's
    ownership registry at teardown so its ``atexit`` hook has nothing left
    to signal.
    """
    import os
    import subprocess

    from a2a_adapter import base_adapter

    spawned: set[int] = set()
    violations: list[int] = []
    original_init = subprocess.Popen.__init__
    original_killpg = getattr(os, "killpg", None)

    def recording_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        pid = getattr(self, "pid", None)
        if pid is not None:
            spawned.add(pid)

    def guarded_killpg(pgid, sig):
        if pgid not in spawned:
            violations.append(pgid)
            raise AssertionError(f"os.killpg({pgid}) targets a group this test did not spawn")
        return original_killpg(pgid, sig)

    monkeypatch.setattr(subprocess.Popen, "__init__", recording_init)
    if original_killpg is not None:
        monkeypatch.setattr(base_adapter.os, "killpg", guarded_killpg)
    try:
        yield
    finally:
        own_groups = getattr(base_adapter, "_OWN_GROUPS", None)
        if own_groups is not None:
            own_groups.clear()
    assert not violations, f"os.killpg targeted foreign process groups: {violations}"
