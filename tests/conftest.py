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
    ``os.killpg`` reach only those and their live descendants (the opt-in
    sweep of detached descendants), found with its own ``ps`` parsing rather
    than the adapter's. A SIGKILL or SIGSTOP sent with ``os.kill`` is held
    to the same rule when the adapter's code sends it. It also empties the adapter's ownership registries at
    teardown so its ``atexit`` hook has nothing left to signal.
    """
    import os
    import signal
    import subprocess
    import sys

    from a2a_adapter import base_adapter

    spawned: set[int] = set()
    violations: list[str] = []
    original_init = subprocess.Popen.__init__
    original_killpg = getattr(os, "killpg", None)
    original_kill = os.kill
    original_run = subprocess.run
    guarded_signals = {
        getattr(signal, name) for name in ("SIGKILL", "SIGSTOP") if hasattr(signal, name)
    }

    def recording_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        pid = getattr(self, "pid", None)
        if pid is not None:
            spawned.add(pid)

    def descendants_of_spawned() -> set[int]:
        # The real subprocess.run, so tests that fake it don't fake this too.
        listing = original_run(
            ["ps", "-A", "-o", "pid=,ppid="], capture_output=True, text=True, check=False
        ).stdout
        parents = {}
        for line in listing.splitlines():
            pid, _, ppid = line.strip().partition(" ")
            parents[int(pid)] = int(ppid.strip())
        found: set[int] = set()
        for pid in parents:
            seen, current = set(), pid
            while current in parents and current not in seen:
                seen.add(current)
                current = parents[current]
                if current in spawned:
                    found.add(pid)
                    break
        return found

    # Every descendant seen so far: one orphaned when its parent was killed
    # first (its PID cannot be reused while it lives) is still fair game.
    known_descendants: set[int] = set()

    def allowed(pid: int) -> bool:
        if pid in spawned or pid in known_descendants:
            return True
        known_descendants.update(descendants_of_spawned())
        return pid in known_descendants

    def guarded_killpg(pgid, sig):
        if not allowed(pgid):
            violations.append(f"killpg({pgid})")
            raise AssertionError(f"os.killpg({pgid}) targets a group this test did not spawn")
        return original_killpg(pgid, sig)

    def guarded_kill(pid, sig):
        # Only the adapter's own calls: tests clean up orphans they never spawned.
        caller = sys._getframe(1).f_globals.get("__name__", "")
        from_adapter = caller.startswith("a2a_adapter")
        if from_adapter and sig in guarded_signals and pid != os.getpid() and not allowed(pid):
            violations.append(f"kill({pid})")
            raise AssertionError(f"os.kill({pid}, {sig}) targets a process this test did not spawn")
        return original_kill(pid, sig)

    monkeypatch.setattr(subprocess.Popen, "__init__", recording_init)
    if original_killpg is not None:
        monkeypatch.setattr(base_adapter.os, "killpg", guarded_killpg)
        monkeypatch.setattr(base_adapter.os, "kill", guarded_kill)
    try:
        yield
    finally:
        for registry in ("_OWN_GROUPS", "_SWEEP_DESCENDANTS"):
            groups = getattr(base_adapter, registry, None)
            if groups is not None:
                groups.clear()
    assert not violations, f"signals reached processes the test did not spawn: {violations}"
