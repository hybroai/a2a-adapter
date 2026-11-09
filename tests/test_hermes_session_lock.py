"""Tests for HermesAdapter per-session lock serialization."""

import asyncio
import gc
import weakref

import pytest

from a2a_adapter.integrations.hermes import HermesAdapter


@pytest.fixture
def adapter():
    """HermesAdapter with stubbed internals — no real Hermes dependency."""
    a = HermesAdapter.__new__(HermesAdapter)
    a._session_db = None
    a._running_agents = {}
    a._session_locks = weakref.WeakValueDictionary()
    a._executor = None
    return a


class TestSessionLock:
    """Verify _session_lock() returns one lock per session_id."""

    def test_same_session_returns_same_lock(self, adapter):
        lock_a = adapter._session_lock("ctx-1")
        lock_b = adapter._session_lock("ctx-1")
        assert lock_a is lock_b

    def test_different_sessions_return_different_locks(self, adapter):
        lock_a = adapter._session_lock("ctx-1")
        lock_b = adapter._session_lock("ctx-2")
        assert lock_a is not lock_b

    def test_lock_is_asyncio_lock(self, adapter):
        assert isinstance(adapter._session_lock("ctx-1"), asyncio.Lock)

    def test_real_adapter_forgets_an_unheld_lock(self):
        real = HermesAdapter()
        lock = real._session_lock("ctx-1")
        assert list(real._session_locks) == ["ctx-1"]
        del lock
        gc.collect()
        assert len(real._session_locks) == 0


class TestSessionLockSerialization:
    """Verify that concurrent calls for the same session are serialized,
    while calls for different sessions run in parallel."""

    @pytest.mark.asyncio
    async def test_same_session_serialized(self, adapter):
        """Two tasks sharing a session_id must not overlap."""
        timeline: list[tuple[str, str]] = []
        barrier = asyncio.Event()

        async def _work(name: str, session_id: str):
            async with adapter._session_lock(session_id):
                timeline.append((name, "start"))
                if name == "first":
                    barrier.set()
                    await asyncio.sleep(0.05)
                else:
                    await asyncio.sleep(0)
                timeline.append((name, "end"))

        task_first = asyncio.create_task(_work("first", "ctx-same"))
        await barrier.wait()
        task_second = asyncio.create_task(_work("second", "ctx-same"))

        await asyncio.gather(task_first, task_second)

        assert timeline == [
            ("first", "start"),
            ("first", "end"),
            ("second", "start"),
            ("second", "end"),
        ]

    @pytest.mark.asyncio
    async def test_different_sessions_parallel(self, adapter):
        """Two tasks with different session_ids should overlap."""
        timeline: list[tuple[str, str]] = []
        both_started = asyncio.Event()
        started_count = 0

        async def _work(name: str, session_id: str):
            nonlocal started_count
            async with adapter._session_lock(session_id):
                timeline.append((name, "start"))
                started_count += 1
                if started_count >= 2:
                    both_started.set()
                await both_started.wait()
                timeline.append((name, "end"))

        await asyncio.gather(
            _work("a", "ctx-1"),
            _work("b", "ctx-2"),
        )

        starts = [e for e in timeline if e[1] == "start"]
        ends = [e for e in timeline if e[1] == "end"]
        assert len(starts) == 2
        assert len(ends) == 2
        assert timeline.index(("a", "start")) < timeline.index(("b", "end"))
        assert timeline.index(("b", "start")) < timeline.index(("a", "end"))
