"""Tests for BaseA2AAdapter contract and AdapterMetadata."""

import asyncio
import contextlib
import logging
import os
from types import SimpleNamespace

import pytest
from a2a.types import Part

from a2a_adapter.base_adapter import (
    AdapterMetadata,
    BaseA2AAdapter,
    CommandResult,
    ParseResult,
    default_state_dir,
)


# ── AdapterMetadata defaults ─────────────────────────────────


class TestAdapterMetadata:
    def test_defaults(self):
        m = AdapterMetadata()
        assert m.name == ""
        assert m.description == ""
        assert m.version == "1.0.0"
        assert m.skills == []
        assert m.input_modes == ["text"]
        assert m.output_modes == ["text"]
        assert m.streaming is False
        assert m.provider is None
        assert m.documentation_url is None
        assert m.icon_url is None

    def test_custom_values(self):
        m = AdapterMetadata(
            name="Test",
            description="desc",
            version="2.0.0",
            streaming=True,
            provider={"organization": "Acme"},
        )
        assert m.name == "Test"
        assert m.streaming is True
        assert m.provider == {"organization": "Acme"}


# ── BaseA2AAdapter contract ──────────────────────────────────


class ConcreteAdapter(BaseA2AAdapter):
    async def invoke(self, user_input, context_id=None, **kwargs):
        return f"echo: {user_input}"


class StreamingConcreteAdapter(BaseA2AAdapter):
    async def invoke(self, user_input, context_id=None, **kwargs):
        return "full"

    async def stream(self, user_input, context_id=None, **kwargs):
        for word in user_input.split():
            yield word


class TestBaseA2AAdapterContract:
    async def test_invoke_returns_string(self):
        adapter = ConcreteAdapter()
        result = await adapter.invoke("hello")
        assert result == "echo: hello"

    async def test_stream_raises_not_implemented(self):
        adapter = ConcreteAdapter()
        with pytest.raises(NotImplementedError):
            async for _ in adapter.stream("hi"):
                pass

    def test_supports_streaming_false_by_default(self):
        adapter = ConcreteAdapter()
        assert adapter.supports_streaming() is False

    def test_supports_streaming_true_when_overridden(self):
        adapter = StreamingConcreteAdapter()
        assert adapter.supports_streaming() is True

    async def test_cancel_is_noop_by_default(self):
        adapter = ConcreteAdapter()
        await adapter.cancel()

    async def test_close_is_noop_by_default(self):
        adapter = ConcreteAdapter()
        await adapter.close()

    def test_get_metadata_returns_empty(self):
        adapter = ConcreteAdapter()
        m = adapter.get_metadata()
        assert isinstance(m, AdapterMetadata)
        assert m.name == ""

    async def test_async_context_manager(self):
        async with ConcreteAdapter() as adapter:
            result = await adapter.invoke("test")
            assert result == "echo: test"

    async def test_streaming_adapter_yields_chunks(self):
        adapter = StreamingConcreteAdapter()
        chunks = []
        async for chunk in adapter.stream("hello world"):
            chunks.append(chunk)
        assert chunks == ["hello", "world"]


# ── Multimodal invoke ────────────────────────────────────────


class MultimodalAdapter(BaseA2AAdapter):
    async def invoke(self, user_input, context_id=None, **kwargs):
        return [Part(text="multi")]


class TestMultimodalInvoke:
    async def test_invoke_returns_parts_list(self):
        adapter = MultimodalAdapter()
        result = await adapter.invoke("hi")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].text == "multi"


# ── Subprocess runtime: cancellation ─────────────────────────


class SleepingCliAdapter(BaseA2AAdapter):
    """CLI adapter whose command runs long enough to be cancelled mid-run."""

    def __init__(self, tmp_path):
        super().__init__(
            working_dir=str(tmp_path),
            session_store_path=str(tmp_path / "sessions.json"),
        )

    async def invoke(self, user_input, context_id=None, **kwargs):
        return await self._run_subprocess(
            context_id or "ctx", kwargs.get("task_id", "task"), user_input=user_input
        )

    def _build_command(self, message: str, context_key: str) -> CommandResult:
        return CommandResult(args=["sleep", "30"], used_resume=False)

    def _parse_invoke_output(self, stdout_text: str, context_key: str) -> ParseResult:
        return ParseResult(text=stdout_text)


class TestRunSubprocessCancellation:
    async def test_cancelled_task_kills_subprocess(self, tmp_path):
        """Cancelling the asyncio task running _run_subprocess must kill the
        child process instead of leaving it orphaned."""
        adapter = SleepingCliAdapter(tmp_path)
        task = asyncio.create_task(adapter.invoke("hi", "ctx", task_id="t1"))

        for _ in range(200):
            if "t1" in adapter._active_processes:
                break
            await asyncio.sleep(0.01)
        proc = adapter._active_processes["t1"]
        assert proc.returncode is None

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert proc.returncode is not None
        assert "t1" not in adapter._active_processes

    async def test_cancel_then_task_cancel_clears_killed_marker(self, tmp_path):
        """tasks/cancel calls adapter.cancel() and then cancels the task; the
        killed-task marker must not outlive the cancelled run."""
        adapter = SleepingCliAdapter(tmp_path)
        task = asyncio.create_task(adapter.invoke("hi", "ctx", task_id="t2"))

        for _ in range(200):
            if "t2" in adapter._active_processes:
                break
            await asyncio.sleep(0.01)
        proc = adapter._active_processes["t2"]

        await adapter.cancel(context=SimpleNamespace(task_id="t2"))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert proc.returncode is not None
        assert "t2" not in adapter._killed_tasks


# ── Subprocess runtime: session store ────────────────────────


class TestSessionStore:
    def test_save_then_load_round_trips(self, tmp_path):
        a1 = SleepingCliAdapter(tmp_path)
        a1._sessions = {"ctx-1": "sess-A", "ctx-2": "sess-B"}
        a1._save_sessions()

        a2 = SleepingCliAdapter(tmp_path)
        assert a2._sessions == {"ctx-1": "sess-A", "ctx-2": "sess-B"}

    def test_malformed_store_loads_empty_with_warning(self, tmp_path, caplog):
        (tmp_path / "sessions.json").write_text('{"ctx-1": "sess-')
        with caplog.at_level(logging.WARNING, logger="a2a_adapter.base_adapter"):
            adapter = SleepingCliAdapter(tmp_path)
        assert adapter._sessions == {}
        assert any(
            r.levelno == logging.WARNING and "sessions.json" in r.getMessage()
            for r in caplog.records
        )

    def test_non_dict_store_loads_empty_with_warning(self, tmp_path, caplog):
        (tmp_path / "sessions.json").write_text("[1, 2, 3]")
        with caplog.at_level(logging.WARNING, logger="a2a_adapter.base_adapter"):
            adapter = SleepingCliAdapter(tmp_path)
        assert adapter._sessions == {}
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_missing_store_does_not_warn(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger="a2a_adapter.base_adapter"):
            adapter = SleepingCliAdapter(tmp_path)
        assert adapter._sessions == {}
        assert not caplog.records

    def test_failed_save_leaves_previous_store_intact(self, tmp_path, monkeypatch):
        """A crash while writing must not truncate or corrupt the existing
        store: the write goes to a temp file that replaces it atomically."""
        store = tmp_path / "sessions.json"
        adapter = SleepingCliAdapter(tmp_path)
        adapter._sessions = {"ctx-1": "sess-A"}
        adapter._save_sessions()
        before = store.read_text()

        def partial_dump(obj, fp, **kwargs):
            fp.write('{"ctx-1": "se')
            raise OSError("disk full")

        monkeypatch.setattr("a2a_adapter.base_adapter.json.dump", partial_dump)
        adapter._sessions = {"ctx-1": "sess-B"}
        adapter._save_sessions()

        assert store.read_text() == before
        assert sorted(p.name for p in tmp_path.iterdir()) == ["sessions.json"]

    def test_save_fsyncs_before_replacing(self, tmp_path, monkeypatch):
        calls = []
        real_fsync, real_replace = os.fsync, os.replace
        monkeypatch.setattr(
            "a2a_adapter.base_adapter.os.fsync",
            lambda fd: (calls.append("fsync"), real_fsync(fd))[1],
        )
        monkeypatch.setattr(
            "a2a_adapter.base_adapter.os.replace",
            lambda a, b: (calls.append("replace"), real_replace(a, b))[1],
        )
        adapter = SleepingCliAdapter(tmp_path)
        adapter._sessions = {"ctx-1": "sess-A"}
        adapter._save_sessions()
        assert calls == ["fsync", "replace"]


class TestDefaultStateDir:
    def test_outside_working_dir_and_per_project(self, tmp_path, monkeypatch):
        state = tmp_path / "state"
        monkeypatch.setenv("XDG_STATE_HOME", str(state))
        proj_a = tmp_path / "proj-a"
        proj_b = tmp_path / "proj-b"

        dir_a = default_state_dir("claude-code", str(proj_a))
        dir_b = default_state_dir("claude-code", str(proj_b))

        assert dir_a != dir_b
        for d, proj in ((dir_a, proj_a), (dir_b, proj_b)):
            assert os.path.commonpath([d, str(proj)]) != str(proj)
            assert os.path.dirname(d) == str(state / "a2a-adapter" / "claude-code")

    def test_same_working_dir_is_stable(self, tmp_path):
        assert default_state_dir("codex", str(tmp_path)) == default_state_dir(
            "codex", str(tmp_path) + os.sep
        )

    def test_relative_working_dir_resolves_to_absolute(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert default_state_dir("codex", ".") == default_state_dir("codex", str(tmp_path))

    def test_agents_do_not_share_a_dir(self, tmp_path):
        assert default_state_dir("codex", str(tmp_path)) != default_state_dir(
            "claude-code", str(tmp_path)
        )

    @pytest.mark.parametrize("xdg", [None, "", "relative/state"])
    def test_falls_back_to_home_local_state(self, tmp_path, monkeypatch, xdg):
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        if xdg is None:
            monkeypatch.delenv("XDG_STATE_HOME", raising=False)
        else:
            monkeypatch.setenv("XDG_STATE_HOME", xdg)
        d = default_state_dir("pi", str(tmp_path / "proj"))
        assert d.startswith(
            os.path.join(str(tmp_path / "home"), ".local", "state", "a2a-adapter", "pi") + os.sep
        )


# ── Process-tree kill (Fix 1) ────────────────────────────────

posix_only = pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-only")


@posix_only
class TestProcessTreeKill:
    async def test_cancel_kills_grandchild(self, tmp_path):
        from a2a_adapter.exceptions import CancelledByAdapterError
        from tests.process_helpers import (
            TreeCliAdapter,
            cancel_and_wait,
            kill_pids,
            wait_dead,
            wait_for_pids,
        )

        adapter = TreeCliAdapter(tmp_path)
        context = SimpleNamespace(task_id="t-cancel")
        task = asyncio.create_task(adapter.invoke("hi", context=context))
        try:
            _, grandchild = await wait_for_pids(tmp_path)
            await adapter.cancel(context=context)
            with pytest.raises(CancelledByAdapterError):
                await asyncio.wait_for(task, 5)
            assert await wait_dead(grandchild), "grandchild survived cancel()"
        finally:
            kill_pids(tmp_path)
            await cancel_and_wait(task)

    async def test_task_cancellation_kills_grandchild(self, tmp_path):
        from tests.process_helpers import (
            TreeCliAdapter,
            cancel_and_wait,
            kill_pids,
            wait_dead,
            wait_for_pids,
        )

        adapter = TreeCliAdapter(tmp_path)
        task = asyncio.create_task(adapter.invoke("hi", context=SimpleNamespace(task_id="t")))
        try:
            _, grandchild = await wait_for_pids(tmp_path)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
            assert await wait_dead(grandchild), "grandchild survived task cancellation"
        finally:
            kill_pids(tmp_path)
            await cancel_and_wait(task)

    async def test_timeout_kills_grandchild(self, tmp_path):
        from tests.process_helpers import (
            TreeCliAdapter,
            cancel_and_wait,
            kill_pids,
            wait_dead,
            wait_for_pids,
        )

        adapter = TreeCliAdapter(tmp_path, timeout=1)
        task = asyncio.create_task(adapter.invoke("hi", context=SimpleNamespace(task_id="t")))
        try:
            _, grandchild = await wait_for_pids(tmp_path)
            with pytest.raises(RuntimeError, match="timed out after 1 seconds"):
                await asyncio.wait_for(task, 5)
            assert await wait_dead(grandchild), "grandchild survived the timeout"
        finally:
            kill_pids(tmp_path)
            await cancel_and_wait(task)

    async def test_close_kills_grandchild(self, tmp_path):
        from tests.process_helpers import (
            TreeCliAdapter,
            cancel_and_wait,
            kill_pids,
            wait_dead,
            wait_for_pids,
        )

        adapter = TreeCliAdapter(tmp_path)
        task = asyncio.create_task(adapter.invoke("hi", context=SimpleNamespace(task_id="t")))
        try:
            _, grandchild = await wait_for_pids(tmp_path)
            await asyncio.wait_for(adapter.close(), 5)
            assert await wait_dead(grandchild), "grandchild survived close()"
        finally:
            kill_pids(tmp_path)
            await cancel_and_wait(task)

    async def test_streaming_cancel_kills_grandchild(self, tmp_path):
        from tests.process_helpers import (
            STREAMING_GRANDCHILD_SCRIPT,
            TreeCliAdapter,
            cancel_and_wait,
            kill_pids,
            wait_dead,
            wait_for_pids,
        )

        adapter = TreeCliAdapter(tmp_path, script=STREAMING_GRANDCHILD_SCRIPT)
        first_chunk = asyncio.Event()

        async def consume():
            async for _ in adapter.stream("hi", context=SimpleNamespace(task_id="t")):
                first_chunk.set()

        task = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(first_chunk.wait(), 5)
            _, grandchild = await wait_for_pids(tmp_path)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
            assert await wait_dead(grandchild), "grandchild survived stream cancellation"
        finally:
            kill_pids(tmp_path)
            await cancel_and_wait(task)

    async def test_streaming_close_kills_grandchild(self, tmp_path):
        from tests.process_helpers import (
            STREAMING_GRANDCHILD_SCRIPT,
            TreeCliAdapter,
            kill_pids,
            wait_dead,
            wait_for_pids,
        )

        adapter = TreeCliAdapter(tmp_path, script=STREAMING_GRANDCHILD_SCRIPT)
        stream = adapter.stream("hi", context=SimpleNamespace(task_id="t"))
        try:
            assert await asyncio.wait_for(anext(stream), 5) == "started"
            _, grandchild = await wait_for_pids(tmp_path)
            await asyncio.wait_for(stream.aclose(), 5)
            assert await wait_dead(grandchild), "grandchild survived closing the stream"
        finally:
            kill_pids(tmp_path)

    async def test_kill_tree_ignores_vanished_group(self, tmp_path, monkeypatch):
        import signal
        from asyncio.subprocess import create_subprocess_exec

        from a2a_adapter import base_adapter
        from a2a_adapter.base_adapter import _kill_process_tree, _spawn_cli

        proc = await _spawn_cli(create_subprocess_exec, "sleep", "30")
        try:

            def vanished(pgid, sig):
                raise ProcessLookupError

            monkeypatch.setattr(base_adapter.os, "killpg", vanished)
            _kill_process_tree(proc)  # must not raise
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.kill(proc.pid, signal.SIGKILL)
            await asyncio.wait_for(proc.wait(), 5)

    def test_mock_proc_is_never_killpg_target(self, monkeypatch):
        import signal
        from unittest.mock import MagicMock

        from a2a_adapter import base_adapter
        from a2a_adapter.base_adapter import _kill_process_tree

        killpg = MagicMock()
        monkeypatch.setattr(base_adapter.os, "killpg", killpg)

        proc = MagicMock(pid=1, returncode=None)
        _kill_process_tree(proc)
        proc.kill.assert_called_once_with()

        proc = MagicMock(pid=1, returncode=None)
        _kill_process_tree(proc, signal.SIGTERM)
        proc.terminate.assert_called_once_with()
        proc.kill.assert_not_called()

        killpg.assert_not_called()

    @pytest.mark.parametrize("streaming", [False, True])
    async def test_spawn_uses_new_session_on_posix(self, tmp_path, monkeypatch, streaming):
        from unittest.mock import AsyncMock, MagicMock

        from tests.process_helpers import TreeCliAdapter

        proc = MagicMock()
        proc.returncode = 0
        proc.pid = 4242
        proc.communicate = AsyncMock(return_value=(b"ok", b""))
        proc.wait = AsyncMock(return_value=0)
        proc.stdout.readline = AsyncMock(side_effect=[b"ok\n", b""])
        proc.stderr.read = AsyncMock(return_value=b"")
        spawn = AsyncMock(return_value=proc)
        monkeypatch.setattr("a2a_adapter.base_adapter.create_subprocess_exec", spawn)

        adapter = TreeCliAdapter(tmp_path)
        if streaming:
            assert [c async for c in adapter.stream("hi")] == ["ok"]
        else:
            assert await adapter.invoke("hi") == "ok"
        assert spawn.call_args.kwargs.get("start_new_session") is True

    @pytest.mark.parametrize("streaming", [False, True])
    async def test_cancel_after_leader_exited_kills_grandchild(self, tmp_path, streaming):
        from a2a_adapter.exceptions import CancelledByAdapterError
        from tests.process_helpers import (
            LEADER_EXITS_SCRIPT,
            STREAMING_LEADER_EXITS_SCRIPT,
            TreeCliAdapter,
            cancel_and_wait,
            kill_pids,
            wait_dead,
            wait_for_pids,
        )

        script = STREAMING_LEADER_EXITS_SCRIPT if streaming else LEADER_EXITS_SCRIPT
        adapter = TreeCliAdapter(tmp_path, script=script, timeout=60)
        context = SimpleNamespace(task_id="t-exited")

        async def run():
            if streaming:
                return [c async for c in adapter.stream("hi", context=context)]
            return await adapter.invoke("hi", context=context)

        task = asyncio.create_task(run())
        try:
            _, grandchild = await wait_for_pids(tmp_path)
            proc = adapter._active_processes["t-exited"]
            loop = asyncio.get_running_loop()
            deadline = loop.time() + 5
            while proc.returncode is None:
                assert loop.time() < deadline, "leader did not exit"
                await asyncio.sleep(0.02)
            await adapter.cancel(context=context)
            with pytest.raises(CancelledByAdapterError):
                await asyncio.wait_for(task, 5)
            assert await wait_dead(grandchild)
        finally:
            kill_pids(tmp_path)
            await cancel_and_wait(task)

    @pytest.mark.parametrize("streaming", [False, True])
    async def test_clean_exit_never_killpgs(self, tmp_path, monkeypatch, streaming):
        from unittest.mock import MagicMock

        from a2a_adapter import base_adapter
        from tests.process_helpers import TreeCliAdapter

        adapter = TreeCliAdapter(tmp_path, script="echo done")
        killpg = MagicMock()
        monkeypatch.setattr(base_adapter.os, "killpg", killpg)
        if streaming:
            assert [c async for c in adapter.stream("hi")] == ["done"]
        else:
            assert await adapter.invoke("hi") == "done"
        assert len(base_adapter._OWN_GROUPS) == 0
        adapter.kill_children_now()
        await adapter.close()
        base_adapter._kill_own_groups_at_exit()
        killpg.assert_not_called()

    async def test_cancelled_stream_leaves_no_killed_marker(self, tmp_path):
        from tests.process_helpers import (
            STREAMING_GRANDCHILD_SCRIPT,
            TreeCliAdapter,
            cancel_and_wait,
            kill_pids,
            wait_for_pids,
        )

        adapter = TreeCliAdapter(tmp_path, script=STREAMING_GRANDCHILD_SCRIPT)
        context = SimpleNamespace(task_id="t-reused")
        started = asyncio.Event()

        async def consume():
            async for _ in adapter.stream("hi", context=context):
                started.set()

        task = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(started.wait(), 5)
            await wait_for_pids(tmp_path)
            # The SDK's order: adapter.cancel() first, then the producer task.
            await adapter.cancel(context=context)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
        finally:
            kill_pids(tmp_path)
            await cancel_and_wait(task)
        assert "t-reused" not in adapter._killed_tasks

    async def test_cancel_during_timeout_reap_leaves_no_killed_marker(self, tmp_path, monkeypatch):
        from unittest.mock import AsyncMock, MagicMock

        from tests.process_helpers import TreeCliAdapter

        adapter = TreeCliAdapter(tmp_path, timeout=0.1)
        context = SimpleNamespace(task_id="t-timeout")

        async def hang(*args, **kwargs):
            await asyncio.sleep(30)

        async def wait_and_get_cancelled():
            # A tasks/cancel arrives while the timed-out run waits for the reap.
            await adapter.cancel(context=context)
            return -9

        proc = MagicMock(pid=4242, returncode=None)
        proc.communicate = hang
        proc.wait = wait_and_get_cancelled
        monkeypatch.setattr(
            "a2a_adapter.base_adapter.create_subprocess_exec", AsyncMock(return_value=proc)
        )
        with pytest.raises(RuntimeError, match="timed out"):
            await asyncio.wait_for(adapter.invoke("hi", context=context), 5)
        proc.kill.assert_called()
        assert "t-timeout" not in adapter._killed_tasks


# ── Streaming total deadline (Fix 2) ─────────────────────────

_TICKER = (
    "import sys, time\n"
    "for i in range(100):\n"
    "    print(f'line {i}', flush=True)\n"
    "    time.sleep(0.1)\n"
)
_CLOSE_STDOUT_THEN_SLEEP = "import os, time\nos.close(1)\ntime.sleep(30)\n"
_LINE_CLOSE_STDOUT_THEN_SLEEP = (
    "import os, sys, time\nprint('first', flush=True)\nos.close(1)\ntime.sleep(30)\n"
)
_QUICK = "print('a'); print('b')\n"


class _PythonStreamAdapter(BaseA2AAdapter):
    """Streams the stdout lines of a ``python -c`` child, one chunk per line."""

    def __init__(self, tmp_path, script: str, *, timeout: float):
        super().__init__(working_dir=str(tmp_path), timeout=timeout)
        self._script = script
        self.spawned = []

    async def invoke(self, user_input, context_id=None, **kwargs):
        raise NotImplementedError

    async def stream(self, user_input, context_id=None, **kwargs):
        context = kwargs.get("context")
        task_id = context.task_id if context is not None else "task-1"
        async for chunk in self._run_subprocess_streaming("ctx", task_id, user_input=user_input):
            yield chunk

    def _build_command(self, message, context_key):
        import sys

        return CommandResult(args=[sys.executable, "-c", self._script], used_resume=False)

    def _handle_stream_line(self, line):
        from a2a_adapter.base_adapter import StreamEvent

        return StreamEvent(text=line)


def _kill_spawned(adapter) -> None:
    import signal

    for proc in adapter.spawned:
        with contextlib.suppress(ProcessLookupError):
            os.kill(proc.pid, signal.SIGKILL)


@pytest.fixture
def record_spawns(monkeypatch):
    """Record every process _run_subprocess_streaming spawns on ``adapter.spawned``."""
    from a2a_adapter import base_adapter

    real = base_adapter.create_subprocess_exec
    adapters = []

    def bind(adapter):
        adapters.append(adapter)
        return adapter

    async def spawning(*args, **kwargs):
        proc = await real(*args, **kwargs)
        for adapter in adapters:
            adapter.spawned.append(proc)
        return proc

    monkeypatch.setattr(base_adapter, "create_subprocess_exec", spawning)
    return bind


@posix_only
class TestStreamingDeadline:
    async def _consume(self, adapter, chunks, **kwargs):
        async for chunk in adapter.stream("hi", **kwargs):
            chunks.append(chunk)

    async def test_streaming_total_deadline(self, tmp_path, record_spawns):
        adapter = record_spawns(_PythonStreamAdapter(tmp_path, _TICKER, timeout=1))
        chunks: list[str] = []
        loop = asyncio.get_running_loop()
        start = loop.time()
        try:
            with pytest.raises(RuntimeError, match="timed out after 1 seconds"):
                await asyncio.wait_for(self._consume(adapter, chunks), 5)
            assert loop.time() - start < 3
            assert chunks and chunks[0] == "line 0"
        finally:
            _kill_spawned(adapter)

    async def test_streaming_deadline_kills_child(self, tmp_path, record_spawns):
        adapter = record_spawns(_PythonStreamAdapter(tmp_path, _TICKER, timeout=1))
        try:
            with pytest.raises(RuntimeError, match="timed out"):
                await asyncio.wait_for(self._consume(adapter, []), 5)
            assert adapter.spawned[0].returncode is not None
            assert adapter._active_processes == {}
        finally:
            _kill_spawned(adapter)

    async def test_streaming_tail_wait_is_bounded(self, tmp_path, record_spawns):
        adapter = record_spawns(_PythonStreamAdapter(tmp_path, _CLOSE_STDOUT_THEN_SLEEP, timeout=1))
        loop = asyncio.get_running_loop()
        start = loop.time()
        try:
            with pytest.raises(RuntimeError, match="timed out after 1 seconds"):
                await asyncio.wait_for(self._consume(adapter, []), 5)
            assert loop.time() - start < 3
            assert adapter.spawned[0].returncode is not None
            assert adapter._active_processes == {}
        finally:
            _kill_spawned(adapter)

    async def test_streaming_fast_run_unaffected(self, tmp_path, record_spawns):
        adapter = record_spawns(_PythonStreamAdapter(tmp_path, _QUICK, timeout=5))
        chunks: list[str] = []
        try:
            await asyncio.wait_for(self._consume(adapter, chunks), 5)
            assert chunks == ["a", "b"]
        finally:
            _kill_spawned(adapter)

    @pytest.mark.parametrize("how", ["cancel", "kill_children_now"])
    async def test_streaming_tail_child_still_cancellable(self, tmp_path, record_spawns, how):
        from a2a_adapter.exceptions import CancelledByAdapterError

        adapter = record_spawns(
            _PythonStreamAdapter(tmp_path, _LINE_CLOSE_STDOUT_THEN_SLEEP, timeout=60)
        )
        context = SimpleNamespace(task_id="t-tail")
        stream = adapter.stream("hi", context=context)
        rest = None
        try:
            assert await asyncio.wait_for(anext(stream), 5) == "first"
            rest = asyncio.create_task(anext(stream))  # now sits in the tail wait
            await asyncio.sleep(0.3)
            assert not rest.done()
            if how == "cancel":
                await adapter.cancel(context=context)
                expected, match = CancelledByAdapterError, "killed by cancel"
            else:
                adapter.kill_children_now()
                expected, match = RuntimeError, "exit code -9"
            with pytest.raises(expected, match=match):
                await asyncio.wait_for(rest, 5)
        finally:
            _kill_spawned(adapter)
            if rest is not None and not rest.done():
                rest.cancel()
                with contextlib.suppress(BaseException):
                    await rest
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(stream.aclose(), 5)

    @pytest.mark.parametrize("adapter_cancel_first", [True, False])
    async def test_consumer_cancelled_in_tail_kills_child(
        self, tmp_path, record_spawns, adapter_cancel_first
    ):
        adapter = record_spawns(
            _PythonStreamAdapter(tmp_path, _LINE_CLOSE_STDOUT_THEN_SLEEP, timeout=60)
        )
        context = SimpleNamespace(task_id="t-tail-cancel")
        stream = adapter.stream("hi", context=context)
        rest = None
        try:
            assert await asyncio.wait_for(anext(stream), 5) == "first"
            rest = asyncio.create_task(anext(stream))  # sits in the tail wait
            await asyncio.sleep(0.3)
            if adapter_cancel_first:  # the SDK's order on tasks/cancel
                await adapter.cancel(context=context)
            rest.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(rest, 5)
            proc = adapter.spawned[0]
            await asyncio.wait_for(proc.wait(), 5)
            assert proc.returncode is not None
            assert adapter._active_processes == {}
            assert "t-tail-cancel" not in adapter._killed_tasks
        finally:
            _kill_spawned(adapter)
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(stream.aclose(), 5)

    async def test_tail_wait_with_no_time_left_still_succeeds(self, tmp_path, monkeypatch):
        """Output that ends right at the deadline is not reported as a timeout."""
        from unittest.mock import AsyncMock, MagicMock

        adapter = _PythonStreamAdapter(tmp_path, _QUICK, timeout=60)
        loop = asyncio.get_running_loop()
        real_time = loop.time
        lines = iter([b"a\n"])

        async def readline():
            try:
                return next(lines)
            except StopIteration:
                # EOF arrives exactly at the deadline: no time left for the tail.
                monkeypatch.setattr(loop, "time", lambda: real_time() + 60)
                return b""

        async def wait():
            await asyncio.sleep(0)
            return 0

        proc = MagicMock(pid=4242, returncode=0)
        proc.stdout.readline = readline
        proc.stderr.read = AsyncMock(return_value=b"")
        proc.wait = wait
        monkeypatch.setattr(
            "a2a_adapter.base_adapter.create_subprocess_exec", AsyncMock(return_value=proc)
        )
        try:
            # No outer wait_for: the shifted clock would fire it at once.
            assert await _collect_chunks(adapter.stream("hi")) == ["a"]
        finally:
            monkeypatch.setattr(loop, "time", real_time)


async def _collect_chunks(stream):
    return [chunk async for chunk in stream]


# ── Stale resume session reported by the CLI (Fix 3) ─────────


class _StaleHookAdapter(BaseA2AAdapter):
    """CLI adapter whose hook treats ``marker`` in stdout/stderr as a stale session."""

    def __init__(self, tmp_path, *, marker: str = "session gone", scripts=None):
        super().__init__(
            working_dir=str(tmp_path),
            timeout=60,
            session_store_path=str(tmp_path / "sessions.json"),
        )
        self.marker = marker
        self.commands: list[list[str]] = []
        self._scripts = scripts  # optional real ``python -c`` scripts, one per run

    async def invoke(self, user_input, context_id=None, **kwargs):
        return await self._run_subprocess("ctx", "task-1", user_input=user_input)

    async def stream(self, user_input, context_id=None, **kwargs):
        async for chunk in self._run_subprocess_streaming("ctx", "task-1", user_input=user_input):
            yield chunk

    def _build_command(self, message, context_key):
        import sys

        session = self._sessions.get(context_key)
        resume = ["--resume", session] if session else []
        if self._scripts is not None:
            args = [sys.executable, "-c", self._scripts[len(self.commands)], *resume]
        else:
            args = ["cli", *resume]
        self.commands.append(args)
        return CommandResult(args=args, used_resume=bool(session))

    def _parse_invoke_output(self, stdout_text, context_key):
        return ParseResult(text=stdout_text.strip(), session_id="new-session")

    def _handle_stream_line(self, line):
        from a2a_adapter.base_adapter import StreamEvent

        if line.startswith("ERR "):
            return StreamEvent(error=line[4:])
        if line.startswith("SID "):
            return StreamEvent(session_id=line[4:])
        return StreamEvent(text=line)

    def _is_stale_session_error(self, stdout_text, stderr_text):
        return self.marker in stdout_text or self.marker in stderr_text


def _invoke_proc(stdout: bytes, stderr: bytes, returncode: int):
    from unittest.mock import AsyncMock, MagicMock

    proc = MagicMock(pid=4242, returncode=returncode)
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.wait = AsyncMock(return_value=returncode)
    return proc


def _stream_proc(lines: list[bytes], stderr: bytes, returncode: int):
    from unittest.mock import AsyncMock, MagicMock

    proc = MagicMock(pid=4242, returncode=returncode)
    proc.stdout.readline = AsyncMock(side_effect=[*lines, b""])
    proc.stderr.read = AsyncMock(side_effect=[stderr, b""] if stderr else [b""])
    proc.wait = AsyncMock(return_value=returncode)
    return proc


def _spawning(monkeypatch, procs):
    from unittest.mock import AsyncMock

    spawn = AsyncMock(side_effect=list(procs))
    monkeypatch.setattr("a2a_adapter.base_adapter.create_subprocess_exec", spawn)
    return spawn


def _stale_adapter(tmp_path, **kwargs):
    adapter = _StaleHookAdapter(tmp_path, **kwargs)
    adapter._sessions["ctx"] = "old-session"
    return adapter


class TestStaleSessionHook:
    def test_default_hook_is_false(self):
        assert BaseA2AAdapter._is_stale_session_error(ConcreteAdapter(), "x", "y") is False

    async def test_invoke_retries_when_hook_matches_stderr(self, tmp_path, monkeypatch):
        import json

        adapter = _stale_adapter(tmp_path)
        _spawning(
            monkeypatch,
            [_invoke_proc(b"", b"session gone", 1), _invoke_proc(b"ok", b"", 0)],
        )
        assert await adapter.invoke("hi") == "ok"
        assert adapter.commands == [["cli", "--resume", "old-session"], ["cli"]]
        assert adapter._sessions["ctx"] == "new-session"
        stored = json.loads((tmp_path / "sessions.json").read_text())
        assert stored == {"ctx": "new-session"}

    async def test_stream_retries_when_hook_matches_stderr(self, tmp_path, monkeypatch):
        import json

        adapter = _stale_adapter(tmp_path)
        _spawning(
            monkeypatch,
            [
                _stream_proc([], b"session gone", 1),
                _stream_proc([b"SID new-session\n", b"ok\n"], b"", 0),
            ],
        )
        assert [c async for c in adapter.stream("hi")] == ["ok"]
        assert adapter.commands[1] == ["cli"]
        assert adapter._sessions["ctx"] == "new-session"
        stored = json.loads((tmp_path / "sessions.json").read_text())
        assert stored == {"ctx": "new-session"}

    async def test_stream_retries_when_hook_matches_error_line(self, tmp_path, monkeypatch):
        adapter = _stale_adapter(tmp_path)
        _spawning(
            monkeypatch,
            [
                _stream_proc([b"ERR session gone\n"], b"", 1),
                _stream_proc([b"ok\n"], b"", 0),
            ],
        )
        assert [c async for c in adapter.stream("hi")] == ["ok"]
        assert adapter.commands[1] == ["cli"]
        assert "ctx" not in adapter._sessions

    @posix_only
    async def test_stream_stale_line_retries_even_if_cli_keeps_running(self, tmp_path):
        import signal

        pidfile = tmp_path / "stale.pid"
        stale = (
            "import os, time\n"
            f"open({str(pidfile)!r}, 'w').write(str(os.getpid()))\n"
            "print('ERR session gone', flush=True)\n"
            "time.sleep(30)\n"
        )
        adapter = _stale_adapter(tmp_path, scripts=[stale, "print('ok')"])
        try:
            chunks = await asyncio.wait_for(_collect(adapter.stream("hi")), 5)
            assert chunks == ["ok"]
            assert len(adapter.commands) == 2
            assert "--resume" not in adapter.commands[1]
        finally:
            if pidfile.exists():
                with contextlib.suppress(ProcessLookupError):
                    os.kill(int(pidfile.read_text()), signal.SIGKILL)

    async def test_no_retry_when_hook_does_not_match(self, tmp_path, monkeypatch):
        adapter = _stale_adapter(tmp_path)
        _spawning(monkeypatch, [_invoke_proc(b"", b"auth failed", 1)])
        with pytest.raises(RuntimeError, match="auth failed"):
            await adapter.invoke("hi")
        assert adapter._sessions["ctx"] == "old-session"
        assert len(adapter.commands) == 1

    async def test_no_retry_after_output_was_streamed(self, tmp_path, monkeypatch):
        adapter = _stale_adapter(tmp_path)
        _spawning(monkeypatch, [_stream_proc([b"chunk\n"], b"session gone", 1)])
        chunks = []
        with pytest.raises(RuntimeError, match="exit code 1"):
            async for chunk in adapter.stream("hi"):
                chunks.append(chunk)
        assert chunks == ["chunk"]
        assert len(adapter.commands) == 1

    async def test_hook_retry_happens_once(self, tmp_path, monkeypatch):
        adapter = _stale_adapter(tmp_path)
        _spawning(
            monkeypatch,
            [_invoke_proc(b"", b"session gone", 1), _invoke_proc(b"", b"session gone", 1)],
        )
        with pytest.raises(RuntimeError, match="session gone"):
            await adapter.invoke("hi")
        assert adapter.commands == [["cli", "--resume", "old-session"], ["cli"]]

    async def test_stream_session_seen_before_stale_line_is_dropped(self, tmp_path, monkeypatch):
        import json

        adapter = _stale_adapter(tmp_path)
        _spawning(
            monkeypatch,
            [
                _stream_proc([b"SID half-made\n", b"ERR session gone\n"], b"", 1),
                _stream_proc([b"ok\n"], b"", 0),
            ],
        )
        assert [c async for c in adapter.stream("hi")] == ["ok"]
        assert "ctx" not in adapter._sessions
        assert json.loads((tmp_path / "sessions.json").read_text()) == {}

    async def test_stream_stale_retry_happens_once(self, tmp_path, monkeypatch):
        adapter = _stale_adapter(tmp_path)
        adapter._sessions["ctx"] = "old-session"
        _spawning(
            monkeypatch,
            [
                _stream_proc([b"ERR session gone\n"], b"", 1),
                _stream_proc([b"ERR session gone\n"], b"", 1),
            ],
        )

        # The retry runs without --resume, so a second stale line is an error.
        with pytest.raises(RuntimeError, match="session gone"):
            async for _ in adapter.stream("hi"):
                pass
        assert len(adapter.commands) == 2

    async def test_stream_retries_on_matching_stderr_with_exit_zero(self, tmp_path, monkeypatch):
        adapter = _stale_adapter(tmp_path)
        _spawning(
            monkeypatch,
            [_stream_proc([], b"session gone", 0), _stream_proc([b"ok\n"], b"", 0)],
        )
        assert [c async for c in adapter.stream("hi")] == ["ok"]
        assert adapter.commands[1] == ["cli"]

    async def test_cancel_during_stale_kill_wins_over_retry(self, tmp_path, monkeypatch):
        from a2a_adapter.exceptions import CancelledByAdapterError

        adapter = _stale_adapter(tmp_path)

        def hook_then_cancel(stdout_text, stderr_text):
            # A tasks/cancel lands while the stale run is being killed.
            adapter._killed_tasks.add("task-1")
            return True

        adapter._is_stale_session_error = hook_then_cancel
        _spawning(monkeypatch, [_stream_proc([b"ERR session gone\n"], b"", -9)])
        with pytest.raises(CancelledByAdapterError):
            async for _ in adapter.stream("hi"):
                pass
        assert len(adapter.commands) == 1


async def _collect(stream):
    return [chunk async for chunk in stream]
