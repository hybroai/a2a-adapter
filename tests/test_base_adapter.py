"""Tests for BaseA2AAdapter contract and AdapterMetadata."""

import asyncio
import logging
import os
from types import SimpleNamespace
from typing import AsyncIterator

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
        assert default_state_dir("codex", ".") == default_state_dir(
            "codex", str(tmp_path)
        )

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
