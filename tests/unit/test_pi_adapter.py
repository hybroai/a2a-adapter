"""Tests for the single-process, single-session Pi adapter."""

import asyncio
import os
from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import httpx
import pytest

from a2a_adapter.base_adapter import default_state_dir
from a2a_adapter.exceptions import CancelledByAdapterError
from a2a_adapter.integrations.pi import PiAdapter, PiTurnAbortedError
from a2a_adapter.loader import load_adapter
from a2a_adapter.server import to_a2a


def _context(task_id: str) -> MagicMock:
    context = MagicMock()
    context.task_id = task_id
    return context


class FakePiRpcProcess:
    def __init__(self, chunks: list[str] | None = None) -> None:
        self.chunks = chunks or ["hello", " world"]
        self.start_calls = 0
        self.prompt_calls: list[str] = []
        self.abort_calls = 0
        self.close_calls = 0
        self.running = False
        self.prompt_started = asyncio.Event()
        self.release_prompt = asyncio.Event()
        self.block_prompt = False
        self.aborted = False

    @property
    def is_running(self) -> bool:
        return self.running

    async def start(self) -> None:
        self.start_calls += 1
        self.running = True

    async def prompt(self, message: str) -> AsyncIterator[str]:
        self.prompt_calls.append(message)
        self.prompt_started.set()
        if self.block_prompt:
            await self.release_prompt.wait()
        if self.aborted:
            raise PiTurnAbortedError("turn aborted")
        for chunk in self.chunks:
            yield chunk

    async def abort(self) -> None:
        self.abort_calls += 1
        self.aborted = True
        self.release_prompt.set()

    async def close(self) -> None:
        self.close_calls += 1
        self.running = False


@pytest.fixture
def fake_rpc() -> FakePiRpcProcess:
    return FakePiRpcProcess()


@pytest.fixture
def adapter(tmp_path, fake_rpc) -> PiAdapter:
    return PiAdapter(
        working_dir=str(tmp_path),
        session_id="test-session",
        _rpc_factory=lambda: fake_rpc,
    )


class TestPiAdapter:
    def test_default_session_dir_is_per_user_and_per_project(self, tmp_path):
        proj_a = tmp_path / "proj-a"
        proj_b = tmp_path / "proj-b"
        a = PiAdapter(working_dir=str(proj_a))
        b = PiAdapter(working_dir=str(proj_b))
        assert a.session_dir == os.path.join(default_state_dir("pi", str(proj_a)), "sessions")
        assert not a.session_dir.startswith(str(proj_a) + os.sep)
        assert a.session_dir != b.session_dir

    def test_explicit_session_dir_wins(self, tmp_path):
        a = PiAdapter(working_dir=str(tmp_path), session_dir=str(tmp_path / "s"))
        assert a.session_dir == str(tmp_path / "s")

    @pytest.mark.asyncio
    async def test_lazy_starts_once_and_reuses_process(self, adapter, fake_rpc):
        first = await adapter.invoke("first", context_id="ctx-1", context=_context("task-1"))
        second = await adapter.invoke("second", context_id="ctx-1", context=_context("task-2"))

        assert first == second == "hello world"
        assert fake_rpc.start_calls == 1
        assert fake_rpc.prompt_calls == ["first", "second"]

    @pytest.mark.asyncio
    async def test_stream_yields_rpc_text_chunks(self, adapter):
        chunks = [
            chunk
            async for chunk in adapter.stream(
                "stream", context_id="ctx-1", context=_context("task-1")
            )
        ]

        assert chunks == ["hello", " world"]

    @pytest.mark.asyncio
    async def test_rejects_a_different_a2a_context(self, adapter):
        await adapter.invoke("first", context_id="ctx-1", context=_context("task-1"))

        with pytest.raises(RuntimeError, match="already bound"):
            await adapter.invoke("second", context_id="ctx-2", context=_context("task-2"))

    @pytest.mark.asyncio
    async def test_same_process_turns_are_serialized(self, adapter, fake_rpc):
        fake_rpc.block_prompt = True

        first = asyncio.create_task(
            adapter.invoke("first", context_id="ctx-1", context=_context("task-1"))
        )
        await fake_rpc.prompt_started.wait()

        second = asyncio.create_task(
            adapter.invoke("second", context_id="ctx-1", context=_context("task-2"))
        )
        await asyncio.sleep(0)

        assert fake_rpc.prompt_calls == ["first"]

        fake_rpc.release_prompt.set()
        await first
        await second
        assert fake_rpc.prompt_calls == ["first", "second"]

    @pytest.mark.asyncio
    async def test_cancel_aborts_the_active_turn(self, adapter, fake_rpc):
        fake_rpc.block_prompt = True
        running = asyncio.create_task(
            adapter.invoke("work", context_id="ctx-1", context=_context("task-1"))
        )
        await fake_rpc.prompt_started.wait()

        await adapter.cancel(context=_context("task-1"))

        with pytest.raises(CancelledByAdapterError):
            await running
        assert fake_rpc.abort_calls == 1

    @pytest.mark.asyncio
    async def test_cancelled_queued_turn_never_reaches_pi(self, adapter, fake_rpc):
        fake_rpc.block_prompt = True
        first = asyncio.create_task(
            adapter.invoke("first", context_id="ctx-1", context=_context("task-1"))
        )
        await fake_rpc.prompt_started.wait()

        second = asyncio.create_task(
            adapter.invoke("second", context_id="ctx-1", context=_context("task-2"))
        )
        await adapter.cancel(context=_context("task-2"))

        fake_rpc.release_prompt.set()
        await first
        with pytest.raises(CancelledByAdapterError):
            await second
        assert fake_rpc.prompt_calls == ["first"]

    @pytest.mark.asyncio
    async def test_close_closes_the_process(self, adapter, fake_rpc):
        await adapter.invoke("first", context_id="ctx-1", context=_context("task-1"))

        await adapter.close()

        assert fake_rpc.close_calls == 1


class TestPiMetadata:
    def test_metadata_declares_streaming(self, adapter):
        metadata = adapter.get_metadata()

        assert metadata.name == "Pi"
        assert metadata.streaming is True


class TestPiRegistration:
    def test_flat_import(self):
        from a2a_adapter import PiAdapter as FlatPiAdapter

        assert FlatPiAdapter is PiAdapter

    def test_loader(self, tmp_path, fake_rpc):
        loaded = load_adapter(
            {
                "adapter": "pi",
                "working_dir": str(tmp_path),
                "session_id": "loaded-session",
                "_rpc_factory": lambda: fake_rpc,
            }
        )

        assert isinstance(loaded, PiAdapter)


class TestPiA2AIntegration:
    @pytest.mark.asyncio
    async def test_message_send_completes_through_real_a2a_stack(self, adapter):
        app = to_a2a(adapter)
        payload = {
            "jsonrpc": "2.0",
            "id": "request-1",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "messageId": "message-1",
                    "parts": [{"kind": "text", "text": "hello Pi"}],
                }
            },
        }

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.post("/", json=payload)

        assert response.status_code == 200
        result = response.json()["result"]
        assert result["status"]["state"] == "completed"
        artifact_text = "".join(
            part["text"]
            for artifact in result["artifacts"]
            for part in artifact["parts"]
            if part.get("kind") == "text"
        )
        assert artifact_text == "hello world"
