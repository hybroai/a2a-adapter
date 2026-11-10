"""Tests for HermesAdapter (v0.2)."""

import asyncio
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from a2a.server.context import ServerCallContext
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import CancelTaskRequest, Message, Part, Role, SendMessageRequest, TaskState

from a2a_adapter.executor import AdapterAgentExecutor
from a2a_adapter.integrations.hermes import HermesAdapter
from a2a_adapter.server import build_agent_card


# ══════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════


def _make_mock_session_db(history=None):
    """Create a mock SessionDB that returns the given history."""
    db = MagicMock()
    db.get_messages_as_conversation = MagicMock(return_value=history or [])
    return db


def _make_mock_agent(response="Hello from Hermes!", error=None, chunks=None):
    """Create a mock AIAgent with controllable run_conversation behavior.

    If `chunks` is provided, run_conversation will call stream_callback
    for each chunk before returning.
    """
    agent = MagicMock()

    def _run_conversation(**kwargs):
        cb = kwargs.get("stream_callback")
        if cb and chunks:
            for chunk in chunks:
                cb(chunk)
        return {
            "final_response": response,
            "messages": [],
            "api_calls": 1,
            "completed": True,
            "interrupted": False,
            "error": error,
        }

    agent.run_conversation = MagicMock(side_effect=_run_conversation)
    agent.interrupt = MagicMock()
    return agent


def _make_context(task_id="test-task-123"):
    return SimpleNamespace(task_id=task_id)


@pytest.fixture
def adapter():
    """HermesAdapter with mocked internals — no real Hermes deps needed."""
    a = HermesAdapter(
        model="test-model",
        provider="test-provider",
        enabled_toolsets=["hermes-cli"],
        name="Test Hermes",
        description="Test adapter",
    )
    a._session_db = _make_mock_session_db()
    return a


# ══════════════════════════════════════════════════════════════
# Constructor
# ══════════════════════════════════════════════════════════════


class TestConstructor:
    def test_defaults(self):
        a = HermesAdapter()
        assert a._model is None
        assert a._provider_name is None
        assert a._enabled_toolsets is None
        assert a._running_agents == {}

    def test_custom_params(self):
        a = HermesAdapter(
            model="claude-opus-4.6",
            provider="openrouter",
            enabled_toolsets=["hermes-cli"],
            max_workers=8,
            name="My Agent",
            description="My desc",
            skills=[{"id": "s1", "name": "skill1", "description": "d"}],
            agent_provider={"organization": "Hybro"},
            documentation_url="https://docs.example.com",
            icon_url="https://img.example.com/icon.png",
        )
        assert a._model == "claude-opus-4.6"
        assert a._provider_name == "openrouter"
        assert a._enabled_toolsets == ["hermes-cli"]
        assert a._name == "My Agent"
        assert a._description == "My desc"
        assert len(a._skills) == 1
        assert a._agent_provider == {"organization": "Hybro"}
        assert a._documentation_url == "https://docs.example.com"
        assert a._icon_url == "https://img.example.com/icon.png"

    def test_session_db_lazy(self):
        a = HermesAdapter()
        assert a._session_db is None


# ══════════════════════════════════════════════════════════════
# _extract_task_id
# ══════════════════════════════════════════════════════════════


class TestExtractTaskId:
    def test_extracts_from_context(self):
        ctx = _make_context("my-task-42")
        tid = HermesAdapter._extract_task_id({"context": ctx})
        assert tid == "my-task-42"

    def test_fallback_uuid_when_no_context(self):
        tid = HermesAdapter._extract_task_id({})
        assert len(tid) == 32  # uuid hex

    def test_fallback_uuid_when_no_task_id(self):
        ctx = SimpleNamespace()  # no task_id attr
        tid = HermesAdapter._extract_task_id({"context": ctx})
        assert len(tid) == 32


# ══════════════════════════════════════════════════════════════
# invoke
# ══════════════════════════════════════════════════════════════


class TestInvoke:
    @pytest.mark.asyncio
    async def test_invoke_returns_response(self, adapter):
        mock_agent = _make_mock_agent(response="Hello!")
        with patch.object(adapter, "_make_agent", return_value=mock_agent):
            result = await adapter.invoke("Hi", context_id="conv-1")
        assert result == "Hello!"

    @pytest.mark.asyncio
    async def test_invoke_passes_correct_args(self, adapter):
        history = [{"role": "user", "content": "prev"}]
        adapter._session_db.get_messages_as_conversation.return_value = history
        mock_agent = _make_mock_agent()
        with patch.object(adapter, "_make_agent", return_value=mock_agent):
            await adapter.invoke("New msg", context_id="sess-1")

        mock_agent.run_conversation.assert_called_once()
        call_kwargs = mock_agent.run_conversation.call_args[1]
        assert call_kwargs["user_message"] == "New msg"
        assert call_kwargs["conversation_history"] == history
        assert call_kwargs["task_id"] == "a2a-sess-1"

    @pytest.mark.asyncio
    async def test_invoke_uses_context_id_as_session_id(self, adapter):
        mock_agent = _make_mock_agent()
        with patch.object(adapter, "_make_agent", return_value=mock_agent) as mock_make:
            await adapter.invoke("Hi", context_id="my-session")
        mock_make.assert_called_once_with("a2a-my-session")
        adapter._session_db.get_messages_as_conversation.assert_called_with("a2a-my-session")

    @pytest.mark.asyncio
    async def test_session_id_is_stable_across_restarts(self, adapter):
        """A new adapter (a restarted server) reads the same SessionDB
        session for the same context_id, so history carries over."""
        restarted = HermesAdapter(model="test-model")
        restarted._session_db = _make_mock_session_db()
        for a in (adapter, restarted):
            with patch.object(a, "_make_agent", return_value=_make_mock_agent()):
                await a.invoke("Hi", context_id="conv-1")
        adapter._session_db.get_messages_as_conversation.assert_called_with("a2a-conv-1")
        restarted._session_db.get_messages_as_conversation.assert_called_with("a2a-conv-1")

    @pytest.mark.asyncio
    async def test_stream_uses_same_session_id_as_invoke(self, adapter):
        with patch.object(adapter, "_make_agent", return_value=_make_mock_agent()) as mock_make:
            async for _ in adapter.stream("Hi", context_id="conv-1"):
                pass
        mock_make.assert_called_once_with("a2a-conv-1")

    @pytest.mark.asyncio
    async def test_invoke_generates_session_id_when_none(self, adapter):
        mock_agent = _make_mock_agent()
        with patch.object(adapter, "_make_agent", return_value=mock_agent) as mock_make:
            await adapter.invoke("Hi", context_id=None)
            await adapter.invoke("Hi again", context_id=None)
        first, second = (c[0][0] for c in mock_make.call_args_list)
        # No context_id: a fresh one-off session per call, never shared
        assert first.startswith("a2a-") and second.startswith("a2a-")
        assert len(first) == 16  # "a2a-" + 12 hex chars
        assert first != second
        # One-off sessions must not leave a lock behind per call
        assert adapter._session_locks == {}

    @pytest.mark.asyncio
    async def test_invoke_empty_response(self, adapter):
        mock_agent = _make_mock_agent(response="")
        with patch.object(adapter, "_make_agent", return_value=mock_agent):
            result = await adapter.invoke("Hi", context_id="c1")
        assert result == ""

    @pytest.mark.asyncio
    async def test_invoke_with_error_still_returns(self, adapter):
        mock_agent = _make_mock_agent(response="partial", error="token limit")
        with patch.object(adapter, "_make_agent", return_value=mock_agent):
            result = await adapter.invoke("Hi", context_id="c1")
        assert result == "partial"

    @pytest.mark.asyncio
    async def test_invoke_tracks_and_removes_agent(self, adapter):
        mock_agent = _make_mock_agent()
        ctx = _make_context("task-99")
        with patch.object(adapter, "_make_agent", return_value=mock_agent):
            assert "task-99" not in adapter._running_agents
            await adapter.invoke("Hi", context_id="c1", context=ctx)
        assert "task-99" not in adapter._running_agents

    @pytest.mark.asyncio
    async def test_invoke_removes_agent_on_exception(self, adapter):
        mock_agent = MagicMock()
        mock_agent.run_conversation = MagicMock(side_effect=RuntimeError("boom"))
        ctx = _make_context("task-err")
        with patch.object(adapter, "_make_agent", return_value=mock_agent):
            with pytest.raises(RuntimeError, match="boom"):
                await adapter.invoke("Hi", context_id="c1", context=ctx)
        assert "task-err" not in adapter._running_agents

    @pytest.mark.asyncio
    async def test_invoke_accepts_context_kwarg(self, adapter):
        """Verify invoke doesn't choke on the context kwarg from the bridge."""
        mock_agent = _make_mock_agent()
        ctx = _make_context()
        with patch.object(adapter, "_make_agent", return_value=mock_agent):
            result = await adapter.invoke("Hi", context_id="c1", context=ctx)
        assert result == "Hello from Hermes!"


# ══════════════════════════════════════════════════════════════
# stream
# ══════════════════════════════════════════════════════════════


class TestStream:
    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self, adapter):
        mock_agent = _make_mock_agent(chunks=["Hello", " ", "world"])
        with patch.object(adapter, "_make_agent", return_value=mock_agent):
            chunks = []
            async for chunk in adapter.stream("Hi", context_id="c1"):
                chunks.append(chunk)
        assert chunks == ["Hello", " ", "world"]

    @pytest.mark.asyncio
    async def test_stream_passes_stream_callback(self, adapter):
        mock_agent = _make_mock_agent(chunks=["x"])
        with patch.object(adapter, "_make_agent", return_value=mock_agent):
            async for _ in adapter.stream("Hi", context_id="c1"):
                pass
        call_kwargs = mock_agent.run_conversation.call_args[1]
        assert call_kwargs["stream_callback"] is not None
        assert callable(call_kwargs["stream_callback"])

    @pytest.mark.asyncio
    async def test_stream_sentinel_on_error(self, adapter):
        """Verify stream doesn't hang when run_conversation raises."""
        mock_agent = MagicMock()
        mock_agent.run_conversation = MagicMock(side_effect=RuntimeError("fail"))
        with patch.object(adapter, "_make_agent", return_value=mock_agent):
            chunks = []
            with pytest.raises(RuntimeError, match="fail"):
                async for chunk in adapter.stream("Hi", context_id="c1"):
                    chunks.append(chunk)
        assert chunks == []

    @pytest.mark.asyncio
    async def test_stream_tracks_and_removes_agent(self, adapter):
        mock_agent = _make_mock_agent(chunks=["x"])
        ctx = _make_context("stream-task")
        with patch.object(adapter, "_make_agent", return_value=mock_agent):
            async for _ in adapter.stream("Hi", context_id="c1", context=ctx):
                pass
        assert "stream-task" not in adapter._running_agents

    @pytest.mark.asyncio
    async def test_stream_accepts_context_kwarg(self, adapter):
        mock_agent = _make_mock_agent(chunks=["ok"])
        ctx = _make_context()
        with patch.object(adapter, "_make_agent", return_value=mock_agent):
            chunks = []
            async for chunk in adapter.stream("Hi", context_id="c1", context=ctx):
                chunks.append(chunk)
        assert chunks == ["ok"]


# ══════════════════════════════════════════════════════════════
# cancel
# ══════════════════════════════════════════════════════════════


class TestCancel:
    @pytest.mark.asyncio
    async def test_cancel_calls_interrupt(self, adapter):
        mock_agent = MagicMock()
        adapter._running_agents["task-1"] = mock_agent
        ctx = _make_context("task-1")
        await adapter.cancel(context_id="c1", context=ctx)
        mock_agent.interrupt.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_noop_when_no_agent(self, adapter):
        ctx = _make_context("nonexistent")
        await adapter.cancel(context_id="c1", context=ctx)

    @pytest.mark.asyncio
    async def test_cancel_accepts_context_kwarg(self, adapter):
        ctx = _make_context("some-task")
        await adapter.cancel(context_id="c1", context=ctx)


# ══════════════════════════════════════════════════════════════
# Metadata
# ══════════════════════════════════════════════════════════════


class TestMetadata:
    def test_default_metadata(self):
        a = HermesAdapter()
        meta = a.get_metadata()
        assert meta.name == "Hermes"
        assert "multi-purpose" in meta.description.lower()
        assert meta.streaming is True

    def test_custom_metadata(self):
        a = HermesAdapter(
            name="Custom Hermes",
            description="Custom desc",
            skills=[{"id": "s1", "name": "sk", "description": "d"}],
            agent_provider={"organization": "Hybro"},
            documentation_url="https://docs.example.com",
            icon_url="https://img.example.com/icon.png",
        )
        meta = a.get_metadata()
        assert meta.name == "Custom Hermes"
        assert meta.description == "Custom desc"
        assert meta.streaming is True
        assert len(meta.skills) == 1
        assert meta.provider == {"organization": "Hybro"}
        assert meta.documentation_url == "https://docs.example.com"
        assert meta.icon_url == "https://img.example.com/icon.png"

    def test_supports_streaming(self, adapter):
        assert adapter.supports_streaming() is True


# ══════════════════════════════════════════════════════════════
# close
# ══════════════════════════════════════════════════════════════


class TestClose:
    @pytest.mark.asyncio
    async def test_close_shuts_down_executor(self, adapter):
        with patch.object(adapter._executor, "shutdown") as mock_shutdown:
            await adapter.close()
        mock_shutdown.assert_called_once_with(wait=False, cancel_futures=True)

    @pytest.mark.asyncio
    async def test_async_context_manager(self, adapter):
        with patch.object(adapter._executor, "shutdown"):
            async with adapter:
                pass


# ══════════════════════════════════════════════════════════════
# _ensure_session_db
# ══════════════════════════════════════════════════════════════


class TestEnsureSessionDb:
    def test_returns_cached_db(self, adapter):
        db = adapter._ensure_session_db()
        assert db is adapter._session_db
        db2 = adapter._ensure_session_db()
        assert db2 is db

    def test_import_error_without_hermes(self):
        a = HermesAdapter()
        with patch.dict("sys.modules", {"hermes_state": None}):
            with pytest.raises(ImportError, match="hermes-agent"):
                a._ensure_session_db()


# ══════════════════════════════════════════════════════════════
# _make_agent
# ══════════════════════════════════════════════════════════════


class TestMakeAgent:
    def test_make_agent_passes_correct_kwargs(self, adapter):
        mock_ai_agent_cls = MagicMock()
        mock_load_config = MagicMock(
            return_value={
                "model": {"default": "fallback-model", "provider": "fallback-prov"},
            }
        )
        mock_resolve = MagicMock(
            return_value={
                "provider": "resolved-prov",
                "api_mode": "chat_completions",
                "base_url": "http://localhost:4000/v1",
                "api_key": "sk-test",
                "command": None,
                "args": None,
            }
        )

        with patch("a2a_adapter.integrations.hermes.HermesAdapter._ensure_session_db") as mock_db:
            mock_db.return_value = _make_mock_session_db()
            with patch.dict(
                "sys.modules",
                {
                    "run_agent": MagicMock(AIAgent=mock_ai_agent_cls),
                    "hermes_cli": MagicMock(),
                    "hermes_cli.config": MagicMock(load_config=mock_load_config),
                    "hermes_cli.runtime_provider": MagicMock(resolve_runtime_provider=mock_resolve),
                },
            ):
                adapter._make_agent("sess-abc")

        mock_ai_agent_cls.assert_called_once()
        kwargs = mock_ai_agent_cls.call_args[1]
        assert kwargs["platform"] == "a2a"
        assert kwargs["enabled_toolsets"] == ["hermes-cli"]
        assert kwargs["quiet_mode"] is True
        assert kwargs["session_id"] == "sess-abc"
        assert kwargs["model"] == "test-model"
        assert kwargs["provider"] == "resolved-prov"
        assert kwargs["api_mode"] == "chat_completions"

    def test_make_agent_falls_back_to_config_model(self):
        a = HermesAdapter(model=None, enabled_toolsets=["hermes-cli"])
        a._session_db = _make_mock_session_db()

        mock_ai_agent_cls = MagicMock()
        mock_load_config = MagicMock(
            return_value={
                "model": {"default": "config-model"},
            }
        )
        mock_resolve = MagicMock(side_effect=Exception("no provider"))

        with patch.dict(
            "sys.modules",
            {
                "run_agent": MagicMock(AIAgent=mock_ai_agent_cls),
                "hermes_cli": MagicMock(),
                "hermes_cli.config": MagicMock(load_config=mock_load_config),
                "hermes_cli.runtime_provider": MagicMock(resolve_runtime_provider=mock_resolve),
            },
        ):
            a._make_agent("sess-x")

        kwargs = mock_ai_agent_cls.call_args[1]
        assert kwargs["model"] == "config-model"

    def test_make_agent_string_model_config(self):
        a = HermesAdapter(model=None, enabled_toolsets=["hermes-cli"])
        a._session_db = _make_mock_session_db()

        mock_ai_agent_cls = MagicMock()
        mock_load_config = MagicMock(return_value={"model": "string-model"})
        mock_resolve = MagicMock(side_effect=Exception("no provider"))

        with patch.dict(
            "sys.modules",
            {
                "run_agent": MagicMock(AIAgent=mock_ai_agent_cls),
                "hermes_cli": MagicMock(),
                "hermes_cli.config": MagicMock(load_config=mock_load_config),
                "hermes_cli.runtime_provider": MagicMock(resolve_runtime_provider=mock_resolve),
            },
        ):
            a._make_agent("sess-str")

        kwargs = mock_ai_agent_cls.call_args[1]
        assert kwargs["model"] == "string-model"

    def test_make_agent_survives_provider_resolution_failure(self):
        a = HermesAdapter(model="explicit-model")
        a._session_db = _make_mock_session_db()

        mock_ai_agent_cls = MagicMock()
        mock_load_config = MagicMock(return_value={})
        mock_resolve = MagicMock(side_effect=RuntimeError("provider down"))

        with patch.dict(
            "sys.modules",
            {
                "run_agent": MagicMock(AIAgent=mock_ai_agent_cls),
                "hermes_cli": MagicMock(),
                "hermes_cli.config": MagicMock(load_config=mock_load_config),
                "hermes_cli.runtime_provider": MagicMock(resolve_runtime_provider=mock_resolve),
            },
        ):
            a._make_agent("sess-y")

        kwargs = mock_ai_agent_cls.call_args[1]
        assert kwargs["model"] == "explicit-model"
        assert "provider" not in kwargs


# ══════════════════════════════════════════════════════════════
# Registration / Loading
# ══════════════════════════════════════════════════════════════


class TestRegistration:
    def test_flat_import(self):
        from a2a_adapter import HermesAdapter as HA

        assert HA is HermesAdapter

    def test_integrations_import(self):
        from a2a_adapter.integrations import HermesAdapter as HA

        assert HA is HermesAdapter

    def test_load_adapter(self):
        from a2a_adapter import load_adapter

        adapter = load_adapter({"adapter": "hermes", "model": "test-m"})
        assert isinstance(adapter, HermesAdapter)
        assert adapter._model == "test-m"

    def test_load_adapter_with_all_params(self):
        from a2a_adapter import load_adapter

        adapter = load_adapter(
            {
                "adapter": "hermes",
                "model": "claude-opus-4.6",
                "provider": "openrouter",
                "enabled_toolsets": ["hermes-cli"],
                "max_workers": 2,
                "name": "Loaded Hermes",
            }
        )
        assert isinstance(adapter, HermesAdapter)
        assert adapter._model == "claude-opus-4.6"
        assert adapter._provider_name == "openrouter"
        assert adapter._enabled_toolsets == ["hermes-cli"]
        assert adapter._name == "Loaded Hermes"


# ══════════════════════════════════════════════════════════════
# Cancelled and abandoned turns
# ══════════════════════════════════════════════════════════════


class _BlockingAgent:
    """Runs in the worker thread until the test sets ``release``."""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.interrupts = 0

    def run_conversation(self, **kwargs):
        self.started.set()
        self.release.wait(10)
        self.finished.set()
        return {"final_response": "done", "completed": True}

    def interrupt(self):
        self.interrupts += 1


async def _started(agent):
    assert await asyncio.to_thread(agent.started.wait, 5)


async def _until(predicate):
    for _ in range(500):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached")


def _handler(adapter):
    return DefaultRequestHandler(
        AdapterAgentExecutor(adapter),
        InMemoryTaskStore(),
        build_agent_card(adapter, streaming=True),
    )


def _request(context_id="ctx-1"):
    return SendMessageRequest(
        message=Message(
            role=Role.ROLE_USER,
            message_id=uuid.uuid4().hex,
            context_id=context_id,
            parts=[Part(text="hi")],
        )
    )


@pytest.fixture
def single_worker(adapter):
    adapter._executor.shutdown(wait=False)
    adapter._executor = ThreadPoolExecutor(max_workers=1)
    yield adapter
    adapter._executor.shutdown(wait=False, cancel_futures=True)


class TestCancelledTurn:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("streaming", [True, False])
    async def test_cancel_interrupts_and_holds_the_lock_until_the_thread_ends(
        self, adapter, streaming
    ):
        agent = _BlockingAgent()
        adapter.supports_streaming = lambda: streaming
        handler = _handler(adapter)
        call = ServerCallContext()
        payloads = []

        async def consume():
            async for payload in handler.on_message_send_stream(_request(), call):
                payloads.append(payload)

        with patch.object(adapter, "_make_agent", return_value=agent):
            lock = adapter._session_lock(adapter._session_key("ctx-1"))
            stream = asyncio.create_task(consume())
            await _started(agent)
            await _until(lambda: len(payloads) >= 2)
            canceled = await asyncio.wait_for(
                handler.on_cancel_task(CancelTaskRequest(id=payloads[0].id), call), 5
            )
            await asyncio.wait_for(stream, 5)

            assert canceled.status.state == TaskState.TASK_STATE_CANCELED
            assert agent.interrupts == 1
            assert not agent.finished.is_set()
            assert lock.locked()

            agent.release.set()
            await _until(lambda: not lock.locked())
            assert agent.interrupts == 1

    @pytest.mark.asyncio
    async def test_a_second_turn_waits_for_the_abandoned_thread(self, adapter):
        first, second = _BlockingAgent(), _make_mock_agent(response="second")
        with patch.object(adapter, "_make_agent", side_effect=[first, second]) as make:
            turn = asyncio.create_task(adapter.invoke("one", context_id="ctx-1"))
            await _started(first)
            turn.cancel()
            with pytest.raises(asyncio.CancelledError):
                await turn

            next_turn = asyncio.create_task(adapter.invoke("two", context_id="ctx-1"))
            await asyncio.sleep(0.05)
            assert not next_turn.done() and make.call_count == 1

            first.release.set()
            assert await asyncio.wait_for(next_turn, 5) == "second"

    @pytest.mark.asyncio
    async def test_a_consumer_closing_the_stream_interrupts(self, adapter):
        class Chatty(_BlockingAgent):
            def run_conversation(self, **kwargs):
                kwargs["stream_callback"]("chunk")
                return super().run_conversation(**kwargs)

        agent = Chatty()
        with patch.object(adapter, "_make_agent", return_value=agent):
            lock = adapter._session_lock(adapter._session_key("ctx-1"))
            stream = adapter.stream("hi", context_id="ctx-1")
            assert await anext(stream) == "chunk"
            await stream.aclose()
            assert agent.interrupts == 1
            assert lock.locked()
            agent.release.set()
            await _until(lambda: not lock.locked())

    @pytest.mark.asyncio
    async def test_abandoned_before_the_pool_ran_it(self, single_worker):
        adapter = single_worker
        blocker, queued = _BlockingAgent(), _make_mock_agent()
        with patch.object(adapter, "_make_agent", side_effect=[blocker, queued]):
            first = asyncio.create_task(adapter.invoke("x", context_id="other"))
            await _started(blocker)
            lock = adapter._session_lock(adapter._session_key("ctx-1"))
            second = asyncio.create_task(adapter.invoke("y", context_id="ctx-1"))
            await _until(lambda: lock.locked())
            await asyncio.sleep(0.01)
            second.cancel()
            with pytest.raises(asyncio.CancelledError):
                await second
            queued.interrupt.assert_not_called()

            blocker.release.set()
            await asyncio.wait_for(first, 5)
            await _until(lambda: not lock.locked())
            queued.run_conversation.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("streaming", [True, False])
    async def test_a_turn_the_pool_dropped_fails_and_releases(self, single_worker, streaming):
        adapter = single_worker
        blocker, queued = _BlockingAgent(), _make_mock_agent()

        async def run_stream():
            return [c async for c in adapter.stream("y", context_id="ctx-1")]

        with patch.object(adapter, "_make_agent", side_effect=[blocker, queued]):
            first = asyncio.create_task(adapter.invoke("x", context_id="other"))
            await _started(blocker)
            lock = adapter._session_lock(adapter._session_key("ctx-1"))
            coro = run_stream() if streaming else adapter.invoke("y", context_id="ctx-1")
            second = asyncio.create_task(coro)
            await _until(lambda: lock.locked())
            await asyncio.sleep(0.01)
            await adapter.close()

            with pytest.raises(RuntimeError, match="closed before the turn started"):
                await asyncio.wait_for(second, 5)
            assert not lock.locked()
            queued.interrupt.assert_not_called()
            blocker.release.set()
            await asyncio.wait_for(first, 5)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("streaming", [True, False])
    async def test_a_failing_interrupt_keeps_the_cancellation(self, adapter, streaming):
        class BrokenInterrupt(_BlockingAgent):
            def interrupt(self):
                super().interrupt()
                raise OSError("interrupt broke")

        async def run_stream():
            return [c async for c in adapter.stream("x", context_id="ctx-1")]

        agent = BrokenInterrupt()
        lock = adapter._session_lock(adapter._session_key("ctx-1"))
        with patch.object(adapter, "_make_agent", return_value=agent):
            coro = run_stream() if streaming else adapter.invoke("x", context_id="ctx-1")
            turn = asyncio.create_task(coro)
            await _started(agent)
            turn.cancel()
            with pytest.raises(asyncio.CancelledError):
                await turn
        assert agent.interrupts == 1
        assert lock.locked()
        agent.release.set()
        await _until(lambda: not lock.locked())

    @pytest.mark.asyncio
    async def test_a_failure_before_submit_releases(self, adapter):
        lock = adapter._session_lock(adapter._session_key("ctx-1"))
        with patch.object(adapter, "_make_agent", side_effect=ValueError("bad config")):
            with pytest.raises(ValueError, match="bad config"):
                await adapter.invoke("x", context_id="ctx-1")
        assert not lock.locked()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("streaming", [True, False])
    async def test_a_normal_turn_never_interrupts(self, adapter, streaming):
        agent = _make_mock_agent(chunks=["a"])
        lock = adapter._session_lock(adapter._session_key("ctx-1"))
        with patch.object(adapter, "_make_agent", return_value=agent):
            if streaming:
                assert [c async for c in adapter.stream("x", context_id="ctx-1")] == ["a"]
            else:
                await adapter.invoke("x", context_id="ctx-1")
        agent.interrupt.assert_not_called()
        await _until(lambda: not lock.locked())

    @pytest.mark.asyncio
    async def test_a_failed_run_releases_without_interrupting(self, adapter):
        agent = _make_mock_agent()
        agent.run_conversation.side_effect = RuntimeError("agent broke")
        lock = adapter._session_lock(adapter._session_key("ctx-1"))
        with patch.object(adapter, "_make_agent", return_value=agent):
            with pytest.raises(RuntimeError, match="agent broke"):
                await adapter.invoke("x", context_id="ctx-1")
        agent.interrupt.assert_not_called()
        await _until(lambda: not lock.locked())
