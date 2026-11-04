"""Tests for the DeepSeek Harness SDK adapter.

The SDK is not installed in CI, so a fake ``deepseek_harness`` package is
placed in ``sys.modules``. Its ``Session.run`` blocks in a worker thread on a
``threading.Event`` so tests can hold a turn open, like the real runtime.
"""

import asyncio
import hashlib
import logging
import os
import re
import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from a2a_adapter.base_adapter import default_state_dir
from a2a_adapter.exceptions import CancelledByAdapterError
from a2a_adapter.integrations.deepseek_harness import DeepSeekHarnessAdapter
from a2a_adapter.loader import load_adapter
from a2a_adapter.server import to_a2a


def _context(task_id: str) -> MagicMock:
    context = MagicMock()
    context.task_id = task_id
    return context


class FakeHarnessError(Exception):
    pass


class FakeTransportClosedError(FakeHarnessError):
    pass


class FakeSdkProtocolError(FakeHarnessError):
    pass


class FakeJsonRpcError(FakeHarnessError):
    def __init__(self, code, message, data=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


class FakeSession:
    def __init__(self, harness: "FakeHarness", session_id: str) -> None:
        self.harness = harness
        self.id = session_id

    def run(self, prompt):
        sdk = self.harness.sdk
        if not self.harness.running or self.harness.exit_code is not None:
            # Like the SDK: Session.run writes to the runtime and never launches it.
            raise FakeTransportClosedError("DeepSeek Harness runtime is not running")
        sdk.run_calls.append((self.id, prompt))
        self.harness.started.set()
        if self.harness.gate is not None:
            self.harness.gate.wait(10)
        if self.harness.closed:
            raise FakeTransportClosedError("DeepSeek Harness runtime closed")
        if sdk.run_error is not None:
            error, sdk.run_error = sdk.run_error, None
            raise error
        sdk.persisted.add(self.id)
        return SimpleNamespace(
            session_id=self.id,
            final_response=sdk.final_response,
            finish_reason=sdk.finish_reason,
            events=list(sdk.events),
            notifications=[],
        )


class FakeHarness:
    """Mirrors the SDK's lazy runtime: session calls launch it, close() before launch is a no-op."""

    def __init__(self, sdk: "FakeSdk", **kwargs) -> None:
        self.sdk = sdk
        self.kwargs = kwargs
        self.running = False
        self.launches = 0
        self.closed = False
        self.close_calls = 0
        self.launch_entered = threading.Event()
        self.launch_gate: threading.Event | None = None
        self.exit_code: int | None = None
        # The SDK keeps the child in client._proc until close(), even after it exits.
        self.client = SimpleNamespace(_proc=SimpleNamespace(poll=lambda: self.exit_code))
        self.resume_calls: list[str] = []
        self.start_calls: list[str] = []
        self.handle_threads: list[int] = []
        self.started = threading.Event()
        self.gate: threading.Event | None = sdk.gates.pop(0) if sdk.gates else None

    def start(self) -> None:
        self.launch_entered.set()
        if self.launch_gate is not None:
            self.launch_gate.wait(10)
        if not self.running:
            self.launches += 1
            self.running = True
            self.closed = False

    def resume_session(self, session_id: str) -> FakeSession:
        self.start()
        self.handle_threads.append(threading.get_ident())
        self.resume_calls.append(session_id)
        if session_id not in self.sdk.persisted:
            raise FakeJsonRpcError(-32603, f'session "{session_id}" not found')
        return FakeSession(self, session_id)

    def start_session(self, session_id: str) -> FakeSession:
        self.start()
        self.handle_threads.append(threading.get_ident())
        self.start_calls.append(session_id)
        return FakeSession(self, session_id)

    def die(self) -> None:
        """The child exits on its own; the SDK object still looks started."""
        self.exit_code = -9

    def close(self) -> None:
        self.close_calls += 1
        if not self.running:
            return
        self.running = False
        self.closed = True
        if self.gate is not None:
            self.gate.set()


class FakeSdk:
    def __init__(self) -> None:
        self.harnesses: list[FakeHarness] = []
        self.gates: list[threading.Event] = []
        self.persisted: set[str] = set()
        self.run_calls: list[tuple[str, str]] = []
        self.run_error: BaseException | None = None
        self.final_response = "hello world"
        self.finish_reason: str | None = "completed"
        self.events: list[dict] = []

    def create(self, **kwargs) -> FakeHarness:
        harness = FakeHarness(self, **kwargs)
        self.harnesses.append(harness)
        return harness

    @property
    def harness(self) -> FakeHarness:
        return self.harnesses[-1]


@pytest.fixture
def sdk(monkeypatch) -> FakeSdk:
    fake = FakeSdk()
    package = types.ModuleType("deepseek_harness")
    package.DeepSeekHarness = fake.create
    package.SdkProtocolError = FakeSdkProtocolError
    errors = types.ModuleType("deepseek_harness.errors")
    errors.HarnessError = FakeHarnessError
    errors.TransportClosedError = FakeTransportClosedError
    errors.SdkProtocolError = FakeSdkProtocolError
    errors.JsonRpcError = FakeJsonRpcError
    package.errors = errors
    monkeypatch.setitem(sys.modules, "deepseek_harness", package)
    monkeypatch.setitem(sys.modules, "deepseek_harness.errors", errors)
    return fake


@pytest.fixture
def adapter(tmp_path, sdk):
    instance = DeepSeekHarnessAdapter(
        working_dir=str(tmp_path / "ws"), dsh_home=str(tmp_path / "home")
    )
    yield instance
    instance._executor.shutdown(wait=False, cancel_futures=True)


async def _wait_started(sdk: FakeSdk) -> None:
    for _ in range(500):
        if sdk.harnesses:
            break
        await asyncio.sleep(0.01)
    assert await asyncio.to_thread(sdk.harness.started.wait, 5)


class TestSessionKey:
    def test_safe_context_id_is_used_verbatim(self):
        assert DeepSeekHarnessAdapter._session_key("ctx-1.a_b") == "a2a-ctx-1.a_b"

    @pytest.mark.parametrize("context_id", ["has space", "-leading", "trailing.", "a/b", "x" * 101])
    def test_unsafe_context_id_is_hashed(self, context_id):
        digest = hashlib.sha256(context_id.encode("utf-8")).hexdigest()[:16]
        assert DeepSeekHarnessAdapter._session_key(context_id) == f"a2ah-{digest}"

    def test_ids_differing_only_in_case_stay_distinct_after_case_folding(self):
        # dsh names session directories after the id, and macOS and Windows
        # filesystems are case-insensitive by default.
        upper = DeepSeekHarnessAdapter._session_key("Foo")
        lower = DeepSeekHarnessAdapter._session_key("foo")
        assert lower == "a2a-foo"
        assert upper.startswith("a2ah-")
        assert upper.lower() != lower.lower()

    def test_hashed_ids_cannot_collide_with_verbatim_ids(self):
        context_id = "has space"
        digest = hashlib.sha256(context_id.encode("utf-8")).hexdigest()[:16]
        assert DeepSeekHarnessAdapter._session_key(digest) != DeepSeekHarnessAdapter._session_key(
            context_id
        )

    def test_missing_context_id_gets_a_one_off_id(self):
        first = DeepSeekHarnessAdapter._session_key(None)
        second = DeepSeekHarnessAdapter._session_key(None)
        assert re.fullmatch(r"a2a-[0-9a-f]{12}", first)
        assert first != second


class TestConfiguration:
    def test_dsh_home_argument_wins_and_is_absolute(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DSH_HOME", str(tmp_path / "env-home"))
        monkeypatch.chdir(tmp_path)
        adapter = DeepSeekHarnessAdapter(working_dir=str(tmp_path), dsh_home="rel-home")
        assert adapter.dsh_home == str(tmp_path / "rel-home")

    def test_dsh_home_env_is_used_when_set(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DSH_HOME", str(tmp_path / "env-home"))
        adapter = DeepSeekHarnessAdapter(working_dir=str(tmp_path))
        assert adapter.dsh_home == str(tmp_path / "env-home")

    @pytest.mark.parametrize("env_value", [None, "", "   "])
    def test_dsh_home_defaults_to_state_dir(self, tmp_path, monkeypatch, env_value):
        if env_value is None:
            monkeypatch.delenv("DSH_HOME", raising=False)
        else:
            monkeypatch.setenv("DSH_HOME", env_value)
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
        adapter = DeepSeekHarnessAdapter(working_dir=str(tmp_path / "ws"))
        assert adapter.dsh_home == default_state_dir("deepseek-harness", str(tmp_path / "ws"))
        assert not os.path.exists(adapter.dsh_home)

    @pytest.mark.asyncio
    async def test_sdk_receives_cwd_home_and_timeouts(self, tmp_path, sdk):
        adapter = DeepSeekHarnessAdapter(
            working_dir=str(tmp_path / "ws"),
            dsh_home=str(tmp_path / "home"),
            timeout=42,
            startup_timeout=7,
            profile="sdk-minimal",
            patches=["p.yml"],
            dsh_bin="/opt/dsh",
            reasoning_effort="high",
            max_tokens=1000,
            base_url="https://example.invalid",
        )
        await adapter.invoke("hi", context_id="ctx-1", context=_context("t1"))

        kwargs = sdk.harness.kwargs
        assert kwargs["cwd"] == str(tmp_path / "ws")
        assert kwargs["dsh_home"] == str(tmp_path / "home")
        assert kwargs["request_timeout_seconds"] == 42
        assert kwargs["initialize_timeout_seconds"] == 7
        assert kwargs["profile"] == "sdk-minimal"
        assert kwargs["patches"] == ("p.yml",)
        assert kwargs["dsh_bin"] == "/opt/dsh"
        assert kwargs["reasoning_effort"] == "high"
        assert kwargs["max_tokens"] == 1000
        assert kwargs["base_url"] == "https://example.invalid"
        assert os.path.isdir(tmp_path / "home")
        await adapter.close()

    @pytest.mark.asyncio
    async def test_bare_dsh_bin_is_found_on_path(self, tmp_path, sdk, monkeypatch):
        monkeypatch.setattr(
            "a2a_adapter.integrations.deepseek_harness.shutil.which",
            lambda name: f"/usr/local/bin/{name}",
        )
        adapter = DeepSeekHarnessAdapter(
            working_dir=str(tmp_path), dsh_home=str(tmp_path / "h"), dsh_bin="dsh"
        )
        await adapter.invoke("hi", context_id="ctx-1", context=_context("t1"))
        assert sdk.harness.kwargs["dsh_bin"] == "/usr/local/bin/dsh"
        await adapter.close()

    @pytest.mark.asyncio
    async def test_bare_dsh_bin_missing_from_path_raises(self, tmp_path, sdk, monkeypatch):
        monkeypatch.setattr(
            "a2a_adapter.integrations.deepseek_harness.shutil.which", lambda name: None
        )
        adapter = DeepSeekHarnessAdapter(
            working_dir=str(tmp_path), dsh_home=str(tmp_path / "h"), dsh_bin="dsh"
        )
        with pytest.raises(FileNotFoundError, match="'dsh' not found on PATH"):
            await adapter.invoke("hi", context_id="ctx-1", context=_context("t1"))
        await adapter.close()

    def test_full_access_profile_logs_a_warning(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING):
            DeepSeekHarnessAdapter(working_dir=str(tmp_path), profile="sdk-minimal")
        assert "danger-full-access" in caplog.text

    def test_default_profile_logs_no_warning(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING):
            DeepSeekHarnessAdapter(working_dir=str(tmp_path))
        assert caplog.text == ""

    @pytest.mark.asyncio
    async def test_unset_options_keep_sdk_defaults(self, adapter, sdk):
        await adapter.invoke("hi", context_id="ctx-1", context=_context("t1"))
        for key in ("provider", "model", "reasoning_effort", "max_tokens", "api_key", "base_url"):
            assert key not in sdk.harness.kwargs

    @pytest.mark.asyncio
    async def test_provider_and_model_are_forwarded(self, tmp_path, sdk):
        adapter = DeepSeekHarnessAdapter(
            working_dir=str(tmp_path), dsh_home=str(tmp_path / "h"), provider="p", model="m"
        )
        await adapter.invoke("hi", context_id="ctx-1", context=_context("t1"))
        assert sdk.harness.kwargs["provider"] == "p"
        assert sdk.harness.kwargs["model"] == "m"
        await adapter.close()

    @pytest.mark.asyncio
    async def test_api_key_reaches_sdk_but_not_logs_or_metadata(self, tmp_path, sdk, caplog):
        secret = "sk-test-not-a-real-key"
        adapter = DeepSeekHarnessAdapter(
            working_dir=str(tmp_path), dsh_home=str(tmp_path / "h"), api_key=secret
        )
        with caplog.at_level(logging.DEBUG):
            await adapter.invoke("hi", context_id="ctx-1", context=_context("t1"))
        assert sdk.harness.kwargs["api_key"] == secret
        assert secret not in caplog.text
        assert secret not in repr(adapter.get_metadata())
        await adapter.close()

    @pytest.mark.asyncio
    async def test_missing_sdk_raises_install_hint(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "deepseek_harness", None)
        adapter = DeepSeekHarnessAdapter(working_dir=str(tmp_path), dsh_home=str(tmp_path / "h"))
        with pytest.raises(ImportError, match=r"a2a-adapter\[deepseek-harness\]"):
            await adapter.invoke("hi", context_id="ctx-1", context=_context("t1"))
        await adapter.close()


class TestTurns:
    @pytest.mark.asyncio
    async def test_first_turn_resumes_then_starts_and_caches_handle(self, adapter, sdk):
        first = await adapter.invoke("one", context_id="ctx-1", context=_context("t1"))
        second = await adapter.invoke("two", context_id="ctx-1", context=_context("t2"))

        assert first == second == "hello world"
        assert sdk.harness.resume_calls == ["a2a-ctx-1"]
        assert sdk.harness.start_calls == ["a2a-ctx-1"]
        assert sdk.run_calls == [("a2a-ctx-1", "one"), ("a2a-ctx-1", "two")]
        assert len(sdk.harnesses) == 1

    @pytest.mark.asyncio
    async def test_persisted_session_is_resumed(self, adapter, sdk):
        sdk.persisted.add("a2a-ctx-1")
        await adapter.invoke("one", context_id="ctx-1", context=_context("t1"))
        assert sdk.harness.resume_calls == ["a2a-ctx-1"]
        assert sdk.harness.start_calls == []

    @pytest.mark.asyncio
    async def test_other_resume_errors_propagate(self, adapter, sdk, monkeypatch):
        def occupied(session_id):
            raise FakeJsonRpcError(-32603, f"SDK session is already open or opening: {session_id}")

        await adapter.invoke("warm", context_id="other", context=_context("t0"))
        monkeypatch.setattr(sdk.harness, "resume_session", occupied)
        with pytest.raises(FakeJsonRpcError, match="already open"):
            await adapter.invoke("one", context_id="ctx-1", context=_context("t1"))
        assert sdk.harness.closed is False

    @pytest.mark.asyncio
    async def test_other_not_found_errors_do_not_start_a_new_session(
        self, adapter, sdk, monkeypatch
    ):
        def missing_workspace(session_id):
            raise FakeJsonRpcError(-32603, "workspace directory not found")

        await adapter.invoke("warm", context_id="other", context=_context("t0"))
        monkeypatch.setattr(sdk.harness, "resume_session", missing_workspace)
        with pytest.raises(FakeJsonRpcError, match="workspace"):
            await adapter.invoke("one", context_id="ctx-1", context=_context("t1"))
        assert "a2a-ctx-1" not in sdk.harness.start_calls

    @pytest.mark.asyncio
    async def test_one_off_session_starts_directly(self, adapter, sdk):
        await adapter.invoke("one", context=_context("t1"))
        assert sdk.harness.resume_calls == []
        assert len(sdk.harness.start_calls) == 1
        assert adapter._session_handles == {}

    @pytest.mark.asyncio
    async def test_handle_acquisition_runs_off_the_event_loop(self, adapter, sdk):
        await adapter.invoke("one", context_id="ctx-1", context=_context("t1"))
        assert sdk.harness.handle_threads
        assert threading.get_ident() not in sdk.harness.handle_threads

    @pytest.mark.asyncio
    @pytest.mark.parametrize("reason", ["error", "max-tokens", None])
    async def test_empty_response_without_completion_raises(self, adapter, sdk, reason):
        sdk.final_response = ""
        sdk.finish_reason = reason
        with pytest.raises(RuntimeError, match="DeepSeek Harness"):
            await adapter.invoke("one", context_id="ctx-1", context=_context("t1"))

    @pytest.mark.asyncio
    async def test_failed_turn_reports_the_runtime_error(self, adapter, sdk, caplog):
        sdk.final_response = ""
        sdk.finish_reason = "error"
        sdk.events = [
            {"type": "turn/start", "data": {}},
            {
                "type": "turn/end",
                "data": {
                    "reason": {
                        "kind": "error",
                        "error": {
                            "message": "DeepSeek Messages transport failed",
                            "code": "TRANSPORT",
                        },
                    }
                },
            },
        ]
        with caplog.at_level(logging.WARNING):
            with pytest.raises(RuntimeError, match="TRANSPORT: DeepSeek Messages transport failed"):
                await adapter.invoke("one", context_id="ctx-1", context=_context("t1"))
        assert "TRANSPORT: DeepSeek Messages transport failed" in caplog.text

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "events",
        [
            [],
            [{"type": "turn/end", "data": {"reason": {"kind": "error"}}}],
            [{"type": "turn/end", "data": {"reason": {"kind": "error", "error": "oops"}}}],
        ],
    )
    async def test_failed_turn_without_error_detail_still_raises(self, adapter, sdk, events):
        sdk.final_response = ""
        sdk.finish_reason = "error"
        sdk.events = events
        with pytest.raises(RuntimeError, match=r"finish reason: 'error'\)$"):
            await adapter.invoke("one", context_id="ctx-1", context=_context("t1"))

    @pytest.mark.asyncio
    async def test_empty_completed_response_returns_empty(self, adapter, sdk):
        sdk.final_response = ""
        assert await adapter.invoke("one", context_id="ctx-1", context=_context("t1")) == ""

    @pytest.mark.asyncio
    async def test_partial_response_is_returned_with_warning(self, adapter, sdk, caplog):
        sdk.final_response = "partial"
        sdk.finish_reason = "max-tokens"
        with caplog.at_level(logging.WARNING):
            result = await adapter.invoke("one", context_id="ctx-1", context=_context("t1"))
        assert result == "partial"
        assert "max-tokens" in caplog.text


class TestRuntimeLoss:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [
            FakeSdkProtocolError("bad frame"),
            FakeTransportClosedError("exited"),
            TimeoutError("session/prompt timed out"),
        ],
    )
    async def test_runtime_error_drops_harness_and_next_turn_relaunches(self, adapter, sdk, error):
        await adapter.invoke("one", context_id="ctx-1", context=_context("t1"))
        old = sdk.harness
        sdk.run_error = error

        with pytest.raises(type(error)):
            await adapter.invoke("two", context_id="ctx-1", context=_context("t2"))
        assert old.closed is True
        assert adapter._harness is None
        assert adapter._session_handles == {}

        await adapter.invoke("three", context_id="ctx-1", context=_context("t3"))
        assert sdk.harness is not old
        assert sdk.harness.resume_calls == ["a2a-ctx-1"]

    @pytest.mark.asyncio
    async def test_runtime_that_died_while_idle_is_replaced_before_the_turn(self, adapter, sdk):
        await adapter.invoke("one", context_id="ctx-1", context=_context("t1"))
        old = sdk.harness
        old.die()

        assert await adapter.invoke("two", context_id="ctx-1", context=_context("t2"))
        assert len(sdk.harnesses) == 2
        assert old.close_calls == 1
        assert sdk.run_calls[-1] == ("a2a-ctx-1", "two")

    @pytest.mark.asyncio
    async def test_runtime_dead_twice_in_a_row_fails(self, adapter, sdk, monkeypatch):
        original = sdk.create

        def dead_on_arrival(**kwargs):
            harness = original(**kwargs)
            harness.die()
            return harness

        monkeypatch.setattr(sys.modules["deepseek_harness"], "DeepSeekHarness", dead_on_arrival)
        with pytest.raises(RuntimeError, match="exited"):
            await adapter.invoke("one", context_id="ctx-1", context=_context("t1"))
        assert len(sdk.harnesses) == 2

    @pytest.mark.asyncio
    async def test_drop_of_old_harness_leaves_current_alone(self, adapter, sdk):
        await adapter.invoke("one", context_id="ctx-1", context=_context("t1"))
        current = sdk.harness
        stale = FakeHarness(sdk)

        await adapter._drop_harness(stale)

        assert adapter._harness is current
        assert current.closed is False
        assert adapter._session_handles

    @pytest.mark.asyncio
    async def test_turn_timeout_drops_harness(self, tmp_path, sdk):
        sdk.gates = [threading.Event()]
        adapter = DeepSeekHarnessAdapter(
            working_dir=str(tmp_path), dsh_home=str(tmp_path / "h"), timeout=0.2
        )
        with pytest.raises(TimeoutError, match="timed out"):
            await adapter.invoke("slow", context_id="ctx-1", context=_context("t1"))
        assert sdk.harnesses[0].closed is True
        assert adapter._harness is None
        assert await adapter.invoke("next", context_id="ctx-1", context=_context("t2"))
        assert len(sdk.harnesses) == 2
        await adapter.close()

    @pytest.mark.asyncio
    async def test_cancelled_coroutine_drops_harness(self, adapter, sdk):
        sdk.gates = [threading.Event()]
        task = asyncio.create_task(
            adapter.invoke("work", context_id="ctx-1", context=_context("t1"))
        )
        await _wait_started(sdk)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert sdk.harness.closed is True
        assert adapter._harness is None


class TestCancel:
    @pytest.mark.asyncio
    async def test_cancel_active_task_drops_harness(self, adapter, sdk):
        sdk.gates = [threading.Event()]
        running = asyncio.create_task(
            adapter.invoke("work", context_id="ctx-1", context=_context("t1"))
        )
        await _wait_started(sdk)

        await adapter.cancel(context=_context("t1"))

        with pytest.raises(CancelledByAdapterError):
            await running
        assert sdk.harnesses[0].closed is True
        assert await adapter.invoke("after", context_id="ctx-1", context=_context("t2"))
        assert len(sdk.harnesses) == 2

    @pytest.mark.asyncio
    async def test_cancel_during_runtime_launch_leaves_no_orphan(self, adapter, sdk):
        await adapter._ensure_harness()
        harness = sdk.harness
        harness.launch_gate = threading.Event()
        running = asyncio.create_task(
            adapter.invoke("work", context_id="ctx-1", context=_context("t1"))
        )
        assert await asyncio.to_thread(harness.launch_entered.wait, 5)

        # A hung launch must not delay the cancel itself.
        await asyncio.wait_for(adapter.cancel(context=_context("t1")), 1)
        harness.launch_gate.set()

        with pytest.raises(CancelledByAdapterError):
            await running
        assert harness.running is False
        assert harness.launches == 1
        assert sdk.run_calls == []

    @pytest.mark.asyncio
    async def test_close_during_runtime_launch_leaves_no_orphan(self, adapter, sdk):
        await adapter._ensure_harness()
        harness = sdk.harness
        harness.launch_gate = threading.Event()
        running = asyncio.create_task(
            adapter.invoke("work", context_id="ctx-1", context=_context("t1"))
        )
        assert await asyncio.to_thread(harness.launch_entered.wait, 5)

        await asyncio.wait_for(adapter.close(), 1)
        harness.launch_gate.set()

        with pytest.raises(Exception):
            await running
        assert harness.running is False
        assert sdk.run_calls == []

    @pytest.mark.asyncio
    async def test_close_during_launch_then_failed_resume_leaves_no_orphan(
        self, adapter, sdk, monkeypatch
    ):
        await adapter._ensure_harness()
        harness = sdk.harness
        harness.launch_gate = threading.Event()

        def occupied(session_id):
            harness.start()
            raise FakeJsonRpcError(-32603, f'session "{session_id}" is already owned')

        monkeypatch.setattr(harness, "resume_session", occupied)
        running = asyncio.create_task(
            adapter.invoke("work", context_id="ctx-1", context=_context("t1"))
        )
        assert await asyncio.to_thread(harness.launch_entered.wait, 5)

        await asyncio.wait_for(adapter.close(), 1)
        harness.launch_gate.set()

        with pytest.raises(Exception):
            await running
        assert harness.running is False

    @pytest.mark.asyncio
    async def test_retired_harness_is_never_relaunched(self, adapter, sdk):
        await adapter.invoke("one", context_id="ctx-1", context=_context("t1"))
        old = sdk.harness
        await adapter._drop_harness(old)

        with pytest.raises(RuntimeError, match="closed"):
            adapter._turn_job(old, {}, "a2a-ctx-2", "late", True)
        assert old.launches == 1
        assert old.running is False

    @pytest.mark.asyncio
    async def test_cancel_after_turn_finished_reports_cancelled(self, adapter, sdk, monkeypatch):
        original = adapter._run_turn

        async def run_then_cancel(*args, **kwargs):
            result = await original(*args, **kwargs)
            adapter._cancelled_tasks.add("t1")
            return result

        monkeypatch.setattr(adapter, "_run_turn", run_then_cancel)
        with pytest.raises(CancelledByAdapterError):
            await adapter.invoke("work", context_id="ctx-1", context=_context("t1"))
        assert adapter._cancelled_tasks == set()

    @pytest.mark.asyncio
    async def test_failed_close_replaces_the_worker(self, adapter, sdk, monkeypatch):
        sdk.gates = [threading.Event()]
        running = asyncio.create_task(
            adapter.invoke("work", context_id="ctx-1", context=_context("t1"))
        )
        await _wait_started(sdk)
        harness = sdk.harness
        old_executor = adapter._executor

        def broken_close():
            harness.gate.set()
            raise OSError("wait failed")

        monkeypatch.setattr(harness, "close", broken_close)
        await adapter.cancel(context=_context("t1"))

        with pytest.raises(CancelledByAdapterError):
            await running
        assert adapter._executor is not old_executor
        assert await adapter.invoke("next", context_id="ctx-1", context=_context("t2"))

    @pytest.mark.asyncio
    async def test_cancel_queued_task_leaves_harness_alone(self, adapter, sdk):
        gate = threading.Event()
        sdk.gates = [gate]
        first = asyncio.create_task(
            adapter.invoke("first", context_id="ctx-a", context=_context("t1"))
        )
        await _wait_started(sdk)
        second = asyncio.create_task(
            adapter.invoke("second", context_id="ctx-b", context=_context("t2"))
        )
        await asyncio.sleep(0)

        await adapter.cancel(context=_context("t2"))
        assert sdk.harness.closed is False

        gate.set()
        assert await first == "hello world"
        with pytest.raises(CancelledByAdapterError):
            await second
        assert sdk.run_calls == [("a2a-ctx-a", "first")]

    @pytest.mark.asyncio
    async def test_cancel_before_invoke_skips_the_turn(self, adapter, sdk):
        await adapter.cancel(context=_context("t1"))
        with pytest.raises(CancelledByAdapterError):
            await adapter.invoke("work", context_id="ctx-1", context=_context("t1"))
        assert sdk.run_calls == []

    @pytest.mark.asyncio
    async def test_cancel_without_context_is_a_no_op(self, adapter, sdk):
        await adapter.cancel()


class TestSerialization:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("second_context", ["ctx-a", "ctx-b"])
    async def test_turns_serialize_through_one_runtime(self, adapter, sdk, second_context):
        gate = threading.Event()
        sdk.gates = [gate]
        first = asyncio.create_task(
            adapter.invoke("first", context_id="ctx-a", context=_context("t1"))
        )
        await _wait_started(sdk)
        second = asyncio.create_task(
            adapter.invoke("second", context_id=second_context, context=_context("t2"))
        )
        await asyncio.sleep(0.05)

        assert [prompt for _, prompt in sdk.run_calls] == ["first"]
        assert adapter._active_task_id == "t1"

        gate.set()
        await first
        await second
        assert [prompt for _, prompt in sdk.run_calls] == ["first", "second"]
        assert adapter._active_task_id is None


class TestClose:
    @pytest.mark.asyncio
    async def test_close_closes_harness_then_executor(self, adapter, sdk):
        await adapter.invoke("one", context_id="ctx-1", context=_context("t1"))
        await adapter.close()
        assert sdk.harness.close_calls == 1
        assert adapter._executor._shutdown is True
        assert adapter._session_handles == {}

    @pytest.mark.asyncio
    async def test_invoke_after_close_raises(self, adapter):
        await adapter.close()
        with pytest.raises(RuntimeError, match="closed"):
            await adapter.invoke("one", context_id="ctx-1", context=_context("t1"))

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self, adapter, sdk):
        await adapter.invoke("one", context_id="ctx-1", context=_context("t1"))
        await adapter.close()
        await adapter.close()
        assert sdk.harness.close_calls == 1


class TestMetadataAndRegistration:
    def test_metadata_defaults(self, adapter):
        metadata = adapter.get_metadata()
        assert metadata.name == "DeepSeek Harness"
        assert metadata.description
        assert metadata.streaming is False
        assert adapter.supports_streaming() is False

    def test_metadata_overrides(self, tmp_path):
        adapter = DeepSeekHarnessAdapter(
            working_dir=str(tmp_path),
            name="Coder",
            description="desc",
            skills=[{"id": "s", "name": "S", "description": "d"}],
            agent_provider={"organization": "Org", "url": "https://example.invalid"},
            documentation_url="https://docs.invalid",
            icon_url="https://icon.invalid",
        )
        metadata = adapter.get_metadata()
        assert metadata.name == "Coder"
        assert metadata.description == "desc"
        assert metadata.skills[0]["id"] == "s"
        assert metadata.provider["organization"] == "Org"
        assert metadata.documentation_url == "https://docs.invalid"
        assert metadata.icon_url == "https://icon.invalid"

    def test_flat_imports(self):
        from a2a_adapter import DeepSeekHarnessAdapter as Flat
        from a2a_adapter.integrations import DeepSeekHarnessAdapter as Integrations

        assert Flat is DeepSeekHarnessAdapter
        assert Integrations is DeepSeekHarnessAdapter

    def test_loader(self, tmp_path):
        loaded = load_adapter({"adapter": "deepseek-harness", "working_dir": str(tmp_path)})
        assert isinstance(loaded, DeepSeekHarnessAdapter)


class TestA2AIntegration:
    @pytest.mark.asyncio
    async def test_message_send_completes_through_real_a2a_stack(self, adapter, sdk):
        app = to_a2a(adapter)
        payload = {
            "jsonrpc": "2.0",
            "id": "request-1",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "messageId": "message-1",
                    "parts": [{"kind": "text", "text": "hello dsh"}],
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
        assert sdk.run_calls[0][1] == "hello dsh"
