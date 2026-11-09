"""Tests for the v0.2 OpenClawAdapter.

Covers: invoke (subprocess execution on the base runtime, prompt via stdin),
cancel (per-task process kill, CancelledByAdapterError), timeout, session ID
sanitization, command building, output extraction, MIME detection, metadata,
and integration with to_a2a / build_agent_card / load_adapter.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from a2a_adapter.integrations.openclaw import (
    _extract_openclaw_failure_message,
    OpenClawAdapter,
    VALID_THINKING_LEVELS,
    _SESSION_ID_MAX_LEN,
)
from a2a.types import TaskState

from a2a_adapter.exceptions import CancelledByAdapterError
from a2a_adapter.server import build_agent_card, to_a2a
from a2a_adapter.loader import load_adapter


# ──── Test: Constructor Validation ────


def test_valid_thinking_levels():
    for level in VALID_THINKING_LEVELS:
        adapter = OpenClawAdapter(thinking=level)
        assert adapter.thinking == level


def test_invalid_thinking_level():
    with pytest.raises(ValueError, match="Invalid thinking level"):
        OpenClawAdapter(thinking="invalid")


def test_auto_session_id():
    adapter = OpenClawAdapter()
    assert adapter.session_id.startswith("a2a-")


def test_custom_session_id():
    adapter = OpenClawAdapter(session_id="my-session")
    assert adapter.session_id == "my-session"


# ──── Test: Session ID Sanitization ────


def test_context_id_to_session_id_none():
    adapter = OpenClawAdapter(session_id="default")
    assert adapter._context_id_to_session_id(None) == "default"


def test_context_id_to_session_id_empty():
    adapter = OpenClawAdapter(session_id="default")
    assert adapter._context_id_to_session_id("") == "default"


def test_context_id_to_session_id_valid():
    adapter = OpenClawAdapter()
    result = adapter._context_id_to_session_id("my-context-123")
    assert result == f"{adapter.session_id}-my-context-123"


def test_context_id_to_session_id_sanitizes():
    adapter = OpenClawAdapter()
    result = adapter._context_id_to_session_id("My Context ID!")
    assert result.startswith("a2a-")
    assert "!" not in result
    # Should be lowercase
    assert result == result.lower()


def test_context_id_to_session_id_truncates():
    adapter = OpenClawAdapter()
    long_id = "a" * 200
    result = adapter._context_id_to_session_id(long_id)
    assert len(result) <= _SESSION_ID_MAX_LEN


def test_context_id_to_session_id_all_invalid_chars():
    adapter = OpenClawAdapter(session_id="fallback")
    result = adapter._context_id_to_session_id("!!!")
    assert result == "fallback"


# ──── Test: Command Building ────


def test_build_command_basic():
    adapter = OpenClawAdapter(
        openclaw_path="/usr/bin/openclaw",
        thinking="medium",
    )
    cmd = adapter._build_command("hello world", "session-1").args
    assert cmd[0] == "/usr/bin/openclaw"
    assert "agent" in cmd
    assert "--local" in cmd
    assert "--json" in cmd
    assert "--session-id" in cmd
    idx = cmd.index("--session-id")
    assert cmd[idx + 1] == "session-1"
    assert "--thinking" in cmd
    idx = cmd.index("--thinking")
    assert cmd[idx + 1] == "medium"


def test_build_command_sends_message_on_stdin():
    """The prompt goes through stdin (--message-file -), never argv."""
    adapter = OpenClawAdapter()
    result = adapter._build_command("hello world", "session-1")
    assert result.stdin == "hello world"
    assert result.used_resume is False
    assert "--message" not in result.args
    assert "hello world" not in result.args
    idx = result.args.index("--message-file")
    assert result.args[idx + 1] == "-"


def test_build_command_with_agent_id():
    adapter = OpenClawAdapter(agent_id="research-agent")
    cmd = adapter._build_command("test", "session-1").args
    assert "--agent" in cmd
    idx = cmd.index("--agent")
    assert cmd[idx + 1] == "research-agent"


def test_build_command_no_agent_id():
    adapter = OpenClawAdapter(agent_id=None)
    cmd = adapter._build_command("test", "session-1").args
    assert "--agent" not in cmd


# ──── Test: Output Extraction ────


def test_extract_response_text_basic():
    output = {
        "payloads": [
            {"text": "Hello from OpenClaw"},
        ]
    }
    result = OpenClawAdapter._extract_response_text(output)
    assert result == "Hello from OpenClaw"


def test_extract_response_text_multiple_payloads():
    output = {
        "payloads": [
            {"text": "Part 1"},
            {"text": "Part 2"},
        ]
    }
    result = OpenClawAdapter._extract_response_text(output)
    assert "Part 1" in result
    assert "Part 2" in result


def test_extract_response_text_empty():
    output = {"payloads": []}
    result = OpenClawAdapter._extract_response_text(output)
    assert result == ""


def test_extract_response_text_no_payloads():
    output = {}
    result = OpenClawAdapter._extract_response_text(output)
    assert result == ""


def test_extract_response_text_skips_empty():
    output = {
        "payloads": [
            {"text": ""},
            {"text": "real content"},
        ]
    }
    result = OpenClawAdapter._extract_response_text(output)
    assert result == "real content"


# ──── Test: MIME Type Detection ────


def test_mime_type_png():
    assert OpenClawAdapter._detect_mime_type("image.png") == "image/png"


def test_mime_type_jpeg():
    assert OpenClawAdapter._detect_mime_type("photo.jpg") == "image/jpeg"
    assert OpenClawAdapter._detect_mime_type("photo.jpeg") == "image/jpeg"


def test_mime_type_pdf():
    assert OpenClawAdapter._detect_mime_type("doc.pdf") == "application/pdf"


def test_mime_type_unknown():
    assert OpenClawAdapter._detect_mime_type("file.xyz") == "application/octet-stream"


def test_mime_type_case_insensitive():
    assert OpenClawAdapter._detect_mime_type("IMAGE.PNG") == "image/png"


# ──── Test: Invoke with mocked subprocess ────

_SPAWN = "a2a_adapter.base_adapter.create_subprocess_exec"


def _ok_output(text: str) -> bytes:
    return json.dumps({"payloads": [{"text": text}], "meta": {}}).encode()


class FakeProcess:
    """Stand-in for an asyncio subprocess.

    ``communicate`` blocks until ``finish()`` or ``kill()`` is called, so a
    test can observe the process while it is running.
    """

    def __init__(self, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
        self._stdout = stdout
        self._stderr = stderr
        self._exit_code = returncode
        self.returncode: int | None = None
        self.stdin_data: bytes | None = None
        self.killed = False
        self.started = asyncio.Event()
        self._done = asyncio.Event()

    def finish(self) -> None:
        self.returncode = self._exit_code
        self._done.set()

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self._stdout = b""
        self._done.set()

    async def wait(self) -> int:
        await self._done.wait()
        return self.returncode

    async def communicate(self, input: bytes | None = None):
        self.stdin_data = input
        self.started.set()
        await self._done.wait()
        return self._stdout, self._stderr


def _finished(stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0) -> FakeProcess:
    proc = FakeProcess(stdout, stderr, returncode)
    proc.finish()
    return proc


def _ctx(task_id: str) -> MagicMock:
    ctx = MagicMock()
    ctx.task_id = task_id
    return ctx


@pytest.mark.asyncio
async def test_invoke_success():
    """Test invoke with a mocked subprocess that returns valid JSON."""
    adapter = OpenClawAdapter(session_id="test")
    proc = _finished(_ok_output("mocked response"))

    with patch(_SPAWN, AsyncMock(return_value=proc)):
        result = await adapter.invoke("test message")

    assert result == "mocked response"


@pytest.mark.asyncio
async def test_invoke_writes_prompt_to_stdin():
    adapter = OpenClawAdapter(session_id="test")
    proc = _finished(_ok_output("ok"))
    spawn = AsyncMock(return_value=proc)

    with patch(_SPAWN, spawn):
        await adapter.invoke("héllo there")

    assert proc.stdin_data == "héllo there".encode("utf-8")
    args = spawn.call_args.args
    assert "héllo there" not in args
    assert args[args.index("--message-file") + 1] == "-"


@pytest.mark.asyncio
async def test_invoke_large_prompt_stays_out_of_argv():
    """Prompts beyond the OS argv limit (~128 KiB) must not go on the command line."""
    adapter = OpenClawAdapter(session_id="test")
    big = "x" * (512 * 1024)
    proc = _finished(_ok_output("ok"))
    spawn = AsyncMock(return_value=proc)

    with patch(_SPAWN, spawn):
        assert await adapter.invoke(big) == "ok"

    assert max(len(a) for a in spawn.call_args.args) < 1024
    assert proc.stdin_data == big.encode()


@pytest.mark.asyncio
async def test_invoke_uses_context_session_id():
    adapter = OpenClawAdapter(session_id="base")
    spawn = AsyncMock(return_value=_finished(_ok_output("ok")))

    with patch(_SPAWN, spawn):
        await adapter.invoke("hi", context_id="Ctx-1")

    args = spawn.call_args.args
    assert args[args.index("--session-id") + 1] == "base-ctx-1"


@pytest.mark.asyncio
async def test_invoke_nonzero_exit():
    """Test invoke raises on non-zero exit code."""
    adapter = OpenClawAdapter(session_id="test")

    with patch(_SPAWN, AsyncMock(return_value=_finished(b"", b"error details", 1))):
        with pytest.raises(RuntimeError, match="exit code 1: error details"):
            await adapter.invoke("test")


@pytest.mark.asyncio
async def test_invoke_nonzero_exit_prefers_json_error():
    adapter = OpenClawAdapter(session_id="test")
    stdout = json.dumps({"error": "model quota exceeded"}).encode()

    with patch(_SPAWN, AsyncMock(return_value=_finished(stdout, b"warn: noise", 1))):
        with pytest.raises(RuntimeError, match="model quota exceeded"):
            await adapter.invoke("test")


@pytest.mark.asyncio
async def test_invoke_empty_output():
    """Test invoke raises on empty stdout and stderr."""
    adapter = OpenClawAdapter(session_id="test")

    with patch(_SPAWN, AsyncMock(return_value=_finished(b"", b""))):
        with pytest.raises(RuntimeError, match="empty output"):
            await adapter.invoke("test")


@pytest.mark.asyncio
async def test_invoke_uses_stderr_when_stdout_empty():
    """Test invoke uses stderr when stdout is empty (OpenClaw stderr output bug)."""
    adapter = OpenClawAdapter(session_id="test")

    with patch(_SPAWN, AsyncMock(return_value=_finished(b"", _ok_output("response from stderr")))):
        result = await adapter.invoke("test message")

    assert result == "response from stderr"


@pytest.mark.asyncio
async def test_invoke_invalid_json():
    """Test invoke raises on invalid JSON output."""
    adapter = OpenClawAdapter(session_id="test")

    with patch(_SPAWN, AsyncMock(return_value=_finished(b"not json"))):
        with pytest.raises(RuntimeError, match="does not contain JSON object"):
            await adapter.invoke("test")


@pytest.mark.asyncio
async def test_invoke_binary_not_found():
    """Test invoke raises FileNotFoundError when binary missing."""
    adapter = OpenClawAdapter(openclaw_path="/nonexistent/openclaw")

    with patch(_SPAWN, AsyncMock(side_effect=FileNotFoundError("not found"))):
        with pytest.raises(FileNotFoundError, match="/nonexistent/openclaw"):
            await adapter.invoke("test")


@pytest.mark.asyncio
async def test_invoke_runs_in_working_directory(tmp_path):
    adapter = OpenClawAdapter(working_directory=str(tmp_path))
    spawn = AsyncMock(return_value=_finished(_ok_output("ok")))

    with patch(_SPAWN, spawn):
        await adapter.invoke("hi")

    assert spawn.call_args.kwargs["cwd"] == str(tmp_path)


@pytest.mark.asyncio
async def test_invoke_passes_env_vars():
    adapter = OpenClawAdapter(env_vars={"OPENCLAW_TEST_VAR": "1"})
    spawn = AsyncMock(return_value=_finished(_ok_output("ok")))

    with patch(_SPAWN, spawn):
        await adapter.invoke("hi")

    assert spawn.call_args.kwargs["env"]["OPENCLAW_TEST_VAR"] == "1"


@pytest.mark.asyncio
async def test_timeout_kills_child():
    adapter = OpenClawAdapter(timeout=0.05)
    proc = FakeProcess(_ok_output("late"))

    with patch(_SPAWN, AsyncMock(return_value=proc)):
        with pytest.raises(RuntimeError, match="timed out"):
            await asyncio.wait_for(adapter.invoke("hi", context=_ctx("t-timeout")), 2)

    assert proc.killed
    assert "t-timeout" not in adapter._active_processes


# ──── Test: errors reported with exit code 0 ────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ({"error": "x-detail"}, "x-detail"),
        ({"error_message": "x-detail"}, "x-detail"),
        ({"message": "x-detail"}, "x-detail"),
        ({"error": {"message": "x-detail"}}, "x-detail"),
        ({"error": {"kind": "x-detail"}}, "x-detail"),
        ({"error": {"code": 429}}, "429"),
        ({"error": ["x-detail"]}, "x-detail"),
        ({"error": 429}, "429"),
        ({"error": True}, "true"),
        ({"meta": {"error": "x-detail"}}, "x-detail"),
        ({"meta": {"error": {"kind": "k", "message": "x-detail"}}}, "x-detail"),
        ({"payloads": [{"text": "partial"}], "error": "x-detail"}, "x-detail"),
        ({"payloads": [{"text": "x-detail", "isError": True}]}, "x-detail"),
        (
            {"payloads": [{"text": "x-detail", "isError": True}, {"isError": True}]},
            "x-detail",
        ),
        ({"payloads": [{"isError": True}]}, "marked as an error"),
    ],
)
async def test_invoke_exit_zero_failure_raises(output, expected):
    adapter = OpenClawAdapter(session_id="test")

    with patch(_SPAWN, AsyncMock(return_value=_finished(json.dumps(output).encode()))):
        with pytest.raises(RuntimeError, match=f"reported a failure: .*{expected}"):
            await adapter.invoke("test")


@pytest.mark.asyncio
async def test_invoke_exit_zero_error_on_stderr_raises():
    adapter = OpenClawAdapter(session_id="test")
    stderr = json.dumps({"error": "x-detail"}).encode()

    with patch(_SPAWN, AsyncMock(return_value=_finished(b"", stderr))):
        with pytest.raises(RuntimeError, match="x-detail"):
            await adapter.invoke("test")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ({"payloads": [{"text": "there was an error"}]}, "there was an error"),
        ({"payloads": [{"text": "Error: file not found"}]}, "Error: file not found"),
        ({"payloads": [{"text": '{"error": "x"}'}]}, '{"error": "x"}'),
        ({"error": None, "payloads": [{"text": "answer"}]}, "answer"),
        ({"error": "", "payloads": [{"text": "answer"}]}, "answer"),
        ({"error": "   ", "payloads": [{"text": "answer"}]}, "answer"),
        ({"error": False, "payloads": [{"text": "answer"}]}, "answer"),
        ({"error": 0, "payloads": [{"text": "answer"}]}, "answer"),
        ({"error": {}, "payloads": [{"text": "answer"}]}, "answer"),
        ({"error": [], "payloads": [{"text": "answer"}]}, "answer"),
        ({"meta": {"error": None}, "payloads": [{"text": "answer"}]}, "answer"),
        ({"message": "note", "payloads": [{"text": "answer"}]}, "answer"),
        (
            {"payloads": [{"text": "warned", "isError": True}, {"text": "answer"}]},
            "warned\n\nanswer",
        ),
        ({"payloads": [{"text": "answer", "isError": "true"}]}, "answer"),
        ({"message": {"a": 1}}, ""),
        ({"message": True}, ""),
        ({"payloads": [], "meta": {"stopReason": "end"}}, ""),
    ],
)
async def test_invoke_exit_zero_success_returns_text(output, expected):
    adapter = OpenClawAdapter(session_id="test")

    with patch(_SPAWN, AsyncMock(return_value=_finished(json.dumps(output).encode()))):
        assert await adapter.invoke("test") == expected


@pytest.mark.asyncio
async def test_invoke_json_array_raises():
    adapter = OpenClawAdapter(session_id="test")

    with patch(_SPAWN, AsyncMock(return_value=_finished(b'["a"]'))):
        with pytest.raises(RuntimeError, match="unexpected JSON"):
            await adapter.invoke("test")


@pytest.mark.asyncio
@pytest.mark.parametrize("payloads", [None, "not-a-list", [1, {"text": "answer"}]])
async def test_invoke_malformed_payloads_raise(payloads):
    adapter = OpenClawAdapter(session_id="test")
    stdout = json.dumps({"payloads": payloads}).encode()

    with patch(_SPAWN, AsyncMock(return_value=_finished(stdout))):
        with pytest.raises(RuntimeError, match="unexpected JSON"):
            await adapter.invoke("test")


@pytest.mark.parametrize(
    "output",
    [
        {"message": "note", "payloads": [{"mediaUrl": "https://example.test/a.png"}]},
        {"message": "note", "payloads": [{"mediaUrls": ["https://example.test/a.png"]}]},
        {"message": "note", "payloads": [1, {"text": "answer"}]},
    ],
)
def test_message_does_not_outrank_a_visible_payload(output):
    assert _extract_openclaw_failure_message(output) is None


@pytest.mark.parametrize(
    "payloads",
    ["not-a-list", [], [1, "x"], [{"text": "   "}], [{"mediaUrl": None, "mediaUrls": []}]],
)
def test_message_is_the_failure_without_a_visible_payload(payloads):
    assert _extract_openclaw_failure_message({"message": "x", "payloads": payloads}) == "x"


def test_error_object_without_readable_fields_is_truncated():
    message = _extract_openclaw_failure_message({"error": {"detail": "y" * 5000}})
    assert message.startswith('{"detail"')
    assert len(message) < 1000


def test_structured_error_text_stays_readable():
    message = _extract_openclaw_failure_message({"error": {"detail": "配额不足"}})
    assert "配额不足" in message


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ({"error": {"message": "from object"}}, "from object"),
        ({"message": "quota", "payloads": [{"text": "partial"}]}, "quota"),
    ],
)
def test_extract_error_on_nonzero_exit(output, expected):
    adapter = OpenClawAdapter(session_id="test")
    assert adapter._extract_error(json.dumps(output)) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("output", "final", "absent"),
    [
        (
            {"error": "x-detail"},
            TaskState.TASK_STATE_FAILED,
            TaskState.TASK_STATE_COMPLETED,
        ),
        (
            {"payloads": [{"text": "fine"}]},
            TaskState.TASK_STATE_COMPLETED,
            TaskState.TASK_STATE_FAILED,
        ),
    ],
)
async def test_exit_zero_result_through_executor(output, final, absent):
    from a2a_adapter.executor import AdapterAgentExecutor

    executor = AdapterAgentExecutor(OpenClawAdapter())
    ctx = MagicMock()
    ctx.task_id = "t-exec"
    ctx.context_id = "c-exec"
    ctx.current_task = None
    ctx.get_user_input.return_value = "hi"
    queue = AsyncMock()

    proc = _finished(json.dumps(output).encode())
    with patch(_SPAWN, AsyncMock(return_value=proc)):
        await asyncio.wait_for(executor.execute(ctx, queue), 2)

    states = [
        call.args[0].status.state
        for call in queue.enqueue_event.call_args_list
        if hasattr(call.args[0], "status")
    ]
    assert states[-1] == final
    assert absent not in states


# ──── Test: Cancel ────


@pytest.mark.asyncio
async def test_cancel_running_task_raises_cancelled_by_adapter():
    """Cancel kills the child and invoke raises CancelledByAdapterError (not RuntimeError)."""
    adapter = OpenClawAdapter()
    proc = FakeProcess(_ok_output("never"))

    with patch(_SPAWN, AsyncMock(return_value=proc)):
        run = asyncio.create_task(adapter.invoke("hi", context=_ctx("t-1")))
        await asyncio.wait_for(proc.started.wait(), 2)
        await adapter.cancel(context=_ctx("t-1"))
        with pytest.raises(CancelledByAdapterError, match="killed by cancel"):
            await asyncio.wait_for(run, 2)

    assert proc.killed


@pytest.mark.asyncio
async def test_cancel_through_executor_reports_canceled_not_failed():
    """End to end through AdapterAgentExecutor: tasks/cancel yields a canceled task."""
    from a2a_adapter.executor import AdapterAgentExecutor

    adapter = OpenClawAdapter()
    executor = AdapterAgentExecutor(adapter)
    proc = FakeProcess(_ok_output("never"))
    ctx = MagicMock()
    ctx.task_id = "t-exec"
    ctx.context_id = "c-exec"
    ctx.current_task = None
    ctx.get_user_input.return_value = "hi"
    queue = AsyncMock()

    with patch(_SPAWN, AsyncMock(return_value=proc)):
        run = asyncio.create_task(executor.execute(ctx, queue))
        await asyncio.wait_for(proc.started.wait(), 2)
        await executor.cancel(ctx, queue)
        await asyncio.wait_for(run, 2)

    states = [
        call.args[0].status.state
        for call in queue.enqueue_event.call_args_list
        if hasattr(call.args[0], "status")
    ]
    assert TaskState.TASK_STATE_CANCELED in states
    assert TaskState.TASK_STATE_FAILED not in states


@pytest.mark.asyncio
async def test_concurrent_invokes_track_independent_processes():
    """Two in-flight tasks keep their own process handle; cancel hits only its own."""
    adapter = OpenClawAdapter()
    proc_a = FakeProcess(_ok_output("from a"))
    proc_b = FakeProcess(_ok_output("from b"))

    with patch(_SPAWN, AsyncMock(side_effect=[proc_a, proc_b])):
        run_a = asyncio.create_task(adapter.invoke("a", context_id="ctx-a", context=_ctx("task-a")))
        await asyncio.wait_for(proc_a.started.wait(), 2)
        run_b = asyncio.create_task(adapter.invoke("b", context_id="ctx-b", context=_ctx("task-b")))
        await asyncio.wait_for(proc_b.started.wait(), 2)

        assert adapter._active_processes["task-a"] is proc_a
        assert adapter._active_processes["task-b"] is proc_b

        await adapter.cancel(context=_ctx("task-a"))
        assert proc_a.killed
        assert not proc_b.killed

        proc_b.finish()
        assert await asyncio.wait_for(run_b, 2) == "from b"
        with pytest.raises(CancelledByAdapterError):
            await asyncio.wait_for(run_a, 2)


@pytest.mark.asyncio
async def test_same_session_runs_one_at_a_time():
    """Two invokes for one OpenClaw session must not overlap."""
    adapter = OpenClawAdapter()
    proc_a = FakeProcess(_ok_output("first"))
    proc_b = _finished(_ok_output("second"))
    spawn = AsyncMock(side_effect=[proc_a, proc_b])

    with patch(_SPAWN, spawn):
        run_a = asyncio.create_task(adapter.invoke("a", context_id="same", context=_ctx("task-a")))
        await asyncio.wait_for(proc_a.started.wait(), 2)
        run_b = asyncio.create_task(adapter.invoke("b", context_id="same", context=_ctx("task-b")))
        for _ in range(5):
            await asyncio.sleep(0)
        assert spawn.call_count == 1

        proc_a.finish()
        assert await asyncio.wait_for(run_a, 2) == "first"
        assert await asyncio.wait_for(run_b, 2) == "second"

    assert spawn.call_count == 2


@pytest.mark.asyncio
async def test_cancel_queued_task_before_start():
    adapter = OpenClawAdapter()
    await adapter.cancel(context=_ctx("t-q"))
    spawn = AsyncMock()

    with patch(_SPAWN, spawn):
        with pytest.raises(CancelledByAdapterError, match="before execution"):
            await adapter.invoke("hi", context=_ctx("t-q"))

    spawn.assert_not_called()


@pytest.mark.asyncio
async def test_cancel_without_context_is_noop():
    adapter = OpenClawAdapter()
    await adapter.cancel()  # Should not raise


@pytest.mark.asyncio
async def test_cancel_already_finished():
    """Cancel does not kill a process that already exited."""
    adapter = OpenClawAdapter()
    proc = _finished()
    proc.kill = MagicMock()
    adapter._active_processes["t-done"] = proc
    await adapter.cancel(context=_ctx("t-done"))
    proc.kill.assert_not_called()


# ──── Test: Close ────


@pytest.mark.asyncio
async def test_close_kills_running_processes():
    adapter = OpenClawAdapter()
    proc = FakeProcess()
    adapter._active_processes["t-close"] = proc
    await adapter.close()
    assert proc.killed
    assert adapter._active_processes == {}


# ──── Test: Streaming NOT supported ────


def test_no_streaming():
    adapter = OpenClawAdapter()
    assert adapter.supports_streaming() is False


# ──── Test: Metadata ────


def test_metadata_defaults():
    adapter = OpenClawAdapter()
    meta = adapter.get_metadata()
    assert meta.name == "OpenClaw"
    assert meta.streaming is False


def test_metadata_custom():
    adapter = OpenClawAdapter(
        name="My OpenClaw Agent",
        description="Does everything",
    )
    meta = adapter.get_metadata()
    assert meta.name == "My OpenClaw Agent"
    assert meta.description == "Does everything"


# ──── Test: Integration with server layer ────


def test_build_agent_card():
    adapter = OpenClawAdapter(
        name="OpenClaw Agent",
        description="A test agent",
    )
    card = build_agent_card(adapter, url="http://localhost:9005")
    assert card.name == "OpenClaw Agent"
    assert card.capabilities.streaming is False


def test_to_a2a_builds_app():
    adapter = OpenClawAdapter()
    app = to_a2a(adapter)
    assert app is not None


def test_load_adapter_openclaw():
    adapter = load_adapter(
        {
            "adapter": "openclaw",
            "session_id": "test-session",
            "thinking": "high",
            "timeout": 120,
        }
    )
    assert isinstance(adapter, OpenClawAdapter)
    assert adapter.session_id == "test-session"
    assert adapter.thinking == "high"
    assert adapter.timeout == 120


# ──── Test: Flat import ────


def test_flat_import():
    from a2a_adapter import OpenClawAdapter as OC

    assert OC is OpenClawAdapter


# ──── Test: Async context manager ────


@pytest.mark.asyncio
async def test_context_manager():
    async with OpenClawAdapter() as adapter:
        assert isinstance(adapter, OpenClawAdapter)
